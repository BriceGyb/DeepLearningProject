"""
LexAI - Pipeline RAG Juridique
Architecture : Nettoyage → Chunking intelligent → FAISS persistant
               → Hybrid Search (BM25 + vectoriel) → GPT-4o-mini
"""

import json
import os
import re
import unicodedata
import hashlib
import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import tiktoken

load_dotenv()

FAISS_PERSIST_DIR  = "./faiss_index"
EMBEDDING_MODEL    = "text-embedding-3-small"
LLM_MODEL          = "gpt-4o-mini"
CHUNK_MAX_TOKENS   = 800
CHUNK_SIZE_CHARS   = 2400
CHUNK_OVERLAP_CHARS = 400
BM25_WEIGHT        = 0.35
VECTOR_WEIGHT      = 0.65
TOP_K              = 5

# ── MODULE 2 — Nettoyage / Normalisation ──────────────────────────────────────

class LegalTextCleaner:
    """Nettoie les textes juridiques bruts."""

    PATTERNS = [
        r"Nota\s*:.*?(?=\n\n|\Z)",
        r"Version\s+en\s+vigueur\s+du.*",
        r"Liens\s+relatifs.*",
        r"<[^>]+>",
    ]

    def nettoyer(self, texte: str) -> str:
        for pattern in self.PATTERNS:
            texte = re.sub(pattern, "", texte, flags=re.DOTALL | re.IGNORECASE)
        texte = re.sub(r"\s+", " ", texte).strip()
        texte = unicodedata.normalize("NFC", texte)
        return texte

    def hash(self, texte: str) -> str:
        return hashlib.sha256(texte.encode("utf-8")).hexdigest()[:16]


# ── MODULE 3 — Chunking Intelligent ──────────────────────────────────────────

class LegalChunker:
    """
    Chunking sémantique par article :
    - Article court (< CHUNK_MAX_TOKENS) → 1 chunk entier
    - Article long → split récursif par alinéa avec overlap
    """

    TOKENIZER = tiktoken.get_encoding("cl100k_base")
    SPLITTER  = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    def compter_tokens(self, texte: str) -> int:
        return len(self.TOKENIZER.encode(texte))

    def chunker(self, article: dict, texte_nettoye: str) -> list[Document]:
        meta = {
            "id":      article["id"],
            "code":    article["code"],
            "article": article["article"],
            "domaine": article["domaine"],
            "url":     f"https://legifrance.gouv.fr/search/code?query={article['article']}",
        }
        prefixe = f"[{article['code']} — {article['article']}]\n"

        if self.compter_tokens(texte_nettoye) <= CHUNK_MAX_TOKENS:
            contenu = prefixe + f"Domaine : {article['domaine']}\n\n" + texte_nettoye
            return [Document(page_content=contenu, metadata={**meta, "chunk": 0, "nb_chunks": 1})]

        morceaux = self.SPLITTER.split_text(texte_nettoye)
        return [
            Document(
                page_content=prefixe + morceau,
                metadata={**meta, "chunk": i, "nb_chunks": len(morceaux)},
            )
            for i, morceau in enumerate(morceaux)
        ]


# ── MODULE 4 — Vectorisation persistante ─────────────────────────────────────

cleaner = LegalTextCleaner()
chunker = LegalChunker()

def charger_corpus(chemin_json: str) -> list[Document]:
    """Charge, nettoie et chunke tous les articles du JSON."""
    with open(chemin_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for article in data["corpus_juridique"]:
        texte_propre = cleaner.nettoyer(article["texte"])
        chunks = chunker.chunker(article, texte_propre)
        documents.extend(chunks)

    nb_articles = len(data["corpus_juridique"])
    print(f"[+] {nb_articles} articles → {len(documents)} chunks après chunking.")
    return documents


def construire_vectorstore(documents: list[Document]) -> FAISS:
    """
    FAISS persistant sur disque.
    Charge depuis le disque si l'index existe, sinon crée et sauvegarde.
    """
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    count_file = os.path.join(FAISS_PERSIST_DIR, "chunk_count.txt")

    if os.path.exists(FAISS_PERSIST_DIR) and os.path.exists(count_file):
        with open(count_file, "r") as f:
            nb_existants = int(f.read().strip())
        if nb_existants == len(documents):
            vs = FAISS.load_local(
                FAISS_PERSIST_DIR,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"[+] Vectorstore FAISS chargé depuis le disque ({nb_existants} chunks — 0 appel API).")
            return vs
        print(f"[~] Corpus modifié ({nb_existants} → {len(documents)} chunks). Réindexation...")

    vs = FAISS.from_documents(documents=documents, embedding=embeddings)
    os.makedirs(FAISS_PERSIST_DIR, exist_ok=True)
    vs.save_local(FAISS_PERSIST_DIR)
    with open(count_file, "w") as f:
        f.write(str(len(documents)))
    print(f"[+] Vectorstore FAISS créé et persisté ({len(documents)} chunks).")
    return vs


# ── MODULE 9 — Hybrid Search BM25 + Vectoriel ────────────────────────────────

class HybridRetriever:
    """
    Reciprocal Rank Fusion entre BM25 (lexical) et vectoriel (sémantique).
    """

    def __init__(self, vectorstore: FAISS, documents: list[Document]):
        self.vs = vectorstore
        self.documents = documents
        textes_tok = [d.page_content.lower().split() for d in documents]
        self.bm25 = BM25Okapi(textes_tok)

    def _rrf(self, rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    def invoke(self, question: str, code_filtre: str = None) -> list[Document]:
        # 1. Recherche vectorielle (post-filtrage si code_filtre)
        fetch_k = TOP_K * 4 if code_filtre else TOP_K * 2
        vec_results = self.vs.similarity_search_with_score(question, k=fetch_k)

        if code_filtre:
            vec_results = [
                (doc, score) for doc, score in vec_results
                if doc.metadata.get("code") == code_filtre
            ]

        # 2. Recherche BM25
        tokens = question.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)

        if code_filtre:
            for i, doc in enumerate(self.documents):
                if doc.metadata.get("code") != code_filtre:
                    bm25_scores[i] = 0.0

        bm25_ranked = np.argsort(bm25_scores)[::-1][:TOP_K * 2]

        # 3. RRF fusion
        scores: dict[int, float] = {}

        for rank, (doc, _score) in enumerate(vec_results):
            idx = next(
                (i for i, d in enumerate(self.documents) if d.page_content == doc.page_content),
                None,
            )
            if idx is not None:
                scores[idx] = scores.get(idx, 0) + VECTOR_WEIGHT * self._rrf(rank)

        for rank, idx in enumerate(bm25_ranked):
            scores[idx] = scores.get(idx, 0) + BM25_WEIGHT * self._rrf(rank)

        top_indices = sorted(scores, key=scores.get, reverse=True)[:TOP_K]
        return [self.documents[i] for i in top_indices]


# ── MODULE 3+6 — Chaîne RAG avec prompt structuré ────────────────────────────

PROMPT_JURIDIQUE = PromptTemplate(
    input_variables=["context", "question", "langue_instruction"],
    template="""You are LexAI, an expert legal assistant specialized in French law.
Absolute rule: every statement must be backed by a precisely cited article of law.
If the answer is not in the provided context, say so clearly without inventing anything.
{langue_instruction}

Legal context (retrieved articles — source texts are in French):
{context}

Question: {question}

Respond using this structure:
**Legal basis**: [Code + applicable Article]
**Applicable text**: [Exact quote from the article]
**Legal analysis**: [Explanation and application to the question]
**Conclusion**: [Direct answer to the question]
"""
)

LANGUE_INSTRUCTIONS = {
    "fr": "Réponds entièrement en français.",
    "en": "Answer entirely in English. Translate the legal article citations into English while keeping the original French article references (e.g. 'Article 1240 of the Civil Code').",
}

def creer_chaine_rag(vectorstore: FAISS, documents: list[Document]):
    """Crée la chaîne RAG avec hybrid retriever (BM25 + vectoriel)."""
    hybrid = HybridRetriever(vectorstore, documents)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    def formater_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def preparer_input(inp: dict) -> dict:
        question = inp["question"]
        langue   = inp.get("langue", "fr")
        return {
            "context":            formater_docs(hybrid.invoke(question, inp.get("code_filtre"))),
            "question":           question,
            "langue_instruction": LANGUE_INSTRUCTIONS.get(langue, LANGUE_INSTRUCTIONS["fr"]),
        }

    chaine = (
        RunnableLambda(preparer_input)
        | PROMPT_JURIDIQUE
        | llm
        | StrOutputParser()
    )
    return chaine, hybrid


# ── Interface CLI ──────────────────────────────────────────────────────────────

def afficher_reponse(reponse: str, sources: list[Document]):
    print("\n" + "=" * 60)
    print("REPONSE LEXAI")
    print("=" * 60)
    print(reponse)
    print("\n--- Sources utilisées (hybrid BM25 + vectoriel) ---")
    vus = set()
    for doc in sources:
        cle = doc.metadata["article"]
        if cle not in vus:
            print(f"  • {doc.metadata['code']} — {doc.metadata['article']} ({doc.metadata['domaine']})")
            vus.add(cle)
    print("=" * 60 + "\n")


def main():
    print("=" * 60)
    print("  LexAI — RAG Juridique (Hybrid Search + FAISS)")
    print("=" * 60)

    documents   = charger_corpus("lois_francaises.json")
    vectorstore = construire_vectorstore(documents)
    chaine, hybrid = creer_chaine_rag(vectorstore, documents)

    print("\n[LexAI] Prêt. Tapez 'quit' pour quitter.\n")

    while True:
        question = input("Question > ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Au revoir.")
            break
        if not question:
            continue

        sources = hybrid.invoke(question)
        reponse = chaine.invoke(question)
        afficher_reponse(reponse, sources)


if __name__ == "__main__":
    main()
