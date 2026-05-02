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

# Détection automatique du modèle fine-tuné (Sprint 3)
# LEXAI_FORCE_OPENAI=1 pour forcer OpenAI (comparaison équitable Sprint 2)
_FORCE_OPENAI = os.getenv("LEXAI_FORCE_OPENAI", "").lower() in ("1", "true", "yes")

if _FORCE_OPENAI:
    FINETUNED_MODEL_PATH = None
    _USE_FINETUNED = False
else:
    # Priorité : var d'env (HF Hub ou local) > local repo > Desktop
    _env_model = os.getenv("LEXAI_EMBEDDING_MODEL", "")
    _local_candidates = [p for p in ["./lexai-embeddings", r"C:\Users\bjgyebre\Desktop\lexai-embeddings"] if os.path.exists(p)]
    if _env_model:
        FINETUNED_MODEL_PATH = _env_model  # peut être un nom HF Hub ex: "DrProfessor/lexai-embeddings"
    elif _local_candidates:
        FINETUNED_MODEL_PATH = _local_candidates[0]
    else:
        FINETUNED_MODEL_PATH = None
    _USE_FINETUNED = FINETUNED_MODEL_PATH is not None

FAISS_PERSIST_DIR  = "./faiss_index_v3" if _USE_FINETUNED else "./faiss_index"
EMBEDDING_MODEL    = "text-embedding-3-small"  # fallback OpenAI si modèle fine-tuné absent
LLM_MODEL          = "gpt-4o-mini"
CHUNK_MAX_TOKENS   = 800
CHUNK_SIZE_CHARS   = 2400
CHUNK_OVERLAP_CHARS = 400
BM25_WEIGHT        = 0.35
VECTOR_WEIGHT      = 0.65
TOP_K              = 5
RERANKER_MODEL     = "BAAI/bge-reranker-base"
RERANKER_TOP_K     = 3

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
    print(f"[+] {nb_articles} articles -> {len(documents)} chunks apres chunking.")
    return documents


def _creer_embeddings():
    """Retourne le bon objet embeddings selon l'environnement."""
    if _USE_FINETUNED:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            from langchain_huggingface import HuggingFaceEmbeddings
        print(f"[+] Embeddings : modele fine-tune LexAI Sprint 3 ({FINETUNED_MODEL_PATH})")
        return HuggingFaceEmbeddings(
            model_name=FINETUNED_MODEL_PATH,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    print(f"[+] Embeddings : OpenAI {EMBEDDING_MODEL} (modele fine-tune non detecte)")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=os.getenv("OPENAI_API_KEY"))


def construire_vectorstore(documents: list[Document]) -> FAISS:
    """
    FAISS persistant sur disque.
    Sprint 3 : utilise le modèle fine-tuné (384 dims) si disponible, sinon OpenAI (1536 dims).
    """
    embeddings = _creer_embeddings()
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
            print(f"[+] FAISS charge depuis le disque ({nb_existants} chunks).")
            return vs
        print(f"[~] Corpus modifie ({nb_existants} -> {len(documents)} chunks). Reindexation...")

    vs = FAISS.from_documents(documents=documents, embedding=embeddings)
    os.makedirs(FAISS_PERSIST_DIR, exist_ok=True)
    vs.save_local(FAISS_PERSIST_DIR)
    with open(count_file, "w") as f:
        f.write(str(len(documents)))
    print(f"[+] FAISS cree et persiste ({len(documents)} chunks).")
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

    def invoke(self, question: str, code_filtre: str = None, top_k: int = TOP_K,
               query_vectorielle: str = None) -> list[Document]:
        # query_vectorielle : si HyDE actif, contient le doc hypothétique à la place de la question
        query_vec = query_vectorielle or question
        # 1. Recherche vectorielle (post-filtrage si code_filtre)
        fetch_k = top_k * 4 if code_filtre else top_k * 2
        vec_results = self.vs.similarity_search_with_score(query_vec, k=fetch_k)

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

        bm25_ranked = np.argsort(bm25_scores)[::-1][:top_k * 2]

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

        top_indices = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [self.documents[i] for i in top_indices]


# ── MODULE 10 — HyDE : Hypothetical Document Embeddings ─────────────────────

class HyDEGenerator:
    """
    Hypothetical Document Embeddings — Gao et al., 2022 (arXiv:2212.10496)

    Principe : au lieu d'embedder la question brute, le LLM génère d'abord
    un 'document hypothétique' (réponse idéale fictive dans le style du corpus).
    Ce document est ensuite embedé pour la recherche vectorielle FAISS.

    Avantage : le document hypothétique est stylistiquement proche des vrais
    articles de loi → meilleure similarité cosinus → meilleur recall.
    Le BM25 continue sur la question originale (complémentarité).
    """

    PROMPT_HYDE = """Tu es un expert en droit français. Génère un court passage (3-4 phrases) dans le style officiel d'un article de loi français qui répondrait directement à cette question juridique. Utilise un langage juridique formel et précis. Ne cite pas d'articles réels.

Question : {question}

Article hypothétique :"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.5,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def generer(self, question: str) -> str:
        """Génère un document hypothétique dans le style d'un article de loi."""
        prompt = self.PROMPT_HYDE.format(question=question)
        return self.llm.invoke(prompt).content


# ── MODULE 11 — Cross-Encoder Reranking (BGE-Reranker) ───────────────────────

class CrossEncoderReranker:
    """
    Reranker Cross-Encoder : re-score les Top-N docs avec attention croisée.
    Modèle : BAAI/bge-reranker-base (open-source, tourne en local).

    Contrairement aux bi-encoders (FAISS), le cross-encoder analyse la question
    et le document ENSEMBLE (attention croisée) → scores de pertinence plus fins.
    Pipeline : Top-10 hybrid → CrossEncoder → Top-3 meilleurs.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        from sentence_transformers import CrossEncoder
        print(f"[+] Chargement reranker : {model_name}")
        self.model = CrossEncoder(model_name)
        print(f"[+] Reranker pret.")

    def rerank(self, question: str, documents: list[Document], top_k: int = RERANKER_TOP_K) -> list[Document]:
        """Re-score les documents et retourne les top_k meilleurs."""
        if not documents:
            return documents
        pairs  = [(question, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]


# ── MODULE 3+6 — Chaîne RAG avec prompt structuré ────────────────────────────

PROMPT_JURIDIQUE = PromptTemplate(
    input_variables=["context", "question", "langue_instruction"],
    template="""You are LexAI, an expert legal assistant specialized in French law.
Absolute rule: every statement must be backed by a precisely cited article of law.
If the answer is not in the provided context, say so clearly without inventing anything.

Legal context (retrieved articles — source texts are in French):
{context}

Question: {question}

{langue_instruction}
"""
)

LANGUE_INSTRUCTIONS = {
    "fr": """Réponds entièrement en français en utilisant cette structure :
**Base légale** : [Code + Article applicable]
**Texte applicable** : [Citation exacte de l'article]
**Analyse juridique** : [Explication et application à la question]
**Conclusion** : [Réponse directe à la question]""",

    "en": """Answer entirely in English using this structure:
**Legal basis**: [Code + applicable Article, e.g. 'Article L1225-5 of the Code du Travail']
**Applicable text**: [ENGLISH TRANSLATION of the French article text — do NOT copy the French, translate it]
**Legal analysis**: [Explanation and application to the question, in English]
**Conclusion**: [Direct answer to the question, in English]""",
}

def creer_chaine_rag(vectorstore: FAISS, documents: list[Document],
                     use_reranking: bool = False, use_hyde: bool = False):
    """
    Crée la chaîne RAG avec hybrid retriever (BM25 + vectoriel).

    use_reranking=True : Cross-Encoder BGE-Reranker — Top-10 → Top-3
    use_hyde=True      : HyDE — génère un doc hypothétique avant la recherche FAISS
    Les deux sont combinables : HyDE améliore le recall, Reranking améliore la précision.
    """
    hybrid   = HybridRetriever(vectorstore, documents)
    reranker = CrossEncoderReranker() if use_reranking else None
    hyde     = HyDEGenerator() if use_hyde else None

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

        # HyDE : génère doc hypothétique pour la recherche vectorielle FAISS
        # BM25 continue sur la question originale (complémentarité)
        hyde_doc = hyde.generer(question) if hyde else None

        # Avec reranking : fetch Top-10 puis rerank → Top-3
        # Sans reranking : fetch Top-5 direct (baseline)
        fetch_k = TOP_K * 2 if reranker else TOP_K
        docs    = hybrid.invoke(question, inp.get("code_filtre"),
                                top_k=fetch_k, query_vectorielle=hyde_doc)
        if reranker:
            docs = reranker.rerank(question, docs)

        return {
            "context":            formater_docs(docs),
            "question":           question,
            "langue_instruction": LANGUE_INSTRUCTIONS.get(langue, LANGUE_INSTRUCTIONS["fr"]),
        }

    chaine = (
        RunnableLambda(preparer_input)
        | PROMPT_JURIDIQUE
        | llm
        | StrOutputParser()
    )
    return chaine, hybrid, reranker, hyde


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
    chaine, hybrid, _, _ = creer_chaine_rag(vectorstore, documents)

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
