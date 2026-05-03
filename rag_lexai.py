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


# ── MODULE — Rédaction de Plainte ────────────────────────────────────────────

TYPE_LITIGE_CONFIG = {
    "Pénal": {
        "destinataire": "Monsieur le Procureur de la République\nprès le Tribunal Judiciaire de [VOTRE VILLE]",
        "destinataire_court": "Monsieur le Procureur de la République",
        "etapes": [
            "**Option 1 — Dépôt physique** : Rendez-vous dans n'importe quel commissariat ou gendarmerie (ils sont tenus de prendre votre plainte).",
            "**Option 2 — Pré-plainte en ligne** : Déposez d'abord sur [pre-plainte.interieur.gouv.fr](https://www.pre-plainte.fr) puis confirmez en personne.",
            "**Option 3 — Courrier recommandé** : Envoyez directement au Procureur de la République du tribunal de votre lieu de résidence.",
            "**Délais de prescription** : 1 an (contraventions) · 6 ans (délits) · 20 ans (crimes).",
        ],
        "code_filtre": "Code Pénal",
        "icone": "🚔",
    },
    "Civil / Contractuel": {
        "destinataire": "Monsieur le Président\ndu Tribunal Judiciaire de [VOTRE VILLE]",
        "destinataire_court": "Monsieur le Président",
        "etapes": [
            "**1. Mise en demeure préalable** : Envoyez d'abord une lettre recommandée AR à la partie adverse (obligatoire avant toute saisine judiciaire).",
            "**2a. Montant ≤ 5 000€** : Saisine simplifiée par formulaire Cerfa n°11764 auprès du tribunal judiciaire.",
            "**2b. Montant > 5 000€** : Assignation en justice via un huissier de justice.",
            "**Médiation** : Tentative de résolution amiable souvent exigée avant le juge — renseignez-vous auprès du tribunal.",
        ],
        "code_filtre": "Code Civil",
        "icone": "⚖️",
    },
    "Travail": {
        "destinataire": "Monsieur le Président\ndu Conseil de Prud'hommes de [VOTRE VILLE]",
        "destinataire_court": "Monsieur le Président du Conseil de Prud'hommes",
        "etapes": [
            "**1. Saisine en ligne** : Formulaire disponible sur [justice.fr](https://www.justice.fr) → section Prud'hommes.",
            "**2. Audience de conciliation** : La procédure commence toujours par une tentative de conciliation obligatoire.",
            "**3. Délais impératifs** : 1 an pour contester un licenciement · 3 ans pour salaires impayés · 5 ans pour harcèlement/discrimination.",
            "**Assistance** : Vous pouvez être accompagné d'un délégué syndical ou d'un avocat lors des audiences.",
        ],
        "code_filtre": "Code du Travail",
        "icone": "👷",
    },
    "Consommation": {
        "destinataire": "[Nom et adresse de l'entreprise concernée]",
        "destinataire_court": "le responsable du service client",
        "etapes": [
            "**1. Lettre recommandée AR** : Envoyez cette plainte à l'entreprise en recommandé avec accusé de réception (premier recours obligatoire).",
            "**2. Médiation** : Sans réponse sous 60 jours → saisissez le médiateur sectoriel sur [mediation-conso.fr](https://www.mediation-conso.fr).",
            "**3. Signalement DGCCRF** : Alertez les autorités de consommation via [signal.conso.gouv.fr](https://signal.conso.gouv.fr).",
            "**4. Justice** : En dernier recours, tribunal judiciaire ou juge de proximité pour les montants < 5 000€.",
        ],
        "code_filtre": "Code de la Consommation",
        "icone": "🛒",
    },
    "Commerce / Affaires": {
        "destinataire": "Monsieur le Président\ndu Tribunal de Commerce de [VOTRE VILLE]",
        "destinataire_court": "Monsieur le Président du Tribunal de Commerce",
        "etapes": [
            "**1. Mise en demeure** : Lettre recommandée AR à la partie adverse avant toute action judiciaire.",
            "**2. Injonction de payer** : Pour créances non contestées, formulaire Cerfa sur justice.fr (procédure rapide et peu coûteuse).",
            "**3. Assignation** : Pour litiges complexes, via huissier de justice devant le tribunal de commerce.",
            "**Médiation commerciale** : Proposée par la plupart des tribunaux de commerce avant l'audience de jugement.",
        ],
        "code_filtre": "Code de Commerce",
        "icone": "💼",
    },
}

PROMPT_PLAINTE = PromptTemplate(
    input_variables=[
        "type_litige", "partie_adverse", "date_faits", "faits",
        "prejudice", "demarches", "destinataire", "context",
        "nom", "adresse", "ville", "date_jour",
    ],
    template="""Tu es LexAI, un assistant juridique expert en droit français. Rédige une plainte officielle formelle et juridiquement fondée.

=== ARTICLES DE LOI APPLICABLES (extraits du corpus Légifrance par recherche RAG) ===
{context}

=== FAITS RAPPORTÉS ===
Type de litige         : {type_litige}
Partie adverse         : {partie_adverse}
Date des faits         : {date_faits}
Description des faits  : {faits}
Préjudice subi         : {prejudice}
Démarches antérieures  : {demarches}

=== IDENTITÉ DU PLAIGNANT (à utiliser telle quelle, sans crochets) ===
Nom complet  : {nom}
Adresse      : {adresse}
Ville        : {ville}
Date du jour : {date_jour}

=== STRUCTURE OBLIGATOIRE ===

{destinataire}

OBJET : Plainte pour [qualifie en 1 ligne le litige juridiquement — ex : "licenciement abusif" ou "escroquerie" ou "vice caché"]

Madame, Monsieur,

Je soussigné(e) {nom}, demeurant {adresse}, ai l'honneur de porter à votre connaissance les faits suivants :

**I. EXPOSÉ DES FAITS**
[Rédige un exposé chronologique, factuel et neutre. Utilise les dates, noms et éléments fournis. Style officiel, impersonnel, sans jugement.]

**II. FONDEMENT JURIDIQUE**
[Pour chaque article pertinent présent dans le contexte RAG fourni :
→ "Aux termes de l'article [numéro] du [Code] : '[citation exacte du texte]'"
→ Explique brièvement en quoi cet article s'applique aux faits décrits.
RÈGLE ABSOLUE : ne cite JAMAIS un article absent du contexte RAG. Si aucun article ne correspond parfaitement, indique-le clairement.]

**III. PRÉJUDICE SUBI**
[Décrit précisément le préjudice : matériel (chiffré si possible), moral, professionnel, physique.]

**IV. DEMANDES**
[Formule les demandes concrètes et spécifiques au type de litige : engager des poursuites / condamner à indemniser / ordonner la réintégration / etc.]

Dans l'attente de votre réponse, je vous prie d'agréer, Madame, Monsieur, l'expression de mes salutations respectueuses.

Fait à {ville}, le {date_jour}

{nom}
""",
)


def generer_plainte(hybrid: "HybridRetriever", reranker, intake: dict) -> tuple:
    """
    Génère une plainte officielle à partir des faits.

    intake: dict — clés: type_litige, partie_adverse, date_faits, faits,
                         prejudice, demarches (optionnel),
                         nom, adresse, ville
    Retourne: (plainte_text: str, sources: list[Document])
    """
    import datetime
    config = TYPE_LITIGE_CONFIG.get(intake["type_litige"], {})
    code_filtre = config.get("code_filtre")

    # Date du jour formatée en français
    mois_fr = ["janvier","février","mars","avril","mai","juin",
               "juillet","août","septembre","octobre","novembre","décembre"]
    aujourd_hui = datetime.date.today()
    date_jour = f"{aujourd_hui.day} {mois_fr[aujourd_hui.month - 1]} {aujourd_hui.year}"

    # Requête RAG construite à partir du type de litige + faits
    query = f"litige {intake['type_litige']} : {intake['faits'][:400]}"

    # HyDE — génère un article hypothétique dans le style Légifrance
    # pour améliorer le recall FAISS du modèle fine-tuné
    hyde = HyDEGenerator()
    hyde_doc = hyde.generer(
        f"Quels articles de loi s'appliquent à ce litige de type {intake['type_litige']} : "
        f"{intake['faits'][:300]}"
    )

    # Retrieval hybrid BM25 + FAISS (embeddings fine-tunés) avec HyDE
    fetch_k = TOP_K * 2
    docs = hybrid.invoke(
        query,
        code_filtre=code_filtre,
        top_k=fetch_k,
        query_vectorielle=hyde_doc,
    )

    # Cross-Encoder reranking — garde les 5 meilleurs (top_k + 2 pour la plainte)
    if reranker:
        docs = reranker.rerank(query, docs, top_k=RERANKER_TOP_K + 2)

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt_text = PROMPT_PLAINTE.format(
        type_litige=intake["type_litige"],
        partie_adverse=intake["partie_adverse"],
        date_faits=intake["date_faits"],
        faits=intake["faits"],
        prejudice=intake["prejudice"],
        demarches=intake.get("demarches") or "Aucune démarche préalable effectuée.",
        destinataire=config.get("destinataire", "Monsieur le Président du Tribunal"),
        context=context,
        nom=intake["nom"],
        adresse=intake["adresse"],
        ville=intake["ville"],
        date_jour=date_jour,
    )

    reponse = llm.invoke(prompt_text)
    return reponse.content, docs


# ── MODULE — Analyse de Contrat ───────────────────────────────────────────────

TYPE_CONTRAT_CONFIG = {
    "Bail d'habitation": {
        "code_filtre": "Code Civil",
        "icone": "🏠",
        "hint": "Contrat de location immobilière (loi 89-462, Code Civil)",
    },
    "Contrat de travail (CDI/CDD)": {
        "code_filtre": "Code du Travail",
        "icone": "👔",
        "hint": "Contrat d'embauche, période d'essai, clauses de non-concurrence",
    },
    "CGV / Contrat commercial (B2B)": {
        "code_filtre": "Code de Commerce",
        "icone": "🤝",
        "hint": "Conditions générales de vente, contrat entre professionnels",
    },
    "Contrat de consommation (B2C)": {
        "code_filtre": "Code de la Consommation",
        "icone": "🛒",
        "hint": "Contrat entre professionnel et consommateur, clauses abusives",
    },
    "Contrat civil / Prestation de service": {
        "code_filtre": "Code Civil",
        "icone": "📋",
        "hint": "Prestation, sous-traitance, reconnaissance de dette",
    },
    "Autre / Non spécifié": {
        "code_filtre": None,
        "icone": "📄",
        "hint": "Analyse tous les codes disponibles",
    },
}

PROMPT_ANALYSE_CONTRAT = PromptTemplate(
    input_variables=["type_contrat", "contrat", "context"],
    template="""Tu es LexAI, un expert en droit français spécialisé dans l'analyse de contrats.
Tu dois identifier les clauses conformes, suspectes et problématiques au regard du droit français.

=== ARTICLES DE LOI APPLICABLES (corpus Légifrance — RAG) ===
{context}

=== CONTRAT À ANALYSER ===
Type déclaré : {type_contrat}
---
{contrat}
---

=== INSTRUCTIONS ===
Analyse chaque clause ou disposition significative du contrat. Utilise EXACTEMENT cette structure :

## Résumé du contrat
[Identifie : type réel du contrat, parties (désignation utilisée), objet principal, durée, rémunération/loyer si mentionnés]

## Analyse des clauses

Pour chaque clause ou groupe de clauses important, utilise ce format exact :

### [Objet de la clause — ex : "Durée et période d'essai", "Clause de non-concurrence", "Résiliation unilatérale"]
**Extrait :** *"[citation littérale courte de la clause — max 160 caractères]"*
**Statut :** ✅ Conforme / ⚠️ À surveiller / ❌ Problématique
**Article applicable :** [cite l'article exact du contexte RAG — ou "Non couvert par le corpus disponible" si absent]
**Analyse :** [2-3 phrases : pourquoi conforme ou risqué, ce que la loi impose réellement]

## Bilan global

**Niveau de risque :** 🟢 Faible / 🟡 Modéré / 🔴 Élevé
**Points critiques :** [liste des clauses ❌ — "Aucun" si tout est conforme]
**Recommandations :** [2-3 actions concrètes : renégocier X / faire supprimer Y / demander à un avocat de vérifier Z]

RÈGLES ABSOLUES :
- Ne cite que les articles présents dans le contexte RAG fourni
- Si une clause viole explicitement un article, cite-le mot pour mot
- Si le contrat est incomplet ou illisible, le signaler dans le résumé
- Reste factuel, juridique, sans dramatiser
""",
)


def analyser_contrat(hybrid: "HybridRetriever", reranker, type_contrat: str, contrat: str) -> tuple:
    """
    Analyse un contrat et identifie les clauses problématiques via RAG.

    Retourne: (analyse_text: str, sources: list[Document])
    """
    config = TYPE_CONTRAT_CONFIG.get(type_contrat, {})
    code_filtre = config.get("code_filtre")

    # Mots-clés juridiques détectés dans le contrat (scan rapide)
    contrat_lower = contrat.lower()
    detected = []
    _kw_map = {
        "résiliation": "résiliation préavis tacite reconduction",
        "résili": "résiliation préavis tacite reconduction",
        "rétractation": "droit de rétractation 14 jours contrat à distance",
        "retractation": "droit de rétractation 14 jours contrat à distance",
        "prix": "modification unilatérale du prix clause abusive consommateur",
        "tarif": "modification unilatérale du tarif clause abusive",
        "responsabilité": "exclusion de responsabilité clause abusive faute lourde",
        "garantie": "garantie légale conformité obligation vendeur",
        "données": "données personnelles cession tiers RGPD consentement",
        "pénalité": "clause pénale forfaitaire résiliation anticipée",
        "tribunal": "clause attributive compétence consommateur domicile",
        "compétence": "clause attributive compétence consommateur",
        "non-concurrence": "clause non-concurrence durée périmètre contrepartie",
        "période d'essai": "période d'essai durée maximale renouvellement",
        "loyer": "loyer révision indice bail habitation",
    }
    for kw, expansion in _kw_map.items():
        if kw in contrat_lower and expansion not in detected:
            detected.append(expansion)

    # Requête enrichie : problématiques légales réelles présentes dans le contrat
    base_terms = f"clauses abusives nulles illégales {type_contrat} obligations légales droits consommateur"
    query = base_terms + (" — " + " ; ".join(detected[:6]) if detected else "")

    # HyDE : article hypothétique rédigé comme un texte de loi sur ce type de contrat
    hyde = HyDEGenerator()
    hyde_doc = hyde.generer(
        f"Rédige un extrait de loi français qui liste les règles impératives applicables "
        f"à un {type_contrat} : clauses réputées abusives ou non écrites, obligations du "
        f"professionnel envers le consommateur, sanctions en cas de violation."
    )

    # Retrieval large (top 12) — pas de filtre si aucun article trouvé avec filtre
    fetch_k = TOP_K * 2 + 2
    docs = hybrid.invoke(
        query,
        code_filtre=code_filtre,
        top_k=fetch_k,
        query_vectorielle=hyde_doc,
    )

    # Fallback : si le filtre a trop réduit les résultats, relancer sans filtre
    if len(docs) < 4 and code_filtre:
        docs = hybrid.invoke(query, code_filtre=None, top_k=fetch_k, query_vectorielle=hyde_doc)

    # Reranking — garde top 7
    if reranker:
        docs = reranker.rerank(query, docs, top_k=RERANKER_TOP_K + 4)

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    # Tronquer le contrat si trop long pour le contexte GPT-4o-mini
    contrat_tronque = contrat[:6000]
    if len(contrat) > 6000:
        contrat_tronque += "\n\n[... contrat tronqué — seules les 6 000 premières caractères ont été analysées ...]"

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt_text = PROMPT_ANALYSE_CONTRAT.format(
        type_contrat=type_contrat,
        contrat=contrat_tronque,
        context=context,
    )

    reponse = llm.invoke(prompt_text)
    return reponse.content, docs


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
