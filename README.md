# ⚖️ LexAI — Assistant Juridique Intelligent

> **Projet académique** — 8INF887 Apprentissage Profond — Université du Québec à Chicoutimi (UQAC) — 2026  
> **Auteur** : GYEBRE Brice Joseph Emeric  
> **Démo live** : [apprentissageprofondprojet.streamlit.app](https://apprentissageprofondprojet.streamlit.app)  
> **Modèle fine-tuné** : [DrProfessor/lexai-embeddings](https://huggingface.co/DrProfessor/lexai-embeddings) (HuggingFace Hub)

---

## Table des matières

1. [Présentation du projet](#1-présentation-du-projet)
2. [Architecture technique](#2-architecture-technique)
3. [Fonctionnalités](#3-fonctionnalités)
4. [Corpus juridique](#4-corpus-juridique)
5. [Fine-tuning des embeddings](#5-fine-tuning-des-embeddings)
6. [Pipeline RAG hybride](#6-pipeline-rag-hybride)
7. [Module de rédaction de plainte](#7-module-de-rédaction-de-plainte)
8. [Évaluation](#8-évaluation)
9. [Installation et démarrage](#9-installation-et-démarrage)
10. [Structure du dépôt](#10-structure-du-dépôt)
11. [Variables d'environnement](#11-variables-denvironnement)
12. [Évolution par sprint](#12-évolution-par-sprint)

---

## 1. Présentation du projet

**LexAI** est un assistant juridique intelligent basé sur une architecture **RAG (Retrieval-Augmented Generation)** hybride, spécialisé dans le droit français. Son objectif est de rendre le droit accessible à tous en permettant à n'importe quel citoyen de :

- **Poser des questions juridiques** en langage naturel et obtenir une réponse structurée citant les articles de loi applicables
- **Rédiger une plainte officielle** fondée sur les textes de loi pertinents, prête à être envoyée à l'autorité compétente
- **Connaître les étapes procédurales** concrètes à suivre après génération du document

LexAI ne se substitue pas à un avocat — il agit comme un premier niveau d'orientation juridique, comparable aux outils comme LegalZoom ou Rocket Lawyer, mais ancré sur les textes officiels français (Légifrance).

---

## 2. Architecture technique

```
┌─────────────────────────────────────────────────────────────────┐
│                        INTERFACE STREAMLIT                      │
│   Mode Consultation (chat)  │  Mode Rédaction de Plainte        │
└────────────────┬────────────────────────────┬───────────────────┘
                 │                            │
                 ▼                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PIPELINE RAG HYBRIDE                       │
│                                                                 │
│  Question / Faits                                               │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────┐     ┌──────────────────────────────────────┐      │
│  │  HyDE   │────▶│  Article hypothétique (GPT-4o-mini)  │      │
│  └─────────┘     └──────────────────┬───────────────────┘      │
│                                     │                           │
│       ┌─────────────────────────────┼──────────────────┐       │
│       ▼                             ▼                   │       │
│  ┌──────────┐              ┌──────────────────┐         │       │
│  │   BM25   │              │  FAISS (dense)   │         │       │
│  │ (lexical)│              │  Fine-tuned      │         │       │
│  │          │              │  embeddings      │         │       │
│  └────┬─────┘              └────────┬─────────┘         │       │
│       │                             │                   │       │
│       └──────────┬──────────────────┘                   │       │
│                  ▼                                       │       │
│         ┌────────────────┐                              │       │
│         │  RRF Fusion    │  (BM25×0.35 + FAISS×0.65)   │       │
│         └───────┬────────┘                              │       │
│                 ▼                                       │       │
│         ┌────────────────┐                              │       │
│         │ Cross-Encoder  │  BGE-Reranker (Top-10→Top-5) │       │
│         │   Reranking    │                              │       │
│         └───────┬────────┘                              │       │
│                 ▼                                       │       │
│         ┌────────────────┐                              │       │
│         │  GPT-4o-mini   │  Génération structurée       │       │
│         └───────┬────────┘                              │       │
│                 ▼                                       │       │
│         Réponse / Plainte + Sources citées              │       │
└─────────────────────────────────────────────────────────────────┘
```

### Stack technologique

| Composant | Technologie |
|---|---|
| Interface | Streamlit |
| LLM (génération) | GPT-4o-mini (OpenAI) |
| Embeddings (fine-tunés) | `paraphrase-multilingual-MiniLM-L12-v2` fine-tuné |
| Embeddings (fallback) | `text-embedding-3-small` (OpenAI) |
| Vectorstore | FAISS (persistant sur disque) |
| Recherche lexicale | BM25Okapi (rank-bm25) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | CrossEncoder `BAAI/bge-reranker-base` |
| HyDE | GPT-4o-mini (température 0.5) |
| Framework RAG | LangChain |
| Source des lois | API Légifrance PISTE (OAuth2) |
| Déploiement | Streamlit Cloud + HuggingFace Hub |

---

## 3. Fonctionnalités

### 3.1 Mode Consultation juridique (chat)

- **Questions en langage naturel** : posez n'importe quelle question juridique en français ou en anglais
- **Réponse structurée** : Base légale → Texte applicable → Analyse juridique → Conclusion
- **Sources citées** : chaque réponse affiche les articles Légifrance utilisés avec lien direct
- **Filtre par code** : limitez la recherche à un code juridique spécifique (Code Civil, Pénal, Travail, etc.)
- **Mode bilingue** : réponses en français ou en anglais (traduction automatique des articles)
- **Streaming** : la réponse s'affiche en temps réel

### 3.2 Mode Rédaction de plainte

Interface dédiée, accessible via le bouton **"📝 Rédiger une plainte"** dans la sidebar.

**Formulaire d'intake :**
- Type de litige (Pénal / Civil / Travail / Consommation / Commerce)
- Partie adverse
- Date des faits
- Description détaillée des faits
- Préjudice subi
- Démarches antérieures (optionnel)

**Génération :**
1. RAG hybride + HyDE pour trouver les articles applicables au type de litige
2. Cross-Encoder reranking pour sélectionner les 5 articles les plus pertinents
3. GPT-4o-mini génère une plainte officielle structurée en 4 sections (Faits / Fondement juridique / Préjudice / Demandes)

**Résultat :**
- Plainte complète avec placeholders `[VOTRE NOM]`, `[VOTRE ADRESSE]` à compléter
- Téléchargement en `.txt`
- Articles de loi utilisés (traçabilité RAG)
- Étapes procédurales concrètes selon le type de litige
- Liens vers les démarches officielles (pre-plainte.fr, justice.fr, etc.)

### 3.3 Interface

- **Thème clair/sombre** complet (toggle dans la sidebar)
- **Filtre juridictionnel** (France active, autres pays à venir)
- **Filtre par code juridique** pour la consultation

---

## 4. Corpus juridique

Le corpus a évolué sur 3 sprints :

| Sprint | Version | Codes couverts | Articles |
|---|---|---|---|
| Sprint 1 | v1 | Code Civil + Code Pénal (partiel) | 60 |
| Sprint 2 | v2 | Code Civil + Code Pénal | 346 |
| Sprint 3 | **v3** | **6 codes juridiques** | **2 419** |

### Codes juridiques (Sprint 3 — `corpus_v3.json`)

| Code | ID Légifrance | Articles |
|---|---|---|
| Code Civil | LEGITEXT000006070721 | 393 |
| Code Pénal | LEGITEXT000006070719 | 363 |
| Code du Travail | LEGITEXT000006072050 | 416 |
| Code de Commerce | LEGITEXT000005634379 | 417 |
| Code de Procédure Civile | LEGITEXT000006070716 | 416 |
| Code de la Consommation | LEGITEXT000006069565 | 414 |
| **Total** | | **2 419** |

### Ingestion

Les articles sont récupérés via l'**API Légifrance PISTE** (OAuth2) — script `build_corpus_v3.py` avec rate limiting (0.15s/requête). Le fetch complet prend ~45 minutes.

### Chunking

- Articles courts (≤ 800 tokens) : 1 chunk unique
- Articles longs : split récursif par alinéa (`RecursiveCharacterTextSplitter`, overlap 400 chars)
- Résultat : **~3 200 chunks** depuis 2 419 articles

---

## 5. Fine-tuning des embeddings

### Modèle de base

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Multilingue (50+ langues, dont FR/EN)
- 384 dimensions (vs 1536 pour OpenAI text-embedding-3-small)
- Apache 2.0, déployable librement

### Génération des paires d'entraînement

GPT-4o-mini génère **3 questions par article** (factuelle, pratique, conditionnelle) → **7 257 paires** au total (~$0.15 de coût API).

```json
{
  "query": "Quand un licenciement est-il considéré comme abusif ?",
  "positive": "[Code du Travail — Article L1232-1]\nDomaine : Licenciement\n\nTout licenciement pour motif personnel doit être justifié par une cause réelle et sérieuse..."
}
```

| Split | Paires |
|---|---|
| Entraînement (90%) | 6 531 |
| Validation (10%) | 726 |
| **Total** | **7 257** |

### Paramètres d'entraînement

| Paramètre | Valeur |
|---|---|
| Loss function | MultipleNegativesRankingLoss |
| Epochs | 25 |
| Batch size | 128 |
| GPU | A100 (Google Colab) |
| Modèle publié | `DrProfessor/lexai-embeddings` (HF Hub) |

### Résultats IR (évaluation sur dataset `lexai-juridique`)

| Métrique | Baseline | Fine-tuné | Gain |
|---|---|---|---|
| Accuracy@1 | 0.501 | **0.730** | +45.5% |
| Accuracy@3 | 0.694 | **0.904** | +30.3% |
| Accuracy@5 | 0.751 | **0.950** | +26.6% |
| NDCG@10 | 0.661 | **0.863** | +30.6% |
| MRR@10 | 0.610 | **0.826** | +35.4% |
| MAP@100 | 0.616 | **0.826** | +34.1% |

---

## 6. Pipeline RAG hybride

### Composants

#### HybridRetriever (BM25 + FAISS + RRF)

Combine deux signaux complémentaires :
- **BM25Okapi** (lexical) : capture les correspondances exactes de termes juridiques (noms d'articles, codes, termes techniques)
- **FAISS** (sémantique) : capture le sens et les reformulations

Fusion par **Reciprocal Rank Fusion** :
```
score_final = 0.65 × RRF(rang_FAISS) + 0.35 × RRF(rang_BM25)
```

#### HyDE (Hypothetical Document Embeddings)

Au lieu d'embedder la question brute, GPT-4o-mini génère d'abord un article de loi hypothétique qui répondrait à la question. Ce document hypothétique, stylistiquement proche des vrais articles Légifrance, produit une meilleure similarité cosinus → meilleur recall.

#### CrossEncoderReranker

Modèle `BAAI/bge-reranker-base` — analyse la question et chaque document **ensemble** (attention croisée) pour un score de pertinence plus fin que les bi-encoders.

Pipeline : Top-10 hybrid → CrossEncoder → Top-3 (consultation) / Top-5 (plainte)

### Prompt de génération

Structure imposée en français :

```
**Base légale** : [Code + Article]
**Texte applicable** : [Citation exacte]
**Analyse juridique** : [Application aux faits]
**Conclusion** : [Réponse directe]
```

---

## 7. Module de rédaction de plainte

### Architecture

```
Faits utilisateur
      │
      ▼
Requête RAG = "litige {type} : {faits[:400]}"
      │
      ├──▶ HyDE → article hypothétique (style Légifrance)
      │
      ▼
HybridRetriever (filtre sur code du type de litige)
      │
      ▼
CrossEncoder Reranking → Top-5 articles
      │
      ▼
PROMPT_PLAINTE (GPT-4o-mini, temp=0.1)
      │
      ▼
Plainte structurée (4 sections) + sources + étapes procédurales
```

### Types de litiges supportés

| Type | Code filtré | Destinataire | Procédure |
|---|---|---|---|
| Pénal | Code Pénal | Procureur de la République | Commissariat / pre-plainte.fr |
| Civil / Contractuel | Code Civil | Tribunal Judiciaire | Mise en demeure → Cerfa / huissier |
| Travail | Code du Travail | Conseil de Prud'hommes | justice.fr → conciliation |
| Consommation | Code de la Consommation | Entreprise adverse | LRAR → médiateur → DGCCRF |
| Commerce / Affaires | Code de Commerce | Tribunal de Commerce | LRAR → injonction / assignation |

---

## 8. Évaluation

### RAGAS (comparaison équitable Sprint 2 vs Sprint 3)

Évaluation sur 60 paires Q/A, même corpus v3, pour une comparaison juste :

| Métrique | Sprint 2 (OpenAI embeddings) | Sprint 3 (fine-tuned) |
|---|---|---|
| Faithfulness | 0.8121 | 0.7896 |
| Context Precision | 0.8696 | 0.8628 |
| Context Recall | 0.9667 | 0.9667 |

> **Note** : Les métriques RAGAS end-to-end sont légèrement inférieures malgré les gains IR massifs. Le modèle fine-tuné (384 dims) reste plus compact qu'OpenAI text-embedding-3-small (1536 dims). Les gains IR (+45% sur Accuracy@1) se traduisent en meilleur rappel des articles exacts, même si le score RAGAS global est très proche.

### Progression RAGAS sur les sprints

| Sprint | Faithfulness | Context Precision | Context Recall |
|---|---|---|---|
| Sprint 1 (baseline) | 0.8322 | 0.8935 | 0.9944 |
| Sprint 2 (+ reranking + HyDE) | 0.8983 | 0.9561 | 0.9708 |
| Sprint 3 (+ fine-tuning + corpus ×7) | 0.7896 | 0.8628 | 0.9667 |

---

## 9. Installation et démarrage

### Prérequis

- Python 3.11+
- Compte OpenAI (clé API)
- Compte Légifrance PISTE (optionnel, pour régénérer le corpus)

### Installation locale

```bash
git clone https://github.com/BriceGyb/DeepLearningProject.git
cd DeepLearningProject
pip install -r requirements.txt
```

### Configuration

Créez un fichier `.env` à la racine :

```env
OPENAI_API_KEY=sk-...
PISTE_CLIENT_ID=...          # Optionnel — API Légifrance
PISTE_CLIENT_SECRET=...      # Optionnel — API Légifrance
LEXAI_EMBEDDING_MODEL=DrProfessor/lexai-embeddings  # HuggingFace Hub
```

### Lancement

```bash
streamlit run app.py
```

L'application charge automatiquement le corpus `corpus_v3.json`, construit l'index FAISS (ou le recharge depuis le disque si déjà existant), et démarre sur `http://localhost:8501`.

### Premier démarrage

Le premier démarrage télécharge le modèle fine-tuné depuis HuggingFace Hub (~450 MB) et le modèle de reranking (~270 MB). Les démarrages suivants sont instantanés (cache disque).

### Déploiement Streamlit Cloud

1. Forkez le dépôt
2. Connectez-le sur [share.streamlit.io](https://share.streamlit.io)
3. Ajoutez les secrets dans les paramètres de l'app :
   ```toml
   OPENAI_API_KEY = "sk-..."
   LEXAI_EMBEDDING_MODEL = "DrProfessor/lexai-embeddings"
   ```

---

## 10. Structure du dépôt

```
DeepLearningProject/
│
├── app.py                        # Interface Streamlit (consultation + rédaction de plainte)
├── rag_lexai.py                  # Pipeline RAG complet + module plainte
├── build_corpus_v3.py            # Script d'ingestion Légifrance (corpus v3)
├── evaluate_ragas.py             # Évaluation RAGAS Sprint 3
├── evaluate_sprint2_fair.py      # Évaluation équitable Sprint 2 vs Sprint 3
│
├── corpus_v3.json                # Corpus principal (2 419 articles, 6 codes)
├── corpus_penal_civil.json       # Corpus Sprint 2 (346 articles)
├── lois_francaises.json          # Corpus Sprint 1 (60 articles)
├── training_pairs_v3.json        # 7 257 paires d'entraînement
├── eval_dataset.json             # Dataset d'évaluation RAGAS (60 Q/A)
│
├── finetune_metrics.json         # Métriques IR baseline vs fine-tuné
├── ragas_results_sprint3.json    # Résultats RAGAS Sprint 3
├── ragas_results_sprint2_fair.json
├── RAGAS_RESULTS.md
│
├── faiss_index/                  # Index FAISS (embeddings OpenAI, corpus v1/v2)
├── faiss_index_v3/               # Index FAISS (embeddings fine-tunés, corpus v3)
│
├── ingestion/
│   └── legifrance_fetcher.py     # Client OAuth2 API Légifrance PISTE
│
├── finetune_embeddings (2).ipynb # Notebook Google Colab — fine-tuning sur A100
│
├── RAPPORT_SPRINT2.md            # Rapport Sprint 2
├── RAPPORT_SPRINT2.pdf
├── RAPPORT_SPRINT3.html          # Rapport Sprint 3
│
├── requirements.txt
├── .devcontainer/                # Dev container (Python 3.11, VSCode)
└── README.md
```

---

## 11. Variables d'environnement

| Variable | Obligatoire | Description |
|---|---|---|
| `OPENAI_API_KEY` | Oui | Clé API OpenAI (GPT-4o-mini + embeddings fallback) |
| `LEXAI_EMBEDDING_MODEL` | Non | Chemin local ou nom HF Hub du modèle fine-tuné. Si absent, utilise OpenAI text-embedding-3-small |
| `LEXAI_FORCE_OPENAI` | Non | Mettre `1` pour forcer OpenAI même si le modèle fine-tuné est présent (comparaison) |
| `PISTE_CLIENT_ID` | Non | ID client OAuth2 Légifrance (uniquement pour régénérer le corpus) |
| `PISTE_CLIENT_SECRET` | Non | Secret client OAuth2 Légifrance |

---

## 12. Évolution par sprint

### Sprint 1 — Prototype RAG de base
- Pipeline RAG simple (FAISS + GPT-4o-mini)
- Corpus initial : 60 articles (Code Civil + Code Pénal partiel)
- Interface Streamlit basique
- Déploiement initial sur Streamlit Cloud

### Sprint 2 — RAG de niveau recherche
- **Hybrid Search** : BM25 + FAISS avec Reciprocal Rank Fusion
- **Cross-Encoder Reranking** : BAAI/bge-reranker-base (Top-10 → Top-3)
- **HyDE** : Hypothetical Document Embeddings pour améliorer le recall
- Corpus étendu à 346 articles
- Framework d'évaluation **RAGAS** intégré
- Interface améliorée (dark mode, filtre par code, bilingue)

### Sprint 3 — Fine-tuning et scaling
- **Corpus v3** : 2 419 articles, 6 codes juridiques (+600%)
- **Fine-tuning d'embeddings** : 7 257 paires, 25 epochs sur A100
- **Déploiement HuggingFace Hub** : `DrProfessor/lexai-embeddings`
- Évaluation équitable Sprint 2 vs Sprint 3 (même conditions)
- Gains IR : +45% Accuracy@1, +35% MRR@10
- **Module de rédaction de plainte** : formulaire d'intake + RAG juridique + génération de plainte officielle + étapes procédurales

---

## Avertissement légal

LexAI est un outil d'aide à la compréhension du droit français, développé dans un cadre académique. Il ne constitue pas un conseil juridique et ne remplace pas l'avis d'un avocat. Pour tout litige important, consultez un professionnel du droit.

---

*Projet développé dans le cadre du cours 8INF887 — Apprentissage Profond — UQAC 2026*
