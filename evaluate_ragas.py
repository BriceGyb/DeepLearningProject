"""
LexAI — Évaluation RAGAS (Sprint 2)
====================================
Framework d'évaluation automatique du pipeline RAG complet.

Métriques évaluées :
  - Faithfulness            : le LLM hallucine-t-il par rapport aux sources ?
  - Response Relevancy      : la réponse est-elle pertinente par rapport à la question ?
  - Context Precision       : les bons articles sont-ils en tête du retrieval ?
  - Context Recall          : tous les articles pertinents sont-ils récupérés ?

Utilisation :
  python evaluate_ragas.py                  # génère le dataset + évalue
  python evaluate_ragas.py --regen          # recrée le dataset Q/A même s'il existe
  python evaluate_ragas.py --skip-generate  # évalue sans relancer le pipeline (si ragas_results.json existe)

Fichiers générés :
  eval_dataset.json    — paires Q/A synthétiques (réutilisable entre variantes)
  ragas_results.json   — scores RAGAS pour cette variante
"""

import argparse
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: F401 (used in pipeline)

load_dotenv()

EVAL_DATASET_PATH = "eval_dataset.json"
RAGAS_RESULTS_PATH = "ragas_results_sprint3.json"
PIPELINE_RESULTS_PATH = "pipeline_outputs_sprint3.json"


# ── 1. Génération du dataset Q/A synthétique ─────────────────────────────────

def generer_paires_qa(articles: list[dict]) -> list[dict]:
    """
    Génère 1 paire (question, ground_truth) par article via GPT-4o-mini.
    Le dataset est sauvegardé en JSON pour être réutilisé entre les variantes.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    paires = []

    for i, article in enumerate(articles):
        print(f"  [{i+1}/{len(articles)}] {article['code']} — {article['article']}")
        texte = article["texte"][:800]

        prompt = f"""Voici un article de loi français. Génère UNE question juridique précise et sa réponse correcte.

Article : {article['article']} ({article['code']})
Texte : {texte}

Retourne un JSON avec exactement ces deux champs :
- "question" : une question juridique spécifique en français à laquelle cet article répond directement
- "ground_truth" : la réponse précise basée UNIQUEMENT sur ce texte (1-3 phrases en français)

Retourne UNIQUEMENT le JSON, sans explication."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            paires.append({
                "question":    data["question"],
                "ground_truth": data["ground_truth"],
                "article_ref": f"{article['code']} — {article['article']}",
            })
        except Exception as e:
            print(f"    [!] Erreur sur {article['article']}: {e}")

    return paires


# ── 2. Exécution du pipeline RAG sur chaque question ─────────────────────────

def executer_pipeline(paires: list[dict]) -> tuple[list, list, list, list]:
    """
    Charge le pipeline LexAI et exécute chaque question.
    Retourne (questions, answers, contexts, ground_truths).
    """
    from rag_lexai import charger_corpus, construire_vectorstore, creer_chaine_rag

    print("\n[~] Chargement du pipeline RAG...")
    documents   = charger_corpus("corpus_v3.json")
    vectorstore = construire_vectorstore(documents)
    chaine, hybrid, _, _ = creer_chaine_rag(vectorstore, documents, use_reranking=True, use_hyde=True)
    print("[+] Pipeline prêt.\n")

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, paire in enumerate(paires):
        print(f"  [{i+1}/{len(paires)}] {paire['question'][:75]}...")
        try:
            docs   = hybrid.invoke(paire["question"])
            answer = chaine.invoke({"question": paire["question"], "langue": "fr"})

            questions.append(paire["question"])
            answers.append(answer)
            contexts.append([doc.page_content for doc in docs])
            ground_truths.append(paire["ground_truth"])
        except Exception as e:
            print(f"    [!] Erreur pipeline: {e}")

    # Sauvegarde intermédiaire (évite de relancer si RAGAS plante)
    with open(PIPELINE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"questions": questions, "answers": answers,
             "contexts": contexts, "ground_truths": ground_truths},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n[+] Sorties pipeline sauvegardées: {PIPELINE_RESULTS_PATH}")
    return questions, answers, contexts, ground_truths


# ── 3. Évaluation RAGAS ───────────────────────────────────────────────────────

def evaluer_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Lance l'évaluation RAGAS et retourne un dict de scores."""
    import warnings
    from datasets import Dataset as HFDataset
    from ragas import evaluate
    # Ancienne API (instances de Metric) — compatible avec evaluate() en ragas 0.4.x
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

    dataset = HFDataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LexAI — Évaluation RAGAS")
    parser.add_argument("--regen",         action="store_true", help="Recrée le dataset Q/A")
    parser.add_argument("--skip-pipeline", action="store_true", help="Réutilise pipeline_outputs.json")
    args = parser.parse_args()

    print("=" * 62)
    print("  LexAI — Évaluation RAGAS  (Sprint 2)")
    print("=" * 62)

    # ── Étape 1 : dataset Q/A ──────────────────────────────────────────────────
    if os.path.exists(EVAL_DATASET_PATH) and not args.regen:
        print(f"\n[+] Dataset existant chargé: {EVAL_DATASET_PATH}")
        with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
            paires = json.load(f)
        print(f"    {len(paires)} paires Q/A.")
    else:
        print("\n[~] Génération du dataset Q/A synthétique...")
        with open("lois_francaises.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        paires = generer_paires_qa(data["corpus_juridique"])
        with open(EVAL_DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(paires, f, ensure_ascii=False, indent=2)
        print(f"[+] Dataset sauvegardé: {EVAL_DATASET_PATH} ({len(paires)} paires)")

    # ── Étape 2 : exécution pipeline ──────────────────────────────────────────
    if args.skip_pipeline and os.path.exists(PIPELINE_RESULTS_PATH):
        print(f"\n[+] Sorties pipeline chargées: {PIPELINE_RESULTS_PATH}")
        with open(PIPELINE_RESULTS_PATH, "r", encoding="utf-8") as f:
            out = json.load(f)
        questions    = out["questions"]
        answers      = out["answers"]
        contexts     = out["contexts"]
        ground_truths = out["ground_truths"]
    else:
        print(f"\n[~] Exécution du pipeline sur {len(paires)} questions...")
        questions, answers, contexts, ground_truths = executer_pipeline(paires)

    # ── Étape 3 : évaluation RAGAS ────────────────────────────────────────────
    print(f"\n[~] Évaluation RAGAS ({len(questions)} questions)...")
    result = evaluer_ragas(questions, answers, contexts, ground_truths)

    # ── Étape 4 : affichage et sauvegarde ────────────────────────────────────
    METRIC_LABELS = {
        "faithfulness":        "Faithfulness        ",
        "answer_relevancy":    "Answer Relevancy    ",
        "context_precision":   "Context Precision   ",
        "context_recall":      "Context Recall      ",
    }

    print("\n" + "=" * 62)
    print("  RESULTATS - LexAI Baseline (Hybrid BM25+FAISS)")
    print("=" * 62)

    # EvaluationResult exposes scores via to_pandas()
    df = result.to_pandas()
    scores = {}
    for key, label in METRIC_LABELS.items():
        if key in df.columns:
            val = df[key].mean(skipna=True)
            import math
            if not math.isnan(val):
                print(f"  {label}: {val:.4f}")
                scores[key] = round(float(val), 4)
            else:
                print(f"  {label}: N/A (all NaN)")
        else:
            print(f"  {label}: not computed")

    print(f"\n  Questions évaluées  : {len(questions)}")
    print("=" * 62)

    output = {
        "variant":     "Sprint 3 — Fine-tuned embeddings + HyDE + Reranking + corpus v3",
        "n_questions": len(questions),
        "metrics":     scores,
    }
    with open(RAGAS_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Résultats sauvegardés: {RAGAS_RESULTS_PATH}")


if __name__ == "__main__":
    main()
