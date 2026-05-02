"""
LexAI — Évaluation Sprint 2 (comparaison équitable Sprint 3)
=============================================================
Même conditions que Sprint 3 :
  - Corpus   : corpus_v3.json (2 419 articles)
  - Questions: eval_dataset.json (60 paires, réutilisées)
  - Pipeline : HyDE + Cross-Encoder Reranking
  - SEULE différence : embeddings OpenAI text-embedding-3-small (pas de fine-tuning)

Sortie : ragas_results_sprint2_fair.json
"""

import os

# Force OpenAI embeddings — doit être fait AVANT l'import de rag_lexai
os.environ["LEXAI_FORCE_OPENAI"] = "1"

import json
import math
import warnings
from dotenv import load_dotenv

load_dotenv()

EVAL_DATASET_PATH    = "eval_dataset.json"
PIPELINE_RESULTS_PATH = "pipeline_outputs_sprint2_fair.json"
RAGAS_RESULTS_PATH   = "ragas_results_sprint2_fair.json"


def executer_pipeline(paires: list[dict]):
    from rag_lexai import charger_corpus, construire_vectorstore, creer_chaine_rag, _USE_FINETUNED, FAISS_PERSIST_DIR

    print(f"\n[~] Chargement pipeline Sprint 2 (OpenAI embeddings)...")
    print(f"    _USE_FINETUNED = {_USE_FINETUNED}  (doit être False)")
    print(f"    FAISS_PERSIST_DIR = {FAISS_PERSIST_DIR}")

    docs = charger_corpus("corpus_v3.json")
    vs   = construire_vectorstore(docs)
    chaine, hybrid, _, _ = creer_chaine_rag(vs, docs, use_reranking=True, use_hyde=True)
    print("[+] Pipeline prêt.\n")

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, paire in enumerate(paires):
        print(f"  [{i+1}/{len(paires)}] {paire['question'][:75]}...")
        try:
            docs_ret = hybrid.invoke(paire["question"])
            answer   = chaine.invoke({"question": paire["question"], "langue": "fr"})
            questions.append(paire["question"])
            answers.append(answer)
            contexts.append([d.page_content for d in docs_ret])
            ground_truths.append(paire["ground_truth"])
        except Exception as e:
            print(f"    [!] Erreur: {e}")

    with open(PIPELINE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"questions": questions, "answers": answers,
                   "contexts": contexts, "ground_truths": ground_truths},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[+] Sorties pipeline sauvegardées: {PIPELINE_RESULTS_PATH}")
    return questions, answers, contexts, ground_truths


def evaluer_ragas(questions, answers, contexts, ground_truths):
    from datasets import Dataset as HFDataset
    from ragas import evaluate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    dataset = HFDataset.from_dict({
        "question": questions, "answer": answers,
        "contexts": contexts,  "ground_truth": ground_truths,
    })
    return evaluate(dataset=dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall])


def main():
    skip_pipeline = os.path.exists(PIPELINE_RESULTS_PATH)

    print("=" * 64)
    print("  LexAI — Évaluation Sprint 2 FAIR (OpenAI + corpus v3)")
    print("=" * 64)

    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        paires = json.load(f)
    print(f"\n[+] Dataset: {EVAL_DATASET_PATH} ({len(paires)} paires)")

    if skip_pipeline:
        print(f"[+] Pipeline déjà calculé: {PIPELINE_RESULTS_PATH}")
        with open(PIPELINE_RESULTS_PATH, "r", encoding="utf-8") as f:
            out = json.load(f)
        questions, answers, contexts, ground_truths = (
            out["questions"], out["answers"], out["contexts"], out["ground_truths"]
        )
    else:
        questions, answers, contexts, ground_truths = executer_pipeline(paires)

    print(f"\n[~] Évaluation RAGAS ({len(questions)} questions)...")
    result = evaluer_ragas(questions, answers, contexts, ground_truths)

    df = result.to_pandas()
    LABELS = {
        "faithfulness":      "Faithfulness      ",
        "answer_relevancy":  "Answer Relevancy  ",
        "context_precision": "Context Precision ",
        "context_recall":    "Context Recall    ",
    }

    print("\n" + "=" * 64)
    print("  RÉSULTATS — Sprint 2 FAIR (OpenAI + corpus v3 + HyDE + Reranking)")
    print("=" * 64)
    scores = {}
    for key, label in LABELS.items():
        if key in df.columns:
            val = df[key].mean(skipna=True)
            if not math.isnan(val):
                print(f"  {label}: {val:.4f}")
                scores[key] = round(float(val), 4)
            else:
                print(f"  {label}: N/A")
        else:
            print(f"  {label}: not computed")

    print(f"\n  Questions évaluées : {len(questions)}")
    print("=" * 64)

    output = {
        "variant":     "Sprint 2 FAIR — OpenAI text-embedding-3-small + HyDE + Reranking + corpus v3",
        "n_questions": len(questions),
        "metrics":     scores,
    }
    with open(RAGAS_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Résultats sauvegardés: {RAGAS_RESULTS_PATH}")


if __name__ == "__main__":
    main()
