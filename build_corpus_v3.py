"""
LexAI — Construction du corpus v3 (Sprint 3)
=============================================
Cible  : 7 codes juridiques × 450 articles = ~3 150 cibles, ~2 500-3 000 réels
Sortie : corpus_v3.json  (corpus v1 et v2 NON modifiés)

Usage :
  python build_corpus_v3.py            # fetch uniquement
  python build_corpus_v3.py --eval     # fetch + génère paires d'entraînement GPT-4o-mini
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

OUTPUT_CORPUS   = "corpus_v3.json"
OUTPUT_TRAINING = "training_pairs_v3.json"
MAX_PAR_CODE    = 450   # 7 codes × 450 = ~3 150 cible

CODES_V3 = {
    "Code Civil":               "LEGITEXT000006070721",
    "Code Pénal":               "LEGITEXT000006070719",
    "Code du Travail":          "LEGITEXT000006072050",
    "Code de Commerce":         "LEGITEXT000005634379",
    "Code de Procédure Civile": "LEGITEXT000006070716",
    "Code de Procédure Pénale": "LEGITEXT000006071154",
    "Code de la Consommation":  "LEGITEXT000006069565",
}


# ── Fetch ──────────────────────────────────────────────────────────────────────

async def fetch_corpus() -> list[dict]:
    from ingestion.legifrance_fetcher import LegiFranceFetcher
    import ingestion.legifrance_fetcher as lf_module

    client_id     = os.getenv("PISTE_CLIENT_ID")
    client_secret = os.getenv("PISTE_CLIENT_SECRET")

    if not client_id or not client_secret:
        print(
            "[STOP] Variables PISTE_CLIENT_ID et PISTE_CLIENT_SECRET manquantes dans .env\n"
            "       Créez un fichier .env avec ces deux variables et relancez."
        )
        return []

    print("=" * 64)
    print("  LexAI — Ingestion corpus v3 (Sprint 3)")
    print(f"  Codes cibles  : {len(CODES_V3)} codes")
    print(f"  Max / code    : {MAX_PAR_CODE} articles")
    print(f"  Cible totale  : ~{len(CODES_V3) * MAX_PAR_CODE} articles")
    print(f"  Date          : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 64)
    for nom in CODES_V3:
        print(f"  • {nom}")
    print("=" * 64)

    codes_originaux = lf_module.CODES_CIBLES.copy()
    lf_module.CODES_CIBLES = CODES_V3

    fetcher  = LegiFranceFetcher(client_id, client_secret, max_articles_par_code=MAX_PAR_CODE)
    articles = []
    par_code: dict[str, int] = {}

    try:
        async for article in fetcher.fetch_all():
            articles.append(article)
            code = article["code"]
            par_code[code] = par_code.get(code, 0) + 1
            print(f"  [{len(articles):>4}] {code[:25]:<25} — {article['article'][:45]}")
    except Exception as e:
        print(f"\n[ERREUR] Ingestion interrompue : {e}")
        if articles:
            print(f"  -> {len(articles)} articles partiels sauvegardés.")
    finally:
        lf_module.CODES_CIBLES = codes_originaux

    print("\n  Résumé par code :")
    for code, nb in par_code.items():
        print(f"    {code:<30} : {nb:>4} articles")

    return articles


# ── Génération paires d'entraînement ──────────────────────────────────────────

def generer_paires_entrainement(articles: list[dict]) -> list[dict]:
    """
    Génère 3 paires (question, article_pertinent) par article via GPT-4o-mini.
    Format cible pour MultipleNegativesRankingLoss (sentence-transformers).
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    paires = []

    print(f"\n[~] Génération paires d'entraînement ({len(articles)} articles × 3 questions)...")
    print(f"    Coût estimé : ~{len(articles) * 3 * 0.00002:.2f} USD (GPT-4o-mini)\n")

    for i, article in enumerate(articles):
        print(f"  [{i+1:>4}/{len(articles)}] {article['code'][:25]:<25} — {article['article']}")
        texte = article["texte"][:1000]

        prompt = f"""Voici un article de loi français. Génère 3 questions juridiques DIFFÉRENTES auxquelles cet article répond directement.

Article : {article['article']} ({article['code']})
Texte : {texte}

Retourne un JSON avec exactement ce format :
{{"questions": ["question 1", "question 2", "question 3"]}}

Les questions doivent être variées : une factuelle, une pratique (cas concret), une sur les conditions/exceptions.
Retourne UNIQUEMENT le JSON, sans explication."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            questions = data.get("questions", [])
            for q in questions[:3]:
                if q.strip():
                    paires.append({
                        "query":    q.strip(),
                        "positive": article["texte"],
                        "metadata": {
                            "code":    article["code"],
                            "article": article["article"],
                            "domaine": article.get("domaine", ""),
                        },
                    })
        except Exception as e:
            print(f"    [!] Erreur: {e}")

    return paires


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="LexAI — Corpus v3 Sprint 3")
    parser.add_argument("--eval",        action="store_true", help="Générer les paires d'entraînement GPT-4o-mini")
    parser.add_argument("--skip-fetch",  action="store_true", help=f"Charger {OUTPUT_CORPUS} existant sans re-fetcher")
    args = parser.parse_args()

    # 1. Fetch ou chargement local
    if args.skip_fetch:
        if not os.path.exists(OUTPUT_CORPUS):
            print(f"[STOP] {OUTPUT_CORPUS} introuvable — relancez sans --skip-fetch.")
            return
        with open(OUTPUT_CORPUS, "r", encoding="utf-8") as f:
            data = json.load(f)
        articles = data.get("corpus_juridique", data) if isinstance(data, dict) else data
        print(f"[+] Corpus chargé depuis {OUTPUT_CORPUS} ({len(articles)} articles)")
    else:
        articles = await fetch_corpus()
        if not articles:
            return

        # Sauvegarde corpus
        corpus = {
            "meta": {
                "version":     "v3",
                "sprint":      3,
                "date":        datetime.now().strftime("%Y-%m-%d"),
                "nb_articles": len(articles),
                "codes":       list(CODES_V3.keys()),
            },
            "corpus_juridique": articles,
        }
        with open(OUTPUT_CORPUS, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        print(f"\n[+] Corpus sauvegardé : {OUTPUT_CORPUS} ({len(articles)} articles)")

    # 2. Paires d'entraînement (optionnel)
    if args.eval:
        paires = generer_paires_entrainement(articles)
        with open(OUTPUT_TRAINING, "w", encoding="utf-8") as f:
            json.dump(paires, f, ensure_ascii=False, indent=2)
        print(f"[+] Paires entraînement : {OUTPUT_TRAINING} ({len(paires)} paires)")

    print("\n" + "=" * 64)
    print("  Récapitulatif des corpus (pour le rapport)")
    print("=" * 64)
    print(f"  lois_francaises.json   : corpus v1 Sprint 1 (NON modifié)")
    print(f"  corpus_penal_civil.json: corpus v2 Sprint 2 (NON modifié)")
    print(f"  {OUTPUT_CORPUS:<23}: corpus v3 Sprint 3 ({len(articles)} articles)")
    if args.eval:
        print(f"  {OUTPUT_TRAINING:<23}: paires entraînement ({len(paires)} paires)")
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(main())
