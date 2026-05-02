"""
LexAI — Fetcher API Légifrance (PISTE)
Authentification OAuth2 → récupération des articles par code juridique.

Prérequis : créer un compte sur https://piste.gouv.fr et obtenir
            client_id + client_secret pour l'API Légifrance.
"""

import httpx
import asyncio
import time
import re
import unicodedata
from datetime import date
from typing import AsyncIterator

# Identifiants des codes principaux (IDs officiels Légifrance)
CODES_CIBLES = {
    "Code Civil":                    "LEGITEXT000006070721",
    "Code Pénal":                    "LEGITEXT000006070719",
    "Code du Travail":               "LEGITEXT000006072050",
    "Code de Commerce":              "LEGITEXT000005634379",
    "Code de Procédure Civile":      "LEGITEXT000006070716",
    "Code de Procédure Pénale":      "LEGITEXT000006071154",
    "Code de la Consommation":       "LEGITEXT000006069565",
}

TOKEN_URL = "https://sandbox-oauth.piste.gouv.fr/api/oauth/token"
BASE_URL  = "https://sandbox-api.piste.gouv.fr/dila/legifrance/lf-engine-app"


class LegiFranceFetcher:
    """
    Client asynchrone pour l'API PISTE / Légifrance.
    Gère : authentification OAuth2, refresh token, rate limiting,
           récupération table des matières + articles.
    """

    def __init__(self, client_id: str, client_secret: str, max_articles_par_code: int = 50):
        self.client_id     = client_id
        self.client_secret = client_secret
        self.max_articles  = max_articles_par_code
        self._token        = None
        self._token_expiry = 0

    # ── Authentification ───────────────────────────────────────────────────────

    async def _get_token(self) -> str:
        """Récupère ou renouvelle le token OAuth2."""
        if self._token and time.time() < self._token_expiry - 30:
            return self._token

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                TOKEN_URL,
                data={
                    "grant_type":    "client_credentials",
                    "client_id":     self.client_id,
                    "client_secret": self.client_secret,
                    "scope":         "openid",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token        = data["access_token"]
            self._token_expiry = time.time() + data.get("expires_in", 3600)
            print(f"[Auth] Token obtenu (expire dans {data.get('expires_in', 3600)}s)")
            return self._token

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }

    # ── Appels API ─────────────────────────────────────────────────────────────

    async def _post(self, client: httpx.AsyncClient, endpoint: str, payload: dict) -> dict:
        """POST avec retry sur expiration de token."""
        await self._get_token()
        for tentative in range(3):
            resp = await client.post(
                f"{BASE_URL}{endpoint}",
                headers=self._headers(),
                json=payload,
                timeout=20,
            )
            if resp.status_code == 401:
                # Token expiré → forcer le renouvellement
                self._token = None
                await self._get_token()
                continue
            if resp.status_code == 429:
                # Rate limit → attendre
                attente = int(resp.headers.get("Retry-After", 5))
                print(f"[RateLimit] Attente {attente}s...")
                await asyncio.sleep(attente)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Échec après 3 tentatives : {endpoint}")

    async def _get_table_matieres(self, client: httpx.AsyncClient, code_id: str) -> dict:
        return await self._post(client, "/consult/code/tableMatieres", {
            "textId": code_id,
            "date":   date.today().isoformat(),
        })

    async def _get_article(self, client: httpx.AsyncClient, article_id: str) -> dict:
        return await self._post(client, "/consult/getArticle", {"id": article_id})

    # ── Extraction récursive des IDs d'articles depuis la TDM ─────────────────

    def _extraire_ids_articles(self, noeud: dict, ids: list, limite: int):
        """Parcourt récursivement la table des matières et collecte les IDs."""
        if len(ids) >= limite:
            return
        # Les articles sont dans "articles" ou dans les sections enfants
        for article in noeud.get("articles", []):
            if len(ids) >= limite:
                break
            art_id = article.get("id") or article.get("cid")
            if art_id:
                ids.append(art_id)
        for section in noeud.get("sections", []):
            if len(ids) >= limite:
                break
            self._extraire_ids_articles(section, ids, limite)

    # ── Nettoyage du texte ────────────────────────────────────────────────────

    def _nettoyer(self, texte: str) -> str:
        if not texte:
            return ""
        # Strip HTML
        texte = re.sub(r"<[^>]+>", " ", texte)
        # Patterns parasites Légifrance
        texte = re.sub(r"Nota\s*:.*?(?=\n\n|\Z)", "", texte, flags=re.DOTALL | re.IGNORECASE)
        texte = re.sub(r"Version\s+en\s+vigueur.*", "", texte, flags=re.IGNORECASE)
        # Normalisation espaces + Unicode
        texte = re.sub(r"\s+", " ", texte).strip()
        texte = unicodedata.normalize("NFC", texte)
        return texte

    # ── Conversion vers le format LexAI JSON ──────────────────────────────────

    def _article_vers_json(self, data: dict, code_nom: str) -> dict | None:
        """Convertit la réponse API en entrée JSON compatible lois_francaises.json."""
        article = data.get("article", data)  # La structure varie selon l'endpoint

        texte = self._nettoyer(
            article.get("texte", "") or
            article.get("texteHtml", "") or
            article.get("content", "")
        )
        if not texte or len(texte) < 20:
            return None

        num = article.get("num", "") or article.get("numero", "") or "Article inconnu"
        art_id = article.get("id", "") or article.get("cid", "")
        etat = article.get("etat", "").upper()

        # Ignorer les articles abrogés
        if etat in ("ABROGE", "ABROGE_DIFF", "TRANSFERE"):
            return None

        # Construire le domaine depuis la hiérarchie titresTM (section la plus précise)
        domaine = self._extraire_domaine(article, code_nom)

        return {
            "id":      art_id,
            "code":    code_nom,
            "article": f"Article {num}" if not num.startswith("Article") else num,
            "domaine": domaine,
            "texte":   texte,
        }

    def _extraire_domaine(self, article: dict, code_nom: str) -> str:
        """Extrait le titre de section le plus précis depuis la réponse API."""
        # Cas 1 : champ string direct
        titre_direct = article.get("sectionParentTitle", "")
        if titre_direct and isinstance(titre_direct, str):
            return f"{code_nom} — {titre_direct}"

        # Cas 2 : champ context avec titresTM (structure hiérarchique Légifrance)
        context = article.get("context", {})
        if isinstance(context, dict):
            titres_tm = context.get("titresTM", [])
            if titres_tm and isinstance(titres_tm, list):
                # Prendre le titre le plus profond (dernier = section la plus précise)
                dernier = titres_tm[-1]
                titre = dernier.get("titre", "")
                if titre:
                    return f"{code_nom} — {titre}"

        return code_nom

    # ── Point d'entrée principal ───────────────────────────────────────────────

    async def fetch_all(self) -> AsyncIterator[dict]:
        """
        Récupère les articles de tous les codes cibles.
        Yields : dict compatible avec le format lois_francaises.json
        """
        async with httpx.AsyncClient() as client:
            for code_nom, code_id in CODES_CIBLES.items():
                print(f"\n[Fetch] Code : {code_nom} (ID: {code_id})")

                # 1. Table des matières
                try:
                    tdm = await self._get_table_matieres(client, code_id)
                except Exception as e:
                    print(f"  [Erreur TDM] {e}")
                    continue

                # 2. Extraction des IDs d'articles
                ids = []
                self._extraire_ids_articles(tdm, ids, self.max_articles)
                print(f"  -> {len(ids)} articles trouves (limite: {self.max_articles})")

                # 3. Récupération article par article
                nb_ok = 0
                for art_id in ids:
                    try:
                        data = await self._get_article(client, art_id)
                        article_json = self._article_vers_json(data, code_nom)
                        if article_json:
                            yield article_json
                            nb_ok += 1
                        await asyncio.sleep(0.15)   # rate limiting conservatif
                    except Exception as e:
                        print(f"  [Erreur article {art_id}] {e}")
                        continue

                print(f"  -> {nb_ok} articles recuperes avec succes.")
