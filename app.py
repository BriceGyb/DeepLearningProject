"""
LexAI — Streamlit Interface
"""

import os
import re
import streamlit as st

# Streamlit Cloud: inject secrets into os.environ before rag_lexai reads them
for _key in ("OPENAI_API_KEY", "PISTE_CLIENT_ID", "PISTE_CLIENT_SECRET"):
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

from rag_lexai import (
    charger_corpus, construire_vectorstore, creer_chaine_rag,
    generer_plainte, TYPE_LITIGE_CONFIG,
    analyser_contrat, TYPE_CONTRAT_CONFIG,
)


# ── Génération PDF ─────────────────────────────────────────────────────────────

def _latin1(t: str) -> str:
    """Encode vers Latin-1 (police Helvetica fpdf2) — retire emojis et symboles hors-range."""
    return t.encode("latin-1", errors="ignore").decode("latin-1")


def generer_pdf(texte: str, titre_doc: str = "Document Juridique") -> bytes:
    """Convertit le texte markdown de la plainte/analyse en PDF imprimable (fpdf2)."""
    from fpdf import FPDF

    L_MARGIN = 20
    R_MARGIN = 20
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(L_MARGIN, 30, R_MARGIN)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Largeur utile fixe — évite tout calcul dynamique lié à la position X courante
    W = pdf.w - L_MARGIN - R_MARGIN  # 210 - 40 = 170 mm

    def _mc(h: float, txt: str) -> None:
        """multi_cell avec reset de X pour garantir la largeur pleine."""
        pdf.set_x(L_MARGIN)
        pdf.multi_cell(W, h, _latin1(txt))

    # ── Bandeau en-tête ──────────────────────────────────────────────────────
    pdf.set_fill_color(26, 35, 126)
    pdf.rect(0, 0, 210, 18, "F")
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(L_MARGIN, 5)
    pdf.cell(W, 8, _latin1(f"LexAI  —  {titre_doc}"), align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(L_MARGIN, 26)

    # ── Rendu ligne par ligne ─────────────────────────────────────────────────
    for raw in texte.split("\n"):
        line = raw.strip()

        if re.match(r"^-{3,}$", line) or line.startswith("Note :"):
            continue

        if not line:
            pdf.ln(3)
            continue

        # En-têtes de section romains : **I. TITRE** / **II. TITRE**
        m = re.match(r"^\*\*([IVX]+\.\s+.+?)\*\*$", line)
        if m:
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 11)
            _mc(7, m.group(1))
            pdf.ln(1)
            continue

        if line.startswith("### "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 10)
            _mc(6, line[4:])
            pdf.ln(1)
            continue
        if line.startswith("## "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 11)
            _mc(7, line[3:])
            pdf.ln(1)
            continue

        if line.upper().startswith("OBJET"):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11)
            _mc(7, line)
            pdf.ln(2)
            continue

        if re.match(r"^\*\*[^*]+\*\*$", line):
            pdf.set_font("Helvetica", "B", 10)
            _mc(6, line[2:-2])
            continue

        # Ligne avec mélange gras / normal — on finit sur une nouvelle ligne
        if "**" in line:
            pdf.set_x(L_MARGIN)
            for part in re.split(r"(\*\*[^*]+\*\*)", line):
                if not part:
                    continue
                if part.startswith("**") and part.endswith("**"):
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.write(6, _latin1(part[2:-2]))
                else:
                    pdf.set_font("Helvetica", "", 10)
                    pdf.write(6, _latin1(part))
            pdf.ln(6)
            continue

        pdf.set_font("Helvetica", "", 10)
        _mc(6, line)

    # ── Pied de page ─────────────────────────────────────────────────────────
    pdf.set_y(-16)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(L_MARGIN, pdf.get_y(), pdf.w - R_MARGIN, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(150, 150, 150)
    pdf.set_x(L_MARGIN)
    pdf.cell(
        W, 5,
        "Document genere par LexAI  |  Aide a la redaction juridique  |  Ne remplace pas l'avis d'un avocat",
        align="C",
    )

    return bytes(pdf.output())

st.set_page_config(
    page_title="LexAI — Legal Assistant",
    page_icon="⚖️",
    layout="wide",
)

# ── Theme toggle ────────────────────────────────────────────────────────────────

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def inject_theme():
    if st.session_state.dark_mode:
        css = """
        <style>
        /* ── Fond global ── */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"],
        [data-testid="block-container"],
        [data-testid="stMainBlockContainer"],
        .main, .block-container {
            background-color: #0f1117 !important;
            color: #fafafa !important;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"],
        section[data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {
            background-color: #1a1c23 !important;
        }
        [data-testid="stSidebar"] * { color: #fafafa !important; }

        /* ── Colonnes et containers (bandes blanches principales) ── */
        [data-testid="stColumn"],
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"],
        [data-testid="stContainer"] {
            background-color: #0f1117 !important;
        }

        /* ── Formulaire ── */
        [data-testid="stForm"] {
            background-color: #1a1c23 !important;
            border: 1px solid #2d3250 !important;
            border-radius: 8px !important;
        }

        /* ── Inputs texte et textareas ── */
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        .stTextInput input,
        .stTextArea textarea {
            background-color: #0f1117 !important;
            color: #fafafa !important;
            border-color: #2d3250 !important;
        }

        /* ── Selectbox ── */
        [data-testid="stSelectbox"] > div > div,
        [data-baseweb="select"] > div {
            background-color: #0f1117 !important;
            color: #fafafa !important;
            border-color: #2d3250 !important;
        }
        [data-baseweb="popover"] ul,
        [data-baseweb="menu"] {
            background-color: #1a1c23 !important;
            color: #fafafa !important;
        }
        [data-baseweb="option"]:hover {
            background-color: #2d3250 !important;
        }

        /* ── Boutons ── */
        .stButton > button,
        [data-testid="stDownloadButton"] button {
            background-color: #1e2130 !important;
            color: #fafafa !important;
            border-color: #2d3250 !important;
        }
        [data-testid="stFormSubmitButton"] button[kind="primary"],
        button[kind="primary"] {
            background-color: #3949ab !important;
            border-color: #3949ab !important;
            color: #ffffff !important;
        }

        /* ── Chat ── */
        [data-testid="stChatMessageContent"] {
            background-color: #1e2130 !important;
            color: #fafafa !important;
        }
        [data-testid="stChatInput"],
        [data-testid="stChatInput"] textarea {
            background-color: #1a1c23 !important;
            color: #fafafa !important;
            border-color: #2d3250 !important;
        }

        /* ── Expander ── */
        [data-testid="stExpander"],
        [data-testid="stExpanderDetails"] {
            background-color: #1a1c23 !important;
            border-color: #2d3250 !important;
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * { color: #fafafa !important; }

        /* ── Alertes / Info ── */
        [data-testid="stAlert"],
        [data-baseweb="notification"],
        .stAlert {
            background-color: #1e2130 !important;
            color: #fafafa !important;
            border-color: #2d3250 !important;
        }

        /* ── Container avec bordure (plainte générée) ── */
        [data-testid="stContainer"][style*="border"],
        div[data-testid="stContainer"] {
            border-color: #2d3250 !important;
        }

        /* ── Texte générique ── */
        p, h1, h2, h3, h4, label, span,
        .stMarkdown, [data-testid="stMarkdownContainer"] {
            color: #fafafa !important;
        }
        .stCaption, [data-testid="stCaptionContainer"] * {
            color: #9ea3b0 !important;
        }

        /* ── Divider et scrollbar ── */
        hr { border-color: #2d3250 !important; }
        ::-webkit-scrollbar { background-color: #1a1c23; }
        ::-webkit-scrollbar-thumb { background-color: #2d3250; border-radius: 4px; }
        </style>
        """
    else:
        css = """
        <style>
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"],
        [data-testid="block-container"],
        [data-testid="stMainBlockContainer"],
        .main, .block-container {
            background-color: #ffffff !important;
            color: #1a1a2e !important;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

inject_theme()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚖️ LexAI")
    st.caption("The law, accessible to everyone")
    st.divider()

    # Dark / Light mode toggle
    col_icon, col_toggle = st.columns([1, 3])
    with col_icon:
        st.markdown("🌙" if not st.session_state.dark_mode else "☀️")
    with col_toggle:
        dark = st.toggle("Dark mode", value=st.session_state.dark_mode)
    if dark != st.session_state.dark_mode:
        st.session_state.dark_mode = dark
        st.rerun()

    st.divider()

    # Language toggle
    langue = st.radio(
        "Language",
        options=["🇫🇷 Français", "🇬🇧 English"],
        horizontal=True,
    )
    langue_code = "en" if "English" in langue else "fr"
    st.divider()

    # Legal code filter
    st.markdown("**Legal Code**")
    CODES_DISPONIBLES = [
        "All codes",
        "Code Civil",
        "Code Pénal",
        "Code du Travail",
        "Code de Commerce",
        "Code de Procédure Civile",
        "Code de la Consommation",
    ]
    code_filtre_label = st.selectbox("Filter by code", CODES_DISPONIBLES)
    code_filtre = None if code_filtre_label == "All codes" else code_filtre_label

    st.divider()

    # Action buttons
    st.markdown("**Actions**")
    _mode = st.session_state.get("mode", "chat")
    if _mode != "chat":
        if st.button("💬 Retour à la consultation", use_container_width=True):
            st.session_state.plainte_result = None
            st.session_state.contrat_result = None
            st.session_state.mode = "chat"
            st.rerun()
    else:
        if st.button("📝 Rédiger une plainte", use_container_width=True):
            st.session_state.mode = "plainte"
            st.rerun()
        if st.button("📄 Analyser un contrat", use_container_width=True):
            st.session_state.mode = "contrat"
            st.rerun()

    if st.button("👨‍⚖️ Find a lawyer", use_container_width=True):
        st.info("Connecting with partner lawyers will be available soon.")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Translation helper ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def traduire_snippet(texte_fr: str) -> str:
    """Traduit un extrait de texte juridique français vers l'anglais (mis en cache)."""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    result = llm.invoke(
        f"Translate this French legal text to English. Return only the translation, no explanation:\n\n{texte_fr}"
    )
    return result.content


# ── Pipeline loading (once, persistent) ───────────────────────────────────────

@st.cache_resource(show_spinner="Loading legal corpus...")
def charger_pipeline(use_reranking: bool = False):
    docs = charger_corpus("corpus_v3.json")
    vs   = construire_vectorstore(docs)
    chaine, hybrid, reranker, _ = creer_chaine_rag(vs, docs, use_reranking=use_reranking)
    return chaine, hybrid, reranker, docs

chaine, hybrid, reranker, documents = charger_pipeline(use_reranking=True)

# ── Session state ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "chat"
if "plainte_result" not in st.session_state:
    st.session_state.plainte_result = None
if "contrat_result" not in st.session_state:
    st.session_state.contrat_result = None

# ── Mode : Consultation ────────────────────────────────────────────────────────

if st.session_state.mode == "chat":

    st.markdown(
        "<h1 style='text-align:center; font-size:2.8rem; margin-bottom:0'>⚖️ LexAI</h1>"
        "<p style='text-align:center; color:gray; font-size:1rem; margin-top:4px'>"
        "The law, accessible to everyone</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📋 Source articles"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"**{src['code']}** — {src['article']}  \n"
                            f"*{src['domaine']}*"
                        )

    col_input, col_attach = st.columns([11, 1])
    with col_attach:
        if st.button("📎", help="Attach a document (contract, deed, decision) — coming soon"):
            st.toast("Document attachment will be available soon.", icon="📎")

    question = st.chat_input("Ask your legal question...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            sources = hybrid.invoke(question, code_filtre=code_filtre)

            placeholder = st.empty()
            reponse_complete = ""

            with st.spinner("Searching..."):
                for chunk in chaine.stream({"question": question, "langue": langue_code, "code_filtre": code_filtre}):
                    reponse_complete += chunk
                    placeholder.markdown(reponse_complete + "▌")

            placeholder.markdown(reponse_complete)

            sources_uniques = []
            vus = set()
            for doc in sources:
                cle = doc.metadata["article"]
                if cle not in vus:
                    sources_uniques.append(doc)
                    vus.add(cle)

            with st.expander(f"📋 {len(sources_uniques)} source article(s) used"):
                for doc in sources_uniques:
                    snippet = doc.page_content[:350] + "..."
                    if langue_code == "en":
                        snippet = traduire_snippet(snippet)
                    st.markdown(
                        f"**{doc.metadata['code']}** — {doc.metadata['article']}  \n"
                        f"*{doc.metadata['domaine']}*  \n"
                        f"[View on Légifrance]({doc.metadata.get('url', '#')})"
                    )
                    st.caption(snippet)
                    st.divider()

        sources_meta = [
            {"code": d.metadata["code"], "article": d.metadata["article"], "domaine": d.metadata["domaine"]}
            for d in sources_uniques
        ]
        st.session_state.messages.append({
            "role": "assistant",
            "content": reponse_complete,
            "sources": sources_meta,
        })


# ── Mode : Rédaction de plainte ────────────────────────────────────────────────

elif st.session_state.mode == "plainte":

    # En-tête du mode plainte avec bouton retour
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Retour", use_container_width=True):
            st.session_state.plainte_result = None
            st.session_state.mode = "chat"
            st.rerun()
    with col_title:
        st.markdown(
            "<h1 style='font-size:2rem; margin:0'>📝 Rédiger une plainte</h1>"
            "<p style='color:gray; font-size:0.9rem; margin-top:2px'>"
            "Décrivez votre situation — LexAI trouve les articles applicables et rédige le document officiel.</p>",
            unsafe_allow_html=True,
        )
    st.divider()

    with st.form("intake_plainte"):
        # --- Identité du plaignant ---
        st.markdown("**Vos informations personnelles**")
        col_nom, col_adresse, col_ville = st.columns([2, 3, 1])
        with col_nom:
            nom_complet = st.text_input(
                "Nom complet *",
                placeholder="Ex : Martin Sophie",
            )
        with col_adresse:
            adresse = st.text_input(
                "Adresse complète *",
                placeholder="Ex : 12 rue de la Paix, 75001 Paris",
            )
        with col_ville:
            ville = st.text_input(
                "Ville *",
                placeholder="Ex : Paris",
            )

        st.markdown("**Votre litige**")
        col1, col2 = st.columns(2)

        with col1:
            type_litige = st.selectbox(
                "Type de litige *",
                list(TYPE_LITIGE_CONFIG.keys()),
                help="Sélectionnez le domaine juridique correspondant à votre situation.",
            )
            partie_adverse = st.text_input(
                "Partie adverse *",
                placeholder="Ex : Mon employeur SAS Dupont SARL — ou : M. Martin Jean-Pierre",
            )
            date_faits = st.text_input(
                "Date des faits *",
                placeholder="Ex : 15 mars 2026 — ou : entre janvier et mars 2026",
            )

        with col2:
            prejudice = st.text_area(
                "Préjudice subi *",
                placeholder="Ex : Perte de salaire de 3 mois (4 500€), préjudice moral, atteinte à la réputation...",
                height=122,
            )
            demarches = st.text_area(
                "Démarches déjà effectuées (optionnel)",
                placeholder="Ex : Mise en demeure envoyée le 01/04/2026, sans réponse. Tentative de médiation refusée.",
                height=122,
            )

        faits = st.text_area(
            "Description détaillée des faits *",
            placeholder="Décrivez chronologiquement ce qui s'est passé. Plus vous êtes précis, plus la plainte sera solide juridiquement...",
            height=160,
        )

        submitted = st.form_submit_button(
            "⚖️ Générer la plainte",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if not all([nom_complet, adresse, ville, type_litige, partie_adverse, date_faits, faits, prejudice]):
            st.error("Veuillez remplir tous les champs obligatoires (*).")
        else:
            intake = {
                "nom": nom_complet,
                "adresse": adresse,
                "ville": ville,
                "type_litige": type_litige,
                "partie_adverse": partie_adverse,
                "date_faits": date_faits,
                "faits": faits,
                "prejudice": prejudice,
                "demarches": demarches,
            }
            with st.spinner("🔍 Recherche des articles applicables (RAG hybride + fine-tuned embeddings)..."):
                plainte_text, plainte_docs = generer_plainte(hybrid, reranker, intake)
            st.session_state.plainte_result = {
                "text": plainte_text,
                "sources": plainte_docs,
                "type_litige": type_litige,
            }

    if st.session_state.plainte_result:
        result = st.session_state.plainte_result

        st.divider()

        # --- Document généré ---
        st.markdown("### 📄 Plainte générée")
        with st.container(border=True):
            st.markdown(result["text"])

        pdf_plainte = generer_pdf(result["text"], "Plainte Officielle")
        st.download_button(
            label="⬇️ Télécharger la plainte (.pdf)",
            data=pdf_plainte,
            file_name="plainte_lexai.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        # --- Articles de loi utilisés (RAG) ---
        plainte_sources_uniques = []
        vus_plainte = set()
        for doc in result["sources"]:
            cle = doc.metadata["article"]
            if cle not in vus_plainte:
                plainte_sources_uniques.append(doc)
                vus_plainte.add(cle)

        with st.expander(f"📋 {len(plainte_sources_uniques)} article(s) de loi utilisés (corpus Légifrance)"):
            for doc in plainte_sources_uniques:
                st.markdown(
                    f"**{doc.metadata['code']}** — {doc.metadata['article']}  \n"
                    f"*{doc.metadata['domaine']}*  \n"
                    f"[Voir sur Légifrance]({doc.metadata.get('url', '#')})"
                )
                st.caption(doc.page_content[:300] + "...")
                st.divider()

        # --- Que faire maintenant ? ---
        config_litige = TYPE_LITIGE_CONFIG.get(result["type_litige"], {})
        etapes = config_litige.get("etapes", [])

        if etapes:
            st.divider()
            st.markdown(
                f"### {config_litige.get('icone', '📌')} Que faire avec cette plainte ?"
            )
            st.markdown(f"*Procédure pour un litige **{result['type_litige']}***")
            for etape in etapes:
                st.markdown(f"- {etape}")

        st.info(
            "⚠️ Cette plainte est une aide à la rédaction générée par IA à partir du corpus Légifrance. "
            "Elle ne remplace pas les conseils d'un avocat. "
            "Pour les litiges complexes ou les enjeux importants, consultez un professionnel du droit."
        )


# ── Mode : Analyse de contrat ──────────────────────────────────────────────────

elif st.session_state.mode == "contrat":

    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("← Retour", use_container_width=True):
            st.session_state.contrat_result = None
            st.session_state.mode = "chat"
            st.rerun()
    with col_title:
        st.markdown(
            "<h1 style='font-size:2rem; margin:0'>📄 Analyser un contrat</h1>"
            "<p style='color:gray; font-size:0.9rem; margin-top:2px'>"
            "Collez votre contrat — LexAI identifie les clauses conformes, suspectes et illégales "
            "en les confrontant aux articles du corpus Légifrance.</p>",
            unsafe_allow_html=True,
        )
    st.divider()

    with st.form("intake_contrat"):
        col_type, col_info = st.columns([2, 3])
        with col_type:
            type_contrat = st.selectbox(
                "Type de contrat *",
                list(TYPE_CONTRAT_CONFIG.keys()),
                help="Sélectionnez le type de contrat pour cibler les bons articles de loi.",
            )
        with col_info:
            cfg = TYPE_CONTRAT_CONFIG.get(type_contrat, {})
            st.markdown(
                f"<div style='padding:10px 14px; background:#f0f4ff; border-radius:8px; "
                f"border-left:4px solid #3949ab; margin-top:24px; font-size:0.88rem;'>"
                f"{cfg.get('icone','📄')} {cfg.get('hint','')}</div>",
                unsafe_allow_html=True,
            )

        contrat_texte = st.text_area(
            "Texte du contrat *",
            placeholder=(
                "Collez ici le texte complet ou partiel de votre contrat...\n\n"
                "Ex : CONTRAT DE TRAVAIL À DURÉE INDÉTERMINÉE\n"
                "Entre la société XYZ SAS, ci-après « l'Employeur »,\n"
                "Et Madame/Monsieur ..., ci-après « le Salarié »,\n\n"
                "Article 1 — Engagement\n..."
            ),
            height=320,
        )

        submitted_contrat = st.form_submit_button(
            "🔍 Analyser le contrat",
            use_container_width=True,
            type="primary",
        )

    if submitted_contrat:
        if not contrat_texte.strip():
            st.error("Veuillez coller le texte du contrat à analyser.")
        elif len(contrat_texte.strip()) < 80:
            st.error("Le texte du contrat est trop court pour être analysé (minimum 80 caractères).")
        else:
            nb_chars = len(contrat_texte)
            with st.spinner(
                f"🔍 Analyse en cours — RAG sur {nb_chars:,} caractères "
                f"({nb_chars // 200} clauses estimées)..."
            ):
                analyse_text, analyse_docs = analyser_contrat(
                    hybrid, reranker, type_contrat, contrat_texte
                )
            st.session_state.contrat_result = {
                "text": analyse_text,
                "sources": analyse_docs,
                "type_contrat": type_contrat,
            }

    if st.session_state.contrat_result:
        result = st.session_state.contrat_result

        st.divider()
        st.markdown("### 📊 Résultat de l'analyse")

        with st.container(border=True):
            st.markdown(result["text"])

        pdf_contrat = generer_pdf(result["text"], "Analyse de Contrat")
        st.download_button(
            label="⬇️ Télécharger l'analyse (.pdf)",
            data=pdf_contrat,
            file_name="analyse_contrat_lexai.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        # --- Articles utilisés ---
        contrat_sources_uniques = []
        vus_contrat = set()
        for doc in result["sources"]:
            cle = doc.metadata["article"]
            if cle not in vus_contrat:
                contrat_sources_uniques.append(doc)
                vus_contrat.add(cle)

        with st.expander(f"📋 {len(contrat_sources_uniques)} article(s) de loi consultés (corpus Légifrance)"):
            for doc in contrat_sources_uniques:
                st.markdown(
                    f"**{doc.metadata['code']}** — {doc.metadata['article']}  \n"
                    f"*{doc.metadata['domaine']}*  \n"
                    f"[Voir sur Légifrance]({doc.metadata.get('url', '#')})"
                )
                st.caption(doc.page_content[:300] + "...")
                st.divider()

        st.info(
            "⚠️ Cette analyse est générée par IA à partir du corpus Légifrance (2 419 articles). "
            "Elle ne remplace pas l'avis d'un avocat spécialisé. "
            "Pour tout contrat à enjeux importants, consultez un professionnel avant de signer."
        )
