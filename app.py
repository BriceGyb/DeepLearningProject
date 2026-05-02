"""
LexAI — Streamlit Interface
"""

import os
import streamlit as st

# Streamlit Cloud: inject secrets into os.environ before rag_lexai reads them
for _key in ("OPENAI_API_KEY", "PISTE_CLIENT_ID", "PISTE_CLIENT_SECRET"):
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

from rag_lexai import charger_corpus, construire_vectorstore, creer_chaine_rag

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
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"],
        [data-testid="block-container"] {
            background-color: #0f1117 !important;
            color: #fafafa !important;
        }
        [data-testid="stSidebar"] {
            background-color: #1a1c23 !important;
        }
        [data-testid="stSidebar"] * { color: #fafafa !important; }
        [data-testid="stChatMessageContent"] {
            background-color: #1e2130 !important;
            color: #fafafa !important;
        }
        [data-testid="stChatInput"] textarea {
            background-color: #1e2130 !important;
            color: #fafafa !important;
        }
        [data-testid="stExpander"] {
            background-color: #1e2130 !important;
            border-color: #2d3250 !important;
        }
        hr { border-color: #2d3250 !important; }
        .stButton > button {
            background-color: #1e2130 !important;
            color: #fafafa !important;
            border-color: #2d3250 !important;
        }
        p, h1, h2, h3, label, .stMarkdown, .stCaption {
            color: #fafafa !important;
        }
        </style>
        """
    else:
        css = """
        <style>
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"],
        [data-testid="block-container"] {
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

    # Country filter (future placeholder)
    st.markdown("**Jurisdiction**")
    pays = st.selectbox(
        "Country",
        ["🇫🇷 France", "🇧🇪 Belgium (coming soon)", "🇨🇭 Switzerland (coming soon)", "🇲🇦 Morocco (coming soon)", "🇸🇳 Senegal (coming soon)"],
        help="More jurisdictions will be available soon.",
    )
    if pays != "🇫🇷 France":
        st.info("This jurisdiction will be available in a future version.")

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

    # Action buttons (future placeholders)
    st.markdown("**Actions**")
    if st.button("📝 File a complaint", use_container_width=True):
        st.info("This module is coming soon. It will allow you to generate and file an AI-assisted complaint.")

    if st.button("📄 Analyze a document", use_container_width=True):
        st.info("Document analysis (contracts, deeds, decisions) is coming soon.")

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

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='text-align:center; font-size:2.8rem; margin-bottom:0'>⚖️ LexAI</h1>"
    "<p style='text-align:center; color:gray; font-size:1rem; margin-top:4px'>"
    "The law, accessible to everyone</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Conversation history ───────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

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

# ── Input & generation ─────────────────────────────────────────────────────────

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
