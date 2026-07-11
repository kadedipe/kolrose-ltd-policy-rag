"""
Kolrose Policy RAG - Streamlit UI ONLY
======================================
Frontend chat interface
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Backend path
BACKEND_PATH = str(Path(__file__).parent.parent.parent / "BACKEND")
sys.path.insert(0, BACKEND_PATH)

from app.config import COMPANY_INFO, DEFAULT_MODEL
from app.rag_system import KolroseRAG
from app.guardrails import GuardrailSystem
from app.ingestion import load_vectorstore


# =========================
# INIT SYSTEM
# =========================

@st.cache_resource
def init():
    vs = load_vectorstore()
    rag = KolroseRAG(vs) if vs else None
    guard = GuardrailSystem() if vs else None
    return rag, guard, vs


rag, guardrails, vectorstore = init()
SYSTEM_READY = rag is not None


# =========================
# STREAMLIT UI
# =========================

def main():

    st.set_page_config(
        page_title="Kolrose Policy Assistant",
        layout="wide"
    )

    st.title("🏢 Kolrose Policy Assistant")

    if not SYSTEM_READY:
        st.error("System not ready")
        return

    question = st.text_area("Ask a question")

    if st.button("Ask"):

        g = guardrails.check_query(question)

        if g.modified_response:
            st.warning(g.modified_response)
            return

        result = rag.query(question)

        st.markdown("### Answer")
        st.write(result.answer)

        st.markdown("### Citations")
        st.write(result.citations)

        st.markdown("### Sources")
        for s in result.sources:
            st.write(s)


if __name__ == "__main__":
    main()