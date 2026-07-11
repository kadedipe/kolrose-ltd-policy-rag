"""
Kolrose Policy RAG - Streamlit UI
======================================
Frontend chat interface communicating over HTTP
"""

import streamlit as st
import requests

# =========================
# CONFIG
# =========================
# Points directly to your live production Railway FastAPI backend
API_URL = "https://melodious-perception-production-32cd.up.railway.app"

# =========================
# STREAMLIT UI LAYOUT
# =========================
st.set_page_config(
    page_title="Kolrose Policy Assistant",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Kolrose Policy Assistant")
st.caption("AI-powered company policy assistant (FastAPI backend + Streamlit frontend)")

# Input text area for user query
question = st.text_area("Ask a company policy question", placeholder="e.g., What is the policy on remote work?")

if st.button("Ask Assistant"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Send HTTP POST request to your backend /chat endpoint
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question, "k_results": 5},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()

                    # Handle blocked query via Guardrails
                    if data.get("refused"):
                        st.warning(data.get("answer"))
                    else:
                        # Render Core Answer
                        st.markdown("### 📋 Answer")
                        st.write(data.get("answer", "No answer returned."))

                        # Render Citations
                        if data.get("citations"):
                            st.markdown("### 📚 Citations")
                            st.write(data["citations"])

                        # Render Structured Sources Dropdowns
                        if data.get("sources"):
                            st.markdown("### 📄 Matching Sources")
                            for source in data["sources"]:
                                title = f"{source.get('policy_name', 'Policy')} - Section: {source.get('section', 'N/A')}"
                                with st.expander(title):
                                    st.markdown(f"**Source File:** `{source.get('source_file')}`")
                                    if source.get("snippet"):
                                        st.markdown(f"**Excerpt:** *\"{source.get('snippet')}\"*")
                                    if source.get("relevance_score"):
                                        st.caption(f"Relevance Score: {round(source.get('relevance_score'), 4)}")

                elif response.status_code in [503, 533]:
                    st.error("❌ The backend RAG system is still initializing its vector database. Please try again in a moment.")
                else:
                    st.error(f"⚠️ Backend returned an unexpected status code: {response.status_code}")

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection error: Could not reach the backend API at {API_URL}. Details: {e}")