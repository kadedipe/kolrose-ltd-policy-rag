"""
Kolrose Policy RAG - Streamlit UI
======================================
Frontend chat interface connecting to production backend
"""

import streamlit as st
import requests

# =========================
# CONFIG
# =========================
# Points to your live production backend running on Railway
API_URL = "https://melodious-perception-production-32cd.up.railway.app"

# =========================
# STREAMLIT UI SETUP
# =========================
st.set_page_config(
    page_title="Kolrose Policy Assistant",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Kolrose Policy Assistant")
st.caption("Production Interface (Connected to Railway Backend)")

# =========================
# CHAT INPUT & ACTION
# =========================
question = st.text_input("Ask a policy question")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Shoot an API request over to your live backend
                res = requests.post(
                    f"{API_URL}/chat", 
                    json={"question": question}
                )

                if res.status_code == 200:
                    data = res.json()

                    st.markdown("### 📋 Answer")
                    st.write(data.get("answer", "No answer returned"))

                    if data.get("citations"):
                        st.markdown("### 📚 Citations")
                        st.write(data["citations"])

                    if data.get("sources"):
                        st.markdown("### 📄 Sources")
                        for s in data["sources"]:
                            st.write(s)
                else:
                    st.error(f"Backend error: {res.status_code} - {res.text}")

            except Exception as e:
                st.error(f"Connection error: Could not reach backend. {str(e)}")