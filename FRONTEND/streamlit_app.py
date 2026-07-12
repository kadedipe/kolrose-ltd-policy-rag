"""
Kolrose Policy RAG - Streamlit UI
======================================
Frontend chat interface connecting to production backend
"""

import streamlit as st
import requests
import os
from datetime import datetime
from PIL import Image

# =========================
# CONFIG
# =========================
BACKEND_URL = os.environ.get(
    "BACKEND_URL", 
    "https://melodious-perception-production-32cd.up.railway.app"
)

# Logo path - tries local file first, falls back to URL
LOGO_PATH = os.path.join(os.path.dirname(__file__), "static", "kolrose.jfif")
LOGO_URL = "https://raw.githubusercontent.com/your-repo/main/docs/images/kolrose.jfif"

# =========================
# STREAMLIT UI SETUP
# =========================
st.set_page_config(
    page_title="Kolrose Policy Assistant",
    page_icon="🏢",
    layout="wide"
)

# =========================
# CUSTOM HEADER WITH LOGO
# =========================
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    try:
        if os.path.exists(LOGO_PATH):
            logo = Image.open(LOGO_PATH)
            st.image(logo, width=150)
        else:
            st.image(LOGO_URL, width=150)
    except:
        st.markdown("<h1 style='text-align: center;'>🏢</h1>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='text-align: center; padding-top: 20px;'>
        <h1>Kolrose Policy Assistant</h1>
        <p style='color: gray; font-size: 1.1rem;'>
            AI-Powered Policy Knowledge Base | 📍 Abuja, FCT, Nigeria
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("<br>", unsafe_allow_html=True)

st.divider()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    # Company branding in sidebar
    try:
        if os.path.exists(LOGO_PATH):
            logo = Image.open(LOGO_PATH)
            st.image(logo, width=100)
    except:
        st.markdown("### 🏢 Kolrose Limited")
    
    st.header("🔗 Connection Status")
    
    with st.spinner("Checking backend..."):
        try:
            health = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if health.status_code == 200:
                data = health.json()
                st.success("✅ Backend Online")
                
                components = data.get("components", {})
                if components.get("policies"):
                    st.metric("📄 Policies Available", components["policies"])
                if components.get("guardrails"):
                    st.metric("🛡️ Guardrails", "Active")
            else:
                st.error(f"❌ Backend Error ({health.status_code})")
        except requests.exceptions.Timeout:
            st.warning("⏱️ Backend Slow")
        except requests.exceptions.ConnectionError:
            st.error("❌ Backend Offline")
        except Exception as e:
            st.warning(f"⚠️ {str(e)[:50]}")
    
    st.divider()
    
    st.subheader("⚙️ Settings")
    k_results = st.slider("Results to retrieve", 3, 10, 5)
    
    st.divider()
    
    # Footer in sidebar
    st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption("© 2024 Kolrose Limited")
    st.caption("Suite 10, Bataiya Plaza")
    st.caption("Area 2 Garki, Abuja, FCT")

# =========================
# MAIN CHAT INTERFACE
# =========================

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message.get("citations"):
            with st.expander("📚 Citations"):
                for citation in message["citations"]:
                    st.markdown(f"- {citation}")
        
        if message.get("sources"):
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.markdown(f"""
                    **{source.get('document_id', 'N/A')}** — {source.get('policy_name', 'Unknown')}  
                    📁 {source.get('source_file', 'N/A')}
                    """)

# Example questions
with st.expander("💡 Example Questions"):
    examples = [
        "What is the annual leave entitlement?",
        "How do I request remote work?",
        "What are the password requirements?",
        "How are travel expenses reimbursed?",
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_question = ex

# Chat input
question = st.chat_input("Ask a policy question...")

# Handle pending question
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question

if question:
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 Searching policies...")
        
        try:
            res = requests.post(
                f"{BACKEND_URL}/chat",
                json={
                    "question": question,
                    "k_results": k_results,
                    "include_snippets": True
                },
                timeout=60
            )
            
            if res.status_code == 200:
                data = res.json()
                answer = data.get("answer", "No answer returned")
                citations = data.get("citations", [])
                sources = data.get("sources", [])
                refused = data.get("refused", False)
                
                if refused:
                    message_placeholder.warning(answer)
                else:
                    message_placeholder.markdown(answer)
                
                if citations:
                    with st.expander("📚 Citations"):
                        for citation in citations:
                            st.markdown(f"- {citation}")
                
                if sources:
                    with st.expander("📄 Sources"):
                        for source in sources:
                            st.markdown(f"""
                            **{source.get('document_id', 'N/A')}** — {source.get('policy_name', 'Unknown')}  
                            📁 {source.get('source_file', 'N/A')}
                            """)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "sources": sources
                })
                
            else:
                message_placeholder.error(f"❌ Backend Error ({res.status_code})")
                
        except requests.exceptions.Timeout:
            message_placeholder.error("⏱️ Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            message_placeholder.error(f"❌ Cannot connect to backend.\n\nURL: {BACKEND_URL}")
        except Exception as e:
            message_placeholder.error(f"❌ Error: {str(e)[:200]}")