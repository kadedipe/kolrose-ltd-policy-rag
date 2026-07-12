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
LOGO_URL = "https://raw.githubusercontent.com/kadedipe/kolrose-ltd-policy-rag/main/docs/images/kolrose.jfif"

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
    
    # Initialize session state for backend status
    if "backend_ready" not in st.session_state:
        st.session_state.backend_ready = False
    if "backend_data" not in st.session_state:
        st.session_state.backend_data = None
    
    with st.spinner("Checking backend (may take up to 30s for cold start)..."):
        try:
            health = requests.get(f"{BACKEND_URL}/health", timeout=30)
            if health.status_code == 200:
                data = health.json()
                st.session_state.backend_data = data
                
                # CHECK IF SYSTEM IS FULLY READY OR STILL WARMING UP
                if data.get("ready", False):
                    st.session_state.backend_ready = True
                    st.success("✅ Backend Online")
                    
                    # Show system components
                    components = data.get("components", {})
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if components.get("policies"):
                            st.metric("📄 Policies", components["policies"])
                    with col_b:
                        if components.get("rag_system"):
                            st.metric("🧠 RAG System", "Ready")
                    
                    if components.get("guardrails"):
                        st.metric("🛡️ Guardrails", "Active")
                        
                elif data.get("status") == "warming_up":
                    st.session_state.backend_ready = False
                    st.warning("⏳ System is warming up...")
                    st.caption("Loading embedding models and initializing RAG system.")
                    st.caption("This usually takes 30-60 seconds.")
                    st.progress(0.5, "Loading models...")
                    
                else:
                    st.session_state.backend_ready = False
                    st.warning("⚠️ Backend status unknown")
                    
            else:
                st.session_state.backend_ready = False
                st.error(f"❌ Backend Error ({health.status_code})")
                
        except requests.exceptions.Timeout:
            st.session_state.backend_ready = False
            st.warning("⏱️ Backend is waking up (cold start). Please wait a moment and refresh.")
        except requests.exceptions.ConnectionError:
            st.session_state.backend_ready = False
            st.error("❌ Backend Offline. Check if backend service is running.")
        except Exception as e:
            st.session_state.backend_ready = False
            st.warning(f"⚠️ {str(e)[:50]}")
    
    # Manual retry button
    if st.button("🔄 Retry Connection", use_container_width=True):
        st.session_state.backend_ready = False
        st.session_state.backend_data = None
        st.rerun()
    
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
        
        # CHECK IF BACKEND IS READY BEFORE MAKING REQUEST
        if not st.session_state.backend_ready:
            message_placeholder.warning(
                "⏳ The backend is still warming up. Please wait a moment and try again.\n\n"
                "Click '🔄 Retry Connection' in the sidebar to check status."
            )
        else:
            message_placeholder.markdown("🔍 Searching policies...")
            
            try:
                res = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "question": question,
                        "k_results": k_results,
                        "include_snippets": True
                    },
                    timeout=120
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
                    
                elif res.status_code == 503:
                    message_placeholder.warning(
                        "⏳ System is still starting up.\n\n"
                        "The embedding models are loading. This happens on cold starts.\n"
                        "Please wait 30-60 seconds and try again."
                    )
                else:
                    message_placeholder.error(f"❌ Backend Error ({res.status_code}): {res.text[:200]}")
                    
            except requests.exceptions.Timeout:
                message_placeholder.error("⏱️ Request timed out. The backend may be starting up. Please try again in 30 seconds.")
            except requests.exceptions.ConnectionError:
                message_placeholder.error(f"❌ Cannot connect to backend.\n\nURL: {BACKEND_URL}")
            except Exception as e:
                message_placeholder.error(f"❌ Error: {str(e)[:200]}")