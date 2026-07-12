"""
Kolrose Policy RAG - Streamlit UI v2.0
======================================
Frontend chat interface with streaming, memory, and authentication.
"""

import streamlit as st
import requests
import os
import json
import time
from datetime import datetime
from PIL import Image

# =========================
# CONFIG
# =========================
BACKEND_URL = os.environ.get(
    "BACKEND_URL", 
    "https://kolrose-backend-production.up.railway.app"
)

# Logo path - tries local file first, falls back to URL
LOGO_PATH = os.path.join(os.path.dirname(__file__), "static", "kolrose.jfif")
LOGO_URL = "https://raw.githubusercontent.com/kadedipe/kolrose-ltd-policy-rag/main/docs/images/kolrose.jfif"

# =========================
# SESSION STATE INIT
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "backend_ready" not in st.session_state:
    st.session_state.backend_ready = False
if "backend_data" not in st.session_state:
    st.session_state.backend_data = None
if "session_id" not in st.session_state:
    st.session_state.session_id = f"conv_{int(time.time())}"
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "use_streaming" not in st.session_state:
    st.session_state.use_streaming = True
if "use_memory" not in st.session_state:
    st.session_state.use_memory = True

# =========================
# STREAMLIT UI SETUP
# =========================
st.set_page_config(
    page_title="Kolrose Policy Assistant",
    page_icon="🏢",
    layout="wide"
)

# =========================
# HELPER FUNCTIONS
# =========================

def call_backend_health():
    """Check backend health with timeout"""
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=30)
        if health.status_code == 200:
            return health.json()
    except:
        pass
    return None


def chat_standard(question: str, k_results: int = 5):
    """Standard chat request"""
    try:
        res = requests.post(
            f"{BACKEND_URL}/chat",
            json={"question": question, "k_results": k_results, "include_snippets": True},
            timeout=120,
            headers=_get_auth_headers(),
        )
        if res.status_code == 200:
            return res.json()
        return {"answer": f"Error ({res.status_code}): {res.text[:200]}", "error": True}
    except Exception as e:
        return {"answer": f"Connection error: {str(e)[:200]}", "error": True}


def chat_with_memory(question: str, session_id: str, k_results: int = 5):
    """Chat with conversation memory"""
    try:
        res = requests.post(
            f"{BACKEND_URL}/chat/memory",
            params={"session_id": session_id},
            json={"question": question, "k_results": k_results, "include_snippets": True},
            timeout=120,
            headers=_get_auth_headers(),
        )
        if res.status_code == 200:
            return res.json()
        return {"answer": f"Error ({res.status_code}): {res.text[:200]}", "error": True}
    except Exception as e:
        return {"answer": f"Connection error: {str(e)[:200]}", "error": True}


def stream_chat_response(question: str, k_results: int = 5):
    """Stream chat response with typing effect"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat/stream",
            json={"question": question, "k_results": k_results, "include_snippets": True},
            stream=True,
            timeout=120,
            headers=_get_auth_headers(),
        )
        if response.status_code != 200:
            yield None, None, None, f"Error: {response.status_code}"
            return

        answer_parts = []
        citations = []
        sources = []
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    msg_type = data.get('type', 'token')

                    if msg_type == 'sources':
                        sources = data.get('sources', [])
                    elif msg_type == 'citations':
                        citations = data.get('citations', [])
                    elif 'token' in data:
                        token = data.get('token', '')
                        done = data.get('done', False)
                        refused = data.get('refused', False)
                        if token:
                            answer_parts.append(token)
                        if done:
                            yield answer_parts, citations, sources, {
                                "refused": refused, "metrics": data.get('metrics', {}), "complete": True
                            }
                            return
                    yield answer_parts, citations, sources, None
    except Exception as e:
        yield None, None, None, f"Error: {str(e)[:200]}"


def _get_auth_headers():
    """Get authentication headers if logged in"""
    headers = {}
    if st.session_state.access_token:
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"
    if st.session_state.session_id:
        headers["X-Session-ID"] = st.session_state.session_id
    return headers


def login_user(username: str, password: str):
    """Login and store access token"""
    try:
        res = requests.post(
            f"{BACKEND_URL}/auth/login",
            data={"username": username, "password": password},
            timeout=30,
        )
        if res.status_code == 200:
            data = res.json()
            st.session_state.access_token = data.get("access_token")
            st.session_state.user_info = {
                "username": data.get("username"),
                "role": data.get("role"),
                "department": data.get("department"),
            }
            return True, "Login successful!"
        return False, res.json().get("detail", "Login failed")
    except Exception as e:
        return False, f"Connection error: {str(e)[:100]}"


def logout_user():
    """Clear authentication state"""
    st.session_state.access_token = None
    st.session_state.user_info = None

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
    # Company branding
    try:
        if os.path.exists(LOGO_PATH):
            logo = Image.open(LOGO_PATH)
            st.image(logo, width=100)
    except:
        st.markdown("### 🏢 Kolrose Limited")

    # =========================
    # LOGIN SECTION
    # =========================
    st.header("🔐 Account")

    if st.session_state.user_info:
        st.success(f"✅ Logged in as **{st.session_state.user_info['username']}**")
        st.caption(f"Role: {st.session_state.user_info['role']}")
        if st.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.rerun()
    else:
        with st.expander("🔑 Login"):
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", use_container_width=True):
                success, msg = login_user(login_username, login_password)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    st.divider()

    # =========================
    # CONNECTION STATUS
    # =========================
    st.header("🔗 Connection Status")

    with st.spinner("Checking backend..."):
        data = call_backend_health()
        if data:
            st.session_state.backend_data = data
            if data.get("ready", False):
                st.session_state.backend_ready = True
                st.success("✅ Backend Online")
                components = data.get("components", {})
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("📄 Policies", "12")
                with col_b:
                    st.metric("🧠 RAG", "Ready" if components.get("rag_system", {}).get("ready") else "Loading")
            elif data.get("status") == "warming_up":
                st.session_state.backend_ready = False
                st.warning("⏳ System warming up...")
                unready = data.get("unready", [])
                if unready:
                    st.caption(f"Loading: {', '.join(unready)}")
            else:
                st.warning("⚠️ Status unknown")
        else:
            st.error("❌ Backend Offline")

    if st.button("🔄 Retry Connection", use_container_width=True):
        st.rerun()

    st.divider()

    # =========================
    # CONVERSATION MEMORY
    # =========================
    st.header("💬 Conversation Memory")

    use_memory = st.checkbox("Enable follow-up memory", value=st.session_state.use_memory,
                             help="Remember context for follow-up questions")
    st.session_state.use_memory = use_memory

    msg_count = len(st.session_state.messages)
    if msg_count > 0:
        st.caption(f"📝 {msg_count} messages in session")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.session_id = f"conv_{int(time.time())}"
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # =========================
    # SETTINGS
    # =========================
    st.subheader("⚙️ Settings")
    k_results = st.slider("Results to retrieve", 3, 10, 5)
    use_streaming = st.checkbox("Streaming responses (typing effect)", value=st.session_state.use_streaming)
    st.session_state.use_streaming = use_streaming

    st.divider()

    # Footer
    st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption("© 2024 Kolrose Limited")
    st.caption("Suite 10, Bataiya Plaza")
    st.caption("Area 2 Garki, Abuja, FCT")

# =========================
# MAIN CHAT INTERFACE
# =========================

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
    # Display user message
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        if not st.session_state.backend_ready:
            message_placeholder.warning(
                "⏳ The backend is still warming up. Please wait a moment and try again.\n\n"
                "Click '🔄 Retry Connection' in the sidebar to check status."
            )
        else:
            # =========================
            # STREAMING MODE
            # =========================
            if st.session_state.use_streaming:
                full_answer = ""
                all_citations = []
                all_sources = []

                message_placeholder.markdown("🔍 Searching policies...")

                for answer_parts, citations, sources, extra in stream_chat_response(question, k_results):
                    if isinstance(extra, str) and extra.startswith("Error"):
                        message_placeholder.error(extra)
                        break

                    if answer_parts is not None:
                        full_answer = "".join(answer_parts)
                        if full_answer.strip():
                            message_placeholder.markdown(full_answer + "▌")
                        else:
                            message_placeholder.markdown("▌")

                    if citations:
                        all_citations = citations
                    if sources:
                        all_sources = sources

                    if extra and extra.get("complete"):
                        message_placeholder.markdown(full_answer)

                        if all_citations:
                            with st.expander("📚 Citations"):
                                for citation in all_citations:
                                    st.markdown(f"- {citation}")
                        if all_sources:
                            with st.expander("📄 Sources"):
                                for source in all_sources:
                                    st.markdown(f"""
                                    **{source.get('document_id', 'N/A')}** — {source.get('policy_name', 'Unknown')}  
                                    📁 {source.get('source_file', 'N/A')}
                                    """)

                        st.session_state.messages.append({
                            "role": "assistant", "content": full_answer,
                            "citations": all_citations, "sources": all_sources,
                        })
                        break

            # =========================
            # STANDARD/MEMORY MODE
            # =========================
            else:
                message_placeholder.markdown("🔍 Searching policies...")

                if st.session_state.use_memory:
                    data = chat_with_memory(question, st.session_state.session_id, k_results)
                    if data.get("metrics", {}).get("is_follow_up"):
                        st.caption("💡 Using conversation context for follow-up")
                else:
                    data = chat_standard(question, k_results)

                answer = data.get("answer", "No answer returned")
                citations = data.get("citations", [])
                sources = data.get("sources", [])
                refused = data.get("refused", False)
                is_error = data.get("error", False)

                if is_error:
                    message_placeholder.error(answer)
                elif refused:
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
                    "role": "assistant", "content": answer,
                    "citations": citations, "sources": sources,
                })