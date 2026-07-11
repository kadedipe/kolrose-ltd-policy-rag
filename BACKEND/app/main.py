# app/main.py
"""
Kolrose Limited - Policy Assistant Web Application
===================================================
Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria

Endpoints:
  /        - Web chat interface (Streamlit)
  /chat    - API endpoint (POST) - returns answers with citations
  /health  - Health check (GET) - returns system status
"""

import streamlit as st
import sys
import os
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import (
    COMPANY_INFO, CHROMA_PATH, POLICIES_PATH, OPENROUTER_API_KEY,
    DEFAULT_MODEL, RETRIEVAL_K, FINAL_K, MAX_OUTPUT_TOKENS,
)
from app.rag_system import KolroseRAG, TopicClassifier
from app.guardrails import GuardrailSystem
from app.ingestion import load_vectorstore, load_embeddings, check_policies_exist

# ============================================================================
# FASTAPI SETUP (for /chat and /health endpoints)
# ============================================================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Create FastAPI app
fastapi_app = FastAPI(
    title=f"{COMPANY_INFO['name']} - Policy Assistant API",
    description="RAG-based API for querying company policies with citations",
    version="1.0.0",
)

# CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for /chat endpoint"""
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Policy question to ask",
        examples=["What is the annual leave entitlement?"]
    )
    user_id: Optional[str] = Field(
        default="anonymous",
        description="User identifier for logging"
    )
    include_snippets: bool = Field(
        default=True,
        description="Include source text snippets in response"
    )
    k_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to retrieve"
    )

class SourceInfo(BaseModel):
    """Source document information"""
    document_id: str
    policy_name: str
    source_file: str
    section: str
    snippet: Optional[str] = None
    relevance_score: Optional[float] = None

class ChatResponse(BaseModel):
    """Response model for /chat endpoint"""
    question: str
    answer: str
    citations: List[str]
    sources: List[SourceInfo]
    refused: bool = False
    category: str = "in_corpus"
    metrics: Dict = {}
    timestamp: str

class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str
    version: str
    company: str
    location: str
    timestamp: str
    components: Dict
    stats: Dict

# ============================================================================
# INITIALIZE RAG SYSTEM (Cached)
# ============================================================================
@st.cache_resource(show_spinner=False)
def init_rag_system():
    """Initialize the RAG system (cached across sessions)"""
    vectorstore = load_vectorstore()
    
    if vectorstore is None:
        return None, "Vector store not found. Run ingestion first."
    
    rag = KolroseRAG(vectorstore)
    guardrails = GuardrailSystem()
    
    return {
        'rag': rag,
        'guardrails': guardrails,
        'vectorstore': vectorstore,
    }, "System ready"

# Initialize
system_data, init_msg = init_rag_system()

if system_data:
    rag = system_data['rag']
    guardrails = system_data['guardrails']
    vectorstore = system_data['vectorstore']
    SYSTEM_READY = True
else:
    SYSTEM_READY = False

# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

@fastapi_app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns system status and component information.
    """
    # Check vector store
    vector_count = 0
    if SYSTEM_READY and vectorstore:
        try:
            vector_count = vectorstore._collection.count()
        except:
            pass
    
    # Check policies
    policies_exist, policy_count, _ = check_policies_exist()
    
    return HealthResponse(
        status="healthy" if SYSTEM_READY else "degraded",
        version="1.0.0",
        company=COMPANY_INFO['name'],
        location=COMPANY_INFO['address'],
        timestamp=datetime.now().isoformat(),
        components={
            "vector_store": {
                "status": "connected" if SYSTEM_READY else "unavailable",
                "vectors": vector_count,
                "path": CHROMA_PATH,
            },
            "policies": {
                "status": "available" if policies_exist else "missing",
                "count": policy_count,
                "path": POLICIES_PATH,
            },
            "llm": {
                "status": "configured" if OPENROUTER_API_KEY else "missing_key",
                "model": DEFAULT_MODEL,
                "provider": "OpenRouter",
            },
            "guardrails": {
                "status": "active",
            },
        },
        stats={
            "total_queries": guardrails.stats['total_queries'] if SYSTEM_READY else 0,
            "blocked_queries": guardrails.stats['blocked_queries'] if SYSTEM_READY else 0,
            "uptime_seconds": time.time() - START_TIME if 'START_TIME' in globals() else 0,
        },
    )


@fastapi_app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat API endpoint.
    Receives a policy question and returns an answer with citations and source snippets.
    """
    if not SYSTEM_READY:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Run ingestion and check configuration."
        )
    
    try:
        # Apply guardrails to query
        guardrail_result = guardrails.check_query(request.question)
        
        if guardrail_result.modified_response:
            # Query was blocked
            return ChatResponse(
                question=request.question,
                answer=guardrail_result.modified_response,
                citations=[],
                sources=[],
                refused=True,
                category=guardrail_result.metadata.get('topic', 'blocked'),
                metrics={'total_ms': 0},
                timestamp=datetime.now().isoformat(),
            )
        
        # Execute RAG query
        result = rag.query(
            request.question,
            k_final=request.k_results,
            enable_rerank=True,
            enable_guardrails=False,  # Already checked above
        )
        
        # Format sources with snippets
        sources = []
        for src in result.sources:
            source_info = SourceInfo(
                document_id=src.get('document_id', 'Unknown'),
                policy_name=src.get('policy_name', 'Unknown'),
                source_file=src.get('source_file', 'Unknown'),
                section=src.get('section', 'N/A'),
                snippet=src.get('snippet', '') if request.include_snippets else None,
                relevance_score=src.get('score'),
            )
            sources.append(source_info)
        
        # Apply response guardrails
        final_response, response_results = guardrails.check_response(
            result.answer,
            source_docs=result.sources,
        )
        
        return ChatResponse(
            question=request.question,
            answer=final_response,
            citations=result.citations,
            sources=sources,
            refused=False,
            category=result.category,
            metrics=result.metrics,
            timestamp=datetime.now().isoformat(),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ============================================================================
# STREAMLIT UI (Web Chat Interface - /)
# ============================================================================

def render_web_ui():
    """Render the Streamlit web chat interface"""
    
    # Page config
    st.set_page_config(
        page_title=f"{COMPANY_INFO['name']} - Policy Assistant",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        /* Main header */
        .main-header {
            background: linear-gradient(135deg, #1a5276, #2e86c1);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 25px;
        }
        .main-header h1 { color: white; margin: 0; font-size: 2.2rem; }
        .main-header p { color: #d4e6f1; margin: 5px 0 0 0; font-size: 1rem; }
        
        /* Answer box */
        .answer-box {
            background: linear-gradient(135deg, #f0f8ff, #e8f4f8);
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #1a5276;
            margin: 15px 0;
            font-size: 1.05rem;
            line-height: 1.6;
        }
        
        /* Refusal box */
        .refusal-box {
            background: linear-gradient(135deg, #fff8e1, #fff3cd);
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #ffc107;
            margin: 15px 0;
        }
        
        /* Citation tag */
        .citation-tag {
            display: inline-block;
            background: #1a5276;
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.85rem;
            margin: 3px;
            font-family: monospace;
        }
        
        /* Source card */
        .source-card {
            background: #f8f9fa;
            padding: 12px 15px;
            border-radius: 8px;
            margin: 8px 0;
            border: 1px solid #dee2e6;
            font-size: 0.9rem;
        }
        .source-card .snippet {
            background: #fff;
            padding: 8px;
            border-radius: 4px;
            margin-top: 8px;
            font-family: monospace;
            font-size: 0.85rem;
            border-left: 3px solid #1a5276;
        }
        
        /* Metrics */
        .metrics-row {
            display: flex;
            justify-content: space-around;
            margin: 15px 0;
        }
        .metric-item {
            text-align: center;
            padding: 10px;
            background: #e8f5e9;
            border-radius: 8px;
            min-width: 100px;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1a5276;
        }
        .metric-label {
            font-size: 0.8rem;
            color: #666;
        }
        
        /* Chat input */
        .stTextArea textarea {
            font-size: 1rem;
            border-radius: 10px;
            border: 2px solid #1a5276;
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #1a5276, #2e86c1);
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 1.1rem;
            border-radius: 10px;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(26, 82, 118, 0.3);
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: #999;
            font-size: 0.8rem;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }
        
        /* API info box */
        .api-info {
            background: #f0f0f0;
            padding: 10px 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.85rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # HEADER
    # ========================================================================
    st.markdown(f"""
    <div class="main-header">
        <h1>🏢 {COMPANY_INFO['name']}</h1>
        <p>AI-Powered Policy Assistant</p>
        <p style="font-size:0.85rem;">📍 {COMPANY_INFO['address']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key
        api_key = st.text_input(
            "OpenRouter API Key",
            value=OPENROUTER_API_KEY,
            type="password",
            help="Get free key at https://openrouter.ai/keys"
        )
        if api_key:
            os.environ['OPENROUTER_API_KEY'] = api_key
        
        st.divider()
        
        # Retrieval settings
        st.subheader("📊 Retrieval")
        k_retrieve = st.slider("Initial candidates", 10, 30, RETRIEVAL_K)
        k_final = st.slider("Final results", 1, 10, FINAL_K)
        use_rerank = st.checkbox("Cross-encoder re-ranking", value=True)
        show_snippets = st.checkbox("Show source snippets", value=True)
        
        st.divider()
        
        # System status
        st.subheader("📊 System Status")
        if SYSTEM_READY:
            st.success("✅ System Ready")
            try:
                count = vectorstore._collection.count()
                st.metric("Indexed Chunks", count)
            except:
                st.metric("Indexed Chunks", "N/A")
        else:
            st.error(f"⚠️ {init_msg}")
        
        st.divider()
        
        # API info
        st.subheader("🔗 API Endpoints")
        st.markdown("""
        <div class="api-info">
        <b>POST /chat</b><br>
        curl -X POST http://localhost:8000/chat \<br>
          -H "Content-Type: application/json" \<br>
          -d '{"question":"..."}'<br><br>
        <b>GET /health</b><br>
        curl http://localhost:8000/health
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        st.caption(f"💰 Running on free infrastructure")
        st.caption(f"📍 {COMPANY_INFO['address']}")
    
    # ========================================================================
    # MAIN CONTENT - TABS
    # ========================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat", "📚 Policies", "🔗 API", "ℹ️ About"
    ])
    
    # ========================================================================
    # TAB 1: CHAT INTERFACE
    # ========================================================================
    with tab1:
        st.subheader("Ask a Policy Question")
        
        # Example questions
        with st.expander("💡 Example Questions"):
            cols = st.columns(3)
            examples = [
                "What is the annual leave entitlement?",
                "How do I request remote work?",
                "What are the password requirements?",
                "How are travel expenses reimbursed?",
                "What training budget is available?",
                "Where is Kolrose Limited located?",
                "Can I carry over unused vacation days?",
                "How do I report a security incident?",
                "What is the procurement threshold?",
            ]
            for i, ex in enumerate(examples):
                if cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
                    st.session_state['question'] = ex
        
        # Question input
        question = st.text_area(
            "Your question:",
            value=st.session_state.get('question', ''),
            placeholder="e.g., What is the annual leave policy for new employees?",
            height=100,
            key="question_input",
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)
        
        # Process question
        if ask_clicked and question and SYSTEM_READY:
            with st.spinner("🔍 Searching policy documents..."):
                try:
                    # Apply guardrails
                    guardrail_result = guardrails.check_query(question)
                    
                    if guardrail_result.modified_response:
                        # Query blocked
                        st.markdown(
                            f'<div class="refusal-box">🛡️ {guardrail_result.modified_response}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        # Execute RAG
                        result = rag.query(
                            question,
                            k_retrieve=k_retrieve,
                            k_final=k_final,
                            enable_rerank=use_rerank,
                        )
                        
                        # Store in session
                        st.session_state['last_answer'] = result
                        
                        # Display answer
                        st.markdown("### 📋 Answer")
                        st.markdown(
                            f'<div class="answer-box">{result.answer}</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Metrics row
                        m1, m2, m3, m4 = st.columns(4)
                        metrics = result.metrics
                        with m1:
                            st.markdown(
                                f'<div class="metric-item">'
                                f'<div class="metric-value">{metrics.get("total_ms", 0)}ms</div>'
                                f'<div class="metric-label">⏱️ Response</div></div>',
                                unsafe_allow_html=True
                            )
                        with m2:
                            st.markdown(
                                f'<div class="metric-item">'
                                f'<div class="metric-value">{metrics.get("num_sources", 0)}</div>'
                                f'<div class="metric-label">📚 Sources</div></div>',
                                unsafe_allow_html=True
                            )
                        with m3:
                            st.markdown(
                                f'<div class="metric-item">'
                                f'<div class="metric-value">{metrics.get("num_citations", 0)}</div>'
                                f'<div class="metric-label">📝 Citations</div></div>',
                                unsafe_allow_html=True
                            )
                        with m4:
                            st.markdown(
                                f'<div class="metric-item">'
                                f'<div class="metric-value">FREE</div>'
                                f'<div class="metric-label">💰 Cost</div></div>',
                                unsafe_allow_html=True
                            )
                        
                        # Citations
                        if result.citations:
                            st.markdown("### 📝 Document Citations")
                            cites_html = " ".join(
                                f'<span class="citation-tag">📄 {c}</span>'
                                for c in result.citations
                            )
                            st.markdown(cites_html, unsafe_allow_html=True)
                        
                        # Sources with snippets
                        if result.sources:
                            st.markdown("### 📚 Source Documents")
                            for src in result.sources:
                                doc_id = src.get('document_id', 'N/A')
                                policy = src.get('policy_name', 'Unknown')
                                section = src.get('section', 'N/A')
                                source_file = src.get('source_file', 'Unknown')
                                
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>[{doc_id}]</strong> {policy}<br>
                                    <small>📁 Section: {section} | 📄 {source_file}</small>
                                </div>
                                """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif ask_clicked and not SYSTEM_READY:
            st.error(f"System not initialized. {init_msg}")
    
    # ========================================================================
    # TAB 2: POLICY BROWSER
    # ========================================================================
    with tab2:
        st.subheader("📚 Available Policy Documents")
        
        policies = [
            ("KOL-HR-001", "Employee Handbook", "HR", "8", "📘"),
            ("KOL-HR-002", "Leave and Time-Off Policy", "HR", "10", "🏖️"),
            ("KOL-HR-003", "Code of Conduct and Ethics", "HR/Legal", "12", "⚖️"),
            ("KOL-HR-005", "Remote Work Policy", "HR", "10", "🏠"),
            ("KOL-IT-001", "IT Security & Acceptable Use", "IT", "14", "🔒"),
            ("KOL-FIN-001", "Expenses & Reimbursement", "Finance", "12", "💰"),
            ("KOL-HR-006", "Performance Management", "HR", "12", "📈"),
            ("KOL-HR-007", "Training & Development", "HR", "10", "🎓"),
            ("KOL-ADMIN-001", "Business Travel Policy", "Admin", "14", "✈️"),
            ("KOL-FIN-002", "Procurement Policy", "Finance", "14", "📦"),
            ("KOL-ADMIN-002", "Health & Safety Policy", "Admin", "12", "🏥"),
            ("KOL-HR-008", "Grievance & Dispute Resolution", "HR", "12", "📋"),
        ]
        
        cols = st.columns(3)
        for i, (doc_id, name, dept, pages, icon) in enumerate(policies):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="background:#f8f9fa; padding:15px; border-radius:10px;
                            margin:10px 0; border-top:3px solid #1a5276;">
                    <span style="font-size:1.5rem;">{icon}</span><br>
                    <strong style="color:#1a5276;">[{doc_id}]</strong><br>
                    <strong>{name}</strong><br>
                    <small>📁 {dept} | 📄 ~{pages} pages</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.caption(f"Total: {len(policies)} policy documents | ~140 pages | 250 indexed chunks")
    
    # ========================================================================
    # TAB 3: API DOCUMENTATION
    # ========================================================================
    with tab3:
        st.subheader("🔗 API Documentation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### POST /chat")
            st.markdown("Send a policy question and get an answer with citations.")
            
            st.markdown("**Request:**")
            st.code("""{
  "question": "What is the annual leave policy?",
  "user_id": "employee-123",
  "include_snippets": true,
  "k_results": 5
}""", language="json")
            
            st.markdown("**cURL Example:**")
            st.code("""curl -X POST http://localhost:8000/chat \\
  -H "Content-Type: application/json" \\
  -d '{"question":"What is the leave policy?"}'""", language="bash")
        
        with col2:
            st.markdown("### GET /health")
            st.markdown("Check system status and component health.")
            
            st.markdown("**Response:**")
            st.code("""{
  "status": "healthy",
  "version": "1.0.0",
  "company": "Kolrose Limited",
  "components": {
    "vector_store": {"status": "connected"},
    "llm": {"status": "configured"}
  }
}""", language="json")
            
            st.markdown("**cURL Example:**")
            st.code("curl http://localhost:8000/health", language="bash")
        
        st.divider()
        st.markdown("### 📋 API Response Fields")
        st.markdown("""
        | Field | Type | Description |
        |-------|------|-------------|
        | `answer` | string | Generated response with citations |
        | `citations` | array | List of document citations [KOL-XX-NNN] |
        | `sources` | array | Source documents with metadata and snippets |
        | `refused` | boolean | Whether query was blocked by guardrails |
        | `metrics` | object | Performance metrics (latency, count) |
        """)
    
    # ========================================================================
    # TAB 4: ABOUT
    # ========================================================================
    with tab4:
        st.subheader("ℹ️ About This System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            ### 🏢 {COMPANY_INFO['name']}
            **Address:** {COMPANY_INFO['address']}
            
            ### 🔧 Technical Stack
            - **Framework:** LangChain + Streamlit + FastAPI
            - **LLM:** OpenRouter ({DEFAULT_MODEL})
            - **Embeddings:** all-MiniLM-L6-v2 (Local, Free)
            - **Vector DB:** ChromaDB (Local)
            - **Chunking:** Header-aware semantic
            - **Retrieval:** MMR + Cross-encoder re-ranking
            
            ### 🛡️ Guardrails
            - ✅ Corpus boundary detection
            - ✅ Output length limiting
            - ✅ Mandatory citation enforcement
            - ✅ Sensitive topic handling
            """)
        
        with col2:
            st.markdown("""
            ### 💰 Costs
            | Component | Cost |
            |-----------|------|
            | LLM (OpenRouter) | FREE |
            | Embeddings | FREE |
            | Vector DB | FREE |
            | Hosting | FREE |
            | **Total** | **$0/month** |
            
            ### 📊 Performance
            - Avg Latency: 1.5-2.5 seconds
            - Documents: 12 policies
            - Chunks: 250 indexed
            - Embedding Dimension: 384
            
            ### 🔗 Endpoints
            - `GET /` - Web chat interface (this page)
            - `POST /chat` - API query endpoint
            - `GET /health` - Health check endpoint
            """)
    
    # Footer
    st.markdown(f"""
    <div class="footer">
        © 2024 {COMPANY_INFO['name']} | {COMPANY_INFO['address']}<br>
        For HR inquiries: {COMPANY_INFO['email_hr']}
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point - runs both Streamlit and FastAPI"""
    import argparse
    import threading
    
    parser = argparse.ArgumentParser(description="Kolrose Policy Assistant")
    parser.add_argument("--mode", choices=["streamlit", "fastapi", "both"], 
                       default="streamlit", help="Run mode")
    parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit")
    parser.add_argument("--api-port", type=int, default=8000, help="Port for FastAPI")
    args = parser.parse_args()
    
    if args.mode == "fastapi":
        # Run FastAPI only
        print(f"🚀 Starting FastAPI on port {args.api_port}")
        print(f"   📡 API Docs: http://localhost:{args.api_port}/docs")
        print(f"   🏥 Health: http://localhost:{args.api_port}/health")
        uvicorn.run(fastapi_app, host="0.0.0.0", port=args.api_port)
    
    elif args.mode == "both":
        # Run FastAPI in background thread
        def run_api():
            uvicorn.run(fastapi_app, host="0.0.0.0", port=args.api_port, log_level="info")
        
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        
        print(f"🚀 FastAPI running on http://localhost:{args.api_port}")
        print(f"   📡 API Docs: http://localhost:{args.api_port}/docs")
        
        # Run Streamlit in main thread
        render_web_ui()
    
    else:
        # Run Streamlit only (default)
        render_web_ui()


# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    # When run as: streamlit run app/main.py
    # Streamlit automatically calls this file
    START_TIME = time.time()
    
    # Check if running under Streamlit
    if 'STREAMLIT_RUNNING' not in os.environ:
        os.environ['STREAMLIT_RUNNING'] = '1'
    
    try:
        main()
    except SystemExit:
        pass