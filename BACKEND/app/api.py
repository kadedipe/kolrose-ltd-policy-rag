"""
Kolrose Policy RAG - FastAPI Service ONLY
==========================================
Handles /, /chat, and /health endpoints.
No Streamlit code here.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add backend path
BACKEND_PATH = str(Path(__file__).parent.parent.parent / "BACKEND")
sys.path.insert(0, BACKEND_PATH)

from app.config import (
    COMPANY_INFO,
    OPENROUTER_API_KEY,
    DEFAULT_MODEL,
    CHROMA_PATH,
    POLICIES_PATH,
    ENABLE_GUARDRAILS,
)

from app.rag_system import KolroseRAG
from app.guardrails import GuardrailSystem
from app.ingestion import load_vectorstore, check_policies_exist


# =========================
# FASTAPI APP
# =========================
# Change the instance name to 'app' so the Uvicorn start command 
# 'app.api:app' finds it successfully without throwing a boot error.
app = FastAPI(
    title="Kolrose Policy Assistant API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# MODELS
# =========================

class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = "anonymous"
    include_snippets: bool = True
    k_results: int = 5


class SourceInfo(BaseModel):
    document_id: str
    policy_name: str
    source_file: str
    section: str
    snippet: Optional[str] = None
    relevance_score: Optional[float] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: List[str]
    sources: List[SourceInfo]
    refused: bool = False
    category: str = "in_corpus"
    metrics: Dict = {}
    timestamp: str


# =========================
# INIT SYSTEM
# =========================

vectorstore = load_vectorstore()
rag = KolroseRAG(vectorstore) if vectorstore else None
guardrails = GuardrailSystem() if vectorstore else None


SYSTEM_READY = rag is not None


# =========================
# ENDPOINTS
# =========================

# ============================================================
# ROOT ROUTE (LANDING PAGE)
# ============================================================
@app.get("/")
def read_root():
    return {
        "message": f"Welcome to the {COMPANY_INFO['name']} Policy API",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "chat": "/chat"
        }
    }


@app.get("/health")
def health():
    policies_exist, policy_count, _ = check_policies_exist()

    return {
        "status": "healthy" if SYSTEM_READY else "degraded",
        "company": COMPANY_INFO["name"],
        "timestamp": datetime.now().isoformat(),
        "components": {
            "vector_store": bool(vectorstore),
            "policies": policy_count,
            "llm": bool(OPENROUTER_API_KEY),
            "guardrails": ENABLE_GUARDRAILS,
        },
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    if not SYSTEM_READY:
        raise HTTPException(503, "System not ready")

    # guardrails
    g = guardrails.check_query(req.question)

    if g.modified_response:
        return ChatResponse(
            question=req.question,
            answer=g.modified_response,
            citations=[],
            sources=[],
            refused=True,
            category="blocked",
            metrics={},
            timestamp=datetime.now().isoformat(),
        )

    result = rag.query(
        req.question,
        k_final=req.k_results,
        enable_rerank=True,
        enable_guardrails=False,
    )

    sources = [
        SourceInfo(
            document_id=s.get("document_id", "N/A"),
            policy_name=s.get("policy_name", "Unknown"),
            source_file=s.get("source_file", "Unknown"),
            section=s.get("section", "N/A"),
            snippet=s.get("snippet"),
            relevance_score=s.get("score"),
        )
        for s in result.sources
    ]

    return ChatResponse(
        question=req.question,
        answer=result.answer,
        citations=result.citations,
        sources=sources,
        refused=False,
        category=result.category,
        metrics=result.metrics,
        timestamp=datetime.now().isoformat(),
    )