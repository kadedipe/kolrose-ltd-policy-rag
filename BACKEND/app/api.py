"""
Kolrose Policy RAG - FastAPI Service ONLY
==========================================
Handles /, /chat, /health, /auth, /admin, /metrics endpoints.
JWT Authentication + RBAC + Streaming + Memory + Lazy Loading + Logging
"""

import sys
import os
import json
import time
import uuid
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

# =========================
# PATH SETUP
# =========================
BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, BACKEND_ROOT)

from app.config import (
    COMPANY_INFO,
    OPENROUTER_API_KEY,
    DEFAULT_MODEL,
    CHROMA_PATH,
    POLICIES_PATH,
    ENABLE_GUARDRAILS,
)

# =========================
# LAZY LOADER IMPORTS
# =========================
from app.lazy_loader import (
    get_embeddings,
    get_vector_store,
    get_rag_system,
    get_guardrails,
    get_cross_encoder,
    component_registry,
    prewarm_all_components,
    get_system_health,
    reset_all_components,
)

# =========================
# AUTHENTICATION IMPORTS
# =========================
from app.auth import (
    Token,
    User,
    UserRole,
    UserCreate,
    Permission,
    authenticate_user,
    create_access_token,
    get_current_user,
    get_user_permissions,
    create_user,
    list_users,
    update_user_role,
    delete_user,
    has_permission,
    can_access_department,
    DEPARTMENT_POLICY_ACCESS,
    ROLE_HIERARCHY,
    # RBAC dependencies
    require_admin,
    require_hr,
    require_manager,
    require_employee,
    require_authenticated,
    require_manage_users,
    require_view_sensitive,
    require_generate_reports,
)

# =========================
# MEMORY IMPORTS
# =========================
from app.memory import (
    conversation_store,
    Conversation,
    ConversationMessage,
    build_context_with_memory,
)

# =========================
# LOGGING IMPORTS
# =========================
from app.logging_config import (
    setup_logging,
    set_request_id,
    set_session_id,
    set_user_id,
    get_request_id,
    get_session_id as get_current_session_id,
    get_user_id as get_current_user_id,
    audit_logger,
    metrics_collector,
    log_performance,
)

# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Kolrose Policy Assistant API",
    version="2.0.0",
    description="AI-Powered Policy Assistant with JWT Auth, RBAC, Streaming, and Memory",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOGGING MIDDLEWARE
# =========================

class LoggingMiddleware(BaseHTTPMiddleware):
    """Add request ID and log all requests with timing"""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        set_request_id(request_id)

        session_id = request.headers.get("X-Session-ID")
        if session_id:
            set_session_id(session_id)

        user_id = request.headers.get("X-User-ID")
        if user_id:
            set_user_id(user_id)

        start_time = time.time()

        try:
            response = await call_next(request)
            duration_ms = round((time.time() - start_time) * 1000, 2)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms}ms"
            return response
        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            print(f"✗ {request.method} {request.url.path} ERROR ({duration_ms}ms): {e}")
            raise


app.add_middleware(LoggingMiddleware)

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
# STARTUP EVENT
# =========================

@app.on_event("startup")
async def startup_event():
    """Pre-warm components in background"""
    print("=" * 50)
    print("🚀 Starting Kolrose Backend API v2.0")
    print(f"📂 ChromaDB Path: {CHROMA_PATH}")
    print(f"📄 Policies Path: {POLICIES_PATH}")
    print("=" * 50)
    print("🔄 Starting background component pre-warming...")
    prewarm_all_components(timeout=180)
    print("✅ API ready (components warming in background)")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    print("🔄 Shutting down Kolrose Backend API...")


# =========================
# ROOT & HEALTH ENDPOINTS
# =========================

@app.get("/")
def read_root():
    return {
        "message": f"Welcome to the {COMPANY_INFO['name']} Policy API",
        "status": "online",
        "version": "2.0.0",
        "documentation": "/docs",
        "components": component_registry.get_status(),
    }


@app.get("/health")
def health():
    system_health = get_system_health()
    return {
        "status": "healthy" if system_health["all_ready"] else "warming_up",
        "ready": system_health["all_ready"],
        "company": COMPANY_INFO["name"],
        "timestamp": datetime.now().isoformat(),
        "components": {
            name: {"ready": info["ready"], "init_time": info.get("init_time_seconds")}
            for name, info in system_health["components"].items()
        },
        "unready": system_health["unready_components"],
    }


@app.get("/health/detailed")
def health_detailed():
    return get_system_health()


# =========================
# AUTHENTICATION ENDPOINTS
# =========================

@app.post("/auth/login", response_model=Token)
@log_performance("auth_login")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login to get JWT access token with RBAC info"""
    user = authenticate_user(form_data.username, form_data.password)

    if not user:
        audit_logger.log_login(form_data.username, success=False)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    audit_logger.log_login(form_data.username, success=True)

    access_token = create_access_token(
        data={"sub": user.username, "role": user.role.value, "department": user.department}
    )

    return Token(
        access_token=access_token,
        username=user.username,
        role=user.role,
        department=user.department,
        permissions=get_user_permissions(user.role),
    )


@app.get("/auth/me", response_model=dict)
async def read_users_me(current_user: User = require_authenticated):
    """Get current user info with permissions"""
    return {
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "department": current_user.department,
        "role": current_user.role.value,
        "permissions": get_user_permissions(current_user.role),
        "accessible_departments": list(DEPARTMENT_POLICY_ACCESS.get(current_user.department, [])),
    }


# =========================
# ADMIN ENDPOINTS (RBAC Protected)
# =========================

@app.post("/admin/users", response_model=dict)
@log_performance("admin_create_user")
async def admin_create_user(
    user_data: UserCreate,
    current_user: User = require_manage_users,
):
    """Create a new user (requires manage_users permission)"""
    try:
        user = create_user(user_data)
        audit_logger.log_sensitive_access(current_user.username, "create_user", f"Created: {user.username}")
        return {
            "message": f"User '{user.username}' created successfully",
            "user": {
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "department": user.department,
                "role": user.role.value,
                "permissions": get_user_permissions(user.role),
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/admin/users", response_model=list)
async def admin_list_users(current_user: User = require_admin):
    """List all users (admin only)"""
    return list_users()


@app.get("/admin/roles", response_model=dict)
async def get_role_definitions(current_user: User = require_admin):
    """Get all role definitions and permissions"""
    return {
        "roles": {r.value: get_user_permissions(r) for r in UserRole},
        "hierarchy": {r.value: [sub.value for sub in subs] for r, subs in ROLE_HIERARCHY.items()},
        "department_access": DEPARTMENT_POLICY_ACCESS,
    }


@app.put("/admin/users/{username}/role", response_model=dict)
async def admin_update_user_role(
    username: str,
    new_role: UserRole,
    current_user: User = require_admin,
):
    """Update a user's role (admin only)"""
    if update_user_role(username, new_role):
        audit_logger.log_sensitive_access(current_user.username, "update_role", f"{username} → {new_role.value}")
        return {
            "message": f"User '{username}' role updated to '{new_role.value}'",
            "permissions": get_user_permissions(new_role),
        }
    raise HTTPException(status_code=404, detail="User not found")


@app.delete("/admin/users/{username}", response_model=dict)
async def admin_delete_user(
    username: str,
    current_user: User = require_admin,
):
    """Delete a user (admin only)"""
    if username == current_user.username:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    if delete_user(username):
        audit_logger.log_sensitive_access(current_user.username, "delete_user", f"Deleted: {username}")
        return {"message": f"User '{username}' deleted successfully"}
    raise HTTPException(status_code=404, detail="User not found")


@app.post("/admin/reset", response_model=dict)
async def reset_components(current_user: User = require_admin):
    """Reset all components for re-initialization (admin only)"""
    reset_all_components()
    prewarm_all_components()
    audit_logger.log_system_event("components_reset")
    return {"message": "Components reset and re-initialization started"}


# =========================
# CHAT ENDPOINTS
# =========================

@app.post("/chat", response_model=ChatResponse)
@log_performance("chat_endpoint")
def chat(req: ChatRequest):
    """Public chat endpoint with lazy-loaded components"""
    request_id = get_request_id()

    if not component_registry.is_all_ready():
        unready = component_registry.get_unready()
        raise HTTPException(
            503,
            f"System is warming up. Unready: {', '.join(unready)}. Try again shortly."
        )

    try:
        guardrails = get_guardrails()
        rag = get_rag_system()
    except Exception as e:
        raise HTTPException(503, f"Init failed: {str(e)[:200]}")

    g = guardrails.check_query(req.question)
    if g.modified_response:
        audit_logger.log_query(req.question, req.user_id, category="blocked", refused=True)
        return ChatResponse(
            question=req.question, answer=g.modified_response,
            citations=[], sources=[], refused=True, category="blocked",
            metrics={"request_id": request_id}, timestamp=datetime.now().isoformat(),
        )

    start_time = time.time()
    try:
        result = rag.query(req.question, k_final=req.k_results, enable_rerank=True, enable_guardrails=False)
        duration_ms = round((time.time() - start_time) * 1000, 2)

        audit_logger.log_query(req.question, req.user_id, session_id=get_current_session_id(),
                               category=result.category, duration_ms=duration_ms)

        sources = [
            SourceInfo(
                document_id=s.get("document_id", "N/A"),
                policy_name=s.get("policy_name", "Unknown"),
                source_file=s.get("source_file", "Unknown"),
                section=s.get("section", "N/A"),
                snippet=s.get("snippet"), relevance_score=s.get("score"),
            ) for s in result.sources
        ]

        return ChatResponse(
            question=req.question, answer=result.answer,
            citations=result.citations, sources=sources,
            refused=False, category=result.category,
            metrics={**result.metrics, "request_id": request_id, "duration_ms": duration_ms},
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)[:200]}")


# =========================
# MEMORY-AWARE CHAT
# =========================

@app.post("/chat/memory", response_model=ChatResponse)
@log_performance("chat_memory_endpoint")
def chat_with_memory(req: ChatRequest, session_id: Optional[str] = None):
    """Chat with conversation memory for follow-up questions"""
    request_id = get_request_id()

    if not component_registry.is_all_ready():
        raise HTTPException(503, "System is warming up. Try again shortly.")

    session_id = session_id or f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    conv = conversation_store.get_or_create_conversation(session_id=session_id, user_id=req.user_id)
    memory_context = build_context_with_memory(req.question, conv)

    try:
        guardrails = get_guardrails()
        rag = get_rag_system()
    except Exception as e:
        raise HTTPException(503, f"Init failed: {str(e)[:200]}")

    g = guardrails.check_query(req.question)
    if g.modified_response:
        conv.add_message("user", req.question)
        conv.add_message("assistant", g.modified_response)
        return ChatResponse(
            question=req.question, answer=g.modified_response,
            citations=[], sources=[], refused=True, category="blocked",
            metrics={"session_id": session_id, "request_id": request_id},
            timestamp=datetime.now().isoformat(),
        )

    enhanced = f"{memory_context}\n\nCurrent Question: {req.question}" if memory_context else req.question

    start_time = time.time()
    try:
        result = rag.query(enhanced, k_final=req.k_results, enable_rerank=True, enable_guardrails=False)
        duration_ms = round((time.time() - start_time) * 1000, 2)

        conv.add_message("user", req.question)
        conv.add_message("assistant", result.answer, citations=result.citations,
                         sources=[{"document_id": s.get("document_id", "N/A"),
                                   "policy_name": s.get("policy_name", "Unknown"),
                                   "source_file": s.get("source_file", "Unknown"),
                                   "section": s.get("section", "N/A")} for s in result.sources])

        sources = [
            SourceInfo(document_id=s.get("document_id", "N/A"), policy_name=s.get("policy_name", "Unknown"),
                       source_file=s.get("source_file", "Unknown"), section=s.get("section", "N/A"),
                       snippet=s.get("snippet"), relevance_score=s.get("score"))
            for s in result.sources
        ]

        return ChatResponse(
            question=req.question, answer=result.answer,
            citations=result.citations, sources=sources,
            refused=False, category=result.category,
            metrics={**result.metrics, "session_id": session_id, "request_id": request_id,
                     "duration_ms": duration_ms, "message_count": len(conv.messages),
                     "is_follow_up": bool(memory_context)},
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)[:200]}")


# =========================
# STREAMING CHAT
# =========================

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream chat responses token by token"""
    if not component_registry.is_all_ready():
        raise HTTPException(503, "System is warming up.")

    try:
        guardrails = get_guardrails()
        rag = get_rag_system()
    except Exception as e:
        raise HTTPException(503, f"Init failed: {str(e)[:200]}")

    g = guardrails.check_query(req.question)
    if g.modified_response:
        async def generate_refusal():
            words = g.modified_response.split()
            for word in words:
                yield f"data: {json.dumps({'token': word + ' ', 'done': False})}\n\n"
                await asyncio.sleep(0.05)
            yield f"data: {json.dumps({'token': '', 'done': True, 'refused': True})}\n\n"

        return StreamingResponse(generate_refusal(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

    result = rag.query(req.question, k_final=req.k_results, enable_rerank=True, enable_guardrails=False)

    async def generate():
        sources_data = [{"document_id": s.get("document_id", "N/A"),
                         "policy_name": s.get("policy_name", "Unknown"),
                         "source_file": s.get("source_file", "Unknown"),
                         "section": s.get("section", "N/A")} for s in result.sources]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources_data})}\n\n"
        if result.citations:
            yield f"data: {json.dumps({'type': 'citations', 'citations': result.citations})}\n\n"

        words = result.answer.split()
        for i, word in enumerate(words):
            is_last = (i == len(words) - 1)
            token = word if word in '.,!?:;)]}' else (word if word in '([{' else word + ' ')
            yield f"data: {json.dumps({'token': token, 'done': is_last})}\n\n"
            await asyncio.sleep(0.03)
        yield f"data: {json.dumps({'token': '', 'done': True, 'metrics': result.metrics})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


# =========================
# AUTHENTICATED CHAT
# =========================

@app.post("/chat/authenticated", response_model=ChatResponse)
@log_performance("chat_authenticated_endpoint")
def chat_authenticated(req: ChatRequest, current_user: User = require_employee):
    """Chat endpoint for authenticated employees with audit logging"""
    audit_logger.log_query(req.question, current_user.username, category="authenticated")
    return chat(req)


@app.post("/chat/hr", response_model=ChatResponse)
@log_performance("chat_hr_endpoint")
def chat_hr_sensitive(req: ChatRequest, current_user: User = require_view_sensitive):
    """Chat endpoint for HR personnel with sensitive data access"""
    audit_logger.log_sensitive_access(current_user.username, "hr_chat_query")
    return chat(req)


# =========================
# CONVERSATION MANAGEMENT
# =========================

@app.get("/conversations", response_model=list)
async def list_conversations(user_id: str = "anonymous"):
    """List all conversations for a user"""
    return conversation_store.get_user_conversations(user_id)


@app.get("/conversations/{session_id}", response_model=dict)
async def get_conversation(session_id: str):
    """Get a specific conversation"""
    conv = conversation_store.get_conversation(session_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return conv.to_dict()


@app.delete("/conversations/{session_id}", response_model=dict)
async def delete_conversation(session_id: str):
    """Delete a conversation"""
    if conversation_store.delete_conversation(session_id):
        return {"message": f"Conversation '{session_id}' deleted"}
    raise HTTPException(404, "Conversation not found")


@app.delete("/conversations", response_model=dict)
async def clear_all_conversations(user_id: str = "anonymous"):
    """Clear all conversations for a user"""
    conversations = conversation_store.get_user_conversations(user_id)
    for conv in conversations:
        conversation_store.delete_conversation(conv["session_id"])
    return {"message": f"Cleared {len(conversations)} conversations"}


@app.get("/memory/stats", response_model=dict)
async def get_memory_stats():
    """Get conversation store statistics"""
    return conversation_store.get_stats()


# =========================
# OBSERVABILITY ENDPOINTS
# =========================

@app.get("/metrics", response_model=dict)
async def get_metrics():
    """Get performance metrics"""
    return {
        "request_metrics": metrics_collector.get_all_stats(),
        "component_status": component_registry.get_status(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/metrics/{metric_name}", response_model=dict)
async def get_metric_stats(metric_name: str):
    """Get stats for a specific metric"""
    stats = metrics_collector.get_stats(metric_name)
    if stats.get("count", 0) == 0:
        raise HTTPException(404, f"Metric '{metric_name}' not found")
    return {"metric": metric_name, "stats": stats}


@app.get("/system/memory", response_model=dict)
async def get_memory_info():
    """Get system memory usage"""
    import gc
    gc.collect()
    from app.memory_config import get_memory_usage
    return {"memory": get_memory_usage(), "gc_collections": gc.get_count()}


@app.get("/policies/accessible", response_model=dict)
async def get_accessible_policies(current_user: User = require_authenticated):
    """Get policies accessible based on user's department"""
    if current_user.role == UserRole.ADMIN:
        accessible = list(DEPARTMENT_POLICY_ACCESS.keys())
    else:
        accessible = DEPARTMENT_POLICY_ACCESS.get(current_user.department, [])
    return {"department": current_user.department, "accessible_categories": accessible, "role": current_user.role.value}


@app.get("/reports/usage", response_model=dict)
async def get_usage_report(current_user: User = require_generate_reports):
    """Get system usage report"""
    return {
        "generated_by": current_user.username, "role": current_user.role.value,
        "timestamp": datetime.now().isoformat(),
        "system_status": {"ready": component_registry.is_all_ready(), "policies_available": 12, "total_chunks": 250},
    }