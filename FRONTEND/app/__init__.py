# kolrose-policy-rag/FRONTEND/app/__init__.py
"""
Kolrose Limited - Frontend Application
=======================================
Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria

Web chat interface and API endpoints for the Policy Assistant.

This package contains:
- main.py: Streamlit web application (/ endpoint)
- API endpoints for /chat and /health
"""

__version__ = "1.0.0"
__author__ = "Kolrose Limited"
__description__ = "AI-Powered Policy Assistant - Frontend"

# Import main UI components
from .main import (
    render_web_ui,
    fastapi_app,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SourceInfo,
)

# Define public API
__all__ = [
    # Version
    "__version__",
    "__author__",
    
    # Web UI
    "render_web_ui",
    
    # FastAPI app
    "fastapi_app",
    
    # Request/Response models
    "ChatRequest",
    "ChatResponse",
    "HealthResponse",
    "SourceInfo",
]