# app/__init__.py
"""
Kolrose Limited - AI Policy Assistant
=====================================
Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria

A Retrieval-Augmented Generation (RAG) system for answering employee questions
about company policies with accurate citations and guardrails.

Usage:
    from app import KolroseRAG, ingest_policies, run_app
    
    # Run the web application
    run_app()
    
    # Or use programmatically
    rag = KolroseRAG(vectorstore, llm)
    result = rag.query("What is the annual leave policy?")
"""

__version__ = "1.0.0"
__author__ = "Kolrose Limited"
__email__ = "hr@kolroselimited.com.ng"
__location__ = "Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria"

# Import main components for easy access
from .config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DEFAULT_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
    POLICIES_PATH,
    COMPANY_INFO,
)

from .rag_system import KolroseRAG, TopicClassifier
from .guardrails import GuardrailSystem
from .ingestion import ingest_policies, load_vectorstore, load_embeddings

# Define what's available on `from app import *`
__all__ = [
    # Version
    "__version__",
    "__author__",
    
    # Configuration
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL", 
    "DEFAULT_MODEL",
    "CHROMA_PATH",
    "COLLECTION_NAME",
    "POLICIES_PATH",
    "COMPANY_INFO",
    
    # Core classes
    "KolroseRAG",
    "TopicClassifier",
    "GuardrailSystem",
    
    # Functions
    "ingest_policies",
    "load_vectorstore",
    "load_embeddings",
]

# Log initialization
import logging
logging.getLogger(__name__).info(
    f"Kolrose Limited RAG System v{__version__} initialized"
)