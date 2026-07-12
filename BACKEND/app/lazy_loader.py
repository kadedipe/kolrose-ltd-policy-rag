"""
Lazy Loading Module for Kolrose Policy Assistant
==================================================
Provides lazy initialization and caching for heavy components
to improve startup time and reduce memory usage.
"""

import os
import time
import logging
import threading
from functools import lru_cache, wraps
from typing import Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# LAZY LOADER CLASS
# ============================================================================

class LazyLoader:
    """
    Thread-safe lazy initializer with retry logic and health monitoring.
    Components are only loaded on first access, not at import time.
    """
    
    def __init__(self, name: str = "component", max_retries: int = 3, retry_delay: float = 2.0):
        self.name = name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._instance: Optional[Any] = None
        self._error: Optional[Exception] = None
        self._initialized = False
        self._initializing = False
        self._lock = threading.Lock()
        self._init_time: Optional[float] = None
        self._access_count: int = 0
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator syntax: @LazyLoader('name')"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.get_instance(func, *args, **kwargs)
        wrapper._loader = self
        return wrapper
    
    def get_instance(self, factory_func: Callable, *args, **kwargs) -> Any:
        """Get or create the singleton instance with retry logic"""
        
        # Fast path: already initialized
        if self._initialized and self._instance is not None:
            self._access_count += 1
            return self._instance
        
        with self._lock:
            # Double-check after acquiring lock
            if self._initialized and self._instance is not None:
                self._access_count += 1
                return self._instance
            
            if self._initializing:
                logger.info(f"⏳ {self.name} is being initialized by another thread, waiting...")
            
            self._initializing = True
            start_time = time.time()
            
            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.info(f"🔄 Initializing {self.name} (attempt {attempt}/{self.max_retries})...")
                    
                    instance = factory_func(*args, **kwargs)
                    
                    self._instance = instance
                    self._initialized = True
                    self._initializing = False
                    self._init_time = time.time() - start_time
                    self._error = None
                    self._access_count = 0
                    
                    logger.info(f"✅ {self.name} initialized in {self._init_time:.1f}s")
                    return instance
                    
                except Exception as e:
                    logger.warning(f"⚠️ {self.name} initialization failed (attempt {attempt}): {e}")
                    self._error = e
                    
                    if attempt < self.max_retries:
                        logger.info(f"⏳ Retrying {self.name} in {self.retry_delay}s...")
                        time.sleep(self.retry_delay)
                    else:
                        self._initializing = False
                        logger.error(f"❌ {self.name} failed after {self.max_retries} attempts")
                        raise RuntimeError(f"Failed to initialize {self.name}: {e}") from e
        
        return None
    
    @property
    def is_ready(self) -> bool:
        return self._initialized and self._instance is not None
    
    @property
    def health(self) -> dict:
        return {
            "name": self.name,
            "ready": self.is_ready,
            "init_time_seconds": round(self._init_time, 2) if self._init_time else None,
            "access_count": self._access_count,
            "error": str(self._error) if self._error else None,
        }
    
    def reset(self):
        """Reset the loader for re-initialization"""
        with self._lock:
            self._instance = None
            self._initialized = False
            self._initializing = False
            self._error = None
            self._init_time = None
            self._access_count = 0


# ============================================================================
# COMPONENT REGISTRY
# ============================================================================

class ComponentRegistry:
    """Registry for all lazy-loaded components with health tracking"""
    
    def __init__(self):
        self._components: dict[str, LazyLoader] = {}
        self._lock = threading.Lock()
    
    def register(self, name: str, loader: LazyLoader):
        with self._lock:
            self._components[name] = loader
    
    def get_status(self) -> dict:
        with self._lock:
            return {name: loader.health for name, loader in self._components.items()}
    
    def is_all_ready(self) -> bool:
        with self._lock:
            return all(loader.is_ready for loader in self._components.values())
    
    def get_unready(self) -> list:
        with self._lock:
            return [name for name, loader in self._components.items() if not loader.is_ready]
    
    def warm_up_all(self, timeout: float = 120):
        """Force initialize all registered components"""
        unready = self.get_unready()
        if not unready:
            logger.info("✅ All components already initialized")
            return
        
        logger.info(f"🔄 Warming up {len(unready)} components...")
        for name in unready:
            try:
                loader = self._components[name]
                loader.get_instance(lambda: None)
            except Exception as e:
                logger.error(f"❌ Failed to warm up {name}: {e}")


# Global component registry
component_registry = ComponentRegistry()


# ============================================================================
# LAZY-LOADED COMPONENTS
# ============================================================================

# --- Embeddings Model ---
_embeddings_loader = LazyLoader(name="embeddings_model", max_retries=3, retry_delay=3.0)

@_embeddings_loader
def get_embeddings():
    """Lazily load the embedding model (cached globally)"""
    from app.ingestion import load_embeddings as _load_embeddings
    return _load_embeddings()

component_registry.register("embeddings_model", _embeddings_loader)


# --- Vector Store ---
_vector_store_loader = LazyLoader(name="vector_store", max_retries=3, retry_delay=5.0)

@_vector_store_loader
def get_vector_store():
    """Lazily load or create the vector store with auto-ingestion fallback"""
    from app.config import CHROMA_PATH, POLICIES_PATH
    from app.ingestion import load_vectorstore, ingest_all, check_policies_exist
    
    logger.info("🔄 Attempting to load existing vector store...")
    vectorstore = load_vectorstore()
    
    if vectorstore is not None:
        return vectorstore
    
    logger.info("⚠️ Vector store not found. Starting auto-ingestion...")
    policies_exist, policy_count, _ = check_policies_exist()
    
    if not policies_exist:
        raise RuntimeError(f"No policy files found in {POLICIES_PATH}")
    
    logger.info(f"📁 Found {policy_count} policy files. Ingesting...")
    vectorstore = ingest_all()
    
    if vectorstore is None:
        raise RuntimeError("Failed to create vector store through auto-ingestion")
    
    return vectorstore

component_registry.register("vector_store", _vector_store_loader)


# --- RAG System ---
_rag_system_loader = LazyLoader(name="rag_system", max_retries=2, retry_delay=2.0)

@_rag_system_loader
def get_rag_system():
    """Lazily initialize the RAG system (depends on vector store)"""
    from app.rag_system import KolroseRAG
    
    vectorstore = get_vector_store()
    rag = KolroseRAG(vectorstore)
    
    try:
        logger.info("🔄 Running warm-up query...")
        _ = rag.query("test warmup", k_final=1, enable_rerank=False, enable_guardrails=False)
        logger.info("✅ RAG system warm-up complete")
    except Exception as e:
        logger.warning(f"⚠️ Warm-up query failed (non-critical): {e}")
    
    return rag

component_registry.register("rag_system", _rag_system_loader)


# --- Guardrails ---
_guardrails_loader = LazyLoader(name="guardrails", max_retries=2, retry_delay=2.0)

@_guardrails_loader
def get_guardrails():
    """Lazily initialize the guardrails system"""
    from app.guardrails import GuardrailSystem
    return GuardrailSystem()

component_registry.register("guardrails", _guardrails_loader)


# --- Cross-Encoder (for re-ranking) ---
@lru_cache(maxsize=1)
def get_cross_encoder():
    """Cached cross-encoder for re-ranking"""
    from sentence_transformers import CrossEncoder
    
    logger.info("🔄 Loading cross-encoder model...")
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    logger.info("✅ Cross-encoder loaded")
    
    return model


# ============================================================================
# PRE-WARM FUNCTION
# ============================================================================

def prewarm_all_components(timeout: float = 180):
    """Pre-warm all lazy components in background thread"""
    import threading
    
    def _warm():
        logger.info("=" * 50)
        logger.info("🔥 Starting component pre-warming...")
        logger.info("=" * 50)
        
        start = time.time()
        
        components = [
            ("Embeddings", get_embeddings),
            ("Vector Store", get_vector_store),
            ("RAG System", get_rag_system),
            ("Guardrails", get_guardrails),
            ("Cross-Encoder", get_cross_encoder),
        ]
        
        for name, loader in components:
            try:
                comp_start = time.time()
                loader()
                elapsed = time.time() - comp_start
                logger.info(f"  ✅ {name}: {elapsed:.1f}s")
            except Exception as e:
                logger.error(f"  ❌ {name}: {e}")
        
        total = time.time() - start
        logger.info(f"\n⏱️ Total pre-warm time: {total:.1f}s")
        logger.info("=" * 50)
    
    thread = threading.Thread(target=_warm, daemon=True)
    thread.start()
    return thread


# ============================================================================
# HEALTH CHECK HELPERS
# ============================================================================

def get_system_health() -> dict:
    """Get comprehensive system health status"""
    return {
        "components": component_registry.get_status(),
        "all_ready": component_registry.is_all_ready(),
        "unready_components": component_registry.get_unready(),
        "timestamp": datetime.now().isoformat(),
    }


def reset_all_components():
    """Reset all components for re-initialization"""
    _embeddings_loader.reset()
    _vector_store_loader.reset()
    _rag_system_loader.reset()
    _guardrails_loader.reset()
    get_cross_encoder.cache_clear()
    logger.info("🔄 All components reset")