"""
Memory Optimization Configuration for Kolrose Policy Assistant
===============================================================
Settings to keep the application within Railway memory limits.
"""

import os
import gc
import logging

logger = logging.getLogger(__name__)


def optimize_memory():
    """Apply memory optimizations for Railway's free tier (8 GB limit)"""
    optimizations = []
    
    # 1. Garbage collection
    gc.enable()
    gc.set_threshold(100, 10, 10)
    optimizations.append("GC optimized")
    
    # 2. PyTorch memory
    try:
        import torch
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(1)
            optimizations.append("PyTorch threads: 1")
    except ImportError:
        pass
    
    # 3. Limit numpy threads
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    optimizations.append("NumPy threads limited")
    
    # 4. ChromaDB settings
    os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"
    optimizations.append("ChromaDB optimized")
    
    # 5. HuggingFace cache
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    optimizations.append("HF cache optimized")
    
    # 6. Embedding batch size
    os.environ["EMBEDDING_BATCH_SIZE"] = "8"
    optimizations.append("Embedding batch: 8")
    
    # 7. CPU-only mode
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    optimizations.append("CPU-only mode")
    
    for opt in optimizations:
        logger.info(f"✅ {opt}")
    
    # Print memory usage
    try:
        import psutil
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"📊 Memory usage: {mem_mb:.1f} MB")
    except ImportError:
        logger.info("📊 psutil not installed - memory monitoring disabled")


def get_memory_usage() -> dict:
    """Get current memory usage stats"""
    try:
        import psutil
        process = psutil.Process()
        mem = process.memory_info()
        
        return {
            "rss_mb": round(mem.rss / 1024 / 1024, 2),
            "vms_mb": round(mem.vms / 1024 / 1024, 2),
            "percent": round(process.memory_percent(), 2),
        }
    except ImportError:
        return {"error": "psutil not installed"}


# Apply optimizations on import
optimize_memory()