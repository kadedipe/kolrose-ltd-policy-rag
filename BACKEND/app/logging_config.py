"""
Structured Logging Module for Kolrose Policy Assistant
========================================================
Provides structured JSON logging with request IDs,
performance metrics, and audit trails.
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from contextvars import ContextVar
from functools import wraps

# ============================================================================
# REQUEST ID CONTEXT
# ============================================================================

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


def get_request_id() -> str:
    """Get current request ID or generate a new one"""
    rid = request_id_var.get()
    if rid is None:
        rid = str(uuid.uuid4())[:8]
        request_id_var.set(rid)
    return rid


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set request ID for the current context"""
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    request_id_var.set(request_id)
    return request_id


def get_session_id() -> Optional[str]:
    """Get current session ID"""
    return session_id_var.get()


def set_session_id(session_id: str):
    """Set session ID for the current context"""
    session_id_var.set(session_id)


def get_user_id() -> Optional[str]:
    """Get current user ID"""
    return user_id_var.get()


def set_user_id(user_id: str):
    """Set user ID for the current context"""
    user_id_var.set(user_id)


# ============================================================================
# STRUCTURED LOG FORMATTER
# ============================================================================

class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for machine-readable logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        request_id = request_id_var.get()
        if request_id:
            log_entry["request_id"] = request_id
        
        session_id = session_id_var.get()
        if session_id:
            log_entry["session_id"] = session_id
        
        user_id = user_id_var.get()
        if user_id:
            log_entry["user_id"] = user_id
        
        log_entry["thread"] = threading.current_thread().name
        
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }
        
        return json.dumps(log_entry, default=str)


class PrettyFormatter(logging.Formatter):
    """Human-readable formatter for development"""
    
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
        'RESET': '\033[0m',
    }
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        reset = self.COLORS['RESET']
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        parts = [
            f"{color}[{timestamp}]{reset}",
            f"{color}[{record.levelname:8}]{reset}",
        ]
        
        request_id = request_id_var.get()
        if request_id:
            parts.append(f"[rid:{request_id}]")
        
        parts.append(f"[{record.module}:{record.lineno}]")
        parts.append(record.getMessage())
        
        return ' '.join(parts)


# ============================================================================
# LOGGER SETUP
# ============================================================================

def setup_logging(
    level: str = "INFO",
    format_type: str = "auto",
    log_file: Optional[str] = None,
):
    """Setup structured logging for the application"""
    if format_type == "auto":
        is_cloud = bool(
            os.environ.get("RAILWAY_ENVIRONMENT") or
            os.environ.get("PORT") or
            os.environ.get("STREAMLIT_SHARING_MODE")
        )
        format_type = "json" if is_cloud else "pretty"
    
    if format_type == "json":
        formatter = StructuredFormatter()
    else:
        formatter = PrettyFormatter()
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)
    
    # Silence noisy libraries
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return root_logger


# ============================================================================
# PERFORMANCE METRICS COLLECTOR
# ============================================================================

class MetricsCollector:
    """Collect performance metrics for API requests"""
    
    def __init__(self):
        self._metrics: Dict[str, list] = {}
        self._lock = threading.Lock()
    
    def record(self, metric_name: str, value: float):
        """Record a metric value"""
        with self._lock:
            if metric_name not in self._metrics:
                self._metrics[metric_name] = []
            self._metrics[metric_name].append(value)
            if len(self._metrics[metric_name]) > 1000:
                self._metrics[metric_name] = self._metrics[metric_name][-1000:]
    
    def get_stats(self, metric_name: str) -> dict:
        """Get statistics for a metric"""
        with self._lock:
            values = self._metrics.get(metric_name, [])
            if not values:
                return {"count": 0}
            
            values.sort()
            return {
                "count": len(values),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "avg": round(sum(values) / len(values), 4),
                "p50": round(values[len(values) // 2], 4),
                "p95": round(values[int(len(values) * 0.95)], 4),
                "p99": round(values[int(len(values) * 0.99)], 4),
            }
    
    def get_all_stats(self) -> dict:
        """Get stats for all metrics"""
        with self._lock:
            return {name: self.get_stats(name) for name in self._metrics}


# Global metrics collector
metrics_collector = MetricsCollector()


# ============================================================================
# AUDIT LOGGER
# ============================================================================

class AuditLogger:
    """Logs security-relevant events for audit trails"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("audit")
    
    def log_query(
        self,
        question: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        category: Optional[str] = None,
        refused: bool = False,
        duration_ms: Optional[float] = None,
    ):
        """Log a policy query for audit"""
        self.logger.info(
            f"QUERY: '{question[:100]}{'...' if len(question) > 100 else ''}'",
            extra={
                "event_type": "policy_query",
                "question_preview": question[:100],
                "question_length": len(question),
                "category": category,
                "refused": refused,
                "duration_ms": duration_ms,
            }
        )
    
    def log_login(self, username: str, success: bool, ip_address: Optional[str] = None):
        """Log a login attempt"""
        level = logging.INFO if success else logging.WARNING
        self.logger.log(
            level,
            f"LOGIN: {username} - {'SUCCESS' if success else 'FAILED'}",
            extra={
                "event_type": "login",
                "username": username,
                "success": success,
                "ip_address": ip_address,
            }
        )
    
    def log_sensitive_access(self, user_id: str, action: str, details: Optional[str] = None):
        """Log access to sensitive data"""
        self.logger.warning(
            f"SENSITIVE: {user_id} performed {action}",
            extra={
                "event_type": "sensitive_access",
                "user_id": user_id,
                "action": action,
                "details": details,
            }
        )
    
    def log_system_event(self, event: str, details: Optional[Dict] = None):
        """Log system events"""
        self.logger.info(
            f"SYSTEM: {event}",
            extra={
                "event_type": "system",
                "event": event,
                "details": details or {},
            }
        )


# ============================================================================
# PERFORMANCE DECORATOR
# ============================================================================

def log_performance(func_name: Optional[str] = None):
    """Decorator to log function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or f"{func.__module__}.{func.__name__}"
            start = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = round((time.time() - start) * 1000, 2)
                
                logger = logging.getLogger(func.__module__)
                logger.debug(f"⏱️ {name}: {duration_ms}ms")
                
                metrics_collector.record(f"function.{name}.duration", duration_ms)
                
                return result
                
            except Exception as e:
                duration_ms = round((time.time() - start) * 1000, 2)
                logger = logging.getLogger(func.__module__)
                logger.error(f"⏱️ {name} FAILED after {duration_ms}ms: {e}")
                raise
        
        return wrapper
    return decorator


# ============================================================================
# INITIALIZATION
# ============================================================================

# Auto-setup logging on import
setup_logging(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format_type="auto",
)

# Create global instances
audit_logger = AuditLogger()