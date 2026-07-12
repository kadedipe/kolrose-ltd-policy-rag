"""
Configuration Management for Kolrose Limited RAG System.
=========================================================
Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria

Handles:
- Environment variable loading (.env file)
- Configuration validation
- Sensible defaults for all settings
- Type casting and error handling
- Configuration categories (API, DB, Guardrails, Auth, Logging, etc.)
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Try to load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from {env_path}")
    else:
        load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
    print("   Using system environment variables only.")

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

class Environment(str, Enum):
    """Application environment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class APIConfig:
    """API-related configuration"""
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "openrouter/free"
    max_tokens: int = 500
    temperature: float = 0.0
    request_timeout: int = 30
    max_retries: int = 3
    
    free_models: List[str] = field(default_factory=lambda: [
        "openrouter/free",
        "google/gemma-4-31b",
        "tencent/hy3",
        "nvidia/nemotron-3-super",
        "meta-llama/llama-3.1-8b-instruct:free",
    ])
    
    @property
    def is_configured(self) -> bool:
        return bool(self.openrouter_api_key and 
                   not self.openrouter_api_key.startswith("sk-or-v1-your"))


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 16
    normalize: bool = True
    dimension: int = 384
    
    alternative_models: Dict[str, Dict] = field(default_factory=lambda: {
        "small": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "size_mb": 80,
        },
        "medium": {
            "name": "sentence-transformers/all-mpnet-base-v2",
            "dimension": 768,
            "size_mb": 420,
        },
    })


@dataclass
class DatabaseConfig:
    """Vector database configuration"""
    persist_directory: str = "./chroma_db"
    collection_name: str = "kolrose_policies_v2"
    distance_metric: str = "cosine"
    
    @property
    def chroma_settings(self) -> Dict:
        return {
            "persist_directory": self.persist_directory,
            "collection_name": self.collection_name,
            "collection_metadata": {"hnsw:space": self.distance_metric},
        }


@dataclass
class RAGConfig:
    """RAG pipeline configuration"""
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_k: int = 20
    final_k: int = 5
    mmr_lambda: float = 0.7
    enable_rerank: bool = True
    enable_hybrid_search: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class GuardrailConfig:
    """Guardrail configuration"""
    enabled: bool = True
    sensitive_topics_enabled: bool = True
    citation_required: bool = True
    max_response_chars: int = 2000
    max_sentences: int = 10
    min_citations: int = 1


@dataclass
class AuthConfig:
    """Authentication configuration"""
    jwt_secret_key: str = "kolrose-secret-key-change-in-production-2024"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours
    bcrypt_rounds: int = 12


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format_type: str = "auto"  # auto, json, pretty
    log_file: Optional[str] = None
    enable_audit: bool = True
    enable_metrics: bool = True


@dataclass
class MemoryConfig:
    """Conversation memory configuration"""
    ttl_minutes: int = 120  # Conversations expire after 2 hours
    max_messages_per_conversation: int = 50
    max_context_exchanges: int = 5


@dataclass
class CompanyConfig:
    """Company-specific configuration"""
    name: str = "Kolrose Limited"
    address: str = "Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria"
    website: str = "https://kolroselimited.com.ng"
    email_hr: str = "hr@kolroselimited.com.ng"
    email_compliance: str = "compliance@kolroselimited.com.ng"
    email_security: str = "security@kolroselimited.com.ng"
    hotline_whistleblower: str = "0800-KOLROSE"
    hotline_it_security: str = "0800-KOL-ITSEC"


@dataclass
class AppConfig:
    """Application-level configuration"""
    env: Environment = Environment.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO
    streamlit_port: int = 8501
    fastapi_port: int = 8000
    debug: bool = False
    
    @property
    def is_production(self) -> bool:
        return self.env == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        return self.env == Environment.DEVELOPMENT


# ============================================================================
# DATA DIRECTORIES
# ============================================================================

# Check if running in a cloud/container system first
if os.path.exists("/app/data"):
    DATA_DIR = Path("/app/data")
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR

# Let environment variables override paths dynamically
POLICIES_PATH: str = os.environ.get("POLICIES_PATH", "./BACKEND/policies")

# Ensure the directories actually exist
Path(POLICIES_PATH).mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

class ConfigLoader:
    """Loads and validates all configuration from environment variables."""
    
    @staticmethod
    def _get_env(key: str, default: Any = None, required: bool = False) -> str:
        value = os.environ.get(key, default)
        if required and (value is None or value == default):
            logger.error(f"Required environment variable not set: {key}")
            if key == "OPENROUTER_API_KEY":
                logger.error(
                    "Get a free API key at: https://openrouter.ai/keys\n"
                    "Then set: export OPENROUTER_API_KEY=sk-or-v1-your-key"
                )
        return value
    
    @staticmethod
    def _get_bool(key: str, default: bool = False) -> bool:
        value = os.environ.get(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")
    
    @staticmethod
    def _get_int(key: str, default: int = 0) -> int:
        try:
            return int(os.environ.get(key, str(default)))
        except ValueError:
            return default
    
    @staticmethod
    def _get_float(key: str, default: float = 0.0) -> float:
        try:
            return float(os.environ.get(key, str(default)))
        except ValueError:
            return default
    
    @classmethod
    def load_all(cls) -> Dict[str, Any]:
        return {
            'api': cls.load_api_config(),
            'embedding': cls.load_embedding_config(),
            'database': cls.load_database_config(),
            'rag': cls.load_rag_config(),
            'guardrails': cls.load_guardrail_config(),
            'auth': cls.load_auth_config(),
            'logging': cls.load_logging_config(),
            'memory': cls.load_memory_config(),
            'company': cls.load_company_config(),
            'app': cls.load_app_config(),
        }
    
    @classmethod
    def load_api_config(cls) -> APIConfig:
        return APIConfig(
            openrouter_api_key=cls._get_env("OPENROUTER_API_KEY", ""),
            openrouter_base_url=cls._get_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            default_model=cls._get_env("LLM_MODEL", "openrouter/free"),
            max_tokens=cls._get_int("MAX_OUTPUT_TOKENS", 500),
            temperature=cls._get_float("LLM_TEMPERATURE", 0.0),
            request_timeout=cls._get_int("LLM_REQUEST_TIMEOUT", 30),
            max_retries=cls._get_int("LLM_MAX_RETRIES", 3),
        )
    
    @classmethod
    def load_embedding_config(cls) -> EmbeddingConfig:
        return EmbeddingConfig(
            model_name=cls._get_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            device=cls._get_env("EMBEDDING_DEVICE", "cpu"),
            batch_size=cls._get_int("EMBEDDING_BATCH_SIZE", 16),
            normalize=cls._get_bool("EMBEDDING_NORMALIZE", True),
        )
    
    @classmethod
    def load_database_config(cls) -> DatabaseConfig:
        return DatabaseConfig(
            persist_directory=cls._get_env("CHROMA_PATH", "./chroma_db"),
            collection_name=cls._get_env("CHROMA_COLLECTION_NAME", "kolrose_policies_v2"),
            distance_metric=cls._get_env("CHROMA_DISTANCE_METRIC", "cosine"),
        )
    
    @classmethod
    def load_rag_config(cls) -> RAGConfig:
        return RAGConfig(
            chunk_size=cls._get_int("CHUNK_SIZE", 500),
            chunk_overlap=cls._get_int("CHUNK_OVERLAP", 100),
            retrieval_k=cls._get_int("RETRIEVAL_K", 20),
            final_k=cls._get_int("FINAL_K", 5),
            mmr_lambda=cls._get_float("MMR_LAMBDA", 0.7),
            enable_rerank=cls._get_bool("ENABLE_RERANK", True),
            enable_hybrid_search=cls._get_bool("ENABLE_HYBRID_SEARCH", False),
        )
    
    @classmethod
    def load_guardrail_config(cls) -> GuardrailConfig:
        return GuardrailConfig(
            enabled=cls._get_bool("ENABLE_GUARDRAILS", True),
            sensitive_topics_enabled=cls._get_bool("SENSITIVE_TOPICS_ENABLED", True),
            citation_required=cls._get_bool("CITATION_REQUIRED", True),
            max_response_chars=cls._get_int("MAX_RESPONSE_CHARS", 2000),
            max_sentences=cls._get_int("MAX_SENTENCES", 10),
        )
    
    @classmethod
    def load_auth_config(cls) -> AuthConfig:
        return AuthConfig(
            jwt_secret_key=cls._get_env("JWT_SECRET_KEY", "kolrose-secret-key-change-in-production-2024"),
            jwt_algorithm=cls._get_env("JWT_ALGORITHM", "HS256"),
            access_token_expire_minutes=cls._get_int("JWT_EXPIRE_MINUTES", 1440),
        )
    
    @classmethod
    def load_logging_config(cls) -> LoggingConfig:
        return LoggingConfig(
            level=cls._get_env("LOG_LEVEL", "INFO"),
            format_type=cls._get_env("LOG_FORMAT", "auto"),
            enable_audit=cls._get_bool("ENABLE_AUDIT_LOG", True),
            enable_metrics=cls._get_bool("ENABLE_METRICS", True),
        )
    
    @classmethod
    def load_memory_config(cls) -> MemoryConfig:
        return MemoryConfig(
            ttl_minutes=cls._get_int("MEMORY_TTL_MINUTES", 120),
            max_messages_per_conversation=cls._get_int("MEMORY_MAX_MESSAGES", 50),
            max_context_exchanges=cls._get_int("MEMORY_CONTEXT_EXCHANGES", 5),
        )
    
    @classmethod
    def load_company_config(cls) -> CompanyConfig:
        return CompanyConfig(
            name=cls._get_env("COMPANY_NAME", "Kolrose Limited"),
            address=cls._get_env("COMPANY_ADDRESS", "Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria"),
            website=cls._get_env("COMPANY_WEBSITE", "https://kolroselimited.com.ng"),
            email_hr=cls._get_env("COMPANY_EMAIL_HR", "hr@kolroselimited.com.ng"),
            email_compliance=cls._get_env("COMPANY_EMAIL_COMPLIANCE", "compliance@kolroselimited.com.ng"),
            email_security=cls._get_env("COMPANY_EMAIL_SECURITY", "security@kolroselimited.com.ng"),
            hotline_whistleblower=cls._get_env("COMPANY_HOTLINE_WHISTLEBLOWER", "0800-KOLROSE"),
            hotline_it_security=cls._get_env("COMPANY_HOTLINE_IT_SECURITY", "0800-KOL-ITSEC"),
        )
    
    @classmethod
    def load_app_config(cls) -> AppConfig:
        env_str = cls._get_env("APP_ENV", "development").lower()
        log_str = cls._get_env("LOG_LEVEL", "INFO").upper()
        return AppConfig(
            env=Environment(env_str) if env_str in [e.value for e in Environment] else Environment.DEVELOPMENT,
            log_level=LogLevel(log_str) if log_str in [l.value for l in LogLevel] else LogLevel.INFO,
            streamlit_port=cls._get_int("STREAMLIT_PORT", 8501),
            fastapi_port=cls._get_int("FASTAPI_PORT", 8000),
            debug=cls._get_bool("DEBUG", False),
        )


# ============================================================================
# GLOBAL CONFIGURATION INSTANCES
# ============================================================================

# Load all configuration
_config = ConfigLoader.load_all()

# API Configuration
API_CONFIG: APIConfig = _config['api']
OPENROUTER_API_KEY: str = API_CONFIG.openrouter_api_key
OPENROUTER_BASE_URL: str = API_CONFIG.openrouter_base_url
DEFAULT_MODEL: str = API_CONFIG.default_model
MAX_OUTPUT_TOKENS: int = API_CONFIG.max_tokens

# Embedding Configuration
EMBEDDING_CONFIG: EmbeddingConfig = _config['embedding']
EMBEDDING_MODEL: str = EMBEDDING_CONFIG.model_name
EMBEDDING_DEVICE: str = EMBEDDING_CONFIG.device

# Database Configuration
DB_CONFIG: DatabaseConfig = _config['database']

# ============================================================
# CHROMA_PATH with cloud-safe defaults
# ============================================================
IS_CLOUD = bool(
    os.environ.get("RAILWAY_ENVIRONMENT") or 
    os.environ.get("PORT") or
    os.environ.get("STREAMLIT_SHARING_MODE")
)

if IS_CLOUD:
    CHROMA_PATH = os.environ.get("CHROMA_PATH", os.path.join(tempfile.gettempdir(), "chroma_db"))
    print(f"☁️ Cloud environment detected. Using ChromaDB path: {CHROMA_PATH}")
else:
    CHROMA_PATH = os.environ.get("CHROMA_PATH", DB_CONFIG.persist_directory)

# Ensure directory exists and is writable
try:
    os.makedirs(CHROMA_PATH, exist_ok=True)
    test_file = os.path.join(CHROMA_PATH, ".write_test")
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
except (IOError, OSError, PermissionError):
    CHROMA_PATH = os.path.join(tempfile.gettempdir(), "chroma_db")
    os.makedirs(CHROMA_PATH, exist_ok=True)
    print(f"⚠️ Configured path not writable. Using: {CHROMA_PATH}")

COLLECTION_NAME: str = DB_CONFIG.collection_name

# RAG Configuration
RAG_CONFIG: RAGConfig = _config['rag']
CHUNK_SIZE: int = RAG_CONFIG.chunk_size
CHUNK_OVERLAP: int = RAG_CONFIG.chunk_overlap
RETRIEVAL_K: int = RAG_CONFIG.retrieval_k
FINAL_K: int = RAG_CONFIG.final_k

# Guardrail Configuration
GUARDRAIL_CONFIG: GuardrailConfig = _config['guardrails']
ENABLE_GUARDRAILS: bool = GUARDRAIL_CONFIG.enabled
SENSITIVE_TOPICS_ENABLED: bool = GUARDRAIL_CONFIG.sensitive_topics_enabled
CITATION_REQUIRED: bool = GUARDRAIL_CONFIG.citation_required
MAX_RESPONSE_CHARS: int = GUARDRAIL_CONFIG.max_response_chars

# Auth Configuration
AUTH_CONFIG: AuthConfig = _config['auth']
JWT_SECRET_KEY: str = AUTH_CONFIG.jwt_secret_key
JWT_ALGORITHM: str = AUTH_CONFIG.jwt_algorithm
JWT_EXPIRE_MINUTES: int = AUTH_CONFIG.access_token_expire_minutes

# Logging Configuration
LOGGING_CONFIG: LoggingConfig = _config['logging']
LOG_FORMAT: str = LOGGING_CONFIG.format_type
ENABLE_AUDIT_LOG: bool = LOGGING_CONFIG.enable_audit
ENABLE_METRICS: bool = LOGGING_CONFIG.enable_metrics

# Memory Configuration
MEMORY_CONFIG: MemoryConfig = _config['memory']
MEMORY_TTL_MINUTES: int = MEMORY_CONFIG.ttl_minutes
MEMORY_MAX_MESSAGES: int = MEMORY_CONFIG.max_messages_per_conversation

# Company Configuration
COMPANY_CONFIG: CompanyConfig = _config['company']
COMPANY_INFO: Dict[str, str] = {
    "name": COMPANY_CONFIG.name,
    "address": COMPANY_CONFIG.address,
    "website": COMPANY_CONFIG.website,
    "email_hr": COMPANY_CONFIG.email_hr,
    "email_compliance": COMPANY_CONFIG.email_compliance,
    "email_security": COMPANY_CONFIG.email_security,
    "hotline_whistleblower": COMPANY_CONFIG.hotline_whistleblower,
    "hotline_it_security": COMPANY_CONFIG.hotline_it_security,
}

# Application Configuration
APP_CONFIG: AppConfig = _config['app']
APP_ENV: str = APP_CONFIG.env.value
LOG_LEVEL: str = APP_CONFIG.log_level.value


# ============================================================================
# CONFIGURATION VALIDATOR
# ============================================================================

def validate_config() -> Dict[str, Any]:
    """Validate the current configuration."""
    report = {'valid': True, 'issues': [], 'warnings': [], 'summary': {}}
    
    if not OPENROUTER_API_KEY:
        report['issues'].append("OPENROUTER_API_KEY is not set.")
        report['valid'] = False
    elif OPENROUTER_API_KEY.startswith("sk-or-v1-your"):
        report['issues'].append("OPENROUTER_API_KEY is still set to placeholder.")
        report['valid'] = False
    
    if JWT_SECRET_KEY == "kolrose-secret-key-change-in-production-2024":
        report['warnings'].append("JWT_SECRET_KEY is using default value. Change in production!")
    
    policies_path = Path(POLICIES_PATH)
    if not policies_path.exists():
        report['warnings'].append(f"Policies directory not found: {POLICIES_PATH}")
    else:
        md_files = list(policies_path.rglob("*.md"))
        if not md_files:
            report['warnings'].append(f"No .md files found in {POLICIES_PATH}")
        report['summary']['policy_files'] = len(md_files)
    
    chroma_path = Path(CHROMA_PATH)
    report['summary']['chroma_exists'] = chroma_path.exists()
    
    return report


def print_config(show_secrets: bool = False):
    """Print current configuration."""
    print("\n" + "=" * 60)
    print(f"  {COMPANY_INFO['name']} - Configuration")
    print("=" * 60)
    print(f"\n🏢 Company: {COMPANY_INFO['name']}")
    print(f"🔗 Model: {DEFAULT_MODEL}")
    print(f"🤖 Embeddings: {EMBEDDING_MODEL}")
    print(f"💾 ChromaDB: {CHROMA_PATH}")
    print(f"📄 Policies: {POLICIES_PATH}")
    print(f"☁️ Cloud: {IS_CLOUD}")
    print(f"🔐 Auth: {'✅ Configured' if JWT_SECRET_KEY != 'kolrose-secret-key-change-in-production-2024' else '⚠️ Default'}")
    print(f"📝 Logging: {LOG_FORMAT}")
    print("=" * 60 + "\n")


def load_streamlit_secrets():
    """Load configuration from Streamlit secrets."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            try:
                secrets = st.secrets
                for key in ['OPENROUTER_API_KEY', 'LLM_MODEL', 'JWT_SECRET_KEY']:
                    if key in secrets:
                        os.environ[key] = secrets[key]
                logger.info("✅ Loaded configuration from Streamlit secrets")
            except Exception:
                pass
    except ImportError:
        pass


# Initialize
load_streamlit_secrets()

if APP_CONFIG.debug:
    print_config()


if __name__ == "__main__":
    print_config()
    report = validate_config()
    if not report['valid']:
        print("\n❌ Configuration issues:")
        for issue in report['issues']:
            print(f"   - {issue}")
    if report['warnings']:
        print("\n⚠️ Warnings:")
        for warning in report['warnings']:
            print(f"   - {warning}")