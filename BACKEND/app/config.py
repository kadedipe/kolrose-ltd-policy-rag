"""
Configuration Management for Kolrose Limited RAG System.
=========================================================
Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria

Handles:
- Environment variable loading (.env file)
- Configuration validation
- Sensible defaults for all settings
- Type casting and error handling
- Configuration categories (API, DB, Guardrails, etc.)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Try to load .env file
try:
    from dotenv import load_dotenv
    # Look for .env in project root
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from {env_path}")
    else:
        # Try current directory
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
    default_model: str = "google/gemini-2.0-flash-001"
    max_tokens: int = 500
    temperature: float = 0.0
    request_timeout: int = 30
    max_retries: int = 3
    
    # Free tier models (fallback options)
    free_models: List[str] = field(default_factory=lambda: [
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.1-8b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
    ])
    
    @property
    def is_configured(self) -> bool:
        return bool(self.openrouter_api_key and 
                   not self.openrouter_api_key.startswith("sk-or-v1-your"))


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"  # cpu or cuda
    batch_size: int = 16
    normalize: bool = True
    dimension: int = 384  # all-MiniLM-L6-v2 dimension
    
    # Alternative models
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
    
    # Re-ranking
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

# ======================================================
# DATA DIRECTORIES
# ======================================================

# Check if running in a cloud/container system first
if os.path.exists("/app/data"):
    DATA_DIR = Path("/app/data")
else:
    # Resolve relative to the location of config.py safely
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR

# Let environment variables override paths dynamically
POLICIES_PATH: str = os.environ.get("POLICIES_PATH", "./BACKEND/policies")
CHROMA_PATH = os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma_db"))

# Ensure the directories actually exist
Path(POLICIES_PATH).mkdir(parents=True, exist_ok=True)
Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

class ConfigLoader:
    """
    Loads and validates all configuration from environment variables.
    """
    
    @staticmethod
    def _get_env(key: str, default: Any = None, required: bool = False) -> str:
        """Get environment variable with validation"""
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
        """Get boolean environment variable"""
        value = os.environ.get(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")
    
    @staticmethod
    def _get_int(key: str, default: int = 0) -> int:
        """Get integer environment variable"""
        try:
            return int(os.environ.get(key, str(default)))
        except ValueError:
            return default
    
    @staticmethod
    def _get_float(key: str, default: float = 0.0) -> float:
        """Get float environment variable"""
        try:
            return float(os.environ.get(key, str(default)))
        except ValueError:
            return default
    
    @classmethod
    def load_all(cls) -> Dict[str, Any]:
        """Load all configuration"""
        return {
            'api': cls.load_api_config(),
            'embedding': cls.load_embedding_config(),
            'database': cls.load_database_config(),
            'rag': cls.load_rag_config(),
            'guardrails': cls.load_guardrail_config(),
            'company': cls.load_company_config(),
            'app': cls.load_app_config(),
        }
    
    @classmethod
    def load_api_config(cls) -> APIConfig:
        """Load API configuration"""
        return APIConfig(
            openrouter_api_key=cls._get_env("OPENROUTER_API_KEY", ""),
            openrouter_base_url=cls._get_env(
                "OPENROUTER_BASE_URL", 
                "https://openrouter.ai/api/v1"
            ),
            default_model=cls._get_env(
                "LLM_MODEL", 
                "google/gemini-2.0-flash-001"
            ),
            max_tokens=cls._get_int("MAX_OUTPUT_TOKENS", 500),
            temperature=cls._get_float("LLM_TEMPERATURE", 0.0),
            request_timeout=cls._get_int("LLM_REQUEST_TIMEOUT", 30),
            max_retries=cls._get_int("LLM_MAX_RETRIES", 3),
        )
    
    @classmethod
    def load_embedding_config(cls) -> EmbeddingConfig:
        """Load embedding configuration"""
        return EmbeddingConfig(
            model_name=cls._get_env(
                "EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            device=cls._get_env("EMBEDDING_DEVICE", "cpu"),
            batch_size=cls._get_int("EMBEDDING_BATCH_SIZE", 16),
            normalize=cls._get_bool("EMBEDDING_NORMALIZE", True),
        )
    
    @classmethod
    def load_database_config(cls) -> DatabaseConfig:
        """Load database configuration"""
        return DatabaseConfig(
            persist_directory=cls._get_env("CHROMA_PATH", "./chroma_db"),
            collection_name=cls._get_env(
                "CHROMA_COLLECTION_NAME",
                "kolrose_policies_v2"
            ),
            distance_metric=cls._get_env("CHROMA_DISTANCE_METRIC", "cosine"),
        )
    
    @classmethod
    def load_rag_config(cls) -> RAGConfig:
        """Load RAG configuration"""
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
        """Load guardrail configuration"""
        return GuardrailConfig(
            enabled=cls._get_bool("ENABLE_GUARDRAILS", True),
            sensitive_topics_enabled=cls._get_bool(
                "SENSITIVE_TOPICS_ENABLED", True
            ),
            citation_required=cls._get_bool("CITATION_REQUIRED", True),
            max_response_chars=cls._get_int("MAX_RESPONSE_CHARS", 2000),
            max_sentences=cls._get_int("MAX_SENTENCES", 10),
        )
    
    @classmethod
    def load_company_config(cls) -> CompanyConfig:
        """Load company configuration"""
        return CompanyConfig(
            name=cls._get_env("COMPANY_NAME", "Kolrose Limited"),
            address=cls._get_env(
                "COMPANY_ADDRESS",
                "Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria"
            ),
            website=cls._get_env(
                "COMPANY_WEBSITE",
                "https://kolroselimited.com.ng"
            ),
            email_hr=cls._get_env("COMPANY_EMAIL_HR", "hr@kolroselimited.com.ng"),
            email_compliance=cls._get_env(
                "COMPANY_EMAIL_COMPLIANCE",
                "compliance@kolroselimited.com.ng"
            ),
            email_security=cls._get_env(
                "COMPANY_EMAIL_SECURITY",
                "security@kolroselimited.com.ng"
            ),
            hotline_whistleblower=cls._get_env(
                "COMPANY_HOTLINE_WHISTLEBLOWER",
                "0800-KOLROSE"
            ),
            hotline_it_security=cls._get_env(
                "COMPANY_HOTLINE_IT_SECURITY",
                "0800-KOL-ITSEC"
            ),
        )
    
    @classmethod
    def load_app_config(cls) -> AppConfig:
        """Load application configuration"""
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
CHROMA_PATH: str = DB_CONFIG.persist_directory
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
    """
    Validate the current configuration.
    Returns a report of any issues found.
    """
    report = {
        'valid': True,
        'issues': [],
        'warnings': [],
        'summary': {},
    }
    
    # Check API key
    if not OPENROUTER_API_KEY:
        report['issues'].append(
            "OPENROUTER_API_KEY is not set. Get a free key at https://openrouter.ai/keys"
        )
        report['valid'] = False
    elif OPENROUTER_API_KEY.startswith("sk-or-v1-your"):
        report['issues'].append(
            "OPENROUTER_API_KEY is still set to the placeholder value"
        )
        report['valid'] = False
    
    # Check policies directory
    policies_path = Path(POLICIES_PATH)
    if not policies_path.exists():
        report['warnings'].append(f"Policies directory not found: {POLICIES_PATH}")
    else:
        md_files = list(policies_path.rglob("*.md"))
        if not md_files:
            report['warnings'].append(f"No .md files found in {POLICIES_PATH}")
        report['summary']['policy_files'] = len(md_files)
    
    # Check ChromaDB
    chroma_path = Path(CHROMA_PATH)
    if chroma_path.exists():
        report['summary']['chroma_exists'] = True
    else:
        report['warnings'].append("ChromaDB not yet created. Run ingestion first.")
        report['summary']['chroma_exists'] = False
    
    # Log report
    if report['issues']:
        for issue in report['issues']:
            logger.error(f"❌ {issue}")
    if report['warnings']:
        for warning in report['warnings']:
            logger.warning(f"⚠️ {warning}")
    
    return report


def print_config(show_secrets: bool = False):
    """Print current configuration (for debugging)"""
    print("\n" + "=" * 60)
    print(f"  {COMPANY_INFO['name']} - Configuration")
    print("=" * 60)
    
    print(f"\n🏢 Company:")
    print(f"   Name: {COMPANY_INFO['name']}")
    print(f"   Location: {COMPANY_INFO['address']}")
    
    print(f"\n🔗 API:")
    print(f"   Provider: OpenRouter")
    print(f"   Model: {DEFAULT_MODEL}")
    if show_secrets:
        print(f"   Key: {OPENROUTER_API_KEY[:20]}...")
    else:
        print(f"   Key: {'✅ Configured' if API_CONFIG.is_configured else '❌ Missing'}")
    
    print(f"\n🤖 Embeddings:")
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"   Device: {EMBEDDING_DEVICE}")
    
    print(f"\n💾 Database:")
    print(f"   Path: {CHROMA_PATH}")
    print(f"   Collection: {COLLECTION_NAME}")
    
    print(f"\n🔍 RAG:")
    print(f"   Chunk Size: {CHUNK_SIZE}")
    print(f"   Retrieval K: {RETRIEVAL_K}")
    print(f"   Final K: {FINAL_K}")
    print(f"   Re-rank: {'✅' if RAG_CONFIG.enable_rerank else '❌'}")
    
    print(f"\n🛡️ Guardrails:")
    print(f"   Enabled: {'✅' if ENABLE_GUARDRAILS else '❌'}")
    print(f"   Sensitive Topics: {'✅' if SENSITIVE_TOPICS_ENABLED else '❌'}")
    print(f"   Citations Required: {'✅' if CITATION_REQUIRED else '❌'}")
    
    print(f"\n⚙️ Application:")
    print(f"   Environment: {APP_ENV}")
    print(f"   Log Level: {LOG_LEVEL}")
    
    print("=" * 60 + "\n")


# ============================================================================
# STREAMLIT SECRETS LOADER (For Streamlit Cloud / Safeguarded for Local/CLI)
# ============================================================================

def load_streamlit_secrets():
    """
    Load configuration from Streamlit secrets (for Streamlit Cloud deployment).
    Safeguarded with a try-except layer so background script invocations don't crash.
    """
    try:
        import streamlit as st
        
        # Override with Streamlit secrets if available
        if hasattr(st, 'secrets'):
            try:
                secrets = st.secrets
                
                # API
                if 'OPENROUTER_API_KEY' in secrets:
                    os.environ['OPENROUTER_API_KEY'] = secrets['OPENROUTER_API_KEY']
                if 'LLM_MODEL' in secrets:
                    os.environ['LLM_MODEL'] = secrets['LLM_MODEL']
                
                # Embeddings
                if 'EMBEDDING_DEVICE' in secrets:
                    os.environ['EMBEDDING_DEVICE'] = secrets['EMBEDDING_DEVICE']
                
                # Guardrails
                if 'ENABLE_GUARDRAILS' in secrets:
                    os.environ['ENABLE_GUARDRAILS'] = secrets['ENABLE_GUARDRAILS']
                
                logger.info("✅ Loaded configuration from Streamlit secrets")
            except Exception:
                # Catching StreamlitSecretNotFoundError when executed as a raw Python CLI command
                logger.info("ℹ️ Streamlit secrets file not found. Falling back to native system environment variables.")
                pass
            
    except ImportError:
        pass  # Not running in Streamlit


# ============================================================================
# INITIALIZATION
# ============================================================================

# Load Streamlit secrets if available
load_streamlit_secrets()

# Log configuration on import
if APP_CONFIG.debug:
    print_config()


if __name__ == "__main__":
    # Print configuration and validate
    print_config()
    report = validate_config()
    
    if not report['valid']:
        print("\n❌ Configuration issues found:")
        for issue in report['issues']:
            print(f"   - {issue}")
    
    if report['warnings']:
        print("\n⚠️ Warnings:")
        for warning in report['warnings']:
            print(f"   - {warning}")