import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any
import json
from contextvars import ContextVar
from dotenv import load_dotenv
from .logging_setup import configure_logging


load_dotenv()
configure_logging()
logger = logging.getLogger("config")


@dataclass
class Settings:
    docs_dir: str = os.getenv("DOCS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs")))
    data_dir: str = os.getenv("DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-5")
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))
    min_context_similarity: float = float(os.getenv("MIN_CONTEXT_SIMILARITY", "0.25"))
    frontend_dir: str = os.getenv("FRONTEND_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend")))
    # Model profiles
    model_profiles: Dict[str, Dict[str, Any]] = None  # type: ignore[assignment]
    default_profile_id: str = os.getenv("DEFAULT_MODEL_PROFILE", "openai-gpt5")
    # Context for per-request selected profile
    current_profile_id: ContextVar[Optional[str]] = ContextVar("current_profile_id", default=None)


settings = Settings()


def ensure_data_dirs() -> None:
    os.makedirs(settings.data_dir, exist_ok=True)
    logger.info("Ensured data directory exists", extra={"data_dir": settings.data_dir})


# Initialize model profiles from environment
def _init_model_profiles() -> None:
    raw = os.getenv("MODEL_PROFILES")
    profiles: Dict[str, Dict[str, Any]]
    if raw:
        try:
            profiles = json.loads(raw)
            if not isinstance(profiles, dict):
                raise ValueError("MODEL_PROFILES must be a JSON object")
        except Exception:
            logger.exception("Failed parsing MODEL_PROFILES; falling back to defaults")
            profiles = {}
    else:
        profiles = {}
    # Defaults
    profiles.setdefault("openai-gpt5", {
        "id": "openai-gpt5",
        "label": "OpenAI GPT-5 (hosted)",
        "chat_model": os.getenv("CHAT_MODEL", settings.chat_model),
        "embedding_model": os.getenv("EMBEDDING_MODEL", settings.embedding_model),
        "base_url": os.getenv("OPENAI_BASE_URL") or None,
        "api_key_env": "OPENAI_API_KEY",
    })
    profiles.setdefault("local-runner", {
        "id": "local-runner",
        "label": "Local Model Runner",
        "chat_model": os.getenv("LOCAL_CHAT_MODEL", "llama-3.1-8b-instruct-q4"),
        "embedding_model": os.getenv("LOCAL_EMBED_MODEL", "nomic-embed-text-v1.5"),
        "base_url": os.getenv("LOCAL_BASE_URL", "http://model-runner:8001/v1"),
        "api_key_env": os.getenv("LOCAL_API_KEY_ENV", "LOCAL_API_KEY"),
    })
    profiles.setdefault("mistral-light", {
        "id": "mistral-light",
        "label": "Mistral Small (hosted)",
        "chat_model": os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest"),
        "embedding_model": os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed"),
        "base_url": os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
        "api_key_env": os.getenv("MISTRAL_API_KEY_ENV", "MISTRAL_API_KEY"),
    })
    settings.model_profiles = profiles


_init_model_profiles()


def get_profile(profile_id: Optional[str]) -> Dict[str, Any]:
    pid = profile_id or settings.current_profile_id.get() or settings.default_profile_id
    mp = settings.model_profiles.get(pid) if settings.model_profiles else None
    if not mp:
        # Fall back to primary settings
        return {
            "id": "fallback",
            "label": "Fallback",
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "base_url": os.getenv("OPENAI_BASE_URL"),
            "api_key_env": "OPENAI_API_KEY",
        }
    return mp

