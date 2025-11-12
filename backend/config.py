import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv
from .logging_setup import configure_logging
from contextvars import ContextVar


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
    # Current profile ID for this request/task; can be overridden via ContextVar in request scope
    current_profile_id: ContextVar[str] = ContextVar(
        "current_profile_id",
        default=os.getenv("DEFAULT_MODEL_PROFILE", "openai-gpt5"),
    )


settings = Settings()


def ensure_data_dirs() -> None:
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.docs_dir, exist_ok=True)
    logger.info("Ensured data directories exist", extra={"data_dir": settings.data_dir, "docs_dir": settings.docs_dir})


def get_profile(profile_id: str | None) -> dict:
    """Resolve a model profile dict for chat/embedding and base URL.

    Returns a mapping with at least keys: id, chat_model, embedding_model, api_key_env, base_url (optional).
    Resolution order:
      1) explicit profile_id argument if provided
      2) Settings.current_profile_id ContextVar
      3) DEFAULT_MODEL_PROFILE env var (docker-compose)
      4) fallback to 'openai-gpt5'
    """
    # Determine desired profile ID
    desired = (
        profile_id
        or (settings.current_profile_id.get() if settings else None)
        or os.getenv("DEFAULT_MODEL_PROFILE")
        or "openai-gpt5"
    )

    # Base defaults from global settings/env
    default_chat = os.getenv("CHAT_MODEL", settings.chat_model)
    default_embed = os.getenv("EMBEDDING_MODEL", settings.embedding_model)

    def _sanitize_model(value: str | None) -> str | None:
        if value is None:
            return None
        v = value.strip()
        if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
            v = v[1:-1].strip()
        return v

    # Known profiles
    profiles: dict[str, dict] = {
        "openai-gpt5": {
            "id": "openai-gpt5",
            "chat_model": _sanitize_model(default_chat) or "gpt-5",
            "embedding_model": _sanitize_model(default_embed) or "text-embedding-3-large",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": os.getenv("OPENAI_BASE_URL"),
        },
        # Local proxy/runner (e.g., LiteLLM/Ollama/OpenAI-compatible server)
        "local-runner": {
            "id": "local-runner",
            # Prefer LOCAL_* if set, then compose-exposed CHAT_LOCAL_MODEL/EMBED_LOCAL_MODEL, else fall back to defaults
            "chat_model": _sanitize_model(os.getenv("LOCAL_CHAT_MODEL") or os.getenv("CHAT_LOCAL_MODEL") or (default_chat or "gpt-4o-mini"))
            ,
            "embedding_model": _sanitize_model(os.getenv("LOCAL_EMBED_MODEL") or os.getenv("EMBED_LOCAL_MODEL") or (default_embed or "text-embedding-3-large")),
            "api_key_env": "LOCAL_API_KEY",
            "base_url": os.getenv("LOCAL_BASE_URL"),
        },
        # Optional hosted alt vendor example (if env provided)
        "mistral": {
            "id": "mistral",
            "chat_model": _sanitize_model(os.getenv("MISTRAL_CHAT_MODEL") or (default_chat or "mistral-small-latest")),
            "embedding_model": _sanitize_model(os.getenv("MISTRAL_EMBED_MODEL") or (default_embed or "mistral-embed")),
            "api_key_env": "MISTRAL_API_KEY",
            "base_url": os.getenv("MISTRAL_BASE_URL") or "https://api.mistral.ai/v1",
        },
    }

    prof = profiles.get(str(desired), profiles["openai-gpt5"]).copy()

    # Allow per-request overrides via env if provided
    # These ensure that global envs like CHAT_MODEL/EMBEDDING_MODEL update the selected profile when set
    if os.getenv("CHAT_MODEL"):
        prof["chat_model"] = _sanitize_model(os.getenv("CHAT_MODEL"))
    if os.getenv("EMBEDDING_MODEL"):
        prof["embedding_model"] = _sanitize_model(os.getenv("EMBEDDING_MODEL"))
    # If an explicit OPENAI_BASE_URL is supplied and profile has no base_url, use it
    if not prof.get("base_url") and os.getenv("OPENAI_BASE_URL"):
        prof["base_url"] = os.getenv("OPENAI_BASE_URL")

    return prof

