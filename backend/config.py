import os
import logging
from dataclasses import dataclass
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


settings = Settings()


def ensure_data_dirs() -> None:
    os.makedirs(settings.data_dir, exist_ok=True)
    logger.info("Ensured data directory exists", extra={"data_dir": settings.data_dir})

