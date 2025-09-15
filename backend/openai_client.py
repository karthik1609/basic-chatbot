import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from .logging_setup import configure_logging


load_dotenv()
configure_logging()
logger = logging.getLogger("openai_client")


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        logger.error("OPENAI_API_KEY not set in environment")
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    if base_url:
        logger.info("Creating OpenAI client with custom base URL", extra={"base_url": base_url})
        return OpenAI(api_key=api_key, base_url=base_url)
    logger.info("Creating OpenAI client with default base URL")
    return OpenAI(api_key=api_key)


