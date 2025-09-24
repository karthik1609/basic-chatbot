import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
import httpx
from .logging_setup import configure_logging
from .config import get_profile, settings


load_dotenv()
configure_logging()
logger = logging.getLogger("openai_client")


def _httpx_with_logging() -> httpx.Client:
    def _mask(value: str) -> str:
        if not value:
            return value
        return value[:8] + "…" if len(value) > 8 else "…"

    def _log_response(resp: httpx.Response) -> None:
        try:
            req = resp.request
            url = str(req.url)
            method = req.method
            status = resp.status_code
            # Build curl with masked auth
            headers = []
            for k, v in req.headers.items():
                if k.lower() == "authorization":
                    v = "Bearer " + _mask(v.split(" ")[-1])
                headers.append(f"-H '{k}: {v}'")
            data = ""
            if req.content:
                try:
                    body = req.content.decode("utf-8", errors="ignore")
                except Exception:
                    body = "<binary>"
                # Truncate overly large bodies
                if len(body) > 2000:
                    body = body[:2000] + "…"
                data = f"--data '{body}'"
            curl = f"curl -X {method} {' '.join(headers)} {data} '{url}'"
            level = logger.error if status >= 400 else logger.info
            level("llm_http", extra={
                "method": method,
                "url": url,
                "status": status,
                "curl": curl,
            })
        except Exception:
            # Never break the request pipeline due to logging
            pass

    return httpx.Client(event_hooks={"response": [_log_response]})


def get_openai_client(profile_id: str | None = None) -> OpenAI:
    prof = get_profile(profile_id)
    api_key_env = prof.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(str(api_key_env) or "OPENAI_API_KEY")
    base_url = prof.get("base_url") or os.getenv("OPENAI_BASE_URL")
    # Allow local-runner without a key (for unsecured local proxies)
    if str(prof.get("id")) == "local-runner":
        # Ensure no hosted telemetry uses a real key; LiteLLM proxy accepts any Bearer if configured
        api_key = api_key or "local"
    if not api_key:
        logger.error("API key not set for profile", extra={"profile": prof.get("id")})
        raise RuntimeError("API key not set in environment for selected profile")
    if base_url:
        logger.info("Creating OpenAI client with base URL", extra={"base_url": base_url, "profile": prof.get("id")})
        return OpenAI(api_key=api_key, base_url=base_url, http_client=_httpx_with_logging())
    logger.info("Creating OpenAI client (default base)", extra={"profile": prof.get("id")})
    return OpenAI(api_key=api_key, http_client=_httpx_with_logging())


def get_resolved_base_url(profile_id: str | None = None) -> str:
    """Return the resolved base URL the OpenAI client will hit for this profile."""
    prof = get_profile(profile_id)
    base_url = prof.get("base_url") or os.getenv("OPENAI_BASE_URL")
    if base_url:
        return str(base_url)
    return "https://api.openai.com/v1"


