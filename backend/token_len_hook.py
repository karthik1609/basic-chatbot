from functools import lru_cache
import glob
import os
from typing import Optional
import httpx
from llama_cpp import Llama


def _pick_gguf(directory: str) -> Optional[str]:
    prefer_order = ["*Q2_K.gguf", "*.Q3_K_S.gguf", "*.Q3_K_M.gguf", "*.Q4_0.gguf", "*.gguf"]
    for pattern in prefer_order:
        matches = sorted(glob.glob(os.path.join(directory, pattern)))
        if matches:
            return matches[0]
    return None


def _resolve_model_path() -> str:
    env_path = os.environ.get("LLAMA_GGUF_PATH")
    if env_path:
        # If it's a directory (or intended to be), try to pick a file from it
        if env_path.endswith(".gguf"):
            # If caller gave a concrete file path, but it doesn't exist yet, ensure parent exists and download there
            if not os.path.exists(env_path):
                os.makedirs(os.path.dirname(env_path) or ".", exist_ok=True)
                _download_gguf(
                    "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q2_K.gguf",
                    env_path,
                )
            return env_path
        # Treat it as a directory path
        if not os.path.isdir(env_path):
            # Directory doesn't exist yet; create and proceed
            os.makedirs(env_path, exist_ok=True)
        picked = _pick_gguf(env_path)
        if picked:
            return picked
        # Empty dir: download smallest into it and return
        dest = os.path.join(env_path, "nomic-embed-text-v1.5.Q2_K.gguf")
        _download_gguf(
            "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q2_K.gguf",
            dest,
        )
        return dest

    # Prefer container locations first when present
    for base in ("/models", "/app/models"):
        if os.path.isdir(base):
            picked = _pick_gguf(base)
            if picked:
                return picked

    # Dev/local or first-run in container: choose a writable target dir for download
    if os.path.isdir("/models") and os.access("/models", os.W_OK):
        target_dir = "/models"
    else:
        target_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(target_dir, exist_ok=True)

    picked = _pick_gguf(target_dir)
    if picked:
        return picked

    # Download Q2_K by default (smallest); persist to target_dir
    dest = os.path.join(target_dir, "nomic-embed-text-v1.5.Q2_K.gguf")
    _download_gguf(
        "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q2_K.gguf",
        dest,
    )
    return dest


def _download_gguf(url: str, dest: str) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes():
                if chunk:
                    f.write(chunk)


@lru_cache(maxsize=1)
def _tok() -> Llama:
    model_path = _resolve_model_path()
    return Llama(model_path=model_path, vocab_only=True)


def token_len(text: str) -> int:
    toks = _tok().tokenize(text.encode("utf-8"), add_bos=False)
    return int(len(toks))


