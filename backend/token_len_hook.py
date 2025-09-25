from functools import lru_cache
import os
from llama_cpp import Llama


@lru_cache(maxsize=1)
def _tok() -> Llama:
    model_path = os.environ.get("LLAMA_GGUF_PATH")
    if not model_path:
        raise RuntimeError("LLAMA_GGUF_PATH not set")
    # Load tokenizer only; fast init
    return Llama(model_path=model_path, vocab_only=True)


def token_len(text: str) -> int:
    toks = _tok().tokenize(text.encode("utf-8"), add_bos=False)
    return int(len(toks))


