import os
import json
import logging
from typing import List, Tuple, Dict, Any

import numpy as np
try:  # Optional FAISS; fallback to numpy retrieval if unavailable
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False
from pypdf import PdfReader

from .config import settings, ensure_data_dirs, get_profile
from .logging_setup import configure_logging
from .openai_client import get_openai_client
from .config import settings


configure_logging()
logger = logging.getLogger("rag")

def _artifact_paths() -> tuple[str, str, str]:
    # Use embedding model from the selected profile to namespace artifacts
    prof = get_profile(None)
    embed_model = str(prof.get("embedding_model") or settings.embedding_model)
    # sanitize filename
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in embed_model)
    index_file = os.path.join(settings.data_dir, f"faiss.index.{safe}")
    embed_file = os.path.join(settings.data_dir, f"embeddings.{safe}.npy")
    meta_file = os.path.join(settings.data_dir, f"meta.{safe}.json")
    return index_file, embed_file, meta_file


def _read_pdfs_from_directory(directory_path: str) -> List[Tuple[str, str]]:
    documents: List[Tuple[str, str]] = []
    logger.info("Scanning directory for PDFs", extra={"directory": directory_path})
    for filename in sorted(os.listdir(directory_path)):
        if not filename.lower().endswith(".pdf"):
            continue
        filepath = os.path.join(directory_path, filename)
        try:
            reader = PdfReader(filepath)
            pages_text = []
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                # Normalize whitespace
                pages_text.append("\n".join(line.strip() for line in text.splitlines()))
            content = "\n\n".join(pages_text)
            documents.append((filename, content))
        except Exception as exc:  # noqa: BLE001
            # Skip corrupted PDFs but continue
            logger.exception("Failed to read PDF", extra={"file": filepath})
    return documents


def _chunk_text(text: str, chunk_budget_tokens: int | None = None) -> List[Dict[str, Any]]:
    """Contiguous, non-overlapping token-budgeted chunks with soft boundaries.

    If sentence boundaries fit within the token budget, prefer them. Otherwise,
    perform a hard token cut (approximated by a conservative chars-per-token ratio
    unless a tokenizer is provided via environment hook).
    """
    if not text:
        return []

    # Token budget configuration (from model card: 2048 default; allow override)
    max_tokens = int(os.getenv("EMBEDDING_MAX_TOKENS", "2048"))
    overhead = int(os.getenv("EMBEDDING_PROMPT_OVERHEAD", "0"))
    budget = max(64, (chunk_budget_tokens or int(os.getenv("CHUNK_TOKEN_BUDGET", "1024"))))
    budget = min(budget, max_tokens - overhead)

    # Require a tokenizer hook for exact token accounting
    def estimate_tokens(s: str) -> int:
        hook = os.getenv("TOKEN_LEN_HOOK")  # e.g., "backend.token_len_hook:token_len"
        if not hook:
            raise RuntimeError("TOKEN_LEN_HOOK must be set for exact tokenization")
        mod, fn = hook.split(":", 1)
        import importlib

        f = getattr(importlib.import_module(mod), fn)
        return int(f(s))

    # Simple sentence split for soft boundaries
    import re
    sentences = [s for s in re.split(r"(?<=[\.!?])\s+", text.strip()) if s]
    # If no obvious sentences, fall back to the whole text
    if not sentences:
        sentences = [text]

    # Build contiguous chunks by walking sentence list, falling back to hard cut
    chunks_out: List[Dict[str, Any]] = []
    buf: list[str] = []
    acc_tokens = 0

    def flush_buf() -> None:
        nonlocal buf, acc_tokens
        if not buf:
            return
        chunk = " ".join(buf).strip()
        if chunk:
            chunks_out.append({"text": chunk, "section": ""})
        buf = []
        acc_tokens = 0

    for s in sentences:
        t = estimate_tokens(s)
        if t > budget:
            # Split this long unit into budgeted pieces using tokenizer length checks
            remaining = s
            while remaining:
                # greedy grow substring until just before budget overflow
                lo, hi = 1, len(remaining)
                best = 1
                # binary search on character boundary guided by exact token count
                while lo <= hi:
                    mid = (lo + hi) // 2
                    cand = remaining[:mid]
                    tk = estimate_tokens(cand)
                    if acc_tokens + tk <= budget and cand.strip():
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                part = remaining[:best].strip()
                if not part:
                    # fallback: force 1 char to progress
                    part = remaining[:1]
                    best = 1
                if acc_tokens + estimate_tokens(part) > budget:
                    flush_buf()
                buf.append(part)
                acc_tokens += estimate_tokens(part)
                if acc_tokens >= budget:
                    flush_buf()
                remaining = remaining[best:]
            continue

        # Normal sentence fits
        if acc_tokens + t <= budget:
            buf.append(s)
            acc_tokens += t
        else:
            flush_buf()
            buf.append(s)
            acc_tokens = t

    flush_buf()
    # Add prev/next pointers in metadata for O(1) neighbor expansion
    for i in range(len(chunks_out)):
        chunks_out[i]["prev_id"] = i - 1 if i > 0 else None
        chunks_out[i]["next_id"] = i + 1 if i < len(chunks_out) - 1 else None
        # Store estimated token length for diagnostics
        try:
            chunks_out[i]["est_tokens"] = estimate_tokens(chunks_out[i]["text"])  # type: ignore[index]
        except Exception:
            pass

    return chunks_out


def _embed_texts(texts: List[str]) -> np.ndarray:
    # Fake embedding mode for CI/Offline runs
    if os.getenv("FAKE_EMBED", "") == "1" or (os.getenv("OPENAI_API_KEY", "").lower() in ("", "dummy")):
        dim = 384
        arr = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = abs(hash(t)) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.normal(size=dim).astype(np.float32)
            # normalize
            v /= (np.linalg.norm(v) + 1e-12)
            arr[i] = v
        return arr

    client = get_openai_client()
    import logging
    _logger = logging.getLogger("llm")
    from .openai_client import get_resolved_base_url
    _base = get_resolved_base_url()
    profile = get_profile(None)
    model_id = str(profile.get("embedding_model") or settings.embedding_model)

    _logger.info(
        "llm_request",
        extra={
            "base_url": _base,
            "endpoint": "/embeddings",
            "model": model_id,
            "compose_injected": {
                "CHAT_LOCAL_URL": os.getenv("CHAT_LOCAL_URL"),
                "EMBED_LOCAL_URL": os.getenv("EMBED_LOCAL_URL"),
                "CHAT_LOCAL_MODEL": os.getenv("CHAT_LOCAL_MODEL"),
                "EMBED_LOCAL_MODEL": os.getenv("EMBED_LOCAL_MODEL"),
            },
        },
    )

    # Batch requests to avoid "input is too large" errors from local runners
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    vectors: list[np.ndarray] = []

    idx = 0
    while idx < len(texts):
        current_batch = texts[idx : idx + max(1, batch_size)]
        kwargs: Dict[str, Any] = {
            "model": model_id,
            "input": current_batch,
        }
        # Never send encoding_format for local runner; many backends don't support it
        if os.getenv("EMBEDDINGS_ENCODING_FORMAT") and str(profile.get("id")) not in ("local-runner",):
            kwargs["encoding_format"] = os.getenv("EMBEDDINGS_ENCODING_FORMAT")

        try:
            response = client.embeddings.create(**kwargs)
            vectors.extend(np.array(item.embedding, dtype=np.float32) for item in response.data)
            idx += len(current_batch)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if ("input is too large" in msg or "too large to process" in msg) and batch_size > 1:
                # Halve the batch size and retry this window
                new_size = max(1, batch_size // 2)
                _logger.warning(
                    "embedding_batch_shrunk",
                    extra={"old_batch_size": batch_size, "new_batch_size": new_size},
                )
                batch_size = new_size
                # do not advance idx; retry with smaller batch
                continue
            if ("input is too large" in msg or "too large to process" in msg) and len(current_batch) == 1:
                # Split this single text in half and embed pieces, then average
                text0 = current_batch[0]
                mid = max(1, len(text0) // 2)
                parts = [text0[:mid], text0[mid:]]
                part_vecs: list[np.ndarray] = []
                for p in parts:
                    sub_kwargs = {"model": model_id, "input": [p]}
                    try:
                        sub_resp = client.embeddings.create(**sub_kwargs)
                        part_vecs.append(np.array(sub_resp.data[0].embedding, dtype=np.float32))
                    except Exception:
                        # If even halves fail, continue halving recursively by shrinking chars
                        quarter = max(1, len(p) // 2)
                        if len(p) == 0 or quarter == len(p):
                            raise
                        sub_kwargs = {"model": model_id, "input": [p[:quarter], p[quarter:]]}
                        sub_resp = client.embeddings.create(**sub_kwargs)
                        e0 = np.array(sub_resp.data[0].embedding, dtype=np.float32)
                        e1 = np.array(sub_resp.data[1].embedding, dtype=np.float32)
                        part_vecs.append((e0 + e1) / 2.0)
                avg_vec = sum(part_vecs) / float(len(part_vecs))
                vectors.append(avg_vec.astype(np.float32))
                idx += 1
                continue
            raise

    return np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)


def build_or_update_index() -> Dict[str, Any]:
    ensure_data_dirs()
    logger.info("Starting index build/update", extra={
        "docs_dir": settings.docs_dir,
        "data_dir": settings.data_dir,
    })
    docs = _read_pdfs_from_directory(settings.docs_dir)
    logger.info("Read PDFs", extra={"num_documents": len(docs)})
    all_chunks: List[str] = []
    chunk_metadata: List[Dict[str, Any]] = []

    for filename, content in docs:
        # semantic-recursive chunks
        chunks = _chunk_text(content)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk["text"])
            chunk_metadata.append({
                "source": filename,
                "chunk_index": idx,
                "section": chunk.get("section") or "",
            })

    # Emit token diagnostics (estimated) if available
    try:
        est_stats = {
            "min": int(min((m.get("est_tokens", 0) for m in chunk_metadata), default=0)),
            "max": int(max((m.get("est_tokens", 0) for m in chunk_metadata), default=0)),
            "avg": float(
                sum((m.get("est_tokens", 0) for m in chunk_metadata)) / max(1, len(chunk_metadata))
            ),
        }
    except Exception:
        est_stats = {"min": 0, "max": 0, "avg": 0.0}
    logger.info("Chunking complete", extra={"num_chunks": len(all_chunks), "est_tokens": est_stats})
    index_file, embed_file, meta_file = _artifact_paths()

    if not all_chunks:
        # Create an empty artifacts set to avoid runtime errors
        if _FAISS_AVAILABLE:
            dim = 3072  # text-embedding-3-large dimension
            index = faiss.IndexFlatIP(dim)
            faiss.write_index(index, index_file)
        else:
            np.save(embed_file, np.zeros((0, 0), dtype=np.float32))
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"chunks": [], "metadata": []}, f)
        logger.warning("No chunks produced; created empty index")
        return {"chunks": 0}

    embeddings = _embed_texts(all_chunks)
    logger.info("Embeddings created", extra={"shape": list(embeddings.shape)})

    # Persist metadata and embeddings (always keep an .npy copy for reranking)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks, "metadata": chunk_metadata}, f)
    try:
        np.save(embed_file, embeddings)
    except Exception:
        logger.exception("Failed to persist embeddings .npy copy")

    if _FAISS_AVAILABLE:
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, index_file)
        logger.info("FAISS index written", extra={"dim": dim, "num": len(all_chunks)})
    else:
        # Save embeddings for numpy-based retrieval
        # Normalize rows for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        np.save(embed_file, embeddings)
        logger.info("Embeddings saved for numpy retrieval", extra={"num": len(all_chunks)})

    return {"chunks": len(all_chunks)}


def _load_index_and_meta():  # type: ignore[no-untyped-def]
    index_file, embed_file, meta_file = _artifact_paths()
    if not os.path.exists(meta_file):
        raise FileNotFoundError("Index metadata not found; run /api/ingest first")
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    chunks = meta.get("chunks", [])
    metadata = meta.get("metadata", [])

    if _FAISS_AVAILABLE:
        if not os.path.exists(index_file):
            raise FileNotFoundError("FAISS index not found; run /api/ingest first")
        index = faiss.read_index(index_file)
        emb_array = None
        try:
            emb_array = np.load(embed_file)
        except Exception:
            pass
        logger.info("Loaded FAISS index", extra={"num_chunks": len(chunks), "embeddings_cached": bool(emb_array is not None)})
        return ("faiss", index, chunks, metadata, emb_array)
    else:
        if not os.path.exists(embed_file):
            raise FileNotFoundError("Embeddings file not found; run /api/ingest first")
        embeddings = np.load(embed_file)
        logger.info("Loaded numpy embeddings", extra={"num_chunks": len(chunks)})
        return ("numpy", embeddings, chunks, metadata, embeddings)


def retrieve_relevant_chunks(query: str, top_k: int | None = None) -> List[Tuple[str, Dict[str, Any], float]]:
    if top_k is None:
        top_k = settings.retrieval_top_k
    backend, index_or_embeds, chunks, metadata, embeddings = _load_index_and_meta()
    query_emb = _embed_texts([query]).astype(np.float32)

    # Primary retrieval
    if backend == "faiss":
        faiss.normalize_L2(query_emb)
        scores, idxs = index_or_embeds.search(query_emb, top_k)
        base_pairs = [(int(i), float(s)) for s, i in zip(scores[0], idxs[0]) if int(i) != -1]
    else:
        doc_embeds = index_or_embeds  # normalized
        q = query_emb[0]
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = (doc_embeds @ q).astype(np.float32)
        top_idx = np.argsort(-sims)[:top_k]
        base_pairs = [(int(i), float(sims[int(i)])) for i in top_idx]

    # Neighbor expansion (±N)
    expand_n = int(os.getenv("RETRIEVAL_EXPAND_NEIGHBORS", "1"))
    candidate_indices: set[int] = set()
    groups: list[list[int]] = []
    for idx, _ in base_pairs:
        start = max(0, idx - expand_n)
        end = min(len(chunks) - 1, idx + expand_n)
        group = list(range(start, end + 1))
        groups.append(group)
        candidate_indices.update(group)

    # Rerank groups by query similarity. If we have embeddings, average group vectors; else use max of base scores.
    group_scores: list[tuple[float, list[int]]] = []
    q = query_emb[0]
    q = q / (np.linalg.norm(q) + 1e-12)
    for group in groups:
        if embeddings is not None and len(group) > 0:
            vecs = embeddings[group]
            # normalize rows
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms
            agg = vecs.mean(axis=0)
            score = float(agg @ q)
        else:
            # fallback: use best base score among members if present
            score = max((s for (i, s) in base_pairs if i in group), default=0.0)
        group_scores.append((score, group))

    # Sort groups by score and flatten into final unique list, preserving order
    group_scores.sort(key=lambda t: t[0], reverse=True)
    seen: set[int] = set()
    ordered_indices: list[int] = []
    for _, group in group_scores:
        for i in group:
            if i not in seen:
                seen.add(i)
                ordered_indices.append(i)

    # Cap final results to around 3*top_k (each group is ±1 expansion)
    max_results = int(os.getenv("RETRIEVAL_MAX_RESULTS", str(top_k * (2 * int(os.getenv("RETRIEVAL_EXPAND_NEIGHBORS", "1")) + 1))))
    final_indices = ordered_indices[:max_results]

    results: List[Tuple[str, Dict[str, Any], float]] = []
    # For scoring, reuse q·embedding if available, else 1.0 baseline
    for i in final_indices:
        if embeddings is not None:
            v = embeddings[i]
            v = v / (np.linalg.norm(v) + 1e-12)
            sc = float(v @ q)
        else:
            # use base score if available, else 0
            sc = next((s for (bi, s) in base_pairs if bi == i), 0.0)
        results.append((chunks[i], metadata[i], sc))

    logger.info("Retrieved results with neighbor expansion", extra={"base": len(base_pairs), "expanded": len(final_indices)})
    return results


