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

from .config import settings, ensure_data_dirs
from .logging_setup import configure_logging
from .openai_client import get_openai_client


configure_logging()
logger = logging.getLogger("rag")

INDEX_FILE = os.path.join(settings.data_dir, "faiss.index")
EMBED_FILE = os.path.join(settings.data_dir, "embeddings.npy")
META_FILE = os.path.join(settings.data_dir, "meta.json")


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


def _chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    if not text:
        return []
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def _embed_texts(texts: List[str]) -> np.ndarray:
    client = get_openai_client()
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    vectors = [np.array(item.embedding, dtype=np.float32) for item in response.data]
    return np.vstack(vectors)


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
        chunks = _chunk_text(content)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                "source": filename,
                "chunk_index": idx,
            })

    logger.info("Chunking complete", extra={"num_chunks": len(all_chunks)})
    if not all_chunks:
        # Create an empty artifacts set to avoid runtime errors
        if _FAISS_AVAILABLE:
            dim = 3072  # text-embedding-3-large dimension
            index = faiss.IndexFlatIP(dim)
            faiss.write_index(index, INDEX_FILE)
        else:
            np.save(EMBED_FILE, np.zeros((0, 0), dtype=np.float32))
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump({"chunks": [], "metadata": []}, f)
        logger.warning("No chunks produced; created empty index")
        return {"chunks": 0}

    embeddings = _embed_texts(all_chunks)
    logger.info("Embeddings created", extra={"shape": list(embeddings.shape)})

    # Persist metadata
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks, "metadata": chunk_metadata}, f)

    if _FAISS_AVAILABLE:
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, INDEX_FILE)
        logger.info("FAISS index written", extra={"dim": dim, "num": len(all_chunks)})
    else:
        # Save embeddings for numpy-based retrieval
        # Normalize rows for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        np.save(EMBED_FILE, embeddings)
        logger.info("Embeddings saved for numpy retrieval", extra={"num": len(all_chunks)})

    return {"chunks": len(all_chunks)}


def _load_index_and_meta():  # type: ignore[no-untyped-def]
    if not os.path.exists(META_FILE):
        raise FileNotFoundError("Index metadata not found; run /api/ingest first")
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    chunks = meta.get("chunks", [])
    metadata = meta.get("metadata", [])

    if _FAISS_AVAILABLE:
        if not os.path.exists(INDEX_FILE):
            raise FileNotFoundError("FAISS index not found; run /api/ingest first")
        index = faiss.read_index(INDEX_FILE)
        logger.info("Loaded FAISS index", extra={"num_chunks": len(chunks)})
        return ("faiss", index, chunks, metadata)
    else:
        if not os.path.exists(EMBED_FILE):
            raise FileNotFoundError("Embeddings file not found; run /api/ingest first")
        embeddings = np.load(EMBED_FILE)
        logger.info("Loaded numpy embeddings", extra={"num_chunks": len(chunks)})
        return ("numpy", embeddings, chunks, metadata)


def retrieve_relevant_chunks(query: str, top_k: int | None = None) -> List[Tuple[str, Dict[str, Any], float]]:
    if top_k is None:
        top_k = settings.retrieval_top_k
    backend, index_or_embeds, chunks, metadata = _load_index_and_meta()
    query_emb = _embed_texts([query]).astype(np.float32)

    results: List[Tuple[str, Dict[str, Any], float]] = []
    if backend == "faiss":
        # Normalize
        faiss.normalize_L2(query_emb)
        scores, idxs = index_or_embeds.search(query_emb, top_k)
        logger.info("FAISS search", extra={"top_k": top_k})
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((chunks[idx], metadata[idx], float(score)))
    else:
        # Numpy cosine similarity
        doc_embeds = index_or_embeds  # normalized
        q = query_emb[0]
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = (doc_embeds @ q).astype(np.float32)
        if sims.size == 0:
            return []
        top_idx = np.argsort(-sims)[:top_k]
        logger.info("NumPy search", extra={"top_k": top_k})
        for idx in top_idx:
            results.append((chunks[int(idx)], metadata[int(idx)], float(sims[int(idx)])))
    logger.info("Retrieved results", extra={"returned": len(results)})
    return results


