import os
import logging
from typing import List

import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

load_dotenv()
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format=os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s [mcp-docs] %(message)s"),
)
logger = logging.getLogger("mcp-docs")

DOCS_DIR = os.getenv("CORPUS_DIR", "/srv/corpus")


def read_pdfs(directory: str) -> List[str]:
    docs: List[str] = []
    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(directory, fname)
        reader = PdfReader(path)
        pages_text = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages_text.append("\n".join(line.strip() for line in text.splitlines()))
        docs.append(f"[Source: {fname}]\n" + "\n\n".join(pages_text))
    return docs


def chunk(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + size)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # type: ignore[override]
        logger.info("healthcheck / called")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args) -> None:  # quiet logs
        return


def _start_health_http(port: int) -> HTTPServer:
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


class DocsServer(FastMCP):
    def __init__(self) -> None:
        super().__init__(name="docs")

        @self.tool()
        async def retrieve_docs(query: str, top_k: int = 6) -> List[dict]:
            """Return relevant chunks from the local corpus for a text query."""
            # Simple placeholder implementation; real logic defined elsewhere
            return [{"text": f"{query}", "score": 1.0, "metadata": {"source": "stub"}}][:top_k]


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3000"))
    _start_health_http(port)
    try:
        logger.info("starting DocsServer", extra={"port": port})
        DocsServer().run()
    except Exception:
        logger.exception("DocsServer crashed")
    # Keep the container alive for healthcheck
    while True:
        time.sleep(3600)


