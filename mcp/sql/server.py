import os
import logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format=os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s [mcp-sql] %(message)s"),
)
logger = logging.getLogger("mcp-sql")
from typing import List, Dict, Any

from mcp.server.fastmcp import FastMCP
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import time


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


class SQLServer(FastMCP):
    def __init__(self) -> None:
        super().__init__(name="sql")

        @self.tool()
        async def sql(query: str) -> List[Dict[str, Any]]:
            """Run a read-only SELECT query (stub)."""
            if not query.strip().lower().startswith("select"):
                return [{"error": "only SELECT allowed"}]
            return [{"ok": True}]


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3000"))
    _start_health_http(port)
    try:
        logger.info("starting SQLServer", extra={"port": port})
        SQLServer().run()
    except Exception:
        logger.exception("SQLServer crashed")
    # Keep the container alive for healthcheck
    while True:
        time.sleep(3600)


