import os
import time
import json
import httpx

API = os.getenv("TEST_API_BASE", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")

headers = {"x-api-key": API_KEY} if API_KEY else {}


def _post(path: str, payload: dict | None = None) -> httpx.Response:
    return httpx.post(f"{API}{path}", json=payload or {}, headers=headers, timeout=60)


def test_ingest_and_db():
    r = _post("/api/ingest")
    assert r.status_code == 200
    data = r.json()
    assert "chunks_indexed" in data

    r = _post("/api/db/init")
    assert r.status_code == 200

    r = _post("/api/db/seed")
    assert r.status_code == 200


def test_agent_chat_happy_path():
    msg = "How many warranty claims are not closed? Group by make and year."
    r = _post("/api/agent/chat", {"message": msg, "language": "en"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    # allow either ask/answer based on pipeline decision
    assert isinstance(data.get("trace"), list) or data.get("ask")
