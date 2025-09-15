## PDF RAG Chatbot (OpenAI GPT-5)

### Features
- RAG over PDFs in `docs/` using FAISS and `text-embedding-3-large`
- Chat via `gpt-5`, answers strictly from PDFs; otherwise polite refusal with support placeholder
- Language-aware responses; default English if not specified
- FastAPI backend with `/ingest` and `/chat` endpoints
- Minimal frontend with language selector and chat UI
 - Agentic mode using tool-calling over RAG and Postgres

### Requirements
- Python 3.13
- `.env` with:
  - `OPENAI_API_KEY=...`
  - Optional: `OPENAI_BASE_URL=...` (for proxies)
  - Optional: `CHAT_MODEL=gpt-5`, `EMBEDDING_MODEL=text-embedding-3-large`

### Install
```
uv sync
```

### Run
```
python main.py
```
Then open `http://localhost:8000/`.

### Postgres (Docker)
```
docker compose up -d postgres
export DATABASE_URL=postgresql+psycopg://chatbot:chatbot@localhost:5432/chatbot
```
Init and seed from the UI buttons or via API:
```
curl -X POST http://localhost:8000/api/db/init
curl -X POST http://localhost:8000/api/db/seed
```

### Usage
1. Place PDFs in `docs/`. Two example files exist.
2. Click "Rebuild Index" to index PDFs.
3. Ask questions; set language from the dropdown.
4. Use Mode: RAG or Agentic. Agentic can query the DB and retrieve docs.

### Notes on approach: openai-agents vs RAG
- For answering strictly from local PDFs, classic RAG is the most reliable and controllable.
- `openai-agents` can orchestrate tools, but here it adds complexity without improving grounding.
- This project uses RAG for retrieval and a simple chat endpoint for generation.

### Project structure
- `backend/`: FastAPI app, RAG, OpenAI client, config
  - `agentic.py`: agent loop with tool-calling for docs and SQL
  - `agent_tools.py`: RAG retrieval and SQL tools
  - `db.py`: schema and seed helpers
- `frontend/`: Static site (HTML/CSS/JS)
- `docs/`: PDFs to index
- `data/`: FAISS index and metadata (gitignored)

