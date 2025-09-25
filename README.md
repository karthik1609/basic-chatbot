## PDF RAG + Agentic Chatbot (FastAPI + Next.js + OpenAI Agents)

### What this repo provides

- RAG over PDFs in `docs/` using FAISS and `text-embedding-3-large`
- Agentic chat pipeline (normalize → extract → decide → retrieve/tools → compose → verify → finalize)
- SQL tool over Postgres with strict guidance and sanitization
- Nitpicker verifier: multi-pass factuality/citation checker with recursive improvements
- Next.js (shadcn/ui + Radix) frontend with ChatGPT-like UI, streaming-ready, in-bubble citation tooltips
- Docker Compose environment: Postgres, FastAPI backend, Next.js frontend (optional MCP servers)

---

### Quick start (Docker Compose)

1) Prereqs: Docker Desktop 4.29+, Node 20+, Make (optional)
2) Configure environment

- Create `.env` in repo root (example):

```
OPENAI_API_KEY=sk-...
# Profiles
DEFAULT_MODEL_PROFILE=local-runner

# Exact tokenizer hook (GGUF via llama.cpp)
TOKEN_LEN_HOOK=backend.token_len_hook:token_len
# Path inside container to the GGUF (provided by ./models volume)
LLAMA_GGUF_PATH=/models/nomic-embed-text-v1.5-gguf.gguf

# Embedding/Chunking
EMBEDDING_MAX_TOKENS=2048
EMBEDDING_PROMPT_OVERHEAD=0
CHUNK_TOKEN_BUDGET=1024
EMBEDDING_BATCH_SIZE=16
RETRIEVAL_EXPAND_NEIGHBORS=1
# Leave blank for local runner (not supported)
EMBEDDINGS_ENCODING_FORMAT=
```

3) Place model files

- Create a local `./models` directory and place the GGUF tokenizer file used by the embedding model there, e.g. `nomic-embed-text-v1.5-gguf.gguf`. The compose file mounts `./models` into the app container at `/models`.

4) Build and run

```
COMPOSE_BAKE=true docker compose up -d --build --force-recreate
```

5) Open the app

- Frontend (Next.js): http://localhost:3000
- Backend (FastAPI): http://localhost:8000

6) From the UI

- Ingest PDFs: Ingest → indexes `docs/*.pdf` into FAISS
- Init DB: creates schema
- Seed DB: inserts demo rows
- Chat: ask questions; use citations on hover in messages

Notes

- In Docker, the frontend talks to FastAPI using `NEXT_PUBLIC_FASTAPI_EXTERNAL_BASE` baked in the image.
- For macOS Docker, the backend origin inside the browser defaults to `http://host.docker.internal:8000`.

---

### Local development

Backend

```
uv python install 3.13
uv sync --all-extras
uv run --python 3.13 python main.py --host 0.0.0.0 --port 8000
```

Frontend

```
cd frontend-react
npm ci
npm run dev
```

Open http://localhost:3000 and ensure `NEXT_PUBLIC_FASTAPI_EXTERNAL_BASE=http://localhost:8000` for local dev.

---

### Endpoints (FastAPI)

- POST `/api/ingest` → { chunks_indexed }
- POST `/api/db/init` → { status }
- POST `/api/db/seed` → { status }
- POST `/api/agent/chat` → {
  answer, trace, citations, session_id, decision, assumptions, confidence
  }

Citations

- type `doc`: { tag, source, chunk_index, score, text }
- type `sql`: { tag, query, rows }

---

### Frontend UX highlights

- Messages render as Markdown; `[D1]/[S1]` tokens become hover-tooltips with full text/SQL
- Per-message citations: each assistant message carries its own evidence; New Chat resets state
- Dark theme; chat bubbles left/right; Shift+Enter = newline, Enter = send

---

### Data & DB

- PDFs in `docs/` are chunked using a semantic-ish recursive chunker with heading awareness
- FAISS index and metadata are stored under `data/` (gitignored)
- Postgres tables (abridged): `car_catalog`, `users`, `warranty_claims`, `sales_pipeline`
  - Proper PK/FK and helpful indexes (see `backend/agentic.py` schema block)

---

### Mermaid diagrams

#### C4 Level 1 — System Context

```mermaid
C4Context
    title Chatbot System Context
    Person(user, "End User")
    System_Boundary(sys, "Chatbot System") {
      System(frontend, "Next.js Frontend", "React + shadcn/ui")
      System(backend, "FastAPI Backend", "Python")
    }
    System_Ext(openai, "OpenAI API", "Models & Embeddings")
    System_Ext(pg, "Postgres", "Operational DB")
    System_Ext(storage, "PDFs + FAISS", "RAG Index")

    Rel(user, frontend, "Chat over HTTPS")
    Rel(frontend, backend, "REST /api/*")
    Rel(backend, openai, "Completions/Responses + Embeddings")
    Rel(backend, pg, "SQL (read-only)")
    Rel(backend, storage, "Ingest/Retrieve chunks")
```


#### C4 Level 4 — Code-Level (Agent Stages)

```mermaid
flowchart TD
    A[Input Normalizer] --> B[Intent & Slots Extractor]
    B --> C{Decision: ASK / ASSUME / PROCEED}
    C -- ASK --> D[Elicitation Question]
    C -- PROCEED --> E[Context Integrator]
    E --> F[Retrieval: Docs]
    E --> G[Tool: SQL]
    F --> H["Composer: Markdown, per-claim citations"]
    G --> H["Composer: Markdown, per-claim citations"]
    H --> I["Nitpicker Verifier: multi-pass, recursive"]
    I -->|score >= 90| J[Finalizer]
    I -->|score < 90| H
```

#### Sequence — Typical doc+SQL turn

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant FE as Next.js Frontend
    participant BE as FastAPI Agent
    participant RAG as FAISS Store
    participant DB as Postgres
    participant OA as OpenAI API

    U->>FE: Type message, Send
    FE->>BE: POST /api/agent/chat { message, session_id }
    BE->>OA: Normalize, Extract (text-only agents)
    alt requires_docs
      BE->>RAG: retrieve_chunks(query)
      RAG-->>BE: top-k chunks
    end
    alt requires_sql
      BE->>OA: generate SQL (constrained prompt)
      BE->>DB: exec SQL (sanitized)
      DB-->>BE: rows
    end
    BE->>OA: Compose (Markdown, one citation per claim)
    loop ≤3 rounds
      BE->>OA: Nitpicker (multi-pass verification)
      alt score < 90
        OA-->>BE: patches / revised answer
        BE->>OA: Re-compose with patches
      else score ≥ 90
        OA-->>BE: pass
      end
    end
    BE-->>FE: { answer, citations, trace, decision, assumptions, confidence }
    FE-->>U: Render bubbles + tooltips
```

#### Sequence — Tool failure with recovery

```mermaid
sequenceDiagram
    participant BE as FastAPI Agent
    participant OA as OpenAI API
    participant DB as Postgres

    BE->>OA: Generate SQL
    BE->>DB: Execute
    DB-->>BE: Error
    BE->>OA: Recovery Supervisor (stage, error, schema, constraints)
    OA-->>BE: Revised constraints/prompt
    BE->>OA: Regenerate SQL
    BE->>DB: Execute → rows
```

---

### Implementation details

- Citations: backend returns full text for doc citations; front-end renders `[Dx]/[Sx]` tokens as hover-tooltips per message
- SQL sanitation: code-fence stripping and percent escaping for psycopg
- Token limits: embeddings batched, long chunks truncated before embed; doc tool still returns full text for UX
- Session memory: short in-process history keyed by `session_id`

### Observability (optional)
- OpenTelemetry hooks are wired. To export traces, set:
  - `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces`
  - `OTEL_SERVICE_NAME=chatbot-backend`
  - Run a local collector (e.g., `otel/opentelemetry-collector` Docker image) and view in Jaeger/Tempo.

---

### Troubleshooting

- Frontend build warnings (unused vars in API routes) are safe; strict `any` is removed in components
- If frontend can’t reach backend in Docker, ensure `NEXT_PUBLIC_FASTAPI_EXTERNAL_BASE` resolves to FastAPI from the browser (macOS: `http://host.docker.internal:8000`)
- If embeddings fail with context length:
  - Ensure TOKEN_LEN_HOOK and LLAMA_GGUF_PATH are set; `./models` is mounted; logs show token stats
  - The chunker is token-budgeted (no truncation). Adjust CHUNK_TOKEN_BUDGET if needed

---

