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

- Create `.env` in repo root:

```
OPENAI_API_KEY=sk-...
# Optional overrides
OPENAI_BASE_URL=
CHAT_MODEL=gpt-5
EMBEDDING_MODEL=text-embedding-3-large
```

3) Build and run

```
COMPOSE_BAKE=true docker compose up -d --build --force-recreate
```

4) Open the app

- Frontend (Next.js): http://localhost:3000
- Backend (FastAPI): http://localhost:8000

5) From the UI

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

#### C4 Level 2 — Container

```mermaid
C4Container
    title Containers
    Person(user, "End User")
    System_Boundary(prod, "Docker Compose") {
      Container(ui, "frontend-react", "Next.js 15", "Chat UI, tooltips, actions")
      Container(api, "app", "FastAPI", "Agent orchestration, RAG, SQL")
      ContainerDb(db, "postgres", "PostgreSQL", "Claims & users data")
      Container(store, "faiss-store", "Filesystem", "PDF chunks + embeddings")
    }
    System_Ext(openai, "OpenAI API")

    Rel(user, ui, "HTTPS")
    Rel(ui, api, "HTTP JSON")
    Rel(api, db, "psycopg/SQLAlchemy")
    Rel(api, store, "FAISS IO")
    Rel(api, openai, "Responses + Embeddings")
```

#### C4 Level 3 — Components (Backend)

```mermaid
C4Component
    title FastAPI Backend Components
    Container(api, "FastAPI")
    Component(app, "app.py", "API routes: ingest, db, agent/chat")
    Component(agentic, "agentic.py", "Agent pipeline + tools + verifier")
    Component(tools, "agent_tools.py", "retrieve_docs, sql")
    Component(rag, "rag.py", "Chunking, embeddings, FAISS")
    Component(db, "db.py", "Schema + seed")

    Rel(app, agentic, "run_agentic_chat()")
    Rel(agentic, tools, "tool calls")
    Rel(agentic, rag, "ingest/retrieve")
    Rel(agentic, db, "SQL via tools")
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

---

### Troubleshooting

- Frontend build warnings (unused vars in API routes) are safe; strict `any` is removed in components
- If frontend can’t reach backend in Docker, ensure `NEXT_PUBLIC_FASTAPI_EXTERNAL_BASE` resolves to FastAPI from the browser (macOS: `http://host.docker.internal:8000`)
- If embeddings fail with context length, confirm PDFs aren’t gigantic and re-run Ingest

---

### License

MIT
