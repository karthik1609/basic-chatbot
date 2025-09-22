from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os

from .config import settings
from .rag import build_or_update_index, retrieve_relevant_chunks
from .openai_client import get_openai_client
from .logging_setup import configure_logging
import logging
from .db import init_schema, seed_data
from .agentic import run_agentic_chat


def create_app() -> FastAPI:
    configure_logging()
    logger = logging.getLogger("app")
    app = FastAPI(title="PDF RAG Chatbot")

    # Optional OpenTelemetry setup (no-op if OTLP not reachable)
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        from .db import get_engine

        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "chatbot-backend")})
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        if otlp_endpoint:
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        FastAPIInstrumentor.instrument_app(app)
        try:
            SQLAlchemyInstrumentor().instrument(engine=get_engine())
        except Exception:
            pass
    except Exception:
        pass

    # CORS: allow only typical browser origins; can be overridden via env ORIGINS
    _origins_env = os.getenv("ALLOWED_ORIGINS")
    if _origins_env:
        allowed_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
    else:
        # Safer defaults for dev instead of wildcard
        allowed_origins = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["content-type", "authorization", "x-request-id", "x-api-key"],
    )

    class IngestResponse(BaseModel):
        chunks_indexed: int

    def _require_api_key(x_api_key: Optional[str]) -> None:
        expected = os.getenv("API_KEY")
        if expected and x_api_key != expected:
            raise HTTPException(status_code=401, detail="invalid api key")

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        req_id = request.headers.get("x-request-id") or request.headers.get("x-correlation-id")
        if not req_id:
            import uuid
            req_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["x-request-id"] = req_id
        return response

    @app.post("/api/ingest", response_model=IngestResponse)
    def ingest(x_api_key: Optional[str] = Header(default=None)) -> Any:  # noqa: ANN401
        _require_api_key(x_api_key)
        logger.info("/api/ingest called")
        result = build_or_update_index()
        logger.info("/api/ingest completed", extra={"chunks_indexed": int(result.get("chunks", 0))})
        return {"chunks_indexed": int(result.get("chunks", 0))}

    class ChatRequest(BaseModel):
        message: str
        language: Optional[str] = None  # e.g., "en", "es", "fr"
        top_k: Optional[int] = None

    class ContextItem(BaseModel):
        text: str
        metadata: Dict[str, Any]
        score: float

    class ChatResponse(BaseModel):
        answer: str
        context: List[ContextItem]

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, x_api_key: Optional[str] = Header(default=None)) -> Any:  # noqa: ANN401
        _require_api_key(x_api_key)
        # Default language
        language = req.language or "en"
        logger.info("/api/chat called", extra={"language": language})

        try:
            retrieved = retrieve_relevant_chunks(req.message, top_k=req.top_k)
        except FileNotFoundError as exc:
            logger.error("Index not found on chat", exc_info=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Filter by similarity threshold
        filtered = [(t, m, s) for (t, m, s) in retrieved if s >= settings.min_context_similarity]

        support_placeholder = "Please contact Course Support Desk at support@example.edu for assistance."

        if not filtered:
            polite_refusal_prompt = (
                f"Respond in language code '{language}'. "
                "Politely explain that the question appears to be out of scope of the provided course PDFs. "
                f"Ask the user to seek help from support: {support_placeholder}"
            )
            client = get_openai_client()
            completion = client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {"role": "system", "content": polite_refusal_prompt},
                    {"role": "user", "content": req.message},
                ],
            )
            text = completion.choices[0].message.content or ""
            logger.info("/api/chat responded with refusal")
            return {
                "answer": text,
                "context": [],
            }

        # Build context string for RAG answer
        context_snippets = [f"[Source: {m['source']} | Score: {s:.3f}]\n{t}" for (t, m, s) in filtered]
        context_text = "\n\n".join(context_snippets)

        system_prompt = (
            f"You are a helpful assistant. Prefer using provided PDF context to ground answers. "
            f"If context is insufficient, ask a brief clarifying question or suggest using Agent mode for database or external info. "
            f"If clearly out-of-scope, state it and refer to {support_placeholder}. "
            f"Always respond in language code '{language}'."
        )
        user_prompt = (
            "Question: " + req.message + "\n\n" +
            "Context from PDFs:\n" + context_text + "\n\n" +
            "When answering, cite only information from the context. If insufficient, ask a concise follow-up or suggest Agent mode."
        )

        client = get_openai_client()
        completion = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer_text = completion.choices[0].message.content or ""
        logger.info("/api/chat responded", extra={"context_used": len(filtered)})

        return {
            "answer": answer_text,
            "context": [
                {"text": t, "metadata": m, "score": s} for (t, m, s) in filtered
            ],
        }

    # DB admin endpoints
    @app.post("/api/db/init")
    def db_init(x_api_key: Optional[str] = Header(default=None)) -> Any:  # noqa: ANN401
        _require_api_key(x_api_key)
        # Prefer Alembic upgrade head if available
        try:
            import subprocess, sys
            env = dict(os.environ)
            env["DB_URL"] = env.get("DATABASE_URL", "postgresql+psycopg://chatbot:chatbot@localhost:5432/chatbot")
            subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True, env=env, cwd=os.path.dirname(os.path.dirname(__file__)))
            return {"status": "ok", "via": "alembic"}
        except Exception:
            init_schema()
            return {"status": "ok", "via": "raw_sql"}

    @app.post("/api/db/seed")
    def db_seed(x_api_key: Optional[str] = Header(default=None)) -> Any:  # noqa: ANN401
        _require_api_key(x_api_key)
        seed_data()
        return {"status": "ok"}

    # Agentic endpoint
    class AgentChatRequest(BaseModel):
        message: str
        language: Optional[str] = None
        session_id: Optional[str] = None

    class AgentChatResponse(BaseModel):
        answer: str
        tools: list[dict]
        session_id: Optional[str] = None
        trace: Optional[list[dict]] = None
        citations: Optional[list[dict]] = None
        ask: Optional[str] = None

    @app.post("/api/agent/chat", response_model=AgentChatResponse)
    async def agent_chat(req: AgentChatRequest, x_api_key: Optional[str] = Header(default=None)) -> Any:  # noqa: ANN401
        _require_api_key(x_api_key)
        result = await run_agentic_chat(req.message, req.language, req.session_id)
        return result

    # Mount static frontend last to avoid overshadowing API routes
    from fastapi.staticfiles import StaticFiles  # local import to avoid unused when not mounting
    try:
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    except Exception:
        pass

    return app


