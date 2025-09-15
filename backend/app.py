from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class IngestResponse(BaseModel):
        chunks_indexed: int

    @app.post("/api/ingest", response_model=IngestResponse)
    def ingest() -> Any:  # noqa: ANN401
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
    def chat(req: ChatRequest) -> Any:  # noqa: ANN401
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
    def db_init() -> Any:  # noqa: ANN401
        init_schema()
        return {"status": "ok"}

    @app.post("/api/db/seed")
    def db_seed() -> Any:  # noqa: ANN401
        seed_data()
        return {"status": "ok"}

    # Agentic endpoint
    class AgentChatRequest(BaseModel):
        message: str
        language: Optional[str] = None

    class AgentChatResponse(BaseModel):
        answer: str
        tools: list[dict]

    @app.post("/api/agent/chat", response_model=AgentChatResponse)
    async def agent_chat(req: AgentChatRequest) -> Any:  # noqa: ANN401
        result = await run_agentic_chat(req.message, req.language)
        return result

    # Mount static frontend last to avoid overshadowing API routes
    from fastapi.staticfiles import StaticFiles  # local import to avoid unused when not mounting
    try:
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    except Exception:
        pass

    return app


