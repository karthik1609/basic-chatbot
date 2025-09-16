import logging
import os
from typing import Any, Dict, Optional, List, Tuple

from openai import OpenAI
from agents import Agent, Runner, function_tool

from .logging_setup import configure_logging
from .openai_client import get_openai_client
from .config import settings
from .agent_tools import tool_retrieve_docs as _tool_retrieve_docs, tool_sql as _tool_sql


configure_logging()
logger = logging.getLogger("agentic")

# Simple in-process conversation memory keyed by session_id
_SESSION_HISTORY: Dict[str, List[Tuple[str, str]]] = {}


def _append_history(session_id: Optional[str], role: str, content: str) -> None:
    if not session_id:
        return
    history = _SESSION_HISTORY.setdefault(session_id, [])
    history.append((role, content))
    # cap memory to last 20 turns
    if len(history) > 40:
        del history[: len(history) - 40]


def _render_history(session_id: Optional[str], max_turns: int = 8) -> str:
    if not session_id or session_id not in _SESSION_HISTORY:
        return ""
    history = _SESSION_HISTORY[session_id][-max_turns * 2 :]
    if not history:
        return ""
    lines = ["Conversation so far (most recent first):"]
    for role, content in reversed(history):
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"- {prefix}: {content}")
    return "\n".join(lines)


@function_tool
def retrieve_docs(query: str, top_k: int = 6) -> list[dict]:
    """Retrieve relevant PDF chunks about the user's question."""
    return _tool_retrieve_docs(query=query, top_k=top_k)


@function_tool
def sql(query: str) -> list[dict]:
    """Run a read-only SQL query against the Postgres database."""
    return _tool_sql(query=query)


async def run_agentic_chat(message: str, language: str | None = None, session_id: Optional[str] = None) -> Dict[str, Any]:
    # Ensure OpenAI client is configured (reads API key / base_url)
    _client: OpenAI = get_openai_client()
    lang = language or "en"

    # Explicit DB dictionary and guidelines (must mirror backend/db.py)
    db_schema = (
        "DB schema (PostgreSQL) with types, PK/FK, nullability, constraints, and indexes:\n"
        "- car_catalog (dictionary of cars)\n"
        "  id SERIAL PRIMARY KEY NOT NULL\n"
        "  make TEXT NOT NULL\n"
        "  model TEXT NOT NULL\n"
        "  year INTEGER NOT NULL\n"
        "  body_type TEXT NULL\n"
        "  fuel_type TEXT NULL\n"
        "  trim TEXT NULL\n"
        "  UNIQUE(make, model, year)\n"
        "\n"
        "- users (existing clients)\n"
        "  id SERIAL PRIMARY KEY NOT NULL\n"
        "  email TEXT UNIQUE NULL\n"
        "  full_name TEXT NULL\n"
        "  car_id INTEGER NULL REFERENCES car_catalog(id)\n"
        "  package_plan TEXT NULL\n"
        "  warranty_status TEXT NULL  -- indexed\n"
        "  warranty_start TIMESTAMP NULL\n"
        "  warranty_end TIMESTAMP NULL\n"
        "  last_interaction_at TIMESTAMP NULL\n"
        "  created_at TIMESTAMP NOT NULL DEFAULT NOW()\n"
        "  INDEX idx_users_warranty_status ON users(warranty_status)\n"
        "  INDEX idx_users_package_plan ON users(package_plan)\n"
        "\n"
        "- warranty_claims (claims)\n"
        "  claim_id SERIAL PRIMARY KEY NOT NULL\n"
        "  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE\n"
        "  car_id INTEGER NOT NULL REFERENCES car_catalog(id) ON DELETE RESTRICT\n"
        "  opened_at TIMESTAMP NOT NULL\n"
        "  closed_at TIMESTAMP NULL\n"
        "  status TEXT NOT NULL  -- indexed; one of {'open','in_review','approved','rejected','closed'}\n"
        "  description TEXT NULL\n"
        "  INDEX idx_claims_status ON warranty_claims(status)\n"
        "  INDEX idx_claims_user_car ON warranty_claims(user_id, car_id)\n"
        "\n"
        "- sales_pipeline (prospects)\n"
        "  id SERIAL PRIMARY KEY NOT NULL\n"
        "  prospect_email TEXT NULL\n"
        "  stage TEXT NOT NULL  -- indexed; one of {'lead','qualified','proposal','negotiation','won','lost'}\n"
        "  car_id INTEGER NULL REFERENCES car_catalog(id)\n"
        "  notes TEXT NULL\n"
        "  created_at TIMESTAMP NOT NULL DEFAULT NOW()\n"
        "  last_activity_at TIMESTAMP NULL\n"
        "  next_follow_up_at TIMESTAMP NULL\n"
        "  INDEX idx_sales_stage ON sales_pipeline(stage)\n"
    )

    disambiguation = (
        "Guidelines:\n"
        f"- Always respond in language code '{lang}'.\n"
        "- Do not invent columns/tables. Use only columns above; if unsure, ask a concise clarifying question.\n"
        "- Map 'pending' warranty claims to status IN ('open','in_review').\n"
        "- There is no towing flag; to detect tow-related cases, filter description via ILIKE '%tow%' OR '%towing%' OR '%tow truck%'.\n"
        "- Prefer COUNT/aggregations and LIMIT for previews; include clear column aliases (e.g., AS num).\n"
        "- Use the defined keys for joins: users.car_id -> car_catalog.id; warranty_claims.user_id -> users.id; warranty_claims.car_id -> car_catalog.id.\n"
        "- Read-only SQL only.\n"
        "- Please do not be verbose, just answer the question."
        "- Be brief and to the point, but not terse or impolite."
        "- Do not answer questions that cannot be answered without using the tools."
        "- Do not answer questions that are not appropriate to the warranty business."
    )

    examples = (
        "Examples:\n"
        "- Count pending towing-related claims:\n"
        "  SELECT COUNT(*) AS num_pending_tow\n"
        "  FROM warranty_claims\n"
        "  WHERE status IN ('open','in_review')\n"
        "    AND (description ILIKE '%tow%' OR description ILIKE '%towing%' OR description ILIKE '%tow truck%');\n"
        "- Open claims by make/year:\n"
        "  SELECT cc.make, cc.year, COUNT(*) AS num\n"
        "  FROM warranty_claims wc\n"
        "  JOIN users u ON wc.user_id = u.id\n"
        "  JOIN car_catalog cc ON u.car_id = cc.id\n"
        "  WHERE wc.status = 'open'\n"
        "  GROUP BY cc.make, cc.year\n"
        "  ORDER BY num DESC;\n"
    )

    convo = _render_history(session_id)
    convo_block = (convo + "\n\n") if convo else ""

    instructions = (
        "You are an analyst assistant. Use the available tools to retrieve PDF context and query Postgres.\n"
        + convo_block + db_schema + disambiguation + examples
    )

    agent = Agent(
        name="Analyst",
        instructions=instructions,
        tools=[retrieve_docs, sql],
    )

    # Do not pass raw string as session (SDK expects a session object)
    result = await Runner.run(agent, message)

    answer_text = getattr(result, "final_output", "") or ""
    tool_uses: list[dict[str, Any]] = []
    if hasattr(result, "tool_calls") and isinstance(result.tool_calls, list):
        try:
            tool_uses = [
                {
                    "name": getattr(tc, "name", None),
                    "args": getattr(tc, "arguments", None),
                }
                for tc in result.tool_calls  # type: ignore[attr-defined]
            ]
        except Exception:
            tool_uses = []

    # Update simple memory
    _append_history(session_id, "user", message)
    _append_history(session_id, "assistant", answer_text)

    return {"answer": answer_text, "tools": tool_uses, "session_id": session_id}


