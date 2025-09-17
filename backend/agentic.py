import logging
import os
from typing import Any, Dict, Optional, List, Tuple

from openai import OpenAI
from agents import Agent, Runner, function_tool
import json
from typing import List
import asyncio

from .logging_setup import configure_logging
from .openai_client import get_openai_client
from .config import settings
from .agent_tools import tool_retrieve_docs as _tool_retrieve_docs, tool_sql as _tool_sql


configure_logging()
logger = logging.getLogger("agentic")

# Simple in-process conversation memory keyed by session_id
_SESSION_HISTORY: Dict[str, List[Tuple[str, str]]] = {}
_SESSION_LOCKS: Dict[str, asyncio.Lock] = {}


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
    # Per-session gating to avoid overlapping turns
    lock: Optional[asyncio.Lock] = None
    if session_id:
        lock = _SESSION_LOCKS.setdefault(session_id, asyncio.Lock())
    if lock is not None:
        async with lock:
            return await _run_agentic_chat_inner(message, language, session_id)
    else:
        return await _run_agentic_chat_inner(message, language, session_id)


async def _run_agentic_chat_inner(message: str, language: str | None = None, session_id: Optional[str] = None) -> Dict[str, Any]:
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
        "- Please be concise and direct, but not impolite.\n"
        "- Do not answer questions that cannot be answered without using the tools.\n"
        "- Do not answer questions that are not appropriate to the warranty business.\n"
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

    # ---- Mini-agent runner for text-only steps ----
    async def _run_text_agent(name: str, instr: str, user_msg: str) -> str:
        mini = Agent(name=name, instructions=instr, tools=[])
        res = await Runner.run(mini, user_msg)
        return getattr(res, "final_output", "") or ""

    trace: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    decision: str = "PROCEED"
    assumptions: List[str] = []
    confidence: float = 0.0

    # 1) Input Normalizer
    normalizer_instr = (
        f"Respond in language code '{lang}'. You normalize user input.\n"
        "- Fix obvious typos and normalize numbers, dates, and units.\n"
        "- Remove irrelevant fluff.\n"
        "- Mask PII: replace emails, phones with placeholders.\n"
        "Return only the cleaned text."
    )
    normalized = await _run_text_agent("InputNormalizer", normalizer_instr, message)
    trace.append({"stage": "normalize", "normalized": normalized})

    # 2) Intent & Slot Extractor
    extractor_instr = (
        f"Respond in language code '{lang}'. Extract task intent and slots as compact JSON.\n"
        "- intents: one of ['policy_question','sql_analytics','doc_lookup','small_talk','other']\n"
        "- slots: key/value pairs relevant to the task\n"
        "- missing_slots: critical slots not provided\n"
        "- requires_docs: boolean\n"
        "- requires_sql: boolean\n"
        "Return JSON only."
    )
    extracted_raw = await _run_text_agent("IntentExtractor", extractor_instr, normalized)
    intent = "other"
    slots: Dict[str, Any] = {}
    missing_slots: List[str] = []
    requires_docs = False
    requires_sql = False
    try:
        j = json.loads(extracted_raw)
        intent = j.get("intent", intent)
        slots = j.get("slots", {}) or {}
        missing_slots = list(j.get("missing_slots", []) or [])
        requires_docs = bool(j.get("requires_docs", False))
        requires_sql = bool(j.get("requires_sql", False))
    except Exception:
        pass
    trace.append({
        "stage": "extract",
        "intent": intent,
        "slots": slots,
        "missing_slots": missing_slots,
        "requires_docs": requires_docs,
        "requires_sql": requires_sql,
    })

    # 3) Uncertainty & Decision Policy
    must_ask = len(missing_slots) > 0
    ask_question = None
    if must_ask:
        ask_instr = (
            f"Respond in language code '{lang}'. You generate one concise elicitation question to collect: {missing_slots}.\n"
            "- Batch into one question.\n"
            "- Offer options if suitable.\n"
            "- Avoid yes/no unless appropriate. Return only the question."
        )
        ask_question = await _run_text_agent("Elicitation", ask_instr, normalized)
        decision = "ASK"
        trace.append({"stage": "ask", "question": ask_question})
        # Memory update and return early awaiting user reply
        _append_history(session_id, "user", message)
        return {
            "ask": ask_question,
            "trace": trace,
            "citations": citations,
            "tools": [],
            "session_id": session_id,
            "answer": "",
            "decision": decision,
            "assumptions": assumptions,
            "confidence": confidence,
        }

    # 4) Retrieval / Tools
    docs_evidence: List[Dict[str, Any]] = []
    sql_rows: List[Dict[str, Any]] = []
    sql_query: Optional[str] = None

    if requires_docs:
        docs_results = _tool_retrieve_docs(query=normalized, top_k=6)
        for idx, item in enumerate(docs_results, start=1):
            citation = {
                "type": "doc",
                "tag": f"D{idx}",
                "source": item.get("metadata", {}).get("source"),
                "chunk_index": item.get("metadata", {}).get("chunk_index"),
                "score": item.get("score"),
            }
            citations.append(citation)
            snippet = (item.get("text") or "")[:600]
            docs_evidence.append({"tag": citation["tag"], "snippet": snippet})
        # crude confidence from doc scores
        if docs_results:
            try:
                scores = [float(item.get("score") or 0.0) for item in docs_results]
                confidence = max(confidence, sum(scores) / max(1, len(scores)))
            except Exception:
                pass
        trace.append({"stage": "retrieve_docs", "num": len(docs_results)})

    if requires_sql:
        sql_gen_instr = (
            f"Respond in language code '{lang}'. Generate a single read-only SQL query for Postgres given the task, slots, and schema.\n"
            "Constraints:\n"
            "- Use only columns/tables in the provided schema.\n"
            "- Map 'pending' to status IN ('open','in_review').\n"
            "- For tow-related, filter description ILIKE '%tow%' OR '%towing%' OR '%tow truck%'.\n"
            "- Prefer COUNT/aggregations where appropriate.\n"
            "- Return only raw SQL, without code fences, backticks, or commentary."
        )
        sql_prompt = (
            "Task intent: " + intent + "\n" +
            "Slots: " + json.dumps(slots, ensure_ascii=False) + "\n" +
            "Schema:\n" + db_schema
        )
        sql_query = await _run_text_agent("SQLGenerator", sql_gen_instr, sql_prompt)
        # Execute
        sql_rows = _tool_sql(query=sql_query)
        citations.append({"type": "sql", "tag": "S1", "query": sql_query, "rows": len(sql_rows)})
        trace.append({"stage": "sql", "query": sql_query, "rows": len(sql_rows)})

    # 5) Draft Answer Composer
    composer_instr = (
        f"Respond in language code '{lang}'. Compose a concise answer.\n"
        "- Use provided evidence.\n"
        "- Include inline citations like [D1], [S1] where used.\n"
        "- Also produce a short reasoning outline (3-5 bullets).\n"
        "Return as JSON with keys: answer, reasoning_bullets (array)."
    )
    evidence_block_lines: List[str] = []
    for d in docs_evidence:
        evidence_block_lines.append(f"[{d['tag']}] {d['snippet']}")
    if sql_rows:
        preview = sql_rows[:5]
        evidence_block_lines.append(f"[S1] SQL preview: {json.dumps(preview, ensure_ascii=False)[:800]}")
    evidence_block = "\n\n".join(evidence_block_lines)
    compose_input = (
        "Normalized question: " + normalized + "\n\n" +
        "Slots: " + json.dumps(slots, ensure_ascii=False) + "\n\n" +
        ("Evidence:\n" + evidence_block if evidence_block else "")
    )
    composed_raw = await _run_text_agent("Composer", composer_instr, compose_input)
    answer_text = composed_raw
    reasoning_bullets: List[str] = []
    try:
        j = json.loads(composed_raw)
        answer_text = j.get("answer", answer_text)
        reasoning_bullets = list(j.get("reasoning_bullets", []) or [])
    except Exception:
        pass
    trace.append({"stage": "compose", "reasoning": reasoning_bullets})

    # 6) Validator/Guardrails (lightweight)
    # Basic heuristic: ensure we didn't answer without any evidence when docs/sql were required
    if (requires_docs or requires_sql) and not citations:
        answer_text = "I'm not fully confident without evidence. Could you rephrase or provide more details?"
        trace.append({"stage": "validate", "status": "low_evidence"})
    else:
        trace.append({"stage": "validate", "status": "ok"})

    # Update memory and return
    _append_history(session_id, "user", message)
    _append_history(session_id, "assistant", answer_text)

    return {
        "answer": answer_text,
        "tools": [],
        "session_id": session_id,
        "trace": trace,
        "citations": citations,
        "decision": decision,
        "assumptions": assumptions,
        "confidence": confidence,
    }


