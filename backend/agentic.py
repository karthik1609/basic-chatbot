import logging
import os
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict

from openai import OpenAI
from agents import function_tool
import json
from typing import List
import asyncio

from .logging_setup import configure_logging
from .openai_client import get_openai_client
from .config import settings
from .agent_tools import tool_retrieve_docs as _tool_retrieve_docs, tool_sql as _tool_sql
from .agent_pipeline.stages import (
    extract_intent_slots as pipeline_extract_intent_slots,
    decide_policy as pipeline_decide_policy,
    generate_elicitation_question as pipeline_generate_elicitation_question,
    retrieve_documents as pipeline_retrieve_documents,
    generate_sql as pipeline_generate_sql,
    compose_answer as pipeline_compose_answer,
    nitpick_verify as pipeline_nitpick_verify,
    finalize_answer as pipeline_finalize_answer,
    best_snippet_for_chunk as pipeline_best_snippet,
)
from .agent_pipeline.stages import normalize_input as pipeline_normalize_input


configure_logging()
logger = logging.getLogger("agentic")

# Simple in-process conversation memory keyed by session_id
_SESSION_HISTORY: Dict[str, List[Tuple[str, str]]] = {}
_SESSION_LOCKS: Dict[str, asyncio.Lock] = {}


@dataclass
class PendingTurn:
    normalized: str
    intent: str
    slots: Dict[str, Any]
    language: str
    trace: List[Dict[str, Any]]
    requires_docs: bool
    requires_sql: bool


_SESSION_PENDING: Dict[str, PendingTurn] = {}


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
    # Offline/CI short-circuit: if no real API key, return a deterministic stub
    import os as _os
    if _os.getenv("OFFLINE_MODE", "") == "1":
        trace = [{"stage": "offline", "note": "OFFLINE_MODE active; returning stub response"}]
        answer_text = "Offline mode: agent pipeline skipped. Ingest and DB init verified."
        _append_history(session_id, "user", message)
        _append_history(session_id, "assistant", answer_text)
        return {
            "answer": answer_text,
            "tools": [],
            "session_id": session_id,
            "trace": trace,
            "citations": [],
            "decision": "PROCEED",
            "assumptions": [],
            "confidence": 0.0,
        }

    # For local-runner, avoid OpenAI telemetry by clearing OPENAI_API_KEY during this request
    _orig_openai = os.environ.get("OPENAI_API_KEY")
    _profile_id = settings.current_profile_id.get()
    if _profile_id == "local-runner":
        os.environ["OPENAI_API_KEY"] = ""

    # Ensure OpenAI client is configured (reads API key / base_url)
    _client: OpenAI = get_openai_client()
    lang = language or "en"

    def _chat_complete(system_instr: str, user_msg: str) -> str:
        try:
            client = get_openai_client()
            completion = client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {"role": "system", "content": system_instr},
                    {"role": "user", "content": user_msg},
                ],
            )
            return completion.choices[0].message.content or ""
        except Exception as _e:
            logger.error("llm_complete_error", extra={"error": str(_e)})
            return ""

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
        "  warranty_status TEXT NULL  -- indexed; allowed values: {'active','expired','cancelled','suspended'}\n"
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
        "  status TEXT NOT NULL  -- indexed; allowed values: {'open','in_review','approved','rejected','closed'}\n"
        "  description TEXT NULL\n"
        "  INDEX idx_claims_status ON warranty_claims(status)\n"
        "  INDEX idx_claims_user_car ON warranty_claims(user_id, car_id)\n"
        "\n"
        "- sales_pipeline (prospects)\n"
        "  id SERIAL PRIMARY KEY NOT NULL\n"
        "  prospect_email TEXT NULL\n"
        "  stage TEXT NOT NULL  -- indexed; allowed values: {'lead','qualified','proposal','negotiation','won','lost'}\n"
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
        "- Map 'pending' warranty claims to warranty_claims.status IN ('open','in_review').\n"
        "- Map 'not closed' claims to warranty_claims.status <> 'closed'.\n"
        "- 'Active users' means users.warranty_status = 'active'. Do NOT confuse users.warranty_status (user-level) with warranty_claims.status (claim-level).\n"
        "- Only select real columns or aggregates; do not add constant text columns.\n"
        "- There is no towing flag; to detect tow-related cases, filter description via ILIKE '%tow%' OR '%towing%' OR '%tow truck%'.\n"
        "- Prefer COUNT/aggregations and LIMIT for previews; include clear column aliases (e.g., AS num).\n"
        "- Use the defined keys for joins: users.car_id -> car_catalog.id; warranty_claims.user_id -> users.id; warranty_claims.car_id -> car_catalog.id.\n"
        "- Read-only SQL only.\n"
        "- Please be concise and direct, but not impolite.\n"
        "- Do not answer questions that cannot be answered without using the tools.\n"
        "- Do not answer questions that are not appropriate to the warranty business.\n"
        "- Do NOT reference non-existent tables; use docs for country coverage or terms/conditions.\n"
        "- Avoid scalar subqueries that return multiple columns; if you need multiple metrics, use CTEs or window functions (e.g., COUNT(DISTINCT cc.make) OVER ()).\n"
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
        "- Active users by make/model/year:\n"
        "  SELECT cc.make, cc.model, cc.year, COUNT(*) AS active_users\n"
        "  FROM users u\n"
        "  JOIN car_catalog cc ON u.car_id = cc.id\n"
        "  WHERE u.warranty_status = 'active'\n"
        "  GROUP BY cc.make, cc.model, cc.year\n"
        "  ORDER BY cc.make, cc.model, cc.year;\n"
        "- Total manufacturers with per-make users (window):\n"
        "  WITH per_make AS (\n"
        "    SELECT cc.make, COUNT(DISTINCT u.id) AS users\n"
        "    FROM users u JOIN car_catalog cc ON u.car_id = cc.id\n"
        "    WHERE u.warranty_status = 'active'\n"
        "    GROUP BY cc.make\n"
        "  )\n"
        "  SELECT make, users, COUNT(*) OVER () AS manufacturers\n"
        "  FROM per_make\n"
        "  ORDER BY make;\n"
    )

    convo = _render_history(session_id)
    convo_block = (convo + "\n\n") if convo else ""

    instructions = (
        "You are an analyst assistant. Use the available tools to retrieve PDF context and query Postgres.\n"
        + convo_block + db_schema + disambiguation + examples
    )

    async def _run_recovery_agent(stage: str, error: str, normalized_q: str, slots_obj: Dict[str, Any], schema_text: str, constraints_text: str, attempt_num: int) -> Dict[str, Any]:
        instr = (
            "You are a Recovery Supervisor Agent. Your job is to help recover from failures.\n"
            "Given the failed stage, precise error message, normalized question, slots, schema, and current constraints,\n"
            "propose a minimal, safe adjustment consistent with the schema and prior guidelines.\n"
            "Do NOT invent tables/columns. Prefer: simplify filters, avoid multi-column scalar subqueries, use CTE/window functions,\n"
            "map statuses correctly. Respond strict JSON: {\"revised_constraints_append\": string, \"revised_prompt_append\": string, \"abort\": boolean}."
        )
        prompt = (
            f"Stage: {stage}\n"
            f"Attempt: {attempt_num}\n"
            f"Error: {error}\n\n"
            f"Normalized question: {normalized_q}\n"
            f"Slots: {json.dumps(slots_obj, ensure_ascii=False)}\n\n"
            "Schema:\n" + schema_text + "\n\n" +
            "Current constraints:\n" + constraints_text + "\n"
        )
        raw = _chat_complete(instr, prompt)
        try:
            return json.loads(raw)
        except Exception:
            return {"revised_constraints_append": "", "revised_prompt_append": "", "abort": False}

    trace: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    decision: str = "PROCEED"
    assumptions: List[str] = []
    confidence: float = 0.0

    # 1) Input Normalizer
    # Modular stage: normalization via pipeline
    # If we have a pending elicitation for this session, resume by combining
    pending = _SESSION_PENDING.get(session_id or "") if session_id else None
    if pending is not None:
        # Seed trace and carry forward language unless overridden
        lang = language or pending.language or lang
        trace.extend(pending.trace)
        trace.append({"stage": "resume", "note": "Resuming after elicitation", "pending_intent": pending.intent})
        combined = f"{pending.normalized}\n\nUser clarification: {message}"
        try:
            normalized = pipeline_normalize_input(combined, lang)
        except Exception:
            normalized = combined
    else:
        try:
            normalized = pipeline_normalize_input(message, lang)
        except Exception:
            # fallback via profile-aware completion
            normalizer_instr = (
                f"Respond in language code '{lang}'. You normalize user input.\n"
                "- Fix obvious typos and normalize numbers, dates, and units.\n"
                "- Remove irrelevant fluff.\n"
                "- Mask PII: replace emails, phones with placeholders.\n"
                "Return only the cleaned text."
            )
            normalized = _chat_complete(normalizer_instr, message)
    logger.info("stage=normalize", extra={"normalized": normalized[:200]})
    trace.append({"stage": "normalize", "normalized": normalized})

    # 2) Intent & Slot Extractor
    # Modular stage: intent & slots
    intent = "other"
    slots: Dict[str, Any] = {}
    if pending is not None:
        # Prefer re-extraction but fall back to pending
        try:
            intent, slots = pipeline_extract_intent_slots(normalized, lang)
        except Exception:
            intent, slots = pending.intent, pending.slots
        # Carry forward requires flags unless intent changed materially
        requires_docs = pending.requires_docs
        requires_sql = pending.requires_sql
    else:
        try:
            intent, slots = pipeline_extract_intent_slots(normalized, lang)
        except Exception:
            pass
        # Heuristic determinations for docs/sql needs based on intent/slots
    if pending is None:
        requires_docs = intent in ("policy_question", "doc_lookup")
        requires_sql = intent in ("sql_analytics",)
    missing_slots: List[str] = []
    logger.info("stage=extract", extra={"intent": intent, "requires_docs": requires_docs, "requires_sql": requires_sql})
    trace.append({
        "stage": "extract",
        "intent": intent,
        "slots": slots,
        "missing_slots": missing_slots,
        "requires_docs": requires_docs,
        "requires_sql": requires_sql,
    })

    # 3) Uncertainty & Decision Policy
    must_ask = pipeline_decide_policy(missing_slots) == "ASK"
    ask_question = None
    if must_ask:
        ask_question = pipeline_generate_elicitation_question(lang, intent, missing_slots, slots)
        decision = "ASK"
        logger.info("stage=ask", extra={"question": ask_question})
        trace.append({"stage": "ask", "question": ask_question})
        # Persist pending turn for this session
        if session_id:
            _SESSION_PENDING[session_id] = PendingTurn(
                normalized=normalized,
                intent=intent,
                slots=slots,
                language=lang,
                trace=trace[:],
                requires_docs=requires_docs,
                requires_sql=requires_sql,
            )
        # Memory update: append both user turn and the question asked
        _append_history(session_id, "user", message)
        _append_history(session_id, "assistant", ask_question)
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

    # Prepare shared SQL guidance
    base_constraints = (
        f"Respond in language code '{lang}'. Generate a single read-only SQL query for Postgres given the task, slots, and schema.\n"
        "Constraints:\n"
        "- Use only columns/tables in the provided schema.\n"
        "- Map 'pending' to warranty_claims.status IN ('open','in_review').\n"
        "- Map 'not closed' to warranty_claims.status <> 'closed'.\n"
        "- For tow-related, filter description ILIKE '%tow%' OR '%towing%' OR '%tow truck%'.\n"
        "- Prefer COUNT/aggregations where appropriate.\n"
        "- Do NOT add constant text columns; select only real columns or aggregates.\n"
        "- 'Active users' strictly refers to users.warranty_status = 'active'.\n"
        "- Distinguish: users.warranty_status ∈ {'active','expired','cancelled','suspended'} vs warranty_claims.status ∈ {'open','in_review','approved','rejected','closed'}.\n"
        "- Do NOT reference non-existent tables; rely on documents via the docs tool for regions/countries and terms/conditions.\n"
        "- If using CTEs, each must be of the form name AS (SELECT ...). Do NOT alias a CTE to a column (e.g., 'AS user_count') without SELECT.\n"
        "- Avoid scalar subqueries that return multiple columns; if you need multiple metrics, use CTEs or window functions (e.g., COUNT(DISTINCT cc.make) OVER ()).\n"
        "- Return only raw SQL, without code fences, backticks, or commentary.\n"
    )
    intent_str = intent if isinstance(intent, str) else json.dumps(intent, ensure_ascii=False)
    slots_str = json.dumps(slots, ensure_ascii=False)
    base_prompt = (
        f"Task intent: {intent_str}\n" +
        f"Slots: {slots_str}\n" +
        "Schema:\n" + db_schema
    )

    constraints_text = base_constraints
    prompt_text = base_prompt

    last_error: Optional[str] = None
    success = False

    # ---- Document retrieval with point-aware evidence selection ----
    if requires_docs:
        try:
            raw_results = pipeline_retrieve_documents(query=normalized, top_k=10)
            # Prepare candidates
            candidates: List[Dict[str, Any]] = []
            for idx, item in enumerate(raw_results, start=1):
                candidates.append({
                    "tag": f"D{idx}",
                    "source": item.get("metadata", {}).get("source"),
                    "chunk_index": item.get("metadata", {}).get("chunk_index"),
                    "score": item.get("score"),
                    # Keep full chunk text for rich tooltips; frontend clamps via CSS
                    "text": (item.get("text") or ""),
                })
            # Build compact snippet list for the selector
            cand_lines: List[str] = []
            for c in candidates:
                score_str = f"{float(c.get('score') or 0.0):.3f}"
                cand_lines.append(f"[{c['tag']}] {c.get('source')}#{c.get('chunk_index')} (score={score_str})\n{c.get('text')}")
            cand_block = "\n\n".join(cand_lines[:12])

            # Extract distinct answer points
            points_instr = (
                f"Respond in language code '{lang}'. Identify 2-6 distinct answer points or sub-questions present in the user's request.\n"
                "Return JSON array of short point labels (strings)." 
            )
            points_raw = _chat_complete(points_instr, normalized)
            try:
                points: List[str] = json.loads(points_raw)
                if not isinstance(points, list):
                    points = []
            except Exception:
                points = []
            if not points:
                points = ["general"]

            # Evidence selector: choose up to 2 chunks per point
            selector_instr = (
                f"Respond in language code '{lang}'. You are selecting evidence chunks to support each point.\n"
                "Given points and candidate chunks with IDs, choose up to 2 chunk IDs per point that best support that point.\n"
                "Prefer higher scores, diversity of sources, and minimal overlap; avoid citing near-duplicates.\n"
                "Return strict JSON object mapping point_label -> array of chunk IDs (e.g., {\"Point A\":[\"D1\",\"D3\"]})."
            )
            selector_input = (
                "Points:\n" + json.dumps(points, ensure_ascii=False) + "\n\n" +
                "Candidates (ID, source, score, text):\n" + cand_block
            )
            selection_raw = _chat_complete(selector_instr, selector_input)
            selected_map: Dict[str, List[str]] = {}
            try:
                selected_map = json.loads(selection_raw)
            except Exception:
                # Fallback: choose top 2 overall
                selected_map = {points[0]: [candidates[0]["tag"]] if candidates else []}
                if len(candidates) > 1:
                    selected_map[points[0]].append(candidates[1]["tag"])

            selected_tags = set(tag for tags in selected_map.values() for tag in (tags or []))
            # If nothing selected, pick top 1-2 by score
            if not selected_tags and candidates:
                top_sorted = sorted(candidates, key=lambda c: float(c.get("score") or 0.0), reverse=True)[:2]
                selected_tags = {c["tag"] for c in top_sorted}

            # Build final citations and evidence only for selected chunks
            # Create reverse map tag -> point label (first match)
            tag_to_point: Dict[str, str] = {}
            for p, tags in selected_map.items():
                for t in (tags or []):
                    if t not in tag_to_point:
                        tag_to_point[t] = p

            for c in candidates:
                if c["tag"] in selected_tags:
                    point_label = tag_to_point.get(c["tag"], points[0])
                    text_full = (c.get("text") or "")
                    snippet = pipeline_best_snippet(text_full, normalized, point_label)
                    citations.append({
                        "type": "doc",
                        "tag": c["tag"],
                        "source": c.get("source"),
                        "chunk_index": c.get("chunk_index"),
                        "score": c.get("score"),
                        "text": text_full,  # keep full chunk text for tooltip display
                        "point": point_label,
                    })
                    docs_evidence.append({"tag": c["tag"], "snippet": snippet, "point": point_label})

            # Confidence from selected docs
            try:
                sel_scores = [float(c.get("score") or 0.0) for c in citations if c.get("type") == "doc"]
                if sel_scores:
                    confidence = max(confidence, sum(sel_scores) / len(sel_scores))
            except Exception:
                pass

            logger.info("stage=evidence_select", extra={"num_candidates": len(candidates), "selected_counts": {k: len(v or []) for k, v in selected_map.items()}})
            trace.append({"stage": "evidence_select", "points": points, "selected": selected_map})
        except Exception as e:
            trace.append({"stage": "retrieve_docs", "status": "error", "error": str(e)})

    # ---- Generic tools loop (docs results already computed above) ----
    for attempt in range(1, 4):
        local_sql_rows: List[Dict[str, Any]] = []
        local_sql_query: Optional[str] = None
        tool_error: Optional[str] = None

        if requires_sql:
            try:
                candidate_sql = pipeline_generate_sql(lang, intent, slots, db_schema)
                try:
                    import re
                    if re.search(r"\bwarranty_status\b", candidate_sql, flags=re.IGNORECASE):
                        if re.search(r"warranty_status\s+IN\s*\(\s*'open'\s*,\s*'in_review'\s*\)", candidate_sql, flags=re.IGNORECASE):
                            candidate_sql = re.sub(r"warranty_status\s+IN\s*\(\s*'open'\s*,\s*'in_review'\s*\)", "warranty_status = 'active'", candidate_sql, flags=re.IGNORECASE)
                        if re.search(r"warranty_status\s*=\s*'(open|in_review|approved|rejected|closed)'", candidate_sql, flags=re.IGNORECASE):
                            candidate_sql = re.sub(r"warranty_status\s*=\s*'(open|in_review|approved|rejected|closed)'", "warranty_status = 'active'", candidate_sql, flags=re.IGNORECASE)
                    candidate_sql = re.sub(r"\(\s*SELECT\s+[^\)]*?,\s*[^\)]*?\)\s+AS", "", candidate_sql)
                except Exception:
                    pass
                logger.info("stage=sql_generate", extra={"attempt": attempt})
                local_sql_rows = _tool_sql(query=candidate_sql)
                local_sql_query = candidate_sql
                citations.append({"type": "sql", "tag": "S1", "query": candidate_sql, "rows": len(local_sql_rows)})
                logger.info("stage=sql_exec", extra={"rows": len(local_sql_rows)})
                trace.append({"stage": "sql", "attempt": attempt, "query": candidate_sql, "rows": len(local_sql_rows)})
            except Exception as e:
                tool_error = f"sql_error: {e}"
                last_error = str(e)
                logger.error("stage=sql_error", extra={"attempt": attempt, "error": last_error})
                trace.append({"stage": "sql_error", "attempt": attempt, "error": last_error})

        if tool_error is None:
            sql_rows = local_sql_rows
            sql_query = local_sql_query
            success = True
            break
        else:
            rec = await _run_recovery_agent(
                stage="tools",
                error=tool_error,
                normalized_q=normalized,
                slots_obj=slots,
                schema_text=db_schema,
                constraints_text=constraints_text,
                attempt_num=attempt,
            )
            constraints_text = constraints_text + ("\n" + rec.get("revised_constraints_append", "")).rstrip()
            prompt_text = prompt_text + ("\n" + rec.get("revised_prompt_append", "")).rstrip()
            if rec.get("abort"):
                break

    if requires_docs:
        trace.append({"stage": "retrieve_docs", "status": "ok" if docs_evidence else "none"})

    # 5) Draft Answer Composer (modular)
    sql_preview = None
    if sql_rows:
        sql_preview = json.dumps(sql_rows[:5], ensure_ascii=False)[:800]
    logger.info("stage=compose", extra={"docs": len(docs_evidence), "has_sql": bool(sql_preview)})
    comp = pipeline_compose_answer(lang, normalized, slots, docs_evidence, sql_preview)
    answer_text = comp.get("answer", "")
    trace.append({"stage": "compose", "reasoning": comp.get("reasoning_bullets", [])})

    # Enforce citation hygiene: ensure each non-empty line has [D#] and/or [S#] if evidence was used
    def _ensure_citations(text: str, used_citations: List[Dict[str, Any]], need_docs: bool, need_sql: bool) -> str:
        try:
            import re as _re
            doc_tags = [c.get("tag") for c in used_citations if c.get("type") == "doc" and c.get("tag")]
            sql_tags = [c.get("tag") for c in used_citations if c.get("type") == "sql" and c.get("tag")]
            if not text:
                return text
            in_code = False
            lines = text.split("\n")
            out_lines: List[str] = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    out_lines.append(line)
                    continue
                if in_code or not stripped:
                    out_lines.append(line)
                    continue
                has_cite = bool(_re.search(r"\[(?:D|S)\d+[^\]]*\]", line))
                if not has_cite:
                    tokens: List[str] = []
                    if need_docs and doc_tags:
                        # attach up to two highest-priority doc tags
                        tokens.extend([f"[{doc_tags[0]}]"])
                        if len(doc_tags) > 1:
                            tokens.append(f"[{doc_tags[1]}]")
                    if need_sql and sql_tags:
                        tokens.append(f"[{sql_tags[0]}]")
                    if tokens:
                        line = line.rstrip() + " " + " ".join(tokens)
                out_lines.append(line)
            return "\n".join(out_lines)
        except Exception:
            return text

    if requires_docs or requires_sql:
        answer_text = _ensure_citations(answer_text, citations, requires_docs, requires_sql)

    # 5.5) Nitpicker Verifier (multi-pass, recursive) — modular
    nitpicker_rounds: List[Dict[str, Any]] = []
    threshold = 90
    for round_idx in range(1, 4):
        docs_map = {}
        for c in citations:
            if c.get("type") == "doc" and c.get("tag") and c.get("text"):
                docs_map[str(c["tag"]) ] = str(c.get("text") or "")
        logger.info("stage=nitpicker_round", extra={"round": round_idx, "docs": len(docs_map)})
        ver = pipeline_nitpick_verify(lang, normalized, answer_text, docs_map, sql_query, sql_rows)
        raw_score = ver.get("compliance_score")
        try:
            score = float(raw_score)
        except Exception:
            score = 0.0
        # Normalize to 0–100 if returned as 0–1
        if 0.0 <= score <= 1.0:
            score = score * 100.0
        nitpicker_rounds.append({"round": round_idx, "score": score, "findings": ver.get("findings", [])})
        if score >= threshold:
            break
        revised = ver.get("revised_answer")
        patches = ver.get("patches")
        # If a revised answer is provided, prefer it; else, re-compose with constraints
        if isinstance(revised, str) and revised.strip():
            answer_text = revised
        else:
            comp2 = pipeline_compose_answer(lang, normalized, slots, docs_evidence, sql_preview)
            answer_text = comp2.get("answer", answer_text)

    if nitpicker_rounds:
        trace.append({"stage": "nitpicker", "rounds": nitpicker_rounds})

    # 6) Validator/Guardrails (lightweight)
    # Basic heuristic: ensure we didn't answer without any evidence when docs/sql were required
    if (requires_docs or requires_sql) and not citations:
        answer_text = "I'm not fully confident without evidence. Could you rephrase or provide more details?"
        trace.append({"stage": "validate", "status": "low_evidence"})
    else:
        trace.append({"stage": "validate", "status": "ok"})

    # Finalize
    try:
        answer_text = pipeline_finalize_answer(answer_text)
    except Exception:
        pass

    # Update memory and return; clear pending if any
    _append_history(session_id, "user", message)
    _append_history(session_id, "assistant", answer_text)
    if session_id and session_id in _SESSION_PENDING:
        try:
            del _SESSION_PENDING[session_id]
        except KeyError:
            pass

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


