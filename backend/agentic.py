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

    # ---- Mini-agent runner for text-only steps ----
    async def _run_text_agent(name: str, instr: str, user_msg: str) -> str:
        mini = Agent(name=name, instructions=instr, tools=[])
        res = await Runner.run(mini, user_msg)
        return getattr(res, "final_output", "") or ""

    async def _run_recovery_agent(stage: str, error: str, normalized_q: str, slots_obj: Dict[str, Any], schema_text: str, constraints_text: str, attempt_num: int) -> Dict[str, Any]:
        instr = (
            "You are a Recovery Supervisor Agent. Your job is to help other agents recover from failures.\n"
            "Given the failed stage, the precise error message, the normalized question, slots, schema, and the current constraints,\n"
            "propose a minimal, safe adjustment that is consistent with the schema and prior guidelines.\n"
            "Do NOT invent tables or columns. Prefer: simplifying filters, avoiding multi-column scalar subqueries, using CTE/window functions,\n"
            "and mapping statuses correctly. Suggest constraints or prompt additions only. Respond in strict JSON with keys: \n"
            "{\"revised_constraints_append\": string, \"revised_prompt_append\": string, \"abort\": boolean}."
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
        raw = await _run_text_agent("RecoverySupervisor", instr, prompt)
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
            raw_results = _tool_retrieve_docs(query=normalized, top_k=10)
            # Prepare candidates
            candidates: List[Dict[str, Any]] = []
            for idx, item in enumerate(raw_results, start=1):
                candidates.append({
                    "tag": f"D{idx}",
                    "source": item.get("metadata", {}).get("source"),
                    "chunk_index": item.get("metadata", {}).get("chunk_index"),
                    "score": item.get("score"),
                    "text": (item.get("text") or "")[:1200],
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
            points_raw = await _run_text_agent("PointExtractor", points_instr, normalized)
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
            selection_raw = await _run_text_agent("EvidenceSelector", selector_instr, selector_input)
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

            def _best_sentences(text: str, point_label: str) -> str:
                import re
                # naive sentence split
                sents = re.split(r"(?<=[\.!?])\s+", text.strip())
                if not sents:
                    return text[:600]
                # keywords from normalized question + point label
                kw_src = (normalized + " " + (point_label or "")).lower()
                kws = [w for w in re.findall(r"[a-zA-Z]{3,}", kw_src) if len(w) > 2]
                if not kws:
                    return " ".join(sents[:3])[:600]
                def score_sent(s: str) -> int:
                    s_low = s.lower()
                    return sum(1 for k in kws if k in s_low)
                ranked = sorted(sents, key=score_sent, reverse=True)
                snippet = " ".join(ranked[:3])
                if len(snippet) < 200 and len(ranked) > 3:
                    snippet = " ".join(ranked[:5])
                return snippet[:600]

            for c in candidates:
                if c["tag"] in selected_tags:
                    point_label = tag_to_point.get(c["tag"], points[0])
                    text_full = (c.get("text") or "")
                    snippet = _best_sentences(text_full, point_label)
                    citations.append({
                        "type": "doc",
                        "tag": c["tag"],
                        "source": c.get("source"),
                        "chunk_index": c.get("chunk_index"),
                        "score": c.get("score"),
                        "text": text_full[:800],
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
                sql_query = await _run_text_agent("SQLGenerator", constraints_text, prompt_text)
                candidate_sql = sql_query
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
                local_sql_rows = _tool_sql(query=candidate_sql)
                local_sql_query = candidate_sql
                citations.append({"type": "sql", "tag": "S1", "query": candidate_sql, "rows": len(local_sql_rows)})
                trace.append({"stage": "sql", "attempt": attempt, "query": candidate_sql, "rows": len(local_sql_rows)})
            except Exception as e:
                tool_error = f"sql_error: {e}"
                last_error = str(e)
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

    # 5) Draft Answer Composer
    composer_instr = (
        f"Respond in language code '{lang}'. Compose a concise answer.\n"
        "- Use provided evidence.\n"
        "- Include inline citations like [D1], [S1] where used.\n"
        "- Prefer Markdown formatting; use Markdown tables for tabular data.\n"
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


