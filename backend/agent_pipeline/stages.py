from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
from ..openai_client import get_openai_client
from ..config import settings
from ..agent_tools import tool_retrieve_docs, tool_sql
from .types import DocEvidence, ComposerResult, NitpickerResult
import re


def normalize_input(message: str, lang: str) -> str:
    instr = (
        f"Respond in language code '{lang}'. You normalize user input conservatively.\n"
        "- Preserve semantics, intent, entities, slot values, and constraints EXACTLY.\n"
        "- Do NOT translate or paraphrase domain terms; do NOT summarize or omit clauses.\n"
        "- Keep quoted strings, numbers, currency, percentages, dates/times/timezones verbatim (just standardize spacing).\n"
        "- Keep special tokens (e.g., [D1], [S1]), SQL keywords/fragments, file paths, and code fences intact.\n"
        "- Fix only obvious spelling, spacing, and casing errors; standardize units (e.g., km → km) without changing values.\n"
        "- Remove irrelevant filler words only if they do not change meaning.\n"
        "- Mask PII (emails, phones) with placeholders like <EMAIL>, <PHONE>.\n"
        "- Never drop constraints like 'not closed', 'last 30 days', country names, section numbers (e.g., 2.23).\n"
        "Return only the cleaned text."
    )
    client = get_openai_client()
    completion = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": instr},
            {"role": "user", "content": message},
        ],
    )
    return completion.choices[0].message.content or message


def extract_intent_slots(text: str, lang: str) -> Tuple[str, Dict[str, Any]]:
    instr = (
        f"Respond in language code '{lang}'. Extract task intent and slots as compact JSON.\n"
        "- intents: one of ['policy_question','sql_analytics','doc_lookup','small_talk','other']\n"
        "- slots: key/value pairs relevant to the task\n"
        "Return JSON only."
    )
    client = get_openai_client()
    completion = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": instr},
            {"role": "user", "content": text},
        ],
    )
    import json
    intent = "other"
    slots: Dict[str, Any] = {}
    try:
        j = json.loads(completion.choices[0].message.content or "{}")
        intent = j.get("intent", intent)
        slots = j.get("slots", {}) or {}
    except Exception:
        pass
    return intent, slots


def decide_policy(missing_slots: List[str]) -> str:
    if missing_slots:
        return "ASK"
    return "PROCEED"


def generate_elicitation_question(lang: str, intent: str, missing_slots: List[str], slots: Dict[str, Any]) -> str:
    if not missing_slots:
        return ""
    instr = (
        f"Respond in language code '{lang}'. Generate ONE concise question to collect the following missing info: {missing_slots}.\n"
        "- Batch into one question.\n- Offer options if suitable.\n- Avoid yes/no unless appropriate.\n"
        "Return only the question."
    )
    client = get_openai_client()
    completion = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": instr},
            {"role": "user", "content": f"Intent: {intent}\nKnown slots: {slots}"},
        ],
    )
    return completion.choices[0].message.content or "Could you clarify the missing details?"


def retrieve_documents(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    return tool_retrieve_docs(query=query, top_k=top_k)


def generate_sql(lang: str, intent: str, slots: Dict[str, Any], schema_text: str) -> str:
    constraints = (
        f"Respond in language code '{lang}'. Generate a single read-only SQL query for Postgres given the task, slots, and schema.\n"
        "Constraints:\n"
        "- Use only columns/tables in the provided schema.\n"
        "- Map 'pending' to warranty_claims.status IN ('open','in_review').\n"
        "- Map 'not closed' to warranty_claims.status <> 'closed'.\n"
        "- For tow-related, filter description ILIKE '%tow%' OR '%towing%' OR '%tow truck%'.\n"
        "- Prefer COUNT/aggregations where appropriate.\n"
        "- Do NOT add constant text columns; select only real columns or aggregates.\n"
        "- Distinguish users.warranty_status vs warranty_claims.status.\n"
        "- Do NOT reference non-existent tables; rely on documents for terms/regions.\n"
        "- Avoid scalar subqueries with multiple columns; use CTEs/window functions if needed.\n"
        "- Return only raw SQL without code fences.\n"
    )
    client = get_openai_client()
    completion = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": constraints},
            {"role": "user", "content": f"Task: {intent}\nSlots: {slots}\nSchema:\n{schema_text}"},
        ],
    )
    return completion.choices[0].message.content or "SELECT 1;"


def run_sql(sql_text: str) -> List[Dict[str, Any]]:
    return tool_sql(query=sql_text)


def best_snippet_for_chunk(full_text: str, normalized_question: str, point_label: str) -> str:
    """Return a short snippet of the chunk text biased toward terms in the question+point."""
    sents = re.split(r"(?<=[\.!?])\s+", (full_text or "").strip())
    if not sents:
        return (full_text or "")[:600]
    kw_src = (normalized_question + " " + (point_label or "")).lower()
    kws = [w for w in re.findall(r"[a-zA-Z]{3,}", kw_src) if len(w) > 2]
    if not kws:
        return " ".join(sents[:3])[:600]
    def score_sent(s: str) -> int:
        low = s.lower()
        return sum(1 for k in kws if k in low)
    ranked = sorted(sents, key=score_sent, reverse=True)
    snippet = " ".join(ranked[:3])
    if len(snippet) < 200 and len(ranked) > 3:
        snippet = " ".join(ranked[:5])
    return snippet[:600]


def compose_answer(lang: str, normalized_q: str, slots: Dict[str, Any], docs: List[DocEvidence], sql_preview: Optional[str]) -> ComposerResult:
    evidence_lines: List[str] = []
    for d in docs:
        evidence_lines.append(f"[{d['tag']}] {d['snippet']}")
    if sql_preview:
        evidence_lines.append(f"[S1] SQL preview: {sql_preview}")
    evidence_block = "\n\n".join(evidence_lines)
    instr = (
        f"Respond in language code '{lang}'. Compose a concise, citation-first answer.\n"
        "- Use ONLY the provided evidence; do not invent facts.\n"
        "- Every atomic claim sentence MUST include all relevant citations at the end (e.g., [D1 §2.23] [D1 §2.24] [D4]).\n"
        "- If a sentence relies on multiple documents, include multiple [D#] tokens in that sentence.\n"
        "- Prefer 2–6 sentences; avoid headings, long quotes, or policy text blocks. DO NOT preface with 'Relevant policy text' or paste large excerpts.\n"
        "- Be to the point.\n"
        "- For SQL-backed answers, include [S1] in the claim sentence that states the numeric/result.\n"
        "Return JSON with keys: answer, reasoning_bullets (array).\n"
    )
    client = get_openai_client()
    completion = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": instr},
            {"role": "user", "content": f"Question: {normalized_q}\nSlots: {slots}\nEvidence:\n{evidence_block}"},
        ],
    )
    import json
    result: ComposerResult = {"answer": completion.choices[0].message.content or "", "reasoning_bullets": []}
    try:
        j = json.loads(completion.choices[0].message.content or "{}")
        result["answer"] = j.get("answer", result["answer"])  # type: ignore[index]
        result["reasoning_bullets"] = list(j.get("reasoning_bullets", []) or [])  # type: ignore[index]
    except Exception:
        pass
    return result


def nitpick_verify(
    lang: str,
    question: str,
    draft: str,
    docs_map: Dict[str, str],
    sql_query: Optional[str],
    sql_rows: Optional[List[Dict[str, Any]]] = None,
) -> NitpickerResult:
    doc_lines = [f"[{k}] {v}" for k, v in docs_map.items()]
    docs_block = "\n\n".join(doc_lines)
    import json
    sql_block = ""
    if sql_query:
        sql_block = f"\nSQL [S1]:\n{sql_query}"
        if sql_rows is not None:
            preview = sql_rows[:5]
            sql_block += "\nSQL rows preview (first 5):\n" + json.dumps(preview, ensure_ascii=False)
    instr = (
        f"Respond in language code '{lang}'. You are Nitpicker, a deterministic verifier (temperature 0).\n"
        "Run multi-pass checks on the draft answer against the provided evidence only.\n"
        "Return strict JSON with keys: compliance_score, findings, patches, revised_answer.\n"
    )
    client = get_openai_client()
    completion = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": instr},
            {"role": "user", "content": f"Question:\n{question}\n\nDraft:\n{draft}\n\nDocs:\n{docs_block}{sql_block}"},
        ],
    )
    import json
    default: NitpickerResult = {"compliance_score": 0.0, "findings": [], "patches": [], "revised_answer": None}
    try:
        return json.loads(completion.choices[0].message.content or "{}")
    except Exception:
        return default


def finalize_answer(answer: str) -> str:
    return answer
