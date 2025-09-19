import logging
from typing import List, Dict, Any
import re

from .logging_setup import configure_logging
from .rag import retrieve_relevant_chunks
from .db import get_engine
from sqlalchemy.exc import SQLAlchemyError


configure_logging()
logger = logging.getLogger("agent_tools")


def tool_retrieve_docs(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    results = retrieve_relevant_chunks(query, top_k=top_k)
    logger.info("tool_retrieve_docs", extra={"returned": len(results)})
    return [
        {"text": t, "metadata": m, "score": s, "source": m.get("source"), "chunk_index": m.get("chunk_index")}
        for (t, m, s) in results
    ]


def tool_sql(query: str, *, statement_timeout_ms: int = 8000) -> List[Dict[str, Any]]:
    # Strip markdown fences like ```sql ... ``` if present
    def _strip_sql_markdown(q: str) -> str:
        q2 = q.strip()
        if q2.startswith("```"):
            # remove opening fence and optional language tag
            q2 = q2[3:]
            if q2.lstrip().lower().startswith("sql"):
                q2 = q2.lstrip()[3:]
            q2 = q2.lstrip("\n\r ")
        if q2.endswith("```"):
            q2 = q2[:-3]
        # Also remove stray triple backticks anywhere
        q2 = re.sub(r"```", "", q2)
        return q2.strip()

    query = _strip_sql_markdown(query)

    # Escape percent signs only inside single-quoted string literals to satisfy psycopg's
    # pyformat scanning while preserving operators like modulo outside strings.
    def _escape_percents_in_string_literals(q: str) -> str:
        out_chars: List[str] = []
        in_str = False
        i = 0
        while i < len(q):
            ch = q[i]
            if ch == "'":
                # toggle string state unless it's an escaped single quote ''
                if in_str:
                    # Lookahead for doubled quote (escaped)
                    if i + 1 < len(q) and q[i + 1] == "'":
                        # Escaped quote inside string
                        out_chars.append("''")
                        i += 2
                        continue
                    else:
                        in_str = False
                        out_chars.append(ch)
                        i += 1
                        continue
                else:
                    in_str = True
                    out_chars.append(ch)
                    i += 1
                    continue
            if in_str and ch == '%':
                out_chars.append('%%')
                i += 1
                continue
            out_chars.append(ch)
            i += 1
        return ''.join(out_chars)

    query = _escape_percents_in_string_literals(query)
    engine = get_engine()
    attempts = 0
    last_err: str | None = None
    while attempts < 2:
        attempts += 1
        try:
            with engine.connect() as conn:
                # Set a per-statement timeout to avoid hanging queries
                try:
                    conn.exec_driver_sql(f"SET LOCAL statement_timeout = {int(statement_timeout_ms)}")
                except Exception:
                    # Some drivers require transaction for LOCAL; fallback to session-level
                    try:
                        conn.exec_driver_sql(f"SET statement_timeout = {int(statement_timeout_ms)}")
                    except Exception:
                        pass
                res = conn.exec_driver_sql(query)
                columns = list(res.keys()) if res.returns_rows else []
                rows = [dict(zip(columns, row)) for row in res.fetchall()] if res.returns_rows else []
            logger.info("tool_sql executed", extra={"rows": len(rows)})
            return rows
        except SQLAlchemyError as exc:  # return structured error for the model to recover
            last_err = str(exc)
            logger.error("tool_sql error", exc_info=True)
            # retry once for transient/timeout-like errors
            low = last_err.lower()
            if attempts < 2 and ("timeout" in low or "connection" in low or "deadlock" in low):
                continue
            break
    return [{"error": last_err or "unknown sql error", "query": query}]


