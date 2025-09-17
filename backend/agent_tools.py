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


def tool_sql(query: str) -> List[Dict[str, Any]]:
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
    # psycopg treats % as placeholder introducer; ensure literal percent usage in LIKE is escaped
    # Replace single % with %% unless it's part of a PostgreSQL concatenation '||' pattern sequence like '%...%'
    # A safe general approach is to double all % characters. SQL literals '%foo%' remain valid as '%%foo%%'.
    query = query.replace('%', '%%')
    engine = get_engine()
    try:
        with engine.connect() as conn:
            res = conn.exec_driver_sql(query)
            columns = list(res.keys()) if res.returns_rows else []
            rows = [dict(zip(columns, row)) for row in res.fetchall()] if res.returns_rows else []
        logger.info("tool_sql executed", extra={"rows": len(rows)})
        return rows
    except SQLAlchemyError as exc:  # return structured error for the model to recover
        logger.error("tool_sql error", exc_info=True)
        return [{"error": str(exc), "query": query}]


