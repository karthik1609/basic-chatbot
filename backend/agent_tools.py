import logging
from typing import List, Dict, Any

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
        {"text": t, "metadata": m, "score": s}
        for (t, m, s) in results
    ]


def tool_sql(query: str) -> List[Dict[str, Any]]:
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


