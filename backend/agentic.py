import logging
import os
from typing import Any, Dict

from openai import OpenAI
from agents import Agent, Runner, function_tool

from .logging_setup import configure_logging
from .openai_client import get_openai_client
from .config import settings
from .agent_tools import tool_retrieve_docs as _tool_retrieve_docs, tool_sql as _tool_sql


configure_logging()
logger = logging.getLogger("agentic")


@function_tool
def retrieve_docs(query: str, top_k: int = 6) -> list[dict]:
    """Retrieve relevant PDF chunks about the user's question."""
    return _tool_retrieve_docs(query=query, top_k=top_k)


@function_tool
def sql(query: str) -> list[dict]:
    """Run a read-only SQL query against the Postgres database."""
    return _tool_sql(query=query)


async def run_agentic_chat(message: str, language: str | None = None) -> Dict[str, Any]:
    # Ensure OpenAI client is configured (reads API key / base_url)
    _client: OpenAI = get_openai_client()
    lang = language or "en"

    instructions = (
        f"You are an analyst assistant. Use the available tools to retrieve PDF context and query Postgres. "
        f"Prefer COUNT/aggregations for totals; never mutate the DB. Always respond in language code '{lang}'."
    )

    agent = Agent(
        name="Analyst",
        instructions=instructions,
        tools=[retrieve_docs, sql],
    )

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

    return {"answer": answer_text, "tools": tool_uses}


