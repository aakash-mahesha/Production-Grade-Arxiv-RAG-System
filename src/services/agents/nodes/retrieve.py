"""Retrieve node: run hybrid/BM25 search for the current query."""

import logging

from src.services.agents.context import Context
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)


async def ainvoke_retrieve_step(state: AgentState, *, context: Context) -> dict:
    query = state.get("query") or state["question"]
    attempts = state.get("retrieval_attempts", 0) + 1

    with context.tracer.span("agent-retrieve", input={"query": query, "attempt": attempts}) as span:
        chunks, mode = await context.retrieve(query)
        span.update(output={"num_chunks": len(chunks), "mode": mode})

    return {
        "chunks": chunks,
        "search_mode": mode,
        "retrieval_attempts": attempts,
        "reasoning_steps": [
            f"Retrieved {len(chunks)} chunks for '{query}' ({mode}, attempt {attempts})"
        ],
    }
