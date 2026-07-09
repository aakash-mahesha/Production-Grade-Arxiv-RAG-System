"""Rewrite node: reformulate the query after a failed grade, then retry retrieval."""

import logging

from src.services.agents.context import Context
from src.services.agents.prompts import rewrite_messages
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)


async def ainvoke_rewrite_query_step(state: AgentState, *, context: Context) -> dict:
    question = state["question"]
    previous = state.get("query") or question

    with context.tracer.span("agent-rewrite", input={"previous_query": previous}) as span:
        try:
            new_query = await context.complete(rewrite_messages(question, previous), temperature=0.3)
            new_query = new_query.strip().strip('"').strip() or previous
        except Exception as e:
            logger.warning("Query rewrite failed, reusing previous query: %s", e)
            new_query = previous
        span.update(output={"rewritten_query": new_query})

    return {
        "query": new_query,
        "rewritten_query": new_query,
        "reasoning_steps": [f"Rewrote query -> '{new_query}'"],
    }
