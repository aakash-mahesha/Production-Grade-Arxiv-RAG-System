"""Grade node: keep only the retrieved chunks that are actually relevant."""

import logging
import re
from typing import Any, Dict, List

from src.services.agents.context import Context
from src.services.agents.prompts import grade_messages
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)


def _select_relevant(text: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if "none" in text.lower():
        return []
    indices = {int(n) for n in re.findall(r"\d+", text)}
    selected = [chunks[i] for i in sorted(indices) if 0 <= i < len(chunks)]
    return selected


async def ainvoke_grade_documents_step(state: AgentState, *, context: Context) -> dict:
    query = state.get("query") or state["question"]
    chunks = state.get("chunks", [])

    with context.tracer.span("agent-grade", input={"query": query, "num_chunks": len(chunks)}) as span:
        if not chunks:
            relevant: List[Dict[str, Any]] = []
        else:
            try:
                text = await context.complete(grade_messages(query, chunks))
                relevant = _select_relevant(text, chunks)
            except Exception as e:
                logger.warning("Grading failed, keeping all chunks: %s", e)
                relevant = chunks
        span.update(output={"num_relevant": len(relevant)})

    return {
        "relevant_chunks": relevant,
        "reasoning_steps": [f"Graded chunks: {len(relevant)}/{len(chunks)} relevant"],
    }
