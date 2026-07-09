"""Guardrail node: score domain relevance and decide in-scope vs out-of-scope."""

import logging
import re

from src.services.agents.context import Context
from src.services.agents.prompts import guardrail_messages
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)


def _parse_score(text: str) -> int:
    match = re.search(r"\d{1,3}", text)
    if not match:
        return 100  # fail open: treat as in-scope if the model didn't cooperate
    return max(0, min(100, int(match.group())))


async def ainvoke_guardrail_step(state: AgentState, *, context: Context) -> dict:
    question = state["question"]
    with context.tracer.span("agent-guardrail", input={"question": question}) as span:
        try:
            text = await context.complete(guardrail_messages(question))
            score = _parse_score(text)
        except Exception as e:
            logger.warning("Guardrail LLM call failed, assuming in-scope: %s", e)
            score = 100

        decision = "in_scope" if score >= context.config.guardrail_threshold else "out_of_scope"
        span.update(output={"score": score, "decision": decision})

    return {
        "query": question,
        "guardrail_score": score,
        "guardrail_decision": decision,
        "retrieval_attempts": 0,
        "reasoning_steps": [f"Guardrail scored domain relevance {score}/100 -> {decision}"],
    }
