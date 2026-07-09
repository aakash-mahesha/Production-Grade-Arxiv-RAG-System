"""Out-of-scope node: handle questions outside the CS.AI paper domain."""

from src.services.agents.context import Context
from src.services.agents.prompts import OUT_OF_SCOPE_ANSWER
from src.services.agents.state import AgentState


async def ainvoke_out_of_scope_step(state: AgentState, *, context: Context) -> dict:
    with context.tracer.span("agent-out-of-scope", input={"question": state.get("question", "")}) as span:
        span.update(output={"answer": OUT_OF_SCOPE_ANSWER})

    return {
        "answer": OUT_OF_SCOPE_ANSWER,
        "sources": [],
        "relevant_chunks": [],
        "reasoning_steps": ["Question judged out of scope; returned domain guidance"],
    }
