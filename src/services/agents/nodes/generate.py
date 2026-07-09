"""Generate node: answer the original question from the relevant chunks."""

import logging

from src.services.agents.context import Context
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)


async def ainvoke_generate_answer_step(state: AgentState, *, context: Context) -> dict:
    question = state["question"]
    chunks = state.get("relevant_chunks") or state.get("chunks", [])
    model_name = context.config.model or getattr(context.llm, "model", "unknown")

    with context.tracer.generation(
        "agent-generate",
        model=model_name,
        input={"question": question, "num_chunks": len(chunks)},
    ) as generation:
        if not chunks:
            answer = "I couldn't find relevant information in the papers to answer your question."
            sources: list[str] = []
        else:
            try:
                result = await context.llm.generate_rag_answer(
                    query=question,
                    chunks=chunks,
                    model=context.config.model,
                )
                answer = result.get("answer", "").strip() or "Unable to generate an answer."
                sources = result.get("sources") or context.sources_from_chunks(chunks)
                usage = result.get("usage_metadata", {})
                generation.update(
                    output=answer,
                    usage_details={
                        "input": usage.get("prompt_tokens", 0),
                        "output": usage.get("completion_tokens", 0),
                        "total": usage.get("total_tokens", 0),
                    },
                )
            except Exception as e:
                logger.warning("Answer generation failed: %s", e)
                answer = "I found relevant papers but couldn't generate an answer due to an LLM error."
                sources = context.sources_from_chunks(chunks)

    return {
        "answer": answer,
        "sources": sources,
        "reasoning_steps": [f"Generated answer from {len(chunks)} chunk(s)"],
    }
