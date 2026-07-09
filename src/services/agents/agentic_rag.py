"""Agentic RAG service (Week 7).

Builds and runs a LangGraph StateGraph that turns naive retrieve->answer into a
stateful decision workflow:

    START -> guardrail
    guardrail --in_scope--> retrieve
    guardrail --out_of_scope--> out_of_scope -> END
    retrieve -> grade
    grade --relevant--> generate -> END
    grade --no relevant, attempts left--> rewrite -> retrieve   (loop)
    grade --no relevant, exhausted--> generate -> END           (best effort)
"""

import functools
import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from src.config import Settings
from src.services.agents.config import GraphConfig
from src.services.agents.context import Context
from src.services.agents.nodes import (
    ainvoke_generate_answer_step,
    ainvoke_grade_documents_step,
    ainvoke_guardrail_step,
    ainvoke_out_of_scope_step,
    ainvoke_retrieve_step,
    ainvoke_rewrite_query_step,
)
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)


def _route_after_guardrail(state: AgentState) -> str:
    return "retrieve" if state.get("guardrail_decision") == "in_scope" else "out_of_scope"


def _make_route_after_grade(config: GraphConfig):
    def _route_after_grade(state: AgentState) -> str:
        relevant = state.get("relevant_chunks", [])
        if len(relevant) >= config.grade_min_relevant:
            return "generate"
        if state.get("retrieval_attempts", 0) < config.max_retrieval_attempts:
            return "rewrite"
        return "generate"  # exhausted retries: answer with whatever we have

    return _route_after_grade


class AgenticRAGService:
    """Owns the shared clients and compiles a fresh graph per request config."""

    def __init__(
        self,
        settings: Settings,
        llm_client: Any,
        opensearch_client: Any,
        embeddings_client: Any,
        tracer: Any,
    ) -> None:
        self.settings = settings
        self.llm = llm_client
        self.opensearch = opensearch_client
        self.embeddings = embeddings_client
        self.tracer = tracer

    def _build_graph(self, context: Context):
        builder = StateGraph(AgentState)

        bind = lambda fn: functools.partial(fn, context=context)  # noqa: E731
        builder.add_node("guardrail", bind(ainvoke_guardrail_step))
        builder.add_node("retrieve", bind(ainvoke_retrieve_step))
        builder.add_node("grade", bind(ainvoke_grade_documents_step))
        builder.add_node("rewrite", bind(ainvoke_rewrite_query_step))
        builder.add_node("generate", bind(ainvoke_generate_answer_step))
        builder.add_node("out_of_scope", bind(ainvoke_out_of_scope_step))

        builder.add_edge(START, "guardrail")
        builder.add_conditional_edges(
            "guardrail",
            _route_after_guardrail,
            {"retrieve": "retrieve", "out_of_scope": "out_of_scope"},
        )
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges(
            "grade",
            _make_route_after_grade(context.config),
            {"generate": "generate", "rewrite": "rewrite"},
        )
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("generate", END)
        builder.add_edge("out_of_scope", END)

        return builder.compile()

    async def run(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_hybrid: Optional[bool] = None,
        categories: Optional[list[str]] = None,
        model: Optional[str] = None,
    ) -> dict:
        config = GraphConfig.from_settings(
            self.settings,
            top_k=top_k,
            use_hybrid=use_hybrid,
            categories=categories,
            model=model,
        )
        context = Context(
            llm_client=self.llm,
            opensearch_client=self.opensearch,
            embeddings_client=self.embeddings,
            tracer=self.tracer,
            config=config,
        )
        graph = self._build_graph(context)

        try:
            with self.tracer.span(
                "agentic-rag",
                input={"question": question},
                metadata={"top_k": config.top_k, "max_attempts": config.max_retrieval_attempts},
            ) as root:
                final_state = await graph.ainvoke({"question": question, "reasoning_steps": []})
                root.update(
                    output={
                        "answer": final_state.get("answer", ""),
                        "guardrail_score": final_state.get("guardrail_score"),
                        "retrieval_attempts": final_state.get("retrieval_attempts", 0),
                    }
                )
        finally:
            self.tracer.flush()

        return {
            "query": question,
            "answer": final_state.get("answer", ""),
            "sources": final_state.get("sources", []),
            "reasoning_steps": final_state.get("reasoning_steps", []),
            "guardrail_score": final_state.get("guardrail_score", 0),
            "retrieval_attempts": final_state.get("retrieval_attempts", 0),
            "rewritten_query": final_state.get("rewritten_query"),
            "search_mode": final_state.get("search_mode", "hybrid"),
            "chunks_used": len(final_state.get("relevant_chunks") or final_state.get("chunks", [])),
        }
