"""Shared state passed between nodes of the agentic RAG graph (Week 7).

LangGraph threads a single dict through every node; each node returns a partial
dict that is merged in. `total=False` lets nodes populate fields incrementally.
"""

import operator
from typing import Annotated, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # Inputs
    question: str  # the original user question (never mutated)
    query: str  # the current query used for retrieval (may be rewritten)

    # Guardrail
    guardrail_score: int  # 0-100 domain relevance
    guardrail_decision: str  # "in_scope" | "out_of_scope"

    # Retrieval / grading
    chunks: List[dict]  # last retrieved chunks
    relevant_chunks: List[dict]  # chunks the grader kept
    search_mode: str  # "hybrid" | "bm25"
    retrieval_attempts: int  # how many retrieval passes have run
    rewritten_query: Optional[str]  # last rewrite, if any

    # Output
    answer: str
    sources: List[str]
    # Accumulated across nodes (operator.add appends instead of overwriting).
    reasoning_steps: Annotated[List[str], operator.add]
