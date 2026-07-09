"""Tunable configuration for the agentic RAG graph (Week 7)."""

from dataclasses import dataclass
from typing import Optional

from src.config import Settings


@dataclass
class GraphConfig:
    """Per-run knobs for the LangGraph workflow.

    Defaults come from Settings but can be overridden per request.
    """

    model: Optional[str] = None
    top_k: int = 5
    use_hybrid: bool = True
    categories: Optional[list[str]] = None
    max_retrieval_attempts: int = 2
    guardrail_threshold: int = 50
    grade_min_relevant: int = 1

    @classmethod
    def from_settings(cls, settings: Settings, **overrides) -> "GraphConfig":
        base = cls(
            top_k=settings.agent_top_k,
            max_retrieval_attempts=settings.agent_max_retrieval_attempts,
            guardrail_threshold=settings.agent_guardrail_threshold,
            grade_min_relevant=settings.agent_grade_min_relevant,
        )
        for key, value in overrides.items():
            if value is not None and hasattr(base, key):
                setattr(base, key, value)
        return base
