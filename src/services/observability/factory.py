from typing import Optional

from src.config import Settings, get_settings
from src.services.observability.tracer import RAGTracer


def make_tracer(settings: Optional[Settings] = None) -> RAGTracer:
    """Create the RAG tracer from settings (no-op when Langfuse keys are absent)."""
    if settings is None:
        settings = get_settings()
    return RAGTracer(settings)
