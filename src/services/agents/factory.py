from typing import Any

from src.config import Settings
from src.services.agents.agentic_rag import AgenticRAGService


def make_agentic_rag_service(
    settings: Settings,
    llm_client: Any,
    opensearch_client: Any,
    embeddings_client: Any,
    tracer: Any,
) -> AgenticRAGService:
    """Create the agentic RAG service wired to the shared clients."""
    return AgenticRAGService(
        settings=settings,
        llm_client=llm_client,
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        tracer=tracer,
    )
