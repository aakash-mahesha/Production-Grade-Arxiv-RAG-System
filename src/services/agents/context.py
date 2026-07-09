"""Dependency container + shared helpers for agent nodes (Week 7).

Nodes are plain async functions; they receive this Context (bound via
functools.partial when the graph is built) to reach the LLM, search, and
embeddings clients without importing app state directly.
"""

import logging
from typing import Any, Dict, List, Optional

from src.services.agents.config import GraphConfig

logger = logging.getLogger(__name__)


def _extract_text(response: Dict[str, Any]) -> str:
    """Normalize a chat_completion response from either LLM client to plain text."""
    message = response.get("message")
    if isinstance(message, dict) and message.get("content"):
        return message["content"].strip()
    choices = response.get("choices")
    if choices:
        return (choices[0].get("message", {}).get("content") or "").strip()
    return ""


class Context:
    def __init__(
        self,
        llm_client: Any,
        opensearch_client: Any,
        embeddings_client: Any,
        tracer: Any,
        config: GraphConfig,
    ) -> None:
        self.llm = llm_client
        self.opensearch = opensearch_client
        self.embeddings = embeddings_client
        self.tracer = tracer
        self.config = config

    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> str:
        """Call the configured LLM and return its text (provider-agnostic)."""
        response = await self.llm.chat_completion(
            messages=messages,
            model=model or self.config.model,
            temperature=temperature,
        )
        return _extract_text(response)

    async def retrieve(self, query: str) -> tuple[List[Dict[str, Any]], str]:
        """Run hybrid (or BM25) search for a query and return (chunks, mode)."""
        query_embedding = None
        if self.config.use_hybrid:
            try:
                query_embedding = await self.embeddings.embed_query(query)
            except Exception as e:
                logger.warning("Embedding failed, falling back to BM25: %s", e)

        use_hybrid = self.config.use_hybrid and query_embedding is not None
        search_mode = "hybrid" if use_hybrid else "bm25"

        results = self.opensearch.search_unified(
            query=query,
            query_embedding=query_embedding,
            size=self.config.top_k,
            from_=0,
            categories=self.config.categories,
            use_hybrid=use_hybrid,
            min_score=0.0,
        )

        chunks: List[Dict[str, Any]] = []
        for hit in results.get("hits", []):
            chunks.append(
                {
                    "arxiv_id": hit.get("arxiv_id", ""),
                    "title": hit.get("title", ""),
                    "chunk_text": hit.get("chunk_text") or hit.get("abstract", ""),
                    "section_name": hit.get("section_name"),
                }
            )
        return chunks, search_mode

    @staticmethod
    def sources_from_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
        sources: List[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            arxiv_id = chunk.get("arxiv_id")
            if not arxiv_id:
                continue
            clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
            url = f"https://arxiv.org/pdf/{clean}.pdf"
            if url not in seen:
                sources.append(url)
                seen.add(url)
        return sources
