from typing import Optional

from src.config import Settings, get_settings
from src.services.cache.cache import RAGCache


def make_cache(settings: Optional[Settings] = None) -> RAGCache:
    """Create the RAG response cache (no-op when REDIS_URL is not set)."""
    if settings is None:
        settings = get_settings()
    return RAGCache(settings)
