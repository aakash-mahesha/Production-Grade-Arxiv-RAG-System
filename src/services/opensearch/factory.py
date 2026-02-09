from functools import lru_cache
from typing import Optional
from src.config import Settings, get_settings
from .client import OpenSearchClient


@lru_cache(maxsize=1)
def make_opensearch_client(settings: Optional[Settings] = None) -> OpenSearchClient:
    """Factory function to create cached OpenSearch client."""
    if settings is None:
        settings = get_settings()
    return OpenSearchClient(host=settings.opensearch.host, settings=settings)


def make_opensearch_client_fresh(settings: Optional[Settings] = None, host: Optional[str] = None) -> OpenSearchClient:
    """Factory function to create a fresh (non-cached) OpenSearch client."""
    if settings is None:
        settings = get_settings()
    opensearch_host = host or settings.opensearch.host
    return OpenSearchClient(host=opensearch_host, settings=settings)