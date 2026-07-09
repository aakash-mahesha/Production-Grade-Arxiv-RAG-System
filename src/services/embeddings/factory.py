from typing import Optional

from src.config import Settings, get_settings

from .jina_client import JinaEmbeddingsClient


def make_embeddings_client(settings: Optional[Settings] = None) -> JinaEmbeddingsClient:
    """Factory function to create embeddings client.

    Creates a new client instance each time to avoid closed client issues.

    :param settings: Optional settings instance
    :returns: JinaEmbeddingsClient instance
    """
    if settings is None:
        settings = get_settings()

    # Get API key from settings
    api_key = settings.jina_api_key.strip()
    if not api_key:
        raise ValueError(
            "JINA_API_KEY is not set. Add it to .env and pass it to the container "
            "(e.g. JINA_API_KEY=... in compose.yaml environment)."
        )

    return JinaEmbeddingsClient(
        api_key=api_key,
        batch_size=settings.jina_batch_size,
        request_delay=settings.jina_request_delay,
    )
