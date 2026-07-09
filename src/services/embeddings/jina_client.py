import asyncio
import logging
from typing import List

import httpx
from src.schemas.embeddings.jina import JinaEmbeddingRequest, JinaEmbeddingResponse

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 503, 504}


class JinaEmbeddingsClient:
    """Client for Jina AI embeddings API.

    Uses Jina embeddings v3 model with 1024 dimensions optimized for retrieval.
    Documentation: https://jina.ai/embeddings
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.jina.ai/v1",
        batch_size: int = 16,
        request_delay: float = 0.7,
        max_retries: int = 6,
    ):
        """Initialize Jina embeddings client.

        :param api_key: Jina API key
        :param base_url: API base URL
        :param batch_size: Texts per API request (keep low to avoid TPM limits)
        :param request_delay: Seconds to wait between batch requests
        :param max_retries: Max retries for rate-limited or transient errors
        """
        self.api_key = api_key
        self.base_url = base_url
        self.batch_size = batch_size
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(timeout=60.0)
        logger.info(
            "Jina embeddings client initialized (batch_size=%s, request_delay=%ss)",
            batch_size,
            request_delay,
        )

    async def _post_with_retry(self, payload: dict) -> JinaEmbeddingResponse:
        """POST to embeddings endpoint with retry/backoff for rate limits."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json=payload,
                )

                if response.status_code in RETRYABLE_STATUS_CODES:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = float(retry_after)
                    else:
                        delay = min(60.0, 2 ** attempt)

                    logger.warning(
                        "Jina rate limit/transient error (%s), retrying in %.1fs (attempt %s/%s)",
                        response.status_code,
                        delay,
                        attempt + 1,
                        self.max_retries,
                    )
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                return JinaEmbeddingResponse(**response.json())

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code not in RETRYABLE_STATUS_CODES:
                    raise
                if attempt >= self.max_retries:
                    raise
            except httpx.HTTPError as e:
                last_error = e
                if attempt >= self.max_retries:
                    raise
                delay = min(60.0, 2 ** attempt)
                logger.warning("Jina request failed (%s), retrying in %.1fs", e, delay)
                await asyncio.sleep(delay)

        if last_error:
            raise last_error
        raise RuntimeError("Jina embeddings request failed after retries")

    async def embed_passages(self, texts: List[str], batch_size: int | None = None) -> List[List[float]]:
        """Embed text passages for indexing.

        :param texts: List of text passages to embed
        :param batch_size: Number of texts to process in each API call
        :returns: List of embedding vectors
        """
        effective_batch_size = batch_size or self.batch_size
        embeddings = []

        for i in range(0, len(texts), effective_batch_size):
            batch = texts[i : i + effective_batch_size]

            request_data = JinaEmbeddingRequest(
                model="jina-embeddings-v3", task="retrieval.passage", dimensions=1024, input=batch
            )

            result = await self._post_with_retry(request_data.model_dump())
            batch_embeddings = [item["embedding"] for item in result.data]
            embeddings.extend(batch_embeddings)

            logger.debug("Embedded batch of %s passages", len(batch))

            if i + effective_batch_size < len(texts):
                await asyncio.sleep(self.request_delay)

        logger.info("Successfully embedded %s passages", len(texts))
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Embed a search query.

        :param query: Query text to embed
        :returns: Embedding vector for the query
        """
        request_data = JinaEmbeddingRequest(
            model="jina-embeddings-v3", task="retrieval.query", dimensions=1024, input=[query]
        )

        result = await self._post_with_retry(request_data.model_dump())
        embedding = result.data[0]["embedding"]

        logger.debug("Embedded query: '%s...'", query[:50])
        return embedding

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
