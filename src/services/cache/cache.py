"""Redis-backed exact-match response cache for the RAG pipeline (Week 6).

If the same question (with the same parameters) is asked again within the TTL,
we return the stored answer and skip embedding, search, and LLM generation
entirely -- turning a multi-second request into a few-millisecond one and
saving the LLM tokens. Every method degrades to a no-op when Redis is not
configured or unreachable, so the cache can never break the app.
"""

import hashlib
import json
import logging
from typing import Any, Dict, Optional

from src.config import Settings

logger = logging.getLogger(__name__)


class RAGCache:
    """Async exact-match cache over Redis keyed by the request contents."""

    def __init__(self, settings: Settings) -> None:
        self.enabled: bool = settings.cache_enabled
        self.ttl: int = settings.cache_ttl_seconds
        self._client: Optional[Any] = None

        if not self.enabled:
            logger.info("Redis cache disabled (REDIS_URL not set)")
            return

        try:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(settings.redis_url, decode_responses=True)
            logger.info("Redis cache configured (url=%s, ttl=%ss)", settings.redis_url, self.ttl)
        except Exception as e:  # never let the cache break startup
            logger.warning("Failed to init Redis cache; caching disabled: %s", e)
            self.enabled = False
            self._client = None

    async def health_check(self) -> bool:
        if not self.enabled or self._client is None:
            return False
        try:
            return bool(await self._client.ping())
        except Exception as e:
            logger.warning("Redis ping failed: %s", e)
            return False

    def _key(self, request: Any) -> str:
        """Build a deterministic key from the request fields that affect the answer."""
        payload = {
            "q": request.question.strip().lower(),
            "top_k": request.top_k,
            "use_hybrid": request.use_hybrid,
            "model": request.model,
            "categories": request.categories,
        }
        raw = json.dumps(payload, sort_keys=True)
        return "rag:ask:" + hashlib.sha256(raw.encode()).hexdigest()

    async def get(self, request: Any) -> Optional[Dict[str, Any]]:
        """Return the cached response dict for this request, or None on miss."""
        if not self.enabled or self._client is None:
            return None
        try:
            cached = await self._client.get(self._key(request))
            return json.loads(cached) if cached else None
        except Exception as e:
            logger.warning("Redis get failed: %s", e)
            return None

    async def set(self, request: Any, response: Dict[str, Any]) -> None:
        """Store a response dict for this request with the configured TTL."""
        if not self.enabled or self._client is None:
            return
        try:
            await self._client.set(self._key(request), json.dumps(response), ex=self.ttl)
        except Exception as e:
            logger.warning("Redis set failed: %s", e)

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
