"""Langfuse-backed tracing for the RAG pipeline.

The tracer wraps the Langfuse v4 SDK behind context managers so routers can
instrument each RAG step (embed -> search -> generate) without caring whether
tracing is actually enabled. When Langfuse keys are missing, every method
degrades to a no-op, so the app runs identically with or without observability.
"""

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from src.config import Settings

logger = logging.getLogger(__name__)


class _NoOpObservation:
    """Stand-in returned when tracing is disabled.

    Implements the same surface the router calls (``update``) but does nothing,
    so callers never need to branch on whether tracing is on.
    """

    def update(self, *args: Any, **kwargs: Any) -> None:
        return None


class RAGTracer:
    """Thin wrapper over the Langfuse client exposing RAG-specific spans."""

    def __init__(self, settings: Settings) -> None:
        self.enabled: bool = settings.langfuse_enabled
        self._client: Optional[Any] = None

        if not self.enabled:
            logger.info("Langfuse tracing disabled (LANGFUSE_PUBLIC_KEY/SECRET_KEY not set)")
            return

        try:
            # Imported lazily so the dependency is only required when enabled.
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            logger.info("Langfuse tracing enabled (host=%s)", settings.langfuse_host)
        except Exception as e:  # never let telemetry break the app
            logger.warning("Failed to initialise Langfuse; tracing disabled: %s", e)
            self.enabled = False
            self._client = None

    @contextmanager
    def span(self, name: str, **kwargs: Any) -> Iterator[Any]:
        """Create a generic observation (embedding, search, whole pipeline).

        Any observation opened inside this ``with`` block nests under it
        automatically via OpenTelemetry context propagation.
        """
        if not self.enabled or self._client is None:
            yield _NoOpObservation()
            return

        with self._client.start_as_current_observation(as_type="span", name=name, **kwargs) as span:
            yield span

    @contextmanager
    def generation(self, name: str, model: str, **kwargs: Any) -> Iterator[Any]:
        """Create an LLM observation that also tracks model + token usage."""
        if not self.enabled or self._client is None:
            yield _NoOpObservation()
            return

        with self._client.start_as_current_observation(
            as_type="generation", name=name, model=model, **kwargs
        ) as generation:
            yield generation

    def flush(self) -> None:
        """Force-send any buffered events (call on shutdown / after a request)."""
        if self._client is not None:
            self._client.flush()
