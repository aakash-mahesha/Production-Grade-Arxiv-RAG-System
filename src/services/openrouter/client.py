import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from src.config import Settings
from src.exceptions import LLMConnectionError, LLMException, LLMTimeoutError
from src.services.llm.prompts import RAGPromptBuilder

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Client for OpenRouter's OpenAI-compatible chat completions API."""

    provider_name = "openrouter"

    def __init__(self, settings: Settings):
        self.base_url = settings.openrouter_base_url.rstrip("/")
        self.api_key = settings.openrouter_api_key
        self.model = settings.openrouter_model
        self.timeout = httpx.Timeout(float(settings.openrouter_timeout))
        self.app_name = settings.openrouter_app_name
        self.app_url = settings.openrouter_app_url
        self.prompt_builder = RAGPromptBuilder()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        if self.app_name:
            headers["X-OpenRouter-Title"] = self.app_name
        return headers

    async def health_check(self) -> Dict[str, Any]:
        """Check OpenRouter connectivity and API key validity."""
        if not self.api_key:
            return {
                "status": "unhealthy",
                "message": "OPENROUTER_API_KEY is not configured",
            }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self._headers(),
                )

            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "message": "OpenRouter API is reachable",
                    "model": self.model,
                }

            return {
                "status": "unhealthy",
                "message": f"OpenRouter returned status {response.status_code}",
            }
        except httpx.ConnectError as e:
            return {"status": "unhealthy", "message": f"Cannot connect to OpenRouter: {e}"}
        except httpx.TimeoutException as e:
            return {"status": "unhealthy", "message": f"OpenRouter timeout: {e}"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"OpenRouter health check failed: {e}"}

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send a chat completion request to OpenRouter."""
        if not self.api_key:
            raise LLMException("OPENROUTER_API_KEY is not configured")

        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            **kwargs,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )

            if response.status_code != 200:
                raise LLMException(
                    f"OpenRouter chat completion failed ({response.status_code}): {response.text}"
                )

            result = response.json()
            usage = result.get("usage", {})
            result["usage_metadata"] = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            return result

        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Cannot connect to OpenRouter: {e}") from e
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"OpenRouter request timed out: {e}") from e
        except LLMException:
            raise
        except Exception as e:
            raise LLMException(f"OpenRouter chat completion error: {e}") from e

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream chat completion chunks from OpenRouter (SSE)."""
        if not self.api_key:
            raise LLMException("OPENROUTER_API_KEY is not configured")

        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            **kwargs,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise LLMException(
                            f"OpenRouter streaming failed ({response.status_code}): {body.decode()}"
                        )

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse streaming chunk: %s", line)
                            continue

        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Cannot connect to OpenRouter: {e}") from e
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"OpenRouter streaming timed out: {e}") from e
        except LLMException:
            raise
        except Exception as e:
            raise LLMException(f"OpenRouter streaming error: {e}") from e

    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> tuple[List[str], List[str]]:
        sources: List[str] = []
        citations: List[str] = []
        seen_urls: set[str] = set()

        for chunk in chunks:
            arxiv_id = chunk.get("arxiv_id")
            if not arxiv_id:
                continue

            citations.append(arxiv_id)
            arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
            if pdf_url not in seen_urls:
                sources.append(pdf_url)
                seen_urls.add(pdf_url)

        return sources, list(dict.fromkeys(citations))[:5]

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a RAG answer using retrieved chunks."""
        messages = self.prompt_builder.create_messages(query, chunks)
        response = await self.chat_completion(messages=messages, model=model, stream=False)

        choices = response.get("choices", [])
        if not choices:
            raise LLMException("No response generated from OpenRouter")

        answer_text = choices[0].get("message", {}).get("content", "").strip()
        if not answer_text:
            raise LLMException("Empty response from OpenRouter")

        sources, citations = self._extract_sources(chunks)
        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": "medium",
            "citations": citations,
            "usage_metadata": response.get("usage_metadata", {}),
        }

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a RAG answer token-by-token."""
        messages = self.prompt_builder.create_messages(query, chunks)
        async for chunk in self.chat_completion_stream(messages=messages, model=model):
            yield chunk

    async def stream_rag_text(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        async for chunk in self.generate_rag_answer_stream(query, chunks, model):
            choices = chunk.get("choices", [])
            if not choices:
                continue
            text = choices[0].get("delta", {}).get("content", "")
            if text:
                yield text
