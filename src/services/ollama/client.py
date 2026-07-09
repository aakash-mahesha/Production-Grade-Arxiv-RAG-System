import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from src.config import Settings
from src.exceptions import LLMConnectionError, LLMException, LLMTimeoutError
from src.services.llm.prompts import RAGPromptBuilder

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for local Ollama LLM service."""

    provider_name = "ollama"

    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_host.rstrip("/")
        self.model = settings.ollama_model
        self.timeout = httpx.Timeout(float(settings.ollama_timeout))
        self.prompt_builder = RAGPromptBuilder()

    async def health_check(self) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                version_response = await client.get(f"{self.base_url}/api/version")
                if version_response.status_code != 200:
                    return {
                        "status": "unhealthy",
                        "message": f"Ollama returned status {version_response.status_code}",
                    }

                version_data = version_response.json()
                tags_response = await client.get(f"{self.base_url}/api/tags")

            if tags_response.status_code != 200:
                return {
                    "status": "unhealthy",
                    "message": "Ollama is running but model list is unavailable",
                }

            models = tags_response.json().get("models", [])
            model_names = {m.get("name", "").split(":")[0] for m in models}
            configured_base = self.model.split(":")[0]

            if configured_base not in model_names and self.model not in {m.get("name") for m in models}:
                return {
                    "status": "unhealthy",
                    "message": (
                        f"Model '{self.model}' not found. "
                        f"Run: docker exec rag-ollama ollama pull {self.model}"
                    ),
                    "version": version_data.get("version", "unknown"),
                }

            return {
                "status": "healthy",
                "message": "Ollama is running with configured model",
                "model": self.model,
                "version": version_data.get("version", "unknown"),
            }
        except httpx.ConnectError as e:
            return {"status": "unhealthy", "message": f"Cannot connect to Ollama: {e}"}
        except httpx.TimeoutException as e:
            return {"status": "unhealthy", "message": f"Ollama timeout: {e}"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"Ollama health check failed: {e}"}

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = False,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)

            if response.status_code != 200:
                raise LLMException(
                    f"Ollama chat failed ({response.status_code}): {response.text}"
                )

            result = response.json()
            prompt_tokens = result.get("prompt_eval_count", 0)
            completion_tokens = result.get("eval_count", 0)
            result["usage_metadata"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            return result

        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Cannot connect to Ollama: {e}") from e
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Ollama request timed out: {e}") from e
        except LLMException:
            raise
        except Exception as e:
            raise LLMException(f"Ollama chat error: {e}") from e

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[Dict[str, Any]]:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise LLMException(
                            f"Ollama streaming failed ({response.status_code}): {body.decode()}"
                        )

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse Ollama stream chunk: %s", line)

        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Cannot connect to Ollama: {e}") from e
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Ollama streaming timed out: {e}") from e
        except LLMException:
            raise
        except Exception as e:
            raise LLMException(f"Ollama streaming error: {e}") from e

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
        messages = self.prompt_builder.create_messages(query, chunks)
        response = await self.chat_completion(messages=messages, model=model, stream=False)

        answer_text = response.get("message", {}).get("content", "").strip()
        if not answer_text:
            raise LLMException("Empty response from Ollama")

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
            text = chunk.get("message", {}).get("content", "")
            if text:
                yield text
