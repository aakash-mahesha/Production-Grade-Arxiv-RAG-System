from typing import Union

from src.config import Settings
from src.services.ollama import OllamaClient
from src.services.openrouter import OpenRouterClient

LLMClient = Union[OllamaClient, OpenRouterClient]


def make_llm_client(settings: Settings) -> LLMClient:
    """Create the configured LLM client (Ollama by default)."""
    provider = settings.llm_provider.lower().strip()

    if provider == "openrouter":
        return OpenRouterClient(settings)

    return OllamaClient(settings)
