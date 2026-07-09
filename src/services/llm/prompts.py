import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class RAGPromptBuilder:
    """Build chat messages for RAG generation."""

    def __init__(self) -> None:
        self._system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        prompt_path = _PROMPTS_DIR / "rag_system.txt"
        if prompt_path.exists():
            return prompt_path.read_text().strip()
        return "You are a helpful assistant that answers questions using only the provided context."

    def _format_chunks(self, chunks: List[Dict[str, Any]], max_chars_per_chunk: int = 800) -> str:
        if not chunks:
            return "No relevant excerpts were retrieved."

        formatted = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("title", "Unknown")
            arxiv_id = chunk.get("arxiv_id", "N/A")
            section = chunk.get("section_name") or chunk.get("section", "")
            text = chunk.get("chunk_text") or chunk.get("text") or chunk.get("raw_text", "")
            if len(text) > max_chars_per_chunk:
                text = text[:max_chars_per_chunk] + "..."

            header = f"[{i}] {title} (arXiv: {arxiv_id})"
            if section:
                header += f" — {section}"

            formatted.append(f"{header}\n{text.strip()}")

        return "\n\n".join(formatted)

    def create_messages(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Build OpenAI/Ollama-compatible chat messages."""
        context = self._format_chunks(chunks)
        user_content = (
            f"Context from retrieved papers:\n\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer based only on the context above."
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    def create_rag_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Return a single user prompt string (legacy helper)."""
        messages = self.create_messages(query, chunks)
        return messages[1]["content"]
