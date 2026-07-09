"""Prompt templates for the agentic RAG decision nodes (Week 7)."""

from typing import Dict, List

DOMAIN = (
    "computer science and artificial intelligence research papers from arXiv "
    "(topics like machine learning, LLMs, agents, NLP, computer vision, "
    "reasoning, robotics, and related CS.AI subjects)"
)


def guardrail_messages(question: str) -> List[Dict[str, str]]:
    system = (
        "You are a domain classifier for a retrieval system whose knowledge base "
        f"contains {DOMAIN}. Rate how likely the user's question can be answered "
        "from that knowledge base on a scale of 0 to 100, where 100 means clearly "
        "on-topic and 0 means clearly unrelated (e.g. cooking, sports, personal chit-chat). "
        "Respond with ONLY the integer score, nothing else."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\nScore (0-100):"},
    ]


def grade_messages(query: str, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    numbered = "\n\n".join(
        f"[{i}] {c.get('title', '')}\n{(c.get('chunk_text') or '')[:600]}"
        for i, c in enumerate(chunks)
    )
    system = (
        "You grade whether retrieved document snippets are relevant to answering a "
        "question. A snippet is relevant if it contains information that helps answer "
        "the question. Return ONLY a comma-separated list of the indices (numbers in "
        "brackets) of the relevant snippets. If none are relevant, return 'none'."
    )
    user = f"Question: {query}\n\nSnippets:\n{numbered}\n\nRelevant indices:"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def rewrite_messages(question: str, previous_query: str) -> List[Dict[str, str]]:
    system = (
        "You improve search queries for a research-paper retrieval system. The previous "
        "query returned no relevant results. Rewrite it to be more specific and use "
        "terminology likely to appear in academic CS/AI papers. Respond with ONLY the "
        "rewritten query, no quotes or explanation."
    )
    user = (
        f"Original question: {question}\n"
        f"Previous query that failed: {previous_query}\n"
        "Rewritten query:"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


OUT_OF_SCOPE_ANSWER = (
    "I'm a research assistant for arXiv computer-science and AI papers, so I can only "
    "answer questions about topics like machine learning, LLMs, agents, NLP, computer "
    "vision, and related CS.AI research. Your question appears to fall outside that "
    "domain. Try rephrasing it around a CS/AI research topic."
)
