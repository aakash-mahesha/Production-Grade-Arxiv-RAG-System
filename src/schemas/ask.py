from typing import List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    """Request schema for asking questions about papers."""

    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        validation_alias=AliasChoices("question", "query"),
        description="Question to ask about arXiv papers",
    )
    use_hybrid: bool = Field(default=True, description="Use hybrid BM25 + vector search")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    categories: Optional[List[str]] = Field(default=None, description="Filter by arXiv categories")
    model: Optional[str] = Field(default=None, description="LLM model override (Ollama or OpenRouter)")


class PaperSource(BaseModel):
    """Schema for paper source information in responses."""

    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list, description="List of paper authors")
    abstract_preview: str = Field(default="", description="Preview of paper abstract")


class AskResponse(BaseModel):
    """Response schema for question answering endpoints."""

    query: str = Field(..., description="Original question")
    answer: str = Field(..., description="Answer to the question")
    sources: List[str] = Field(default_factory=list, description="Source PDF URLs used for the answer")
    chunks_used: int = Field(default=0, description="Number of chunks used as context")
    search_mode: str = Field(default="hybrid", description="Search mode used: bm25, vector, or hybrid")
