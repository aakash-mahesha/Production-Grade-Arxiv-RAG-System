from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"
class DefaultSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        frozen=True,
        env_nest_delimeter="__")

class ArxivSettings(DefaultSettings):
    """arXiv API client settings."""

    base_url: str = "https://export.arxiv.org/api/query"
    namespaces: dict = Field(
        default={
            "atom": "http://www.w3.org/2005/Atom",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
    )
    pdf_cache_dir: str = "./data/arxiv_pdfs"
    rate_limit_delay: float = 3.0  # seconds between requests
    timeout_seconds: int = 30
    max_results: int = 100
    search_category: str = "cs.AI"  # Default category to search
    max_concurrent_downloads: int = 5  # Max parallel PDF downloads
    max_concurrent_parsing: int = 1  # Max parallel PDF parsing (keep low for memory)

    
class PDFParserSettings(DefaultSettings):
    """PDF parser service settings."""

    max_pages: int = 25  # Reduced for faster processing
    max_file_size_mb: int = 20
    do_ocr: bool = False
    do_table_structure: bool = True

class ChunkingSettings(DefaultSettings):
    """Chunking settings for text indexing."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="CHUNKING__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    chunk_size: int = 600
    overlap_size: int = 100
    min_chunk_size: int = 100

class OpenSearchSettings(DefaultSettings):
    """Opensearch settings"""
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="OPENSEARCH__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "http://opensearch:9200"  # Docker service name for container-to-container
    index_name: str = "arxiv-papers"
    chunk_index_suffix: str = "chunks"
    max_text_size: int = 1000000  # Max chars of raw_text to index

    #Vector search settings
    vector_dimension: int = 1024 # Jina embeddings dimension
    vector_space_type: str = "cosinesiml" # cosinesimil, l2, innerproduct

    # Hybrid search settings
    rrf_pipeline_name: str = "hybrid-rrf-pipeline"
    hybrid_search_size_multiplier: int = 2 # Get k* multiplier for better recall


class Settings(DefaultSettings):
    app_version: str = "0.0.1"
    debug: bool = True
    environment: str = "development"
    service_name: str = "rag-api"

    postgres_database_url: str = Field(default="postgresql+psycopg2://rag_user:rag_password@postgres:5432/rag_db")
    postgres_echo_sql: bool = False
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 0

    llm_provider: str = Field(default="ollama", description="LLM provider: ollama or openrouter")

    ollama_host: str = Field(default="http://ollama:11434")
    ollama_model: str = Field(default="llama3.2")
    ollama_timeout: int = 300

    openrouter_api_key: str = ""
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_model: str = Field(default="meta-llama/llama-3.2-3b-instruct")
    openrouter_timeout: int = 300
    openrouter_app_name: str = Field(default="arXiv Paper Curator")
    openrouter_app_url: str = Field(default="http://localhost:8000")

    jina_api_key: str = ""
    jina_batch_size: int = 16
    jina_request_delay: float = 0.7

    # Langfuse observability (Week 6). Tracing is a no-op unless both keys are set.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = Field(default="https://us.cloud.langfuse.com")

    # RAGAS evaluation (Week 6). Judge LLM runs through OpenRouter (OpenAI-compatible),
    # so any OpenRouter model id works: openai/gpt-4o-mini, anthropic/claude-sonnet-5, etc.
    # Reuses openrouter_api_key + openrouter_base_url above. Embeddings reuse jina_api_key.
    ragas_judge_model: str = Field(default="openai/gpt-4o-mini")
    ragas_embedding_model: str = Field(default="jina-embeddings-v3")

    # Redis exact-match response cache (Week 6). Disabled when redis_url is blank.
    redis_url: str = ""
    cache_ttl_seconds: int = 3600

    # Agentic RAG (Week 7). Controls the LangGraph decision workflow.
    agent_top_k: int = 5
    agent_max_retrieval_attempts: int = 2  # how many times the agent may rewrite + retry
    agent_guardrail_threshold: int = 50  # 0-100 domain-relevance cutoff for in-scope
    agent_grade_min_relevant: int = 1  # min relevant chunks needed to answer

    @property
    def langfuse_enabled(self) -> bool:
        """Tracing is active only when both Langfuse keys are configured."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def cache_enabled(self) -> bool:
        """Response caching is active only when a Redis URL is configured."""
        return bool(self.redis_url)

    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)

def get_settings() -> Settings:
    return Settings()