from typing import List
from pydantic import Field, field_validator
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
    max_text_size: int = 1000000  # Max chars of raw_text to index

class Settings(DefaultSettings):
    app_version: str = "0.0.1"
    debug: bool = True
    environment: str = "development"
    service_name: str = "rag-api"

    postgres_database_url: str = Field(default="postgresql+psycopg2://rag_user:rag_password@postgres:5432/rag_db")
    postgres_echo_sql: bool = False
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 0

    opensearch_host: str = Field(default="http://opensearch:9200")

    ollama_host: str = Field(default="http://localhost:11434")
    ollama_models: list[str] = Field(default=["llama3.2:1b"])
    ollama_default_model: str = Field(default="llama3.2:1b")
    ollama_timeout: int = 300

    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)

    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)

    @field_validator("ollama_models", mode = "before")
    @classmethod
    def parse_ollama_models(cls, value: str) -> list[str]:
        if isinstance(value, str):
            return [model.strip() for model in value.split(",") if model.strip()]
        return value

def get_settings() -> Settings:
    return Settings()