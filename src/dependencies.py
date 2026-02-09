from functools import lru_cache
from typing import Annotated, Generator

from fastapi import Depends, Request

# Week 1: Removed API key authentication for simplicity
from sqlalchemy.orm import Session
from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.services.arxiv.client import ArxivClient
from src.services.pdf_parser.parser import PDFParserService
from src.services.opensearch.client import OpenSearchClient
from src.services.embeddings.jina_client import JinaEmbeddingsClient

# Week 1: Simplified - no API key authentication needed for local learning

@lru_cache
def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def get_request_settings(request: Request) -> Settings:
    """Get settings from the request state."""
    return request.app.state.settings


def get_database(request: Request) -> BaseDatabase:
    """Get database from the request state."""
    return request.app.state.database


def get_db_session(database: Annotated[BaseDatabase, Depends(get_database)]) -> Generator[Session, None, None]:
    """Get database session dependency."""
    with database.get_session() as session:
        yield session


# Week 2+: PDF parser service (not implemented in Week 1)
def get_arxiv_client(request: Request) -> ArxivClient:
    """Get ArXiv client from app state."""
    return request.app.state.arxiv_client


def get_pdf_parser(request: Request) -> PDFParserService:
    """Get PDF parser service from app state."""
    return request.app.state.pdf_parser


# Week 1: OpenSearch service (placeholder - full implementation in Week 3+)
def get_opensearch_service(request: Request) -> OpenSearchClient:
    """Get OpenSearch service from app state (Week 3+ - placeholder for Week 1)."""
    return request.app.state.opensearch_client


# Phase 3: LLM service (skeleton only)
def get_llm_service(request: Request):
    """Get LLM service from app state (Phase 3 - not implemented yet)."""
    # Phase 3: Will return actual LLM service
    return None

def get_embeddings_client(request: Request) -> JinaEmbeddingsClient:
    """Get embeddings client from app state."""
    return request.app.state.embeddings_client


# Dependency type aliases for better type hints
SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
PDFParserServiceDep = Annotated[object, Depends(get_pdf_parser)]
ArxivClientDep = Annotated[ArxivClient, Depends(get_arxiv_client)]
OpenSearchServiceDep = Annotated[object, Depends(get_opensearch_service)]
EmbeddingsServiceDep = Annotated[JinaEmbeddingsClient, Depends(get_embeddings_client)]

# Phase 3: LLM service dependency (not used in Phase 2)
# LLMServiceDep = Annotated[object, Depends(get_llm_service)]
