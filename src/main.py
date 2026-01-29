import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from src.config import get_settings
from src.db.factory import make_database
from src.services.arxiv.factory import make_arxiv_client
from src.services.pdf_parser.factory import make_pdf_parser_service

# Week 1: No complex middleware needed
from src.routers import ask, papers, ping

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Week 1: Simplified lifespan for learning purposes."""
    logger.info("Starting RAG API...")

    # Initialize settings and database (Week 1 essentials)
    settings = get_settings()
    app.state.settings = settings

    database = make_database()
    app.state.database = database
    logger.info("Database connected")

    # Placeholders for future weeks
    app.state.arxiv_client = make_arxiv_client()
    app.state.pdf_parser = make_pdf_parser_service()
    app.state.llm_service = None

    logger.info("Services initialized: arXiv API client, PDF parser")

    logger.info("API ready")
    yield

    # Cleanup
    database.teardown()
    logger.info("API shutdown complete")


app = FastAPI(
    title="arXiv Paper Curator API",
    description="Personal arXiv CS.AI paper curator with RAG capabilities",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

# Include routers
app.include_router(ping.router, prefix="/api/v1")
app.include_router(papers.router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, host="0.0.0.0")
