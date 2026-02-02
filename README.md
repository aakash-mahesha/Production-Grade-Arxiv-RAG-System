# Production-Grade Arxiv RAG System

A production-ready Retrieval-Augmented Generation (RAG) system built with modern Python best practices, designed to demonstrate enterprise-grade software engineering patterns.

## Overview

This project implements a full-stack RAG pipeline for processing and querying academic papers from arXiv. It showcases production-grade architecture patterns including clean architecture, dependency injection, factory patterns, and comprehensive containerization.

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | FastAPI, Python 3.12, Pydantic v2 |
| **Database** | PostgreSQL 16, SQLAlchemy 2.0 |
| **Vector Search** | OpenSearch 2.19 |
| **LLM Integration** | Ollama (Llama 3.2) |
| **Orchestration** | Apache Airflow 3.0 |
| **Infrastructure** | Docker Compose, UV Package Manager |
| **Testing** | Pytest, TestContainers |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Client                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Routers   │  │   Schemas   │  │Dependencies │  │ Middleware  │    │
│  │ /health     │  │ Pydantic    │  │ DI Container│  │ Logging     │    │
│  │ /papers     │  │ Validation  │  │ Sessions    │  │ Error Handle│    │
│  │ /ask        │  │             │  │             │  │             │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
        ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
        │  PostgreSQL   │  │  OpenSearch   │  │    Ollama     │
        │  (Metadata)   │  │  (Vectors)    │  │    (LLM)      │
        └───────────────┘  └───────────────┘  └───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Apache Airflow           │
                    │   (Data Pipeline Orchestration)│
                    └───────────────────────────────┘
```

## Key Features

### Implemented

- **RESTful API** with FastAPI and automatic OpenAPI documentation
- **Health Check Endpoints** with service dependency monitoring
- **Database Layer** with Factory Pattern and Repository Pattern
- **Configuration Management** using Pydantic Settings with environment variable support
- **Dependency Injection** for clean, testable code
- **Docker Compose** orchestration with health checks for all services
- **Airflow Integration** for data pipeline orchestration
- **arXiv Paper Ingestion** - Automated fetching of papers from arXiv API
- **PDF Parsing with Docling** - Extract structured content from academic PDFs
- **Sequential Pipeline Processing** - Memory-efficient one-at-a-time PDF processing
- **Immediate DB Updates** - Papers saved to database as soon as parsed (no batch waiting)
- **OpenSearch Integration** - Full-text BM25 search across paper content
- **Search API** - REST endpoint for searching papers by keywords

### Design Patterns Used

| Pattern | Implementation |
|---------|---------------|
| **Factory Pattern** | `db/factory.py` - Database instance creation |
| **Repository Pattern** | `repositories/paper.py` - Data access abstraction |
| **Dependency Injection** | `dependencies.py` - FastAPI DI with type hints |
| **Abstract Base Classes** | `db/interfaces/base.py` - Database contracts |
| **Settings Pattern** | `config.py` - Centralized configuration |

## Project Structure

```
src/
├── config.py              # Pydantic Settings configuration
├── main.py                # FastAPI application entry point
├── dependencies.py        # Dependency injection definitions
├── exceptions.py          # Custom exception classes
├── middleware.py          # Request/response middleware
│
├── db/                    # Database layer
│   ├── factory.py         # Database factory
│   └── interfaces/
│       ├── base.py        # Abstract base classes
│       └── postgresql.py  # PostgreSQL implementation
│
├── models/                # SQLAlchemy ORM models
│   └── paper.py           # Paper entity model
│
├── schemas/               # Pydantic request/response schemas
│   ├── api/               # API response schemas
│   │   └── health.py      # Health check schemas
│   ├── arxiv/             # arXiv data schemas
│   │   └── paper.py       # ArxivPaper, PaperCreate schemas
│   └── pdf_parser/        # PDF parsing schemas
│       └── models.py      # PdfContent, ParsedPaper schemas
│
├── repositories/          # Data access layer
│   └── paper.py           # Paper repository (CRUD + upsert)
│
├── routers/               # API route handlers
│   ├── ping.py            # Health endpoints
│   ├── papers.py          # Paper CRUD endpoints
│   ├── search.py          # Search endpoints (BM25)
│   └── ask.py             # RAG query endpoint
│
└── services/              # Business logic services
    ├── arxiv/             # arXiv API client
    │   └── client.py      # Fetch papers, download PDFs
    ├── metadata_fetcher.py # Pipeline orchestrator
    ├── ollama/            # LLM client
    ├── opensearch/        # Full-text search service
    │   ├── client.py      # OpenSearch client
    │   ├── factory.py     # Client factory
    │   ├── query_builder.py # BM25 query builder
    │   └── index_config.py # Index mappings
    └── pdf_parser/        # Document processing
        ├── docling.py     # Docling PDF parser
        └── parser.py      # Parser service wrapper

scripts/
├── manual_ingest.py       # Manual paper ingestion script
└── test_search.py         # Test OpenSearch search

airflow/
├── Dockerfile             # Custom Airflow image with dependencies
├── entrypoint.sh          # Container entrypoint script
├── requirements-airflow.txt # Airflow Python dependencies
└── dags/
    ├── arxiv_paper_ingestion.py  # Main DAG definition
    └── arxiv_ingestion/
        └── tasks.py       # Task functions for paper processing
```

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Python 3.12+
- [UV Package Manager](https://github.com/astral-sh/uv)

### Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd prod-grade-rag-system

# Start all services
make start

# Check service health
make health

# View logs
make logs
```

### Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| **RAG API** | http://localhost:8000 | FastAPI application |
| **API Docs** | http://localhost:8000/docs | Swagger UI |
| **OpenSearch** | http://localhost:9200 | Search engine |
| **OpenSearch Dashboards** | http://localhost:5601 | Search visualization |
| **Airflow** | http://localhost:8080 | Workflow orchestration |
| **Ollama** | http://localhost:11434 | LLM inference |

### Airflow Credentials

Credentials are stored in `airflow/simple_auth_manager_passwords.json.generated`:
```json
{"admin": "admin"}
```

## Available Make Commands

```bash
make help        # Show all available commands
make start       # Start all services
make stop        # Stop all services
make restart     # Restart all services
make status      # Show service status
make logs        # Show service logs
make health      # Check all services health
make setup       # Install Python dependencies
make format      # Format code with Ruff
make lint        # Lint and type check
make test        # Run tests
make test-cov    # Run tests with coverage
make clean       # Clean up everything
```

## API Endpoints

### Health

```bash
# Simple ping
GET /ping
# Response: {"status": "ok", "message": "pong"}

# Comprehensive health check
GET /health
# Response: Service health with database and Ollama status
```

### Papers (CRUD)

```bash
# List papers
GET /papers

# Get paper by ID
GET /papers/{paper_id}

# Create paper
POST /papers
```

### Search (BM25)

```bash
# Search papers by keyword
POST /search
{
  "query": "transformer attention mechanism",
  "size": 10
}
# Response: Matching papers ranked by BM25 score
```

### Ask (RAG Query)

```bash
# Query the RAG system
POST /ask
{
  "question": "What is transformer architecture?"
}
```

## Configuration

Environment variables are managed via Pydantic Settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_DATABASE_URL` | `postgresql+psycopg2://...` | Database connection |
| `OPENSEARCH_HOST` | `http://opensearch:9200` | OpenSearch endpoint |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `ENVIRONMENT` | `development` | Environment name |

## Development

### Local Setup

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Run locally (requires services running)
uv run uvicorn src.main:app --reload
```

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Run tests
make test
```

## Roadmap

- [x] **Phase 1**: FastAPI setup with PostgreSQL, health checks, CRUD endpoints
- [x] **Phase 2**: PDF parsing with Docling (optimized for academic papers)
- [x] **Phase 3**: Airflow DAGs for arXiv paper ingestion pipeline
- [x] **Phase 4**: OpenSearch BM25 search integration
- [ ] **Phase 5**: LLM integration for RAG queries
- [ ] **Phase 6**: Production deployment with Kubernetes

## License

MIT License - see [LICENSE](License) for details.

---

Built with modern Python practices for production-grade applications.
