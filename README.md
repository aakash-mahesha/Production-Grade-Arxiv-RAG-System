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
│   ├── paper.py           # Paper schemas (Create, Response)
│   ├── health.py          # Health check schemas
│   └── ask.py             # Q&A schemas
│
├── repositories/          # Data access layer
│   └── paper.py           # Paper repository (CRUD operations)
│
├── routers/               # API route handlers
│   ├── ping.py            # Health endpoints
│   ├── papers.py          # Paper CRUD endpoints
│   └── ask.py             # RAG query endpoint
│
└── services/              # Business logic services
    ├── ollama/            # LLM client
    ├── opensearch/        # Vector search service
    └── pdf_parser/        # Document processing
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

- [ ] **Phase 2**: PDF parsing with Docling
- [ ] **Phase 3**: OpenSearch vector indexing
- [ ] **Phase 4**: LLM integration for RAG queries
- [ ] **Phase 5**: Airflow DAGs for arXiv paper ingestion
- [ ] **Phase 6**: Production deployment with Kubernetes

## License

MIT License - see [LICENSE](License) for details.

---

Built with modern Python practices for production-grade applications.
