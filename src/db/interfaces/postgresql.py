import logging
from contextlib import contextmanager
from typing import Generator

from pydantic import Field
from pydantic_settings import BaseSettings
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from src.db.interfaces.base import BaseDatabase

logger = logging.getLogger(__name__)

class PostgreSQLSettings(BaseSettings):
    database_url: str = Field(default="postgresql://rag_user:rag_password@localhost:5432/rag_db")
    echo_sql: bool = Field(default=False)
    pool_size: int = Field(default=20)
    max_overflow: int = Field(default=0)

    class Config:
        env_prefix = "POSTGRES_"


Base = declarative_base()

class PostgreSQLDatabase(BaseDatabase):
    def __init__(self, config: PostgreSQLSettings):
        self.config = config
        self.engine = None
        self.session_factory = None

    def startup(self) -> None:
        self.engine = create_engine(
            self.config.database_url,
            echo=self.config.echo_sql,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_pre_ping=True,
        )
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            
        Base.metadata.create_all(bind=self.engine)
        logger.info("PostgreSQL database initialized successfully")

    def teardown(self) -> None:
        if self.engine:
            self.engine.dispose()

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        if not self.session_factory:
            raise RuntimeError("Database not initialized. Call startup() first.")
        session = self.session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()