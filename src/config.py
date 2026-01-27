from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class DefaultSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        frozen=True,
        env_nest_delimeter="__")
    
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

    @field_validator("ollama_models", mode = "before")
    @classmethod
    def parse_ollama_models(cls, value: str) -> list[str]:
        if isinstance(value, str):
            return [model.strip() for model in value.split(",") if model.strip()]
        return value

def get_settings() -> Settings:
    return Settings()