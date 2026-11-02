from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "fake")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "fake")
    embedding_service_url: str = os.getenv("EMBEDDING_SERVICE_URL", "")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "160"))
    vector_store: str = os.getenv("VECTOR_STORE", "memory")
    database_url: str = os.getenv("DATABASE_URL", "")
    redis_url: str = os.getenv("REDIS_URL", "")
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.72"))
    prompt_version: str = os.getenv("PROMPT_VERSION", "v1")
    mcp_bearer_token: str = os.getenv("MCP_BEARER_TOKEN", "")
    max_request_bytes: int = int(os.getenv("MAX_REQUEST_BYTES", "1048576"))
    allowed_origins: str = os.getenv("ALLOWED_ORIGINS", "")
    environment: str = os.getenv("APP_ENV", "development")

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")
        if not 1024 <= self.max_request_bytes <= 16 * 1024 * 1024:
            raise ValueError("MAX_REQUEST_BYTES must be between 1 KiB and 16 MiB")
        if not 1 <= self.embedding_dimension <= 16384:
            raise ValueError("EMBEDDING_DIMENSION must be between 1 and 16384")
        if not 128 <= self.chunk_size <= 10000:
            raise ValueError("CHUNK_SIZE must be between 128 and 10000")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.embedding_provider.lower() in {"http", "service"} and not self.embedding_service_url:
            raise ValueError("EMBEDDING_SERVICE_URL is required for the http embedding provider")
        if self.vector_store.lower() in {"postgres", "pgvector"} and not self.database_url:
            raise ValueError("DATABASE_URL is required for the postgres vector store")
        if self.environment.lower() == "production" and not self.mcp_bearer_token:
            raise ValueError("MCP_BEARER_TOKEN is required when APP_ENV=production")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
