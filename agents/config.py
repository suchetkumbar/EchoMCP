from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "fake")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "fake")
    vector_store: str = os.getenv("VECTOR_STORE", "memory")
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
        if self.environment.lower() == "production" and not self.mcp_bearer_token:
            raise ValueError("MCP_BEARER_TOKEN is required when APP_ENV=production")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
