from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "fake")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "fake")
    vector_store: str = os.getenv("VECTOR_STORE", "memory")
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.72"))
    prompt_version: str = os.getenv("PROMPT_VERSION", "v1")


def get_settings() -> Settings:
    return Settings()
