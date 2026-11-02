"""Embedding adapters. The HTTP client speaks only to a configured service URL."""
from __future__ import annotations

import math
from urllib.parse import urlsplit
from typing import Sequence

import requests

from agents.config import Settings
from agents.rag import EmbeddingProvider, FakeEmbeddingProvider


class EmbeddingServiceError(RuntimeError):
    """A safe, provider-neutral embedding service failure."""


class HttpEmbeddingProvider:
    """Call a private embedding service with strict response validation."""

    def __init__(self, base_url: str, dimension: int = 1024, api_key: str = "", timeout: tuple[float, float] = (1.0, 8.0), session: requests.Session | None = None) -> None:
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("embedding service URL must be an http(s) URL without credentials, query, or fragment")
        if not 1 <= dimension <= 16384:
            raise ValueError("embedding dimension must be between 1 and 16384")
        self.url = base_url.rstrip("/") + "/embed"
        self.dimension = dimension
        self.timeout = timeout
        self.session = session
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("embedding input must contain non-empty text")
        if any(len(text) > 10000 for text in texts):
            raise ValueError("embedding inputs may not exceed 10000 characters")
        output: list[list[float]] = []
        for offset in range(0, len(texts), 32):
            output.extend(self._embed_batch(texts[offset:offset + 32]))
        return output

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            post = self.session.post if self.session is not None else requests.post
            response = post(self.url, json={"texts": list(texts)}, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise EmbeddingServiceError("embedding service request failed") from exc
        vectors = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingServiceError("embedding service returned an invalid batch")
        output: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != self.dimension:
                raise EmbeddingServiceError("embedding dimension mismatch")
            try:
                clean = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise EmbeddingServiceError("embedding service returned non-numeric values") from exc
            if not all(math.isfinite(value) for value in clean):
                raise EmbeddingServiceError("embedding service returned non-finite values")
            output.append(clean)
        return output


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider = settings.embedding_provider.lower()
    if provider == "fake":
        return FakeEmbeddingProvider()
    if provider in {"http", "service"}:
        return HttpEmbeddingProvider(settings.embedding_service_url, settings.embedding_dimension, settings.embedding_api_key)
    raise ValueError(f"unsupported embedding provider: {provider}")
