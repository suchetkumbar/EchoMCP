from __future__ import annotations

from typing import Any, Optional

_VECTOR_STORE: Optional[Any] = None
_EMBEDDING_PROVIDER: Optional[Any] = None


def set_vector_store(store: Any) -> None:
    global _VECTOR_STORE
    _VECTOR_STORE = store


def get_vector_store() -> Optional[Any]:
    return _VECTOR_STORE


def set_embedding_provider(provider: Any) -> None:
    global _EMBEDDING_PROVIDER
    _EMBEDDING_PROVIDER = provider


def get_embedding_provider() -> Optional[Any]:
    return _EMBEDDING_PROVIDER
