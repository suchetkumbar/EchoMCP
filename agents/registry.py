from __future__ import annotations

from typing import Any, Optional

_VECTOR_STORE: Optional[Any] = None


def set_vector_store(store: Any) -> None:
    global _VECTOR_STORE
    _VECTOR_STORE = store


def get_vector_store() -> Optional[Any]:
    return _VECTOR_STORE
