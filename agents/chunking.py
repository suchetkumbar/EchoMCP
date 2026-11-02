"""Deterministic bounded chunking for document ingestion."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    start: int
    end: int


def chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 160) -> list[TextChunk]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be non-empty")
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")
    normalized = " ".join(text.split())
    chunks: list[TextChunk] = []
    start = 0
    while start < len(normalized):
        hard_end = min(len(normalized), start + chunk_size)
        end = hard_end
        if hard_end < len(normalized):
            boundary = normalized.rfind(" ", start + chunk_size // 2, hard_end)
            if boundary > start:
                end = boundary
        value = normalized[start:end].strip()
        if value:
            chunks.append(TextChunk(len(chunks), value, start, end))
        if end >= len(normalized):
            break
        next_start = max(start + 1, end - overlap)
        while next_start < end and normalized[next_start] == " ":
            next_start += 1
        start = next_start
    return chunks
