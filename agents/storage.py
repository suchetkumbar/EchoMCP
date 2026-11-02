"""Durable Postgres/pgvector and Redis adapters, selected only when configured."""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import UUID

from agents.config import Settings


class PostgresVectorStore:
    """Transactional document/chunk repository backed by PostgreSQL and pgvector."""

    def __init__(self, database_url: str, dimension: int = 1024) -> None:
        try:
            from sqlalchemy import create_engine, text
        except ImportError as exc:
            raise RuntimeError("Postgres storage requires SQLAlchemy and psycopg; install requirements.txt") from exc
        if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError("DATABASE_URL must use PostgreSQL with the psycopg driver")
        if database_url.startswith("postgresql://"):
            database_url = "postgresql+psycopg://" + database_url.removeprefix("postgresql://")
        if not 1 <= dimension <= 2000:
            raise ValueError("pgvector HNSW dimension must be between 1 and 2000")
        self.dimension = dimension
        self._create_engine = create_engine
        self._text = text
        self.engine = create_engine(database_url, pool_pre_ping=True, pool_size=5, max_overflow=10,
                                   pool_timeout=5, pool_recycle=1800)
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        with self.engine.begin() as conn:
            conn.execute(self._text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(self._text("""
                CREATE TABLE IF NOT EXISTS echomcp_documents (
                    document_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """))
            conn.execute(self._text(f"""
                CREATE TABLE IF NOT EXISTS echomcp_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES echomcp_documents(document_id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    embedding VECTOR({self.dimension}) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """))
            conn.execute(self._text("CREATE INDEX IF NOT EXISTS echomcp_chunks_document_idx ON echomcp_chunks(document_id)"))
            conn.execute(self._text("CREATE INDEX IF NOT EXISTS echomcp_chunks_embedding_hnsw ON echomcp_chunks USING hnsw (embedding vector_cosine_ops)"))
            conn.execute(self._text("CREATE INDEX IF NOT EXISTS echomcp_chunks_fts_gin ON echomcp_chunks USING gin (to_tsvector('simple', content))"))
            actual_type = conn.execute(self._text("""
                SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a
                JOIN pg_class c ON c.oid=a.attrelid
                WHERE c.relname='echomcp_chunks' AND a.attname='embedding' AND NOT a.attisdropped
            """)).scalar_one()
            match = re.fullmatch(r"vector\((\d+)\)", actual_type or "")
            if not match or int(match.group(1)) != self.dimension:
                raise RuntimeError("database embedding dimension differs from configured provider; migrate/reindex embeddings")
        self._initialized = True

    def _vector(self, embedding: Sequence[float]) -> str:
        if len(embedding) != self.dimension or not all(math.isfinite(float(value)) for value in embedding):
            raise ValueError(f"embedding must contain {self.dimension} finite values")
        return "[" + ",".join(repr(float(value)) for value in embedding) + "]"

    def replace_document(self, document_id: str, content: str, metadata: dict[str, Any],
                         chunks: Sequence[tuple[str, str, Sequence[float], dict[str, Any]]]) -> None:
        self.initialize()
        document_metadata = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        rows = [{"chunk_id": chunk_id, "document_id": document_id, "content": chunk_text,
                 "embedding": self._vector(vector), "metadata": json.dumps(chunk_meta, ensure_ascii=False, allow_nan=False)}
                for chunk_id, chunk_text, vector, chunk_meta in chunks]
        with self.engine.begin() as conn:
            conn.execute(self._text("""
                INSERT INTO echomcp_documents(document_id, content, metadata, updated_at)
                VALUES (:id, :content, CAST(:metadata AS jsonb), now())
                ON CONFLICT(document_id) DO UPDATE SET content=EXCLUDED.content,
                    metadata=EXCLUDED.metadata, updated_at=now()
            """), {"id": document_id, "content": content, "metadata": document_metadata})
            conn.execute(self._text("DELETE FROM echomcp_chunks WHERE document_id=:id"), {"id": document_id})
            if rows:
                conn.execute(self._text("""
                    INSERT INTO echomcp_chunks(chunk_id, document_id, content, embedding, metadata)
                    VALUES (:chunk_id, :document_id, :content, CAST(:embedding AS vector), CAST(:metadata AS jsonb))
                """), rows)

    def add(self, document_id: str, text_value: str, embedding: list[float], *, chunk_id: str | None = None,
            metadata: dict[str, Any] | None = None) -> None:
        self.initialize()
        chunk_id = chunk_id or document_id
        with self.engine.begin() as conn:
            conn.execute(self._text("""
                INSERT INTO echomcp_documents(document_id, content) VALUES (:id, :content)
                ON CONFLICT(document_id) DO NOTHING
            """), {"id": document_id, "content": text_value})
            conn.execute(self._text("""
                INSERT INTO echomcp_chunks(chunk_id, document_id, content, embedding, metadata)
                VALUES (:chunk_id, :document_id, :content, CAST(:embedding AS vector), CAST(:metadata AS jsonb))
                ON CONFLICT(chunk_id) DO UPDATE SET content=EXCLUDED.content,
                    embedding=EXCLUDED.embedding, metadata=EXCLUDED.metadata
            """), {"chunk_id": chunk_id, "document_id": document_id, "content": text_value,
                  "embedding": self._vector(embedding), "metadata": json.dumps(metadata or {}, allow_nan=False)})

    def delete_document(self, document_id: str) -> None:
        self.initialize()
        with self.engine.begin() as conn:
            conn.execute(self._text("DELETE FROM echomcp_documents WHERE document_id=:id"), {"id": document_id})

    def search(self, embedding: list[float], top_k: int = 3) -> list[dict[str, Any]]:
        if not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if not embedding or not any(float(value) != 0 for value in embedding):
            return []
        self.initialize()
        query_vector = self._vector(embedding)
        with self.engine.connect() as conn:
            rows = conn.execute(self._text("""
                SELECT document_id, chunk_id, content, metadata,
                       1 - (embedding <=> CAST(:embedding AS vector)) AS score
                FROM echomcp_chunks
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :top_k
            """), {"embedding": query_vector, "top_k": top_k}).mappings().all()
        return [{"document_id": row["document_id"], "chunk_id": row["chunk_id"], "content": row["content"],
                 "metadata": row["metadata"], "score": float(row["score"])} for row in rows]

    def list_documents(self) -> list[dict[str, Any]]:
        self.initialize()
        with self.engine.connect() as conn:
            rows = conn.execute(self._text("""
                SELECT document_id, content, metadata, updated_at
                FROM echomcp_documents ORDER BY updated_at DESC, document_id
            """)).mappings().all()
        return [{"id": row["document_id"], "content": row["content"], **row["metadata"],
                 "updated_at": row["updated_at"].isoformat()} for row in rows]

    def keyword_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        self.initialize()
        with self.engine.connect() as conn:
            rows = conn.execute(self._text("""
                SELECT document_id, chunk_id, content, metadata,
                       ts_rank(to_tsvector('simple', content), plainto_tsquery('simple', :query)) AS score
                FROM echomcp_chunks
                WHERE to_tsvector('simple', content) @@ plainto_tsquery('simple', :query)
                ORDER BY score DESC, chunk_id
                LIMIT :top_k
            """), {"query": query, "top_k": top_k}).mappings().all()
        return [{"document_id": row["document_id"], "chunk_id": row["chunk_id"], "content": row["content"],
                 "metadata": row["metadata"], "score": float(row["score"])} for row in rows]

    def health(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(self._text("SELECT 1"))
            return True
        except Exception:
            return False


class RedisSessionMemory:
    """Short-lived, bounded conversation memory with atomic append and TTL."""

    def __init__(self, redis_url: str, *, ttl_seconds: int = 1800, max_turns: int = 20) -> None:
        import redis
        self.client = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2,
                                           health_check_interval=30, decode_responses=True,
                                           max_connections=20, retry_on_timeout=False)
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns

    @staticmethod
    def _key(conversation_id: str) -> str:
        return f"echomcp:session:{UUID(conversation_id)}"

    def get_recent(self, conversation_id: str, limit: int = 10) -> list[dict[str, Any]]:
        if not 1 <= limit <= 40:
            raise ValueError("limit must be between 1 and 40")
        items = self.client.lrange(self._key(conversation_id), -limit, -1)
        output = []
        for item in items:
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict) and parsed.get("role") in {"user", "assistant"}:
                    output.append(parsed)
            except (TypeError, json.JSONDecodeError):
                continue
        return output

    def append_turn(self, conversation_id: str, question: str, answer: str) -> None:
        key = self._key(conversation_id)
        now = datetime.now(timezone.utc).isoformat()
        records = [json.dumps({"role": "user", "content": question, "timestamp": now}, ensure_ascii=False),
                   json.dumps({"role": "assistant", "content": answer, "timestamp": now}, ensure_ascii=False)]
        with self.client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *records)
            pipe.ltrim(key, -2 * self.max_turns, -1)
            pipe.expire(key, self.ttl_seconds)
            pipe.execute()

    def health(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:
            return False


def build_vector_store(settings: Settings, dimension: int | None = None):
    backend = settings.vector_store.lower()
    if backend == "memory":
        from agents.rag import InMemoryVectorStore
        return InMemoryVectorStore()
    if backend in {"postgres", "pgvector"}:
        return PostgresVectorStore(settings.database_url, dimension or settings.embedding_dimension)
    raise ValueError(f"unsupported vector store: {backend}")


def build_session_memory(settings: Settings) -> RedisSessionMemory | None:
    if not settings.redis_url:
        return None
    return RedisSessionMemory(settings.redis_url)
