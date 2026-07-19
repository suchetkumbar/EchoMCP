from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> List[float]:
        ...


class LLMProvider(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class VectorStore(Protocol):
    def add(self, document_id: str, text: str, embedding: List[float]) -> None:
        ...

    def search(self, embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        ...


class FakeEmbeddingProvider:
    def embed(self, text: str) -> List[float]:
        return [1.0 if "france" in text.lower() else 0.0]


class FakeLLMProvider:
    def complete(self, prompt: str) -> str:
        return "Based on the retrieved context, the answer is Paris."


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._documents: List[Dict[str, Any]] = []

    def add(self, document_id: str, text: str, embedding: List[float]) -> None:
        self._documents.append({"document_id": document_id, "text": text, "embedding": embedding})

    def search(self, embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        scored = []
        for document in self._documents:
            similarity = sum(min(a, b) for a, b in zip(document["embedding"], embedding))
            scored.append((similarity, document))
        scored.sort(reverse=True)
        return [{"document_id": doc["document_id"], "content": doc["text"], "score": score} for score, doc in scored[:top_k]]


@dataclass
class RAGPipeline:
    embedding_provider: EmbeddingProvider
    llm_provider: LLMProvider
    vector_store: VectorStore
    documents: List[Dict[str, Any]] = field(default_factory=list)

    def ingest_documents(self, documents: List[Dict[str, Any]]) -> None:
        for document in documents:
            embedding = self.embedding_provider.embed(document["content"])
            self.vector_store.add(document["id"], document["content"], embedding)
            self.documents.append(document)

    def query(self, question: str) -> Dict[str, Any]:
        embedding = self.embedding_provider.embed(question)
        results = self.vector_store.search(embedding, top_k=3)
        prompt = f"Question: {question}\nContext: {results[0]['content'] if results else 'No relevant context'}"
        answer = self.llm_provider.complete(prompt)
        return {
            "answer": answer,
            "citations": [{"document_id": item["document_id"], "content": item["content"], "score": item["score"]} for item in results],
            "confidence": 0.8 if results else 0.2,
        }
