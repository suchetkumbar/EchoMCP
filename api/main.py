from fastapi import FastAPI

from agents.rag import FakeEmbeddingProvider, FakeLLMProvider, InMemoryVectorStore, RAGPipeline

app = FastAPI(title="EchoMCP API")

pipeline = RAGPipeline(
    embedding_provider=FakeEmbeddingProvider(),
    llm_provider=FakeLLMProvider(),
    vector_store=InMemoryVectorStore(),
)

pipeline.ingest_documents([
    {
        "id": "doc-1",
        "title": "Geography",
        "content": "The capital of France is Paris.",
    }
])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(question: str) -> dict[str, object]:
    return pipeline.query(question)
