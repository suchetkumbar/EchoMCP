from fastapi import FastAPI
from pydantic import BaseModel

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


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query(req: QueryRequest) -> dict[str, object]:
    return pipeline.query(req.question)


@app.get("/documents")
def list_documents() -> list[dict]:
    return pipeline.documents


# include MCP router
from .mcp import router as mcp_router

app.include_router(mcp_router)
