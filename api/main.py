from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field

from agents.rag import FakeEmbeddingProvider, FakeLLMProvider, InMemoryVectorStore, RAGPipeline
from .mcp import require_mcp_auth

app = FastAPI(title="EchoMCP API", version="0.1.0")


@app.middleware("http")
async def security_and_body_limits(request: Request, call_next):
    """Apply inexpensive edge protections; deploy behind TLS and a trusted proxy."""
    from agents.config import get_settings
    limit = get_settings().max_request_bytes
    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > limit:
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "request too large"}, status_code=413)
        except ValueError:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "invalid content length"}, status_code=400)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/mcp") else "no-cache"
    return response

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
    model_config = {"str_strip_whitespace": True, "extra": "forbid"}
    question: str = Field(min_length=1, max_length=2000)


@app.post("/query", dependencies=[Depends(require_mcp_auth)])
def query(req: QueryRequest) -> dict[str, object]:
    return pipeline.query(req.question)


@app.get("/documents", dependencies=[Depends(require_mcp_auth)])
def list_documents() -> list[dict]:
    return pipeline.documents


# include MCP router
from .mcp import router as mcp_router

app.include_router(mcp_router)
