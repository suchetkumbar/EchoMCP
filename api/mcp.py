from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any

from agents.registry import get_vector_store
from agents.rag import FakeEmbeddingProvider

router = APIRouter()


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/mcp/semantic_search")
def semantic_search(req: SemanticSearchRequest) -> Any:
    vs = get_vector_store()
    if vs is None:
        raise HTTPException(status_code=503, detail="vector store not available")
    embedding = FakeEmbeddingProvider().embed(req.query)
    results = vs.search(embedding, top_k=req.top_k)
    return {"results": results, "provider": "fake-memory"}
