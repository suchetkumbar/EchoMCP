from fastapi.testclient import TestClient

from api.main import app
from agents.workflow import AgentWorkflow


def test_mcp_semantic_search_uses_registered_vector_store():
    workflow = AgentWorkflow()
    workflow.ingest_documents([
        {"id": "doc-x", "title": "Landmark", "content": "The Grand Canyon is in Arizona."}
    ])

    client = TestClient(app)
    resp = client.post("/mcp/semantic_search", json={"query": "Where is the Grand Canyon?", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert body["results"]
    assert body["results"][0]["document_id"] == "doc-x"
