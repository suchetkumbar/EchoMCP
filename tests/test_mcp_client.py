from fastapi.testclient import TestClient

from api.main import app
from agents.workflow import AgentWorkflow
from agents.mcp_client import MCPClient


def test_mcp_client_with_testclient():
    workflow = AgentWorkflow()
    workflow.ingest_documents([
        {"id": "doc-client", "title": "Park", "content": "Central Park is in New York City."}
    ])

    tc = TestClient(app)
    client = MCPClient(test_client=tc)
    resp = client.semantic_search("Where is Central Park?", top_k=2)
    assert "results" in resp
    assert resp["results"]
    assert resp["results"][0]["document_id"] == "doc-client"
