from fastapi.testclient import TestClient

from agents.workflow import AgentWorkflow
from api.main import app


def test_workflow_routes_time_sensitive_questions_to_web_fallback():
    workflow = AgentWorkflow()
    workflow.ingest_documents([
        {
            "id": "doc-3",
            "title": "Travel",
            "content": "Paris is a major European city.",
        }
    ])

    result = workflow.run("What is the latest news about Paris?")

    assert result["tool_plan"]["route"] == "web_fallback"
    assert "time-sensitive" in result["tool_plan"]["reason"]


def test_query_endpoint_accepts_json_body_and_lists_documents():
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"question": "What is the capital of France?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["citations"]

    docs_response = client.get("/documents")
    assert docs_response.status_code == 200
    docs = docs_response.json()
    assert any(item["id"] == "doc-1" for item in docs)
