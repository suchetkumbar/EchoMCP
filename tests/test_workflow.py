from agents.workflow import AgentWorkflow


def test_workflow_uses_prompt_and_returns_result():
    workflow = AgentWorkflow()
    workflow.ingest_documents([
        {
            "id": "doc-2",
            "title": "History",
            "content": "The Eiffel Tower is in Paris.",
        }
    ])

    result = workflow.run("Where is the Eiffel Tower?")

    assert result["answer"]
    assert result["prompt_version"] == "v1"
    assert "EchoMCP" in result["system_prompt"]
