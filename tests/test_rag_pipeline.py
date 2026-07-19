from agents.rag import FakeEmbeddingProvider, FakeLLMProvider, InMemoryVectorStore, RAGPipeline


def test_pipeline_returns_relevant_answer_and_citations():
    pipeline = RAGPipeline(
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=FakeLLMProvider(),
        vector_store=InMemoryVectorStore(),
    )

    pipeline.ingest_documents(
        [
            {
                "id": "doc-1",
                "title": "Geography",
                "content": "The capital of France is Paris.",
            }
        ]
    )

    result = pipeline.query("What is the capital of France?")

    assert result["answer"].startswith("Based on")
    assert result["citations"][0]["document_id"] == "doc-1"
    assert result["confidence"] >= 0.5
