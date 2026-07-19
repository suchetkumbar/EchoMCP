from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from agents.config import Settings, get_settings
from agents.rag import FakeEmbeddingProvider, FakeLLMProvider, InMemoryVectorStore, RAGPipeline


@dataclass
class AgentWorkflow:
    settings: Settings | None = None

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()
        self.pipeline = RAGPipeline(
            embedding_provider=FakeEmbeddingProvider(),
            llm_provider=FakeLLMProvider(),
            vector_store=InMemoryVectorStore(),
        )
        self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).resolve().parents[1] / "prompts" / f"system_prompt_{self.settings.prompt_version}.txt"
        return prompt_path.read_text(encoding="utf-8")

    def ingest_documents(self, documents: List[Dict[str, Any]]) -> None:
        self.pipeline.ingest_documents(documents)

    def run(self, question: str) -> Dict[str, Any]:
        result = self.pipeline.query(question)
        result["prompt_version"] = self.settings.prompt_version
        result["system_prompt"] = self.system_prompt
        return result
