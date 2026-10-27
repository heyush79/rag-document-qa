from __future__ import annotations

import json
import uuid
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import build_container
from app.main import create_app
from app.services.answer_generation import (
    ExtractiveAnswerGenerator,
    OpenAICompatibleAnswerGenerator,
)
from app.services.embeddings import HashingEmbedder
from app.services.vector_store import InMemoryVectorStore, SearchHit


def chat_completion(content: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ]
    }


class MockLLM:
    """Stands in for an OpenAI-compatible server: records requests and replies from a queue."""

    base_url = "http://mock-llm/v1"
    default_answer = "Mock answer. [S1]"

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._responses: list[httpx.Response] = []
        self.client = httpx.Client(transport=httpx.MockTransport(self._handle))

    def reply(self, content: str) -> None:
        self._responses.append(httpx.Response(200, json=chat_completion(content)))

    def fail(self, status_code: int = 500) -> None:
        self._responses.append(
            httpx.Response(status_code, json={"error": {"message": "mock failure"}})
        )

    def last_json(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)

    def generator(self) -> OpenAICompatibleAnswerGenerator:
        return OpenAICompatibleAnswerGenerator(
            base_url=self.base_url,
            model="mock-model",
            api_key="test-key",
            timeout_seconds=5,
            client=self.client,
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._responses:
            return self._responses.pop(0)
        return httpx.Response(200, json=chat_completion(self.default_answer))


@pytest.fixture
def settings(tmp_path):
    return Settings(
        sqlite_path=tmp_path / "rag.db",
        chroma_path=tmp_path / "chroma",
        chroma_collection=f"test_{uuid.uuid4().hex}",
        embedding_backend="hashing",
        embedding_dimension=64,
        chunk_size_words=20,
        chunk_overlap_words=5,
        min_relevance_score=0.0,
        answer_provider="extractive",
    )


@pytest.fixture
def embedder():
    return HashingEmbedder(dimension=64)


@pytest.fixture
def vector_store(embedder):
    return InMemoryVectorStore(embedder)


@pytest.fixture
def container(settings, vector_store):
    result = build_container(
        settings,
        vector_store=vector_store,
        answer_generator=ExtractiveAnswerGenerator(min_relevance_score=0.0),
    )
    result.db.initialize()
    return result


@pytest.fixture
def client(settings, vector_store):
    app = create_app(
        settings,
        vector_store=vector_store,
        answer_generator=ExtractiveAnswerGenerator(min_relevance_score=0.0),
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def make_hit():
    def factory(
        index: int = 0,
        score: float = 0.9,
        text: str = "Employees receive twenty four paid leave days each year.",
    ) -> SearchHit:
        return SearchHit(
            chunk_id=f"doc-1:{index}",
            document_id="doc-1",
            document_title="Doc",
            source="test",
            chunk_index=index,
            text=text,
            score=score,
            start_char=0,
            end_char=len(text),
        )

    return factory


@pytest.fixture
def mock_llm():
    llm = MockLLM()
    yield llm
    llm.client.close()


@pytest.fixture
def llm_client(settings, vector_store, mock_llm):
    app = create_app(settings, vector_store=vector_store, answer_generator=mock_llm.generator())
    with TestClient(app) as test_client:
        yield test_client
