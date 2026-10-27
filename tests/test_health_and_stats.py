from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.embeddings import HashingEmbedder
from app.services.vector_store import ChromaVectorStore

DOCUMENT = {"title": "Leave", "text": "Employees receive twenty four paid leave days each year."}
QUESTION = {"question": "How many paid leave days?"}


def test_shallow_health_keeps_its_original_shape(client) -> None:
    payload = client.get("/api/v1/health").json()
    assert set(payload) == {"status", "sqlite", "chroma", "embedding_backend"}
    assert payload["status"] == "ok"


def test_deep_health_checks_the_embedder_and_each_provider(llm_client, mock_llm) -> None:
    payload = llm_client.get("/api/v1/health?deep=true").json()
    assert payload["status"] == "ok"
    assert payload["embedder"] == {"status": "ok"}
    assert payload["llm"] == [
        {"status": "ok", "base_url": "http://mock-llm/v1", "model": "mock-model"}
    ]
    request = mock_llm.requests[-1]
    assert (request.method, str(request.url)) == ("GET", "http://mock-llm/v1/models")
    assert request.headers["Authorization"] == "Bearer test-key"


def test_deep_health_reports_an_unreachable_provider(llm_client, mock_llm) -> None:
    mock_llm.models_status = 503
    payload = llm_client.get("/api/v1/health?deep=true").json()
    assert payload["status"] == "degraded"
    assert payload["llm"][0]["status"] == "error"
    assert "503" in payload["llm"][0]["detail"]
    assert payload["sqlite"] == "ok"


def test_deep_health_without_an_llm_provider(client) -> None:
    payload = client.get("/api/v1/health?deep=true").json()
    assert payload["status"] == "ok"
    assert payload["embedder"] == {"status": "ok"}
    assert payload["llm"] == []


def test_deep_health_reports_an_embedder_failure(settings, vector_store) -> None:
    class BrokenEmbedder:
        signature = "broken"

        def embed_documents(self, texts):
            raise RuntimeError("model server down")

        def embed_query(self, text):
            raise RuntimeError("model server down")

    app = create_app(settings, embedder=BrokenEmbedder(), vector_store=vector_store)
    with TestClient(app) as test_client:
        payload = test_client.get("/api/v1/health?deep=true").json()
    assert payload["status"] == "degraded"
    assert payload["embedder"] == {"status": "error", "detail": "model server down"}


def test_chroma_ping_returns_false_instead_of_raising() -> None:
    class ExplodingCollection:
        metadata: dict = {}

        def count(self):
            raise RuntimeError("chroma down")

    class StubChromaClient:
        def get_or_create_collection(self, **kwargs):
            return ExplodingCollection()

    store = ChromaVectorStore(
        embedder=HashingEmbedder(16), collection_name="docs", client=StubChromaClient()
    )
    assert store.ping() is False


def test_stats_include_llm_usage_totals(llm_client, mock_llm) -> None:
    assert llm_client.get("/api/v1/stats").json()["llm_usage"] == {
        "calls": 0,
        "failed_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    llm_client.post("/api/v1/documents/text", json=DOCUMENT)
    assert llm_client.post("/api/v1/query", json=QUESTION).status_code == 200
    usage = llm_client.get("/api/v1/stats").json()["llm_usage"]
    assert (usage["calls"], usage["failed_calls"]) == (1, 0)
    assert usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0

    mock_llm.fail(500)
    assert llm_client.post("/api/v1/query", json=QUESTION).status_code == 502
    usage = llm_client.get("/api/v1/stats").json()["llm_usage"]
    assert (usage["calls"], usage["failed_calls"]) == (2, 1)


def test_llm_calls_are_logged_against_their_query(llm_client) -> None:
    llm_client.post("/api/v1/documents/text", json=DOCUMENT)
    query_id = llm_client.post("/api/v1/query", json=QUESTION).json()["query_id"]
    calls = llm_client.app.state.container.llm_calls
    (row,) = calls.list_for_query(query_id)
    assert (row.purpose, row.provider, row.model, row.ok) == (
        "answer", "http://mock-llm/v1", "mock-model", True
    )
    assert row.tokens_estimated is True  # the mock reports no usage


def test_streamed_answers_are_logged_too(llm_client) -> None:
    llm_client.post("/api/v1/documents/text", json=DOCUMENT)
    response = llm_client.post("/api/v1/query/stream", json=QUESTION)
    end = json.loads(response.text.strip().split("\n\n")[-1].removeprefix("data: "))
    assert end["type"] == "end"
    calls = llm_client.app.state.container.llm_calls
    (row,) = calls.list_for_query(end["query_id"])
    assert (row.ok, row.purpose) == (True, "answer")
    assert row.completion_tokens > 0


def test_failed_query_keeps_its_call_rows(llm_client, mock_llm) -> None:
    llm_client.post("/api/v1/documents/text", json=DOCUMENT)
    mock_llm.fail(500)
    assert llm_client.post("/api/v1/query", json=QUESTION).status_code == 502
    (row,) = llm_client.app.state.container.llm_calls.list_recent(1)
    assert row.ok is False
    assert row.query_id  # the id the query would have had
    assert llm_client.get("/api/v1/stats").json()["query_count"] == 0
