from __future__ import annotations

import json
import math

import httpx
import pytest

from app.core.exceptions import ConflictError, UpstreamServiceError
from app.services.embeddings import (
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
)
from app.services.vector_store import ChromaVectorStore


class FakeEmbeddingServer:
    """Answers /embeddings with one un-normalized vector per input: [len(text), 1, 0, 0]."""

    def __init__(self, *, with_index: bool = True) -> None:
        self.with_index = with_index
        self.status = 200
        self.drop_last = False
        self.requests: list[httpx.Request] = []
        self.client = httpx.Client(transport=httpx.MockTransport(self._handle))

    def inputs(self) -> list[list[str]]:
        return [json.loads(request.content)["input"] for request in self.requests]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "nope"}})
        data = []
        for index, text in enumerate(json.loads(request.content)["input"]):
            entry = {"object": "embedding", "embedding": [float(len(text)), 1.0, 0.0, 0.0]}
            if self.with_index:
                entry["index"] = index
            data.append(entry)
        if self.with_index:
            data.reverse()  # out of order on purpose: the index is what counts
        if self.drop_last:
            data.pop()
        return httpx.Response(200, json={"object": "list", "data": data, "model": "emb-model"})


def embedder(server: FakeEmbeddingServer, *, batch_size: int = 64) -> OpenAICompatibleEmbedder:
    return OpenAICompatibleEmbedder(
        base_url="http://embed/v1/",
        model="emb-model",
        api_key="emb-key",
        batch_size=batch_size,
        timeout_seconds=5,
        client=server.client,
    )


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def test_sends_batches_of_the_configured_size() -> None:
    server = FakeEmbeddingServer()
    vectors = embedder(server, batch_size=2).embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])
    assert len(vectors) == 5
    assert server.inputs() == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    request = server.requests[0]
    assert str(request.url) == "http://embed/v1/embeddings"
    assert request.headers["Authorization"] == "Bearer emb-key"
    assert json.loads(request.content)["model"] == "emb-model"


def test_vectors_follow_input_order_and_are_normalized() -> None:
    vectors = embedder(FakeEmbeddingServer()).embed_documents(["a", "bbb"])
    assert vectors[0] == pytest.approx([1 / math.sqrt(2), 1 / math.sqrt(2), 0.0, 0.0])
    assert vectors[1] == pytest.approx([3 / math.sqrt(10), 1 / math.sqrt(10), 0.0, 0.0])
    assert all(norm(vector) == pytest.approx(1.0) for vector in vectors)


def test_accepts_responses_without_an_index() -> None:
    vectors = embedder(FakeEmbeddingServer(with_index=False)).embed_documents(["a", "bbb"])
    assert vectors[0][0] < vectors[1][0]


def test_embed_query_returns_one_vector() -> None:
    server = FakeEmbeddingServer()
    vector = embedder(server).embed_query("hello")
    assert norm(vector) == pytest.approx(1.0)
    assert server.inputs() == [["hello"]]


def test_signature_names_the_model() -> None:
    assert embedder(FakeEmbeddingServer()).signature == "openai-compatible:emb-model"


def test_raises_upstream_error_on_http_failure() -> None:
    server = FakeEmbeddingServer()
    server.status = 500
    with pytest.raises(UpstreamServiceError, match="Embedding request failed"):
        embedder(server).embed_query("hello")


def test_raises_upstream_error_on_a_short_response() -> None:
    server = FakeEmbeddingServer()
    server.drop_last = True
    with pytest.raises(UpstreamServiceError, match="expected 2 embeddings, got 1"):
        embedder(server).embed_documents(["a", "b"])


def test_sentence_transformers_backend_needs_the_local_extra(monkeypatch) -> None:
    monkeypatch.setattr("app.services.embeddings.importlib.util.find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match=r"local"):
        SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")


def test_chroma_refuses_a_collection_made_with_another_model() -> None:
    chromadb = pytest.importorskip("chromadb")
    client = chromadb.EphemeralClient()
    ChromaVectorStore(embedder=HashingEmbedder(16), collection_name="docs", client=client)
    with pytest.raises(ConflictError, match="different embedding model"):
        ChromaVectorStore(embedder=HashingEmbedder(32), collection_name="docs", client=client)
