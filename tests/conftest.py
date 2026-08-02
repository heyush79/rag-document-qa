from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import build_container
from app.main import create_app
from app.services.answer_generation import ExtractiveAnswerGenerator
from app.services.embeddings import HashingEmbedder
from app.services.vector_store import InMemoryVectorStore


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
