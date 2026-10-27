from __future__ import annotations

import pytest

from app.config import Settings
from app.container import build_container, llm_providers
from app.services.embeddings import OpenAICompatibleEmbedder


def test_llm_fallbacks_parse_from_a_json_environment_variable(monkeypatch) -> None:
    monkeypatch.setenv(
        "RAG_LLM_FALLBACKS",
        '[{"base_url": "http://b/v1", "model": "m2", "api_key": "k2"},'
        ' {"base_url": "http://c/v1", "model": "m3"}]',
    )
    settings = Settings(
        _env_file=None,
        answer_provider="openai-compatible",
        llm_base_url="http://a/v1",
        llm_model="m1",
        llm_api_key="k1",
    )
    providers = llm_providers(settings)
    assert [(p.base_url, p.model, p.api_key) for p in providers] == [
        ("http://a/v1", "m1", "k1"),
        ("http://b/v1", "m2", "k2"),
        ("http://c/v1", "m3", None),
    ]


def test_llm_fallbacks_default_to_the_primary_alone() -> None:
    settings = Settings(
        _env_file=None,
        answer_provider="openai-compatible",
        llm_base_url="http://a/v1",
        llm_model="m1",
    )
    assert settings.llm_fallbacks == []
    assert [p.model for p in llm_providers(settings)] == ["m1"]


def test_fallback_entries_need_a_base_url_and_a_model() -> None:
    with pytest.raises(ValueError, match="base_url"):
        Settings(_env_file=None, llm_fallbacks=[{"base_url": "", "model": "m"}])
    with pytest.raises(ValueError, match="model"):
        Settings(_env_file=None, llm_fallbacks=[{"base_url": "http://b/v1"}])


def test_openai_compatible_embeddings_need_a_base_url() -> None:
    with pytest.raises(ValueError, match="embedding_base_url"):
        Settings(_env_file=None, embedding_backend="openai-compatible")


def test_container_builds_the_remote_embedder(settings, vector_store) -> None:
    remote = settings.model_copy(
        update={
            "embedding_backend": "openai-compatible",
            "embedding_base_url": "http://embed/v1",
            "embedding_model": "emb-model",
            "embedding_batch_size": 16,
        }
    )
    container = build_container(remote, vector_store=vector_store)
    assert isinstance(container.embedder, OpenAICompatibleEmbedder)
    assert container.embedder.signature == "openai-compatible:emb-model"
    assert container.embedder.batch_size == 16
    assert container.llm is None  # extractive answers need no LLM
