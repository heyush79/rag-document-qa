from __future__ import annotations

import pytest

from app.agents.retrieval_agent import RetrievalAgent
from app.config import Settings
from app.container import build_container


class StubVectorStore:
    """Returns preset hits, best first, and records each query."""

    def __init__(self, hits) -> None:
        self.hits = hits
        self.calls: list[dict] = []

    def query(self, *, question, top_k, document_id=None):
        self.calls.append({"question": question, "top_k": top_k, "document_id": document_id})
        return self.hits[:top_k]


def retrieval_agent(store, *, min_relevance_score=0.0, token_budget=10_000) -> RetrievalAgent:
    return RetrievalAgent(
        store, min_relevance_score=min_relevance_score, token_budget=token_budget
    )


def test_fetch_size_over_fetches_within_bounds() -> None:
    assert RetrievalAgent.fetch_size(4) == 20
    assert RetrievalAgent.fetch_size(15) == 30
    assert RetrievalAgent.fetch_size(400) == 500


def test_store_is_asked_for_fetch_size_hits() -> None:
    store = StubVectorStore([])
    retrieval_agent(store).run(question="  leave policy  ", top_k=4, document_id="doc-1")
    assert store.calls == [{"question": "leave policy", "top_k": 20, "document_id": "doc-1"}]


def test_keeps_at_most_top_k_hits(make_hit) -> None:
    store = StubVectorStore([make_hit(index, 0.9 - index * 0.01) for index in range(10)])
    hits = retrieval_agent(store).run(question="leave", top_k=3)
    assert [hit.chunk_index for hit in hits] == [0, 1, 2]


def test_drops_hits_below_relevance_threshold(make_hit) -> None:
    store = StubVectorStore([make_hit(0, 0.8), make_hit(1, 0.31), make_hit(2, 0.29)])
    hits = retrieval_agent(store, min_relevance_score=0.3).run(question="leave", top_k=4)
    assert [hit.chunk_index for hit in hits] == [0, 1]


def test_skips_hits_that_would_exceed_token_budget(make_hit) -> None:
    short = "y" * 36  # with the "Doc\n" title prefix: 40 characters, 10 tokens
    long = "x" * 400  # 404 characters, 101 tokens
    store = StubVectorStore(
        [make_hit(0, 0.9, short), make_hit(1, 0.8, long), make_hit(2, 0.7, short)]
    )
    hits = retrieval_agent(store, token_budget=50).run(question="leave", top_k=4)
    assert [hit.chunk_index for hit in hits] == [0, 2]


def test_retrieval_budget_comes_from_context_window(settings) -> None:
    tuned = settings.model_copy(update={"llm_context_window": 4096, "llm_reserved_tokens": 2000})
    assert tuned.retrieval_token_budget == int((4096 - 2000) * 0.9)


def test_reserved_tokens_must_fit_in_context_window(settings) -> None:
    values = settings.model_dump() | {"llm_context_window": 2000, "llm_reserved_tokens": 2000}
    with pytest.raises(ValueError, match="llm_reserved_tokens"):
        Settings(**values)


def test_llm_path_only_sees_relevant_hits(settings, mock_llm, make_hit) -> None:
    store = StubVectorStore(
        [
            make_hit(0, 0.9, "Employees receive twenty four paid leave days each year."),
            make_hit(1, 0.1, "The cafeteria opens at eight in the morning."),
        ]
    )
    container = build_container(
        settings.model_copy(update={"min_relevance_score": 0.5}),
        vector_store=store,
        answer_generator=mock_llm.generator(),
    )
    container.db.initialize()
    result = container.rag_orchestrator.run(question="How many leave days?", top_k=4)
    assert [hit.chunk_index for hit in result.sources] == [0]
    prompt = mock_llm.last_json()["messages"][-1]["content"]
    assert "paid leave days" in prompt
    assert "cafeteria" not in prompt
