from __future__ import annotations


def handbook_text() -> str:
    return (
        "Employees receive twenty four paid leave days each year. "
        "Requests longer than five working days need two weeks notice. "
        "Multi factor authentication is mandatory for source control and cloud accounts."
    )


def test_ingestion_agent_persists_document_and_chunks(container) -> None:
    result = container.ingestion_agent.run(
        title="Handbook", source="test", text=handbook_text()
    )
    assert result.duplicate is False
    assert result.document.status == "ready"
    assert result.document.chunk_count >= 1
    assert container.vector_store.count() == result.document.chunk_count


def test_duplicate_ingestion_reuses_existing_document(container) -> None:
    first = container.ingestion_agent.run(title="Handbook", source="test", text=handbook_text())
    second = container.ingestion_agent.run(title="Copy", source="another", text=handbook_text())
    assert second.duplicate is True
    assert second.document.id == first.document.id
    assert container.documents.count() == 1


def test_rag_orchestrator_returns_sources_and_stores_history(container) -> None:
    container.ingestion_agent.run(title="Handbook", source="test", text=handbook_text())
    result = container.rag_orchestrator.run(question="How many paid leave days?", top_k=2)
    assert result.sources
    assert "[S1]" in result.answer
    assert container.queries.count() == 1


def test_rag_orchestrator_handles_empty_index(container) -> None:
    result = container.rag_orchestrator.run(question="What is the leave policy?", top_k=2)
    assert result.sources == []
    assert "could not find" in result.answer.lower()
