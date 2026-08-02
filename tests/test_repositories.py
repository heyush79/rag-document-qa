from __future__ import annotations

from app.db.repositories import DocumentRepository, QueryRepository
from app.db.sqlite import SQLiteDatabase


def build_repositories(tmp_path):
    db = SQLiteDatabase(tmp_path / "repository.db")
    db.initialize()
    return DocumentRepository(db), QueryRepository(db)


def test_create_and_get_document(tmp_path) -> None:
    documents, _ = build_repositories(tmp_path)
    created = documents.create_processing(
        document_id="doc-1", title="Guide", source="unit-test", content_hash="hash-1"
    )
    fetched = documents.get("doc-1")
    assert fetched == created
    assert fetched.status == "processing"


def test_list_documents_returns_created_rows(tmp_path) -> None:
    documents, _ = build_repositories(tmp_path)
    documents.create_processing(
        document_id="doc-1", title="One", source="test", content_hash="hash-1"
    )
    documents.create_processing(
        document_id="doc-2", title="Two", source="test", content_hash="hash-2"
    )
    assert {record.id for record in documents.list()} == {"doc-1", "doc-2"}


def test_mark_ready_updates_chunk_count(tmp_path) -> None:
    documents, _ = build_repositories(tmp_path)
    documents.create_processing(
        document_id="doc-1", title="Guide", source="test", content_hash="hash-1"
    )
    documents.mark_ready("doc-1", 4)
    updated = documents.get("doc-1")
    assert updated.status == "ready"
    assert updated.chunk_count == 4


def test_delete_document_removes_row(tmp_path) -> None:
    documents, _ = build_repositories(tmp_path)
    documents.create_processing(
        document_id="doc-1", title="Guide", source="test", content_hash="hash-1"
    )
    assert documents.delete("doc-1") is True
    assert documents.get("doc-1") is None


def test_query_history_is_persisted(tmp_path) -> None:
    _, queries = build_repositories(tmp_path)
    queries.create(
        query_id="query-1",
        question="What is RAG?",
        answer="Retrieval augmented generation.",
        top_k=3,
        latency_ms=12.5,
        sources=[],
    )
    history = queries.list_recent()
    assert len(history) == 1
    assert history[0].id == "query-1"
