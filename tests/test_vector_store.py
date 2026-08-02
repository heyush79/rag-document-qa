from __future__ import annotations

from app.services.chunking import TextChunk


def add_document(vector_store, document_id: str, title: str, text: str) -> None:
    vector_store.upsert_document(
        document_id=document_id,
        title=title,
        source=f"source:{document_id}",
        chunks=[TextChunk(index=0, text=text, start_char=0, end_char=len(text))],
    )


def test_upsert_increases_vector_count(vector_store) -> None:
    add_document(vector_store, "doc-1", "Security", "multi factor authentication is mandatory")
    assert vector_store.count() == 1


def test_query_returns_matching_chunk(vector_store) -> None:
    add_document(vector_store, "doc-1", "Leave", "employees receive twenty four paid leave days")
    hits = vector_store.query(question="paid leave days", top_k=1)
    assert hits[0].document_id == "doc-1"
    assert "paid leave" in hits[0].text


def test_query_can_filter_by_document(vector_store) -> None:
    add_document(vector_store, "doc-1", "Leave", "paid leave policy and approval")
    add_document(vector_store, "doc-2", "Security", "security incident response policy")
    hits = vector_store.query(question="policy", top_k=2, document_id="doc-2")
    assert len(hits) == 1
    assert hits[0].document_id == "doc-2"


def test_delete_removes_document_vectors(vector_store) -> None:
    add_document(vector_store, "doc-1", "Guide", "a document that will be deleted")
    vector_store.delete_document("doc-1")
    assert vector_store.count() == 0
