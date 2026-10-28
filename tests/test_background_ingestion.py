from __future__ import annotations

from fastapi.testclient import TestClient

from app.db.repositories import DocumentRepository
from app.db.sqlite import SQLiteDatabase
from app.main import create_app
from app.services.answer_generation import ExtractiveAnswerGenerator
from app.services.chunking import TextChunk
from app.services.document_service import INTERRUPTED_MESSAGE
from app.services.vector_store import InMemoryVectorStore

DOCUMENT = {"title": "Leave", "text": "Employees receive twenty four paid leave days each year."}


def test_background_text_ingestion_is_accepted_then_indexed(client) -> None:
    response = client.post("/api/v1/documents/text?background=true", json=DOCUMENT)
    assert response.status_code == 202, response.text
    accepted = response.json()
    assert (accepted["status"], accepted["chunk_count"], accepted["duplicate"]) == (
        "processing", 0, False
    )
    # The test client runs background tasks before returning, so the document is done by now.
    done = client.get(f"/api/v1/documents/{accepted['id']}").json()
    assert done["status"] == "ready"
    assert done["chunk_count"] >= 1
    assert client.get("/api/v1/stats").json()["vector_count"] == done["chunk_count"]


def test_background_upload_is_accepted_then_indexed(client) -> None:
    response = client.post(
        "/api/v1/documents/upload?background=true",
        files={"file": ("leave.md", DOCUMENT["text"].encode(), "text/markdown")},
    )
    assert response.status_code == 202
    document_id = response.json()["id"]
    assert client.get(f"/api/v1/documents/{document_id}").json()["status"] == "ready"


def test_background_duplicate_returns_the_existing_document(client) -> None:
    first = client.post("/api/v1/documents/text", json=DOCUMENT).json()
    response = client.post("/api/v1/documents/text?background=true", json=DOCUMENT)
    assert response.status_code == 200
    assert (response.json()["id"], response.json()["duplicate"]) == (first["id"], True)


def test_background_upload_still_rejects_bad_files_at_once(client) -> None:
    response = client.post(
        "/api/v1/documents/upload?background=true",
        files={"file": ("notes.xlsx", b"PK\x03\x04", "application/octet-stream")},
    )
    assert response.status_code == 415
    response = client.post(
        "/api/v1/documents/text?background=true", json=dict(DOCUMENT, collection_id="nope")
    )
    assert response.status_code == 404
    assert client.get("/api/v1/documents").json()["count"] == 0


def test_background_failure_marks_the_document_failed(settings, embedder) -> None:
    class FailingStore(InMemoryVectorStore):
        def upsert_document(self, **kwargs):
            raise RuntimeError("embedding service down")

    store = FailingStore(embedder)
    app = create_app(settings, vector_store=store, answer_generator=ExtractiveAnswerGenerator())
    with TestClient(app) as client:
        response = client.post("/api/v1/documents/text?background=true", json=DOCUMENT)
        assert response.status_code == 202
        document = client.get(f"/api/v1/documents/{response.json()['id']}").json()
        assert document["status"] == "failed"
        assert document["error_message"] == "embedding service down"
        assert store.count() == 0


def test_documents_left_processing_are_failed_at_startup(settings, vector_store) -> None:
    db = SQLiteDatabase(settings.sqlite_path)
    db.initialize()
    DocumentRepository(db).create_processing(
        document_id="cut-off", title="Half done", source="test", content_hash="h1"
    )
    vector_store.upsert_document(
        document_id="cut-off",
        title="Half done",
        source="test",
        chunks=[TextChunk(index=0, text="partial", start_char=0, end_char=7)],
    )
    app = create_app(
        settings, vector_store=vector_store, answer_generator=ExtractiveAnswerGenerator()
    )
    with TestClient(app) as client:
        document = client.get("/api/v1/documents/cut-off").json()
        assert document["status"] == "failed"
        assert document["error_message"] == INTERRUPTED_MESSAGE
        assert vector_store.count() == 0
        # A healthy document is untouched by the sweep.
        created = client.post("/api/v1/documents/text", json=DOCUMENT).json()
        assert created["status"] == "ready"
