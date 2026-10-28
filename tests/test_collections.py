from __future__ import annotations

import json

import pytest

from app.core.exceptions import ConflictError, NotFoundError
from app.db.repositories import DocumentRepository
from app.db.sqlite import SQLiteDatabase
from app.services.chunking import TextChunk
from app.services.embeddings import HashingEmbedder
from app.services.vector_store import ChromaVectorStore

LEAVE = {"title": "Leave", "text": "Employees receive twenty four paid leave days each year."}
MENU = {"title": "Menu", "text": "The cafeteria serves lentil soup and rice on Mondays."}


def create_collection(client, name: str) -> str:
    response = client.post("/api/v1/collections", json={"name": name})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def ingest(client, document: dict, collection_id: str | None = None) -> dict:
    payload = dict(document, collection_id=collection_id) if collection_id else document
    response = client.post("/api/v1/documents/text", json=payload)
    assert response.status_code in (200, 201), response.text
    return response.json()


def ask(client, question: str, **extra) -> dict:
    response = client.post("/api/v1/query", json={"question": question, **extra})
    assert response.status_code == 200, response.text
    return response.json()


# --- storage ----------------------------------------------------------------


def test_duplicates_are_detected_per_collection(tmp_path) -> None:
    db = SQLiteDatabase(tmp_path / "rag.db")
    db.initialize()
    with db.connect() as connection:
        connection.execute(
            "INSERT INTO collections(id, name, created_at) VALUES ('hr', 'hr', '2026-01-01')"
        )
    documents = DocumentRepository(db)
    documents.create_processing(document_id="d1", title="A", source="t", content_hash="h1")
    documents.create_processing(
        document_id="d2", title="A", source="t", content_hash="h1", collection_id="hr"
    )
    with pytest.raises(ConflictError):
        documents.create_processing(document_id="d3", title="A", source="t", content_hash="h1")
    assert documents.get_by_hash("h1").id == "d1"
    assert documents.get_by_hash("h1", "hr").id == "d2"
    assert documents.get_by_hash("h1", "other") is None


def test_documents_need_an_existing_collection(tmp_path) -> None:
    db = SQLiteDatabase(tmp_path / "rag.db")
    db.initialize()
    with pytest.raises(NotFoundError, match="nope"):
        DocumentRepository(db).create_processing(
            document_id="d1", title="A", source="t", content_hash="h1", collection_id="nope"
        )


# --- vector stores ----------------------------------------------------------


def test_in_memory_store_filters_by_document_ids(vector_store) -> None:
    for document_id in ["a", "b", "c"]:
        vector_store.upsert_document(
            document_id=document_id,
            title=document_id,
            source="t",
            chunks=[TextChunk(index=0, text="leave policy", start_char=0, end_char=12)],
        )
    hits = vector_store.query(question="leave policy", top_k=10, document_ids=["a", "c"])
    assert sorted(hit.document_id for hit in hits) == ["a", "c"]
    assert vector_store.query(question="leave policy", top_k=10, document_ids=[]) == []


def test_chroma_store_filters_by_document_ids() -> None:
    chromadb = pytest.importorskip("chromadb")
    store = ChromaVectorStore(
        embedder=HashingEmbedder(16), collection_name="docs", client=chromadb.EphemeralClient()
    )
    for document_id in ["a", "b", "c"]:
        store.upsert_document(
            document_id=document_id,
            title=document_id,
            source="t",
            chunks=[TextChunk(index=0, text="leave policy", start_char=0, end_char=12)],
        )
    hits = store.query(question="leave policy", top_k=10, document_ids=["a", "c"])
    assert sorted(hit.document_id for hit in hits) == ["a", "c"]
    assert store.query(question="leave policy", top_k=10, document_ids=[]) == []
    only_b = store.query(question="leave policy", top_k=10, document_id="b")
    assert [hit.document_id for hit in only_b] == ["b"]


# --- API --------------------------------------------------------------------


def test_default_collection_exists_and_counts_documents(client) -> None:
    listing = client.get("/api/v1/collections").json()
    assert [item["name"] for item in listing["items"]] == ["default"]
    assert listing["items"][0]["id"] == "default"
    created = ingest(client, LEAVE)
    assert created["collection_id"] == "default"
    assert client.get("/api/v1/collections/default").json()["document_count"] == 1
    assert client.get("/api/v1/stats").json()["collection_count"] == 1


def test_create_get_and_list_collections(client) -> None:
    response = client.post(
        "/api/v1/collections", json={"name": "hr", "description": "People policies"}
    )
    assert response.status_code == 201
    created = response.json()
    assert (created["name"], created["description"], created["document_count"]) == (
        "hr", "People policies", 0
    )
    assert client.get(f"/api/v1/collections/{created['id']}").json() == created
    names = [item["name"] for item in client.get("/api/v1/collections").json()["items"]]
    assert names == ["default", "hr"]
    assert client.get("/api/v1/collections/nope").status_code == 404


def test_collection_names_are_unique(client) -> None:
    create_collection(client, "hr")
    response = client.post("/api/v1/collections", json={"name": "hr"})
    assert response.status_code == 409
    assert response.json()["error"] == "conflict"


def test_documents_can_be_ingested_into_and_listed_by_collection(client) -> None:
    hr = create_collection(client, "hr")
    ingest(client, LEAVE, hr)
    ingest(client, MENU)
    in_hr = client.get(f"/api/v1/documents?collection_id={hr}").json()
    assert [item["title"] for item in in_hr["items"]] == ["Leave"]
    assert in_hr["items"][0]["collection_id"] == hr
    everything = client.get("/api/v1/documents").json()
    assert sorted(item["title"] for item in everything["items"]) == ["Leave", "Menu"]
    assert client.get("/api/v1/documents?collection_id=nope").status_code == 404
    response = client.post("/api/v1/documents/text", json=dict(LEAVE, collection_id="nope"))
    assert response.status_code == 404


def test_upload_accepts_a_collection(client) -> None:
    hr = create_collection(client, "hr")
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("leave.md", LEAVE["text"].encode(), "text/markdown")},
        data={"collection_id": hr},
    )
    assert response.status_code == 201
    assert response.json()["collection_id"] == hr


def test_same_text_may_live_in_two_collections_but_not_twice_in_one(client) -> None:
    hr = create_collection(client, "hr")
    first = client.post("/api/v1/documents/text", json=dict(LEAVE, collection_id=hr))
    again = client.post("/api/v1/documents/text", json=dict(LEAVE, collection_id=hr))
    elsewhere = client.post("/api/v1/documents/text", json=LEAVE)
    assert (first.status_code, again.status_code, elsewhere.status_code) == (201, 200, 201)
    assert again.json()["duplicate"] is True
    assert again.json()["id"] == first.json()["id"]
    assert elsewhere.json()["id"] != first.json()["id"]


def test_queries_can_be_scoped_to_a_collection(client) -> None:
    hr = create_collection(client, "hr")
    food = create_collection(client, "food")
    leave = ingest(client, LEAVE, hr)
    ingest(client, MENU, food)

    scoped = ask(client, "leave days", collection_id=hr)
    assert {hit["document_id"] for hit in scoped["sources"]} == {leave["id"]}
    everything = ask(client, "leave days")
    assert len(everything["sources"]) == 2
    assert ask(client, "leave days", collection_id=food)["sources"][0]["document_title"] == "Menu"

    empty = create_collection(client, "empty")
    assert ask(client, "leave days", collection_id=empty)["sources"] == []
    # A document outside the collection is not searched.
    outside = ask(client, "leave days", collection_id=food, document_id=leave["id"])
    assert outside["sources"] == []
    response = client.post("/api/v1/query", json={"question": "leave", "collection_id": "nope"})
    assert response.status_code == 404


def test_streamed_queries_are_scoped_too(client) -> None:
    hr = create_collection(client, "hr")
    leave = ingest(client, LEAVE, hr)
    ingest(client, MENU)
    response = client.post(
        "/api/v1/query/stream", json={"question": "leave days", "collection_id": hr}
    )
    first = json.loads(response.text.split("\n\n")[0].removeprefix("data: "))
    assert [hit["document_id"] for hit in first["sources"]] == [leave["id"]]
    response = client.post(
        "/api/v1/query/stream", json={"question": "leave days", "collection_id": "nope"}
    )
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/json"


def test_delete_refuses_a_non_empty_collection_unless_cascade(client) -> None:
    hr = create_collection(client, "hr")
    leave = ingest(client, LEAVE, hr)
    ingest(client, MENU)
    response = client.delete(f"/api/v1/collections/{hr}")
    assert response.status_code == 409
    assert "cascade" in response.json()["message"]
    assert client.get(f"/api/v1/documents/{leave['id']}").status_code == 200

    assert client.delete(f"/api/v1/collections/{hr}?cascade=true").status_code == 204
    assert client.get(f"/api/v1/collections/{hr}").status_code == 404
    assert client.get(f"/api/v1/documents/{leave['id']}").status_code == 404
    stats = client.get("/api/v1/stats").json()
    assert (stats["document_count"], stats["vector_count"]) == (1, 1)


def test_default_collection_cannot_be_deleted(client) -> None:
    response = client.delete("/api/v1/collections/default?cascade=true")
    assert response.status_code == 409
    assert client.delete("/api/v1/collections/nope").status_code == 404
    empty = create_collection(client, "empty")
    assert client.delete(f"/api/v1/collections/{empty}").status_code == 204
