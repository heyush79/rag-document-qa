from __future__ import annotations


DOCUMENT = {
    "title": "Engineering Handbook",
    "source": "api-test",
    "text": (
        "Employees may work remotely three days per week. "
        "Core collaboration hours are ten to four. "
        "The annual learning budget is forty thousand rupees."
    ),
}


def test_health_endpoint_reports_dependencies(client) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["X-Request-ID"]


def test_ingest_and_get_document(client) -> None:
    created = client.post("/api/v1/documents/text", json=DOCUMENT)
    assert created.status_code == 201
    document_id = created.json()["id"]
    fetched = client.get(f"/api/v1/documents/{document_id}")
    assert fetched.status_code == 200
    assert fetched.json()["title"] == DOCUMENT["title"]


def test_query_endpoint_returns_grounded_sources(client) -> None:
    client.post("/api/v1/documents/text", json=DOCUMENT)
    response = client.post(
        "/api/v1/query",
        json={"question": "What is the annual learning budget?", "top_k": 2},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["sources"]
    assert "[S1]" in payload["answer"]


def test_delete_document_removes_it(client) -> None:
    created = client.post("/api/v1/documents/text", json=DOCUMENT)
    document_id = created.json()["id"]
    deleted = client.delete(f"/api/v1/documents/{document_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/v1/documents/{document_id}").status_code == 404
