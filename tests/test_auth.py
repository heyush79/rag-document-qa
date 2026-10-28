from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.answer_generation import ExtractiveAnswerGenerator

ADMIN = {"Authorization": "Bearer admin-secret"}
DOCUMENT = {"title": "Leave", "text": "Employees receive twenty four paid leave days each year."}
QUESTION = {"question": "How many paid leave days?"}


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def app_client(settings, vector_store, **overrides) -> TestClient:
    app = create_app(
        settings.model_copy(update=overrides),
        vector_store=vector_store,
        answer_generator=ExtractiveAnswerGenerator(0.0),
    )
    return TestClient(app)


@pytest.fixture
def auth_client(settings, vector_store):
    with app_client(settings, vector_store, auth_enabled=True, admin_api_key="admin-secret") as c:
        yield c


def create_key(client: TestClient, name: str = "ci") -> dict:
    response = client.post("/api/v1/api-keys", json={"name": name}, headers=ADMIN)
    assert response.status_code == 201, response.text
    return response.json()


def test_public_routes_need_no_key(auth_client) -> None:
    for path in ["/", "/api/v1/health", "/api/v1/health?deep=true", "/docs", "/openapi.json"]:
        assert auth_client.get(path).status_code == 200, path


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/api/v1/documents/text", DOCUMENT),
        ("GET", "/api/v1/documents", None),
        ("POST", "/api/v1/query", QUESTION),
        ("POST", "/api/v1/query/stream", QUESTION),
        ("GET", "/api/v1/queries", None),
        ("GET", "/api/v1/stats", None),
    ],
)
def test_protected_routes_reject_a_missing_key(auth_client, method, path, body) -> None:
    response = auth_client.request(method, path, json=body)
    assert response.status_code == 401
    assert response.headers["content-type"] == "application/json"
    assert response.headers["WWW-Authenticate"] == "Bearer"
    payload = response.json()
    assert payload["error"] == "unauthorized"
    assert payload["request_id"]


@pytest.mark.parametrize(
    "header",
    ["Basic YWJj", "Bearer", "Bearer rag_not-a-real-key", "Token admin-secret"],
)
def test_wrong_scheme_or_unknown_key_is_rejected(auth_client, header) -> None:
    response = auth_client.get("/api/v1/stats", headers={"Authorization": header})
    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"


def test_created_key_is_returned_once_and_stored_as_a_hash(auth_client) -> None:
    created = create_key(auth_client, name="reporting")
    key = created["key"]
    assert key.startswith("rag_") and len(key) > 40
    assert created["prefix"] == key[:12]
    assert created["name"] == "reporting"
    assert created["last_used_at"] is None and created["revoked_at"] is None

    listing = auth_client.get("/api/v1/api-keys", headers=ADMIN).json()
    assert listing["count"] == 1
    assert "key" not in listing["items"][0]
    assert listing["items"][0]["prefix"] == key[:12]

    db = auth_client.app.state.container.db
    with db.connect() as connection:
        (row,) = connection.execute("SELECT * FROM api_keys").fetchall()
    assert row["key_hash"] == hashlib.sha256(key.encode()).hexdigest()
    assert key not in " ".join(str(value) for value in dict(row).values())


def test_valid_key_gets_through_and_records_its_use(auth_client) -> None:
    created = create_key(auth_client)
    headers = bearer(created["key"])
    ingest = auth_client.post("/api/v1/documents/text", json=DOCUMENT, headers=headers)
    assert ingest.status_code == 201
    response = auth_client.post("/api/v1/query", json=QUESTION, headers=headers)
    assert response.status_code == 200
    assert "twenty four" in response.json()["answer"]

    (listed,) = auth_client.get("/api/v1/api-keys", headers=ADMIN).json()["items"]
    assert listed["id"] == created["id"]
    assert listed["last_used_at"] is not None


def test_revoked_key_is_rejected(auth_client) -> None:
    created = create_key(auth_client)
    headers = bearer(created["key"])
    assert auth_client.get("/api/v1/stats", headers=headers).status_code == 200

    assert auth_client.delete(f"/api/v1/api-keys/{created['id']}", headers=ADMIN).status_code == 204
    assert auth_client.get("/api/v1/stats", headers=headers).status_code == 401
    (listed,) = auth_client.get("/api/v1/api-keys", headers=ADMIN).json()["items"]
    assert listed["revoked_at"] is not None

    # Revoking again is harmless; an unknown id is a 404.
    assert auth_client.delete(f"/api/v1/api-keys/{created['id']}", headers=ADMIN).status_code == 204
    assert auth_client.delete("/api/v1/api-keys/nope", headers=ADMIN).status_code == 404


def test_admin_routes_need_the_admin_key(auth_client) -> None:
    regular = bearer(create_key(auth_client)["key"])
    for headers in [None, regular, bearer("admin-wrong")]:
        response = auth_client.post("/api/v1/api-keys", json={"name": "x"}, headers=headers)
        assert response.status_code == 401, headers
        assert auth_client.get("/api/v1/api-keys", headers=headers).status_code == 401, headers


def test_admin_key_also_works_as_a_regular_key(auth_client) -> None:
    assert auth_client.get("/api/v1/stats", headers=ADMIN).status_code == 200
    ingest = auth_client.post("/api/v1/documents/text", json=DOCUMENT, headers=ADMIN)
    assert ingest.status_code == 201


def test_auth_is_off_by_default_and_key_management_needs_an_admin_key(client) -> None:
    assert client.post("/api/v1/documents/text", json=DOCUMENT).status_code == 201
    assert client.get("/api/v1/stats").status_code == 200
    response = client.post("/api/v1/api-keys", json={"name": "x"})
    assert response.status_code == 403
    assert response.json()["error"] == "forbidden"
    assert "RAG_ADMIN_API_KEY" in response.json()["message"]


def test_keys_can_be_managed_while_auth_is_off(settings, vector_store) -> None:
    with app_client(settings, vector_store, admin_api_key="admin-secret") as client:
        created = create_key(client)
        assert client.get("/api/v1/stats").status_code == 200  # still open
        assert client.get("/api/v1/stats", headers=bearer(created["key"])).status_code == 200


def test_enabling_auth_requires_an_admin_key() -> None:
    with pytest.raises(ValueError, match="admin_api_key"):
        Settings(_env_file=None, auth_enabled=True)


def test_startup_warns_while_auth_is_off(settings, vector_store, caplog) -> None:
    with caplog.at_level("WARNING"), app_client(settings, vector_store):
        pass
    assert "authentication is disabled" in caplog.text


def test_openapi_declares_the_bearer_scheme(auth_client) -> None:
    spec = auth_client.get("/openapi.json").json()
    assert "HTTPBearer" in spec["components"]["securitySchemes"]
    assert spec["paths"]["/api/v1/query"]["post"]["security"] == [{"HTTPBearer": []}]
    assert "security" not in spec["paths"]["/api/v1/health"]["get"]
