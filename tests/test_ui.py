from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.answer_generation import ExtractiveAnswerGenerator


def test_ui_page_is_served(client) -> None:
    response = client.get("/ui/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>RAG Document Q&amp;A</title>" in response.text
    assert 'src="app.js"' in response.text and 'href="styles.css"' in response.text


def test_ui_assets_are_served(client) -> None:
    script = client.get("/ui/app.js")
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert '"/api/v1"' in script.text
    assert client.get("/ui/styles.css").status_code == 200
    assert client.get("/ui/missing.js").status_code == 404


def test_ui_root_redirects_to_the_page(client) -> None:
    response = client.get("/ui", follow_redirects=False)
    assert response.status_code in (301, 307)
    assert response.headers["location"].endswith("/ui/")


def test_root_points_at_the_ui(client) -> None:
    assert client.get("/").json()["ui"] == "/ui/"


def test_ui_needs_no_api_key(settings, vector_store) -> None:
    app = create_app(
        settings.model_copy(update={"auth_enabled": True, "admin_api_key": "admin-secret"}),
        vector_store=vector_store,
        answer_generator=ExtractiveAnswerGenerator(),
    )
    with TestClient(app) as test_client:
        assert test_client.get("/ui/").status_code == 200
        assert test_client.get("/api/v1/documents").status_code == 401
