from __future__ import annotations

import pytest

from app.core.exceptions import UpstreamServiceError
from app.services.vector_store import SearchHit

DOCUMENT = {
    "title": "Engineering Handbook",
    "source": "api-test",
    "text": "The annual learning budget is forty thousand rupees per employee.",
}


def leave_hit() -> SearchHit:
    text = "Employees receive twenty four paid leave days each year."
    return SearchHit(
        chunk_id="doc-1:0",
        document_id="doc-1",
        document_title="Handbook",
        source="test",
        chunk_index=0,
        text=text,
        score=0.9,
        start_char=0,
        end_char=len(text),
    )


def test_llm_generator_returns_model_answer(mock_llm) -> None:
    mock_llm.reply("Twenty four days. [S1]")
    answer = mock_llm.generator().generate("How many leave days?", [leave_hit()])
    assert answer == "Twenty four days. [S1]"


def test_llm_generator_sends_chat_completion_request(mock_llm) -> None:
    mock_llm.generator().generate("How many leave days?", [leave_hit()])
    request = mock_llm.requests[-1]
    assert str(request.url) == "http://mock-llm/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    body = mock_llm.last_json()
    assert body["model"] == "mock-model"
    user_message = body["messages"][-1]["content"]
    assert "How many leave days?" in user_message
    assert "twenty four paid leave days" in user_message


def test_llm_generator_raises_upstream_error_on_failure(mock_llm) -> None:
    mock_llm.fail(500)
    with pytest.raises(UpstreamServiceError):
        mock_llm.generator().generate("How many leave days?", [leave_hit()])


def test_query_endpoint_returns_llm_answer(llm_client, mock_llm) -> None:
    llm_client.post("/api/v1/documents/text", json=DOCUMENT)
    mock_llm.reply("Forty thousand rupees. [S1]")
    response = llm_client.post(
        "/api/v1/query", json={"question": "What is the annual learning budget?"}
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "Forty thousand rupees. [S1]"


def test_query_endpoint_returns_502_when_llm_fails(llm_client, mock_llm) -> None:
    llm_client.post("/api/v1/documents/text", json=DOCUMENT)
    mock_llm.fail(503)
    response = llm_client.post(
        "/api/v1/query", json={"question": "What is the annual learning budget?"}
    )
    assert response.status_code == 502
    assert response.json()["error"] == "upstream_service_error"
