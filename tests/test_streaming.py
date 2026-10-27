from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import UpstreamServiceError
from app.main import create_app

LEAVE_TEXT = "Employees receive twenty four paid leave days each calendar year."
QUESTION = {"question": "How many paid leave days?", "top_k": 2}


def text_chunk(content: str) -> str:
    return f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': content}}]})}"


def parse_events(body: str) -> list[dict]:
    return [json.loads(block.removeprefix("data: ")) for block in body.strip().split("\n\n")]


def ingest_leave_policy(client: TestClient) -> None:
    response = client.post("/api/v1/documents/text", json={"title": "Leave", "text": LEAVE_TEXT})
    assert response.status_code == 201


def stream_events(client: TestClient) -> list[dict]:
    response = client.post("/api/v1/query/stream", json=QUESTION)
    assert response.status_code == 200
    return parse_events(response.text)


def test_generator_yields_text_pieces_in_order(mock_llm, make_hit) -> None:
    mock_llm.stream_reply("Twenty four", " paid leave days.", " [S1]")
    pieces = list(mock_llm.generator().stream("How many leave days?", [make_hit()]))
    assert pieces == ["Twenty four", " paid leave days.", " [S1]"]
    assert mock_llm.last_json()["stream"] is True


def test_generator_skips_non_text_lines_and_stops_at_done(mock_llm, make_hit) -> None:
    mock_llm.stream_lines(
        ": keep-alive comment",
        "event: message",
        'data: {"choices": [], "usage": {"total_tokens": 5}}',
        text_chunk("kept"),
        "data: [DONE]",
        text_chunk("after done"),
    )
    assert list(mock_llm.generator().stream("q", [make_hit()])) == ["kept"]


def test_generator_raises_on_http_error(mock_llm, make_hit) -> None:
    mock_llm.fail(429)
    with pytest.raises(UpstreamServiceError):
        list(mock_llm.generator().stream("q", [make_hit()]))


def test_generator_raises_on_error_event(mock_llm, make_hit) -> None:
    mock_llm.stream_lines(text_chunk("partial"), 'data: {"error": {"message": "quota"}}')
    stream = mock_llm.generator().stream("q", [make_hit()])
    assert next(stream) == "partial"
    with pytest.raises(UpstreamServiceError, match="quota"):
        next(stream)


def test_stream_sends_sources_then_answer_pieces_then_end(llm_client, mock_llm) -> None:
    ingest_leave_policy(llm_client)
    mock_llm.stream_reply("Twenty four", " paid leave days.", " [S1]")
    response = llm_client.post("/api/v1/query/stream", json=QUESTION)

    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    events = parse_events(response.text)
    assert [event["type"] for event in events] == ["source", "answer", "answer", "answer", "end"]
    assert events[0]["sources"][0]["rank"] == 1
    assert events[0]["sources"][0]["text"] == LEAVE_TEXT
    assert "".join(event["answer"] for event in events[1:-1]) == "Twenty four paid leave days. [S1]"
    assert events[-1]["latency_ms"] >= 0

    saved = llm_client.get("/api/v1/queries").json()
    assert [(item["id"], item["answer"]) for item in saved] == [
        (events[-1]["query_id"], "Twenty four paid leave days. [S1]")
    ]


def test_extractive_stream_sends_one_answer_event(client) -> None:
    ingest_leave_policy(client)
    events = stream_events(client)
    assert [event["type"] for event in events] == ["source", "answer", "end"]
    expected = client.post("/api/v1/query", json=QUESTION).json()
    assert events[1]["answer"] == expected["answer"]
    assert events[0]["sources"] == expected["sources"]


def test_failure_partway_sends_error_event_and_saves_nothing(llm_client, mock_llm) -> None:
    ingest_leave_policy(llm_client)
    mock_llm.stream_reply("Twenty four", error=httpx.ReadError("connection reset"))
    events = stream_events(llm_client)
    assert [event["type"] for event in events] == ["source", "answer", "error"]
    assert events[-1]["error"] == "upstream_service_error"
    assert llm_client.get("/api/v1/stats").json()["query_count"] == 0


def test_unexpected_failure_sends_internal_error_event(settings, vector_store) -> None:
    class BrokenGenerator:
        def generate(self, question, hits, *, query_id=None):
            raise NotImplementedError

        def stream(self, question, hits, *, query_id=None):
            yield "partial"
            raise RuntimeError("bug")

    app = create_app(settings, vector_store=vector_store, answer_generator=BrokenGenerator())
    with TestClient(app) as test_client:
        events = stream_events(test_client)
        assert events[-1] == {
            "type": "error",
            "error": "internal_error",
            "message": "The answer stream failed",
        }
        assert test_client.get("/api/v1/stats").json()["query_count"] == 0


def test_retrieval_errors_keep_the_json_error_shape(llm_client, mock_llm) -> None:
    response = llm_client.post("/api/v1/query/stream", json={"question": "   "})
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert response.json()["error"] == "validation_error"
    assert mock_llm.requests == []
