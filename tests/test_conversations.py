from __future__ import annotations

import json

import pytest

from app.db.repositories import AuditRepository, CollectionRepository, ConversationRepository
from app.db.sqlite import SQLiteDatabase
from app.services.conversation_service import ConversationService
from app.services.prompting import Turn

LEAVE = {"title": "Leave", "text": "Employees receive twenty four paid leave days each year."}
REMOTE = {
    "title": "Remote",
    "text": "Employees may work remotely three days per week with manager approval.",
}
FIRST = "How many paid leave days do employees get?"
FOLLOW_UP = "And how many remote days?"


def new_conversation(client, **body) -> dict:
    response = client.post("/api/v1/conversations", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def ask(client, question: str, **extra) -> dict:
    response = client.post("/api/v1/query", json={"question": question, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def messages_of(client, conversation_id: str) -> list[tuple[str, str]]:
    detail = client.get(f"/api/v1/conversations/{conversation_id}").json()
    return [(m["role"], m["content"]) for m in detail["messages"]]


def stream_events(client, body: dict) -> list[dict]:
    response = client.post("/api/v1/query/stream", json=body)
    assert response.status_code == 200, response.text
    return [json.loads(b.removeprefix("data: ")) for b in response.text.strip().split("\n\n")]


# --- history selection --------------------------------------------------------


@pytest.fixture
def service(tmp_path):
    db = SQLiteDatabase(tmp_path / "rag.db")
    db.initialize()
    return ConversationService(
        conversations=ConversationRepository(db),
        collections=CollectionRepository(db),
        audit=AuditRepository(db),
        history_max_turns=2,
        history_token_limit=40,
    )


def exchange(service, conversation_id: str, question: str, answer: str) -> None:
    service.record_exchange(
        conversation_id=conversation_id, question=question, answer=answer, query_id=None
    )


def test_history_keeps_only_the_latest_turns(service) -> None:
    conversation = service.create(collection_id=None, title=None).conversation
    assert service.history(conversation.id) == []
    for index in range(4):
        exchange(service, conversation.id, f"q{index}", f"a{index}")
    assert service.history(conversation.id) == [Turn("q2", "a2"), Turn("q3", "a3")]


def test_history_drops_older_turns_beyond_the_token_limit(service) -> None:
    conversation = service.create(collection_id=None, title=None).conversation
    exchange(service, conversation.id, "x" * 100, "y" * 100)  # about 51 tokens: over the limit
    exchange(service, conversation.id, "short", "reply")
    assert service.history(conversation.id) == [Turn("short", "reply")]
    exchange(service, conversation.id, "z" * 200, "w" * 200)  # the newest alone is too big
    assert service.history(conversation.id) == []


def test_zero_turns_disables_history(tmp_path) -> None:
    db = SQLiteDatabase(tmp_path / "rag.db")
    db.initialize()
    service = ConversationService(
        conversations=ConversationRepository(db),
        collections=CollectionRepository(db),
        audit=AuditRepository(db),
        history_max_turns=0,
        history_token_limit=1000,
    )
    conversation = service.create(collection_id=None, title=None).conversation
    exchange(service, conversation.id, "q", "a")
    assert service.history(conversation.id) == []


def test_untitled_conversation_takes_its_first_question(service) -> None:
    conversation = service.create(collection_id=None, title=None).conversation
    exchange(service, conversation.id, "A" * 100, "answer")
    exchange(service, conversation.id, "second question", "answer")
    assert service.get(conversation.id).title == "A" * 80
    titled = service.create(collection_id=None, title="  Leave chat ").conversation
    exchange(service, titled.id, "question", "answer")
    assert service.get(titled.id).title == "Leave chat"


# --- conversation endpoints ---------------------------------------------------


def test_create_list_get_and_delete_a_conversation(client) -> None:
    created = new_conversation(client, title="Leave chat")
    assert (created["title"], created["collection_id"], created["message_count"]) == (
        "Leave chat", None, 0
    )
    listing = client.get("/api/v1/conversations").json()
    assert listing["count"] == 1 and listing["items"][0]["id"] == created["id"]
    detail = client.get(f"/api/v1/conversations/{created['id']}").json()
    assert detail["messages"] == []
    assert client.get("/api/v1/stats").json()["conversation_count"] == 1

    assert client.delete(f"/api/v1/conversations/{created['id']}").status_code == 204
    assert client.get(f"/api/v1/conversations/{created['id']}").status_code == 404
    assert client.delete(f"/api/v1/conversations/{created['id']}").status_code == 404


def test_conversation_needs_an_existing_collection(client) -> None:
    response = client.post("/api/v1/conversations", json={"collection_id": "nope"})
    assert response.status_code == 404
    hr = client.post("/api/v1/collections", json={"name": "hr"}).json()["id"]
    assert new_conversation(client, collection_id=hr)["collection_id"] == hr


def test_deleting_a_collection_removes_its_conversations(client) -> None:
    hr = client.post("/api/v1/collections", json={"name": "hr"}).json()["id"]
    conversation = new_conversation(client, collection_id=hr)
    assert client.delete(f"/api/v1/collections/{hr}").status_code == 204
    assert client.get(f"/api/v1/conversations/{conversation['id']}").status_code == 404


# --- queries in a conversation ------------------------------------------------


def test_query_in_a_conversation_saves_both_messages(client) -> None:
    client.post("/api/v1/documents/text", json=LEAVE)
    conversation = new_conversation(client)
    result = ask(client, FIRST, conversation_id=conversation["id"])
    assert result["conversation_id"] == conversation["id"]
    assert result["search_query"] == FIRST  # the extractive provider never rewrites
    assert "twenty four" in result["answer"]

    detail = client.get(f"/api/v1/conversations/{conversation['id']}").json()
    assert [(m["role"], m["content"]) for m in detail["messages"]] == [
        ("user", FIRST),
        ("assistant", result["answer"]),
    ]
    assert {m["query_id"] for m in detail["messages"]} == {result["query_id"]}
    assert (detail["title"], detail["message_count"]) == (FIRST, 2)

    again = ask(client, FOLLOW_UP, conversation_id=conversation["id"])
    assert again["search_query"] == FOLLOW_UP
    assert len(messages_of(client, conversation["id"])) == 4
    history = client.get("/api/v1/queries").json()
    assert history[0]["search_query"] == FOLLOW_UP


def test_query_without_a_conversation_is_unchanged(client) -> None:
    client.post("/api/v1/documents/text", json=LEAVE)
    result = ask(client, FIRST)
    assert (result["conversation_id"], result["search_query"]) == (None, FIRST)
    assert client.get("/api/v1/stats").json()["conversation_count"] == 0


def test_unknown_conversation_is_a_404(client) -> None:
    response = client.post("/api/v1/query", json={"question": FIRST, "conversation_id": "nope"})
    assert response.status_code == 404
    response = client.post(
        "/api/v1/query/stream", json={"question": FIRST, "conversation_id": "nope"}
    )
    assert response.status_code == 404


def test_streamed_query_in_a_conversation_saves_messages(client) -> None:
    client.post("/api/v1/documents/text", json=LEAVE)
    conversation = new_conversation(client)
    events = stream_events(client, {"question": FIRST, "conversation_id": conversation["id"]})
    assert (events[0]["conversation_id"], events[0]["search_query"]) == (
        conversation["id"], FIRST
    )
    answer = "".join(e["answer"] for e in events if e["type"] == "answer")
    assert messages_of(client, conversation["id"]) == [("user", FIRST), ("assistant", answer)]


def test_conversation_scopes_queries_to_its_collection(client) -> None:
    hr = client.post("/api/v1/collections", json={"name": "hr"}).json()["id"]
    client.post("/api/v1/documents/text", json=dict(LEAVE, collection_id=hr))
    client.post("/api/v1/documents/text", json=REMOTE)
    conversation = new_conversation(client, collection_id=hr)
    result = ask(client, "leave days", conversation_id=conversation["id"])
    assert {s["document_title"] for s in result["sources"]} == {"Leave"}
    body = {"question": "leave", "conversation_id": conversation["id"], "collection_id": "default"}
    response = client.post("/api/v1/query", json=body)
    assert response.status_code == 400
    assert "belongs to collection" in response.json()["message"]


# --- rewriting with an LLM ----------------------------------------------------


def start_conversation(llm_client, mock_llm) -> str:
    llm_client.post("/api/v1/documents/text", json=LEAVE)
    llm_client.post("/api/v1/documents/text", json=REMOTE)
    conversation = new_conversation(llm_client)
    mock_llm.reply("Twenty four days. [S1]")
    ask(llm_client, FIRST, conversation_id=conversation["id"])
    assert len(mock_llm.requests) == 1  # no history yet, so no rewrite
    return conversation["id"]


def test_follow_up_is_rewritten_for_retrieval_but_answered_as_asked(llm_client, mock_llm):
    conversation_id = start_conversation(llm_client, mock_llm)
    mock_llm.reply("remote work days per week policy")  # the rewrite
    mock_llm.reply("Three days. [S1]")  # the answer
    result = ask(llm_client, FOLLOW_UP, conversation_id=conversation_id)
    assert result["search_query"] == "remote work days per week policy"
    assert result["sources"][0]["document_title"] == "Remote"

    rewrite = json.loads(mock_llm.requests[-2].content)
    assert (rewrite["temperature"], rewrite["max_tokens"]) == (0.0, 120)
    prompt = rewrite["messages"][-1]["content"]
    assert prompt.startswith("Given the following conversation history:")
    assert f"User: {FIRST}\nAssistant: Twenty four days. [S1]" in prompt
    assert prompt.endswith(FOLLOW_UP)

    answer = mock_llm.last_json()["messages"]
    assert answer[1] == {"role": "user", "content": FIRST}
    assert answer[2] == {"role": "assistant", "content": "Twenty four days. [S1]"}
    assert answer[-1]["content"].endswith(f"Question: {FOLLOW_UP}")
    assert "remote work days per week policy" not in answer[-1]["content"]

    calls = llm_client.app.state.container.llm_calls.list_for_query(result["query_id"])
    assert [(c.purpose, c.ok) for c in calls] == [("rewrite", True), ("answer", True)]
    assert messages_of(llm_client, conversation_id)[-2:] == [
        ("user", FOLLOW_UP),
        ("assistant", "Three days. [S1]"),
    ]


def test_failed_rewrite_falls_back_to_the_question(llm_client, mock_llm) -> None:
    conversation_id = start_conversation(llm_client, mock_llm)
    mock_llm.fail(500)  # the rewrite
    mock_llm.reply("Three days. [S1]")  # the answer
    result = ask(llm_client, FOLLOW_UP, conversation_id=conversation_id)
    assert (result["search_query"], result["answer"]) == (FOLLOW_UP, "Three days. [S1]")
    calls = llm_client.app.state.container.llm_calls.list_for_query(result["query_id"])
    assert [(c.purpose, c.ok) for c in calls] == [("rewrite", False), ("answer", True)]


def test_empty_rewrite_falls_back_to_the_question(llm_client, mock_llm) -> None:
    conversation_id = start_conversation(llm_client, mock_llm)
    mock_llm.reply("  ")
    mock_llm.reply("Three days. [S1]")
    assert ask(llm_client, FOLLOW_UP, conversation_id=conversation_id)["search_query"] == FOLLOW_UP


def test_streamed_follow_up_rewrites_then_streams(llm_client, mock_llm) -> None:
    conversation_id = start_conversation(llm_client, mock_llm)
    mock_llm.reply("remote work days per week policy")
    mock_llm.stream_reply("Three", " days.", " [S1]")
    events = stream_events(llm_client, {"question": FOLLOW_UP, "conversation_id": conversation_id})
    assert events[0]["search_query"] == "remote work days per week policy"
    assert [e["type"] for e in events] == ["source", "answer", "answer", "answer", "end"]
    assert messages_of(llm_client, conversation_id)[-1] == ("assistant", "Three days. [S1]")
