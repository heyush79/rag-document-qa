from __future__ import annotations

import json

import httpx
import pytest

from app.core.exceptions import UpstreamServiceError
from app.db.repositories import LLMCallRepository
from app.db.sqlite import SQLiteDatabase
from app.services.llm import FallbackLLM, LLMProvider
from app.services.tokens import estimate_tokens

PRIMARY = LLMProvider(base_url="http://primary/v1", model="primary-model", api_key="key-1")
BACKUP = LLMProvider(base_url="http://backup/v1", model="backup-model", api_key="key-2")
MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "How many leave days?"},
]
USAGE = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}


def completion(content: str, usage: dict | None = None) -> httpx.Response:
    payload: dict = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ]
    }
    if usage is not None:
        payload["usage"] = usage
    return httpx.Response(200, json=payload)


def chunk(content: str | None = None, usage: dict | None = None) -> str:
    payload: dict = {"choices": []}
    if content is not None:
        payload["choices"] = [{"index": 0, "delta": {"content": content}}]
    if usage is not None:
        payload["usage"] = usage
    return "data: " + json.dumps(payload)


def sse(*lines: str, done: bool = True) -> httpx.Response:
    body = "".join(f"{line}\n\n" for line in lines)
    if done:
        body += "data: [DONE]\n\n"
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=body.encode()
    )


class FakeProviders:
    """Routes requests by host. Each host has a queue of responses or exceptions to raise;
    with nothing queued it answers with a completion that names the host."""

    def __init__(self) -> None:
        self.queues: dict[str, list[httpx.Response | Exception]] = {}
        self.requests: list[httpx.Request] = []
        self.client = httpx.Client(transport=httpx.MockTransport(self._handle))

    def queue(self, provider: LLMProvider, *items: httpx.Response | Exception) -> None:
        self.queues.setdefault(httpx.URL(provider.base_url).host, []).extend(items)

    def hosts_called(self) -> list[str]:
        return [request.url.host for request in self.requests]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        queue = self.queues.get(request.url.host, [])
        if queue:
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return completion(f"answer from {request.url.host}")


@pytest.fixture
def providers():
    fake = FakeProviders()
    yield fake
    fake.client.close()


@pytest.fixture
def calls(tmp_path) -> LLMCallRepository:
    db = SQLiteDatabase(tmp_path / "rag.db")
    db.initialize()
    return LLMCallRepository(db)


def llm(providers: FakeProviders, *, calls=None, chain=(PRIMARY, BACKUP)) -> FallbackLLM:
    return FallbackLLM(list(chain), timeout_seconds=5, calls=calls, client=providers.client)


def test_requires_at_least_one_provider(providers) -> None:
    with pytest.raises(ValueError):
        FallbackLLM([], timeout_seconds=5, client=providers.client)


def test_uses_the_primary_provider_when_it_works(providers) -> None:
    result = llm(providers).complete(MESSAGES)
    assert result.text == "answer from primary"
    assert result.provider == PRIMARY
    assert providers.hosts_called() == ["primary"]
    request = providers.requests[0]
    assert str(request.url) == "http://primary/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer key-1"
    assert json.loads(request.content)["model"] == "primary-model"


@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(500, json={"error": {"message": "boom"}}),
        httpx.Response(429, json={"error": {"message": "rate limited"}}),
        httpx.ReadTimeout("slow"),
        httpx.ConnectError("refused"),
        httpx.Response(200, json={"unexpected": True}),
    ],
    ids=["500", "429", "timeout", "connect-error", "malformed"],
)
def test_falls_back_when_the_primary_fails(providers, failure) -> None:
    providers.queue(PRIMARY, failure)
    result = llm(providers).complete(MESSAGES)
    assert result.text == "answer from backup"
    assert providers.hosts_called() == ["primary", "backup"]
    assert json.loads(providers.requests[-1].content)["model"] == "backup-model"


def test_raises_when_every_provider_fails(providers) -> None:
    providers.queue(PRIMARY, httpx.Response(500))
    providers.queue(BACKUP, httpx.Response(503))
    with pytest.raises(UpstreamServiceError, match="All 2 LLM providers failed"):
        llm(providers).complete(MESSAGES)


def test_single_provider_error_is_passed_through(providers) -> None:
    providers.queue(PRIMARY, httpx.Response(503))
    with pytest.raises(UpstreamServiceError, match="Answer model request failed"):
        llm(providers, chain=(PRIMARY,)).complete(MESSAGES)


def test_stream_falls_back_before_any_text_was_sent(providers) -> None:
    providers.queue(PRIMARY, httpx.Response(429))
    providers.queue(BACKUP, sse(chunk("Twenty"), chunk(" four")))
    items = list(llm(providers).stream(MESSAGES))
    assert items[:-1] == ["Twenty", " four"]
    assert items[-1].text == "Twenty four"
    assert items[-1].provider == BACKUP
    assert providers.hosts_called() == ["primary", "backup"]
    assert json.loads(providers.requests[-1].content)["stream_options"] == {
        "include_usage": True
    }


def test_stream_does_not_fall_back_after_text_was_sent(providers) -> None:
    providers.queue(
        PRIMARY, sse(chunk("Twenty"), 'data: {"error": {"message": "quota"}}', done=False)
    )
    stream = llm(providers).stream(MESSAGES)
    assert next(stream) == "Twenty"
    with pytest.raises(UpstreamServiceError, match="quota"):
        next(stream)
    assert providers.hosts_called() == ["primary"]


def test_records_reported_usage(providers, calls) -> None:
    providers.queue(PRIMARY, completion("ok", usage=USAGE))
    llm(providers, calls=calls).complete(MESSAGES, purpose="answer", query_id="q-1")
    (row,) = calls.list_for_query("q-1")
    assert (row.ok, row.error) == (True, None)
    assert (row.provider, row.model) == ("http://primary/v1", "primary-model")
    assert row.purpose == "answer"
    assert (row.prompt_tokens, row.completion_tokens, row.tokens_estimated) == (12, 3, False)
    assert row.latency_ms >= 0


def test_estimates_usage_when_the_server_reports_none(providers, calls) -> None:
    llm(providers, calls=calls).complete(MESSAGES, query_id="q-1")
    (row,) = calls.list_for_query("q-1")
    assert row.tokens_estimated is True
    assert row.prompt_tokens == estimate_tokens("sys\nHow many leave days?")
    assert row.completion_tokens == estimate_tokens("answer from primary")


def test_records_the_failed_attempt_and_the_successful_one(providers, calls) -> None:
    providers.queue(PRIMARY, httpx.Response(500, json={"error": {"message": "boom"}}))
    llm(providers, calls=calls).complete(MESSAGES, query_id="q-1")
    failed, succeeded = calls.list_for_query("q-1")
    assert (failed.ok, failed.provider, failed.prompt_tokens) == (False, "http://primary/v1", None)
    assert "500" in failed.error
    assert (succeeded.ok, succeeded.provider) == (True, "http://backup/v1")
    totals = calls.totals()
    assert (totals.calls, totals.failed_calls) == (2, 1)
    assert totals.completion_tokens == estimate_tokens("answer from backup")


def test_stream_reads_usage_from_the_last_text_chunk(providers) -> None:
    # Gemini's OpenAI-compatible endpoint reports usage on the final content chunk.
    providers.queue(PRIMARY, sse(chunk("ok", usage=USAGE)))
    *_, result = llm(providers).stream(MESSAGES)
    assert (result.prompt_tokens, result.completion_tokens, result.tokens_estimated) == (
        12, 3, False
    )


def test_stream_reads_usage_from_a_trailing_chunk(providers) -> None:
    # OpenAI sends usage in a final chunk with no choices.
    providers.queue(PRIMARY, sse(chunk("ok"), chunk(usage=USAGE)))
    *pieces, result = llm(providers).stream(MESSAGES)
    assert pieces == ["ok"]
    assert (result.prompt_tokens, result.tokens_estimated) == (12, False)


def test_stream_estimates_usage_when_none_is_reported(providers, calls) -> None:
    providers.queue(PRIMARY, sse(chunk("ok"), chunk(" then")))
    *_, result = llm(providers, calls=calls).stream(MESSAGES, query_id="q-1")
    assert result.tokens_estimated is True
    assert result.completion_tokens == estimate_tokens("ok then")
    (row,) = calls.list_for_query("q-1")
    assert (row.ok, row.tokens_estimated) == (True, True)
    assert row.completion_tokens == estimate_tokens("ok then")


def test_check_providers_reports_each_one(providers) -> None:
    providers.queue(BACKUP, httpx.Response(401, json={"error": {"message": "bad key"}}))
    primary, backup = llm(providers).check_providers()
    assert (primary.base_url, primary.model, primary.ok, primary.detail) == (
        "http://primary/v1", "primary-model", True, None
    )
    assert (backup.ok, backup.base_url) == (False, "http://backup/v1")
    assert "401" in backup.detail
    request = providers.requests[0]
    assert (request.method, str(request.url)) == ("GET", "http://primary/v1/models")
    assert request.headers["Authorization"] == "Bearer key-1"
