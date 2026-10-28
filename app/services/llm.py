from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.exceptions import UpstreamServiceError
from app.db.repositories import LLMCallRepository
from app.services.tokens import estimate_tokens

Message = dict[str, str]


@dataclass(frozen=True)
class LLMProvider:
    """One OpenAI-compatible `/chat/completions` server and the model to ask it for."""

    base_url: str
    model: str
    api_key: str | None = None

    @property
    def url(self) -> str:
        return self.base_url.rstrip("/")

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key or 'local-key'}"}


@dataclass(frozen=True)
class Completion:
    text: str
    provider: LLMProvider
    prompt_tokens: int
    completion_tokens: int
    # True when the server reported no usage and the counts are chars/4 estimates.
    tokens_estimated: bool


@dataclass(frozen=True)
class ProviderStatus:
    base_url: str
    model: str
    ok: bool
    detail: str | None = None


_PAYLOAD_ERRORS = (httpx.HTTPError, AttributeError, KeyError, IndexError, TypeError, ValueError)


def _prompt_text(messages: Sequence[Message]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages)


def _token_counts(
    usage: Any, messages: Sequence[Message], text: str
) -> tuple[int, int, bool]:
    """Counts from the server's `usage`, or estimates when it reported none."""
    if isinstance(usage, dict):
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if isinstance(prompt, int) and isinstance(completion, int):
            return prompt, completion, False
    return estimate_tokens(_prompt_text(messages)), estimate_tokens(text), True


class FallbackLLM:
    """Tries providers in order, and records every attempt when given a call repository.

    A non-streaming call moves to the next provider on any upstream error. A streaming call
    moves on only while nothing has been yielded yet: once the caller has seen part of an
    answer, a second provider would start a different one.
    """

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        *,
        timeout_seconds: float,
        calls: LLMCallRepository | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not providers:
            raise ValueError("At least one LLM provider is required")
        self.providers = list(providers)
        self.timeout_seconds = timeout_seconds
        self.calls = calls
        # Long-lived so connections are reused; tests pass one backed by a mock transport.
        self.client = client or httpx.Client()

    def complete(
        self,
        messages: Sequence[Message],
        *,
        purpose: str = "answer",
        query_id: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> Completion:
        last_error: UpstreamServiceError | None = None
        for provider in self.providers:
            started = time.perf_counter()
            try:
                completion = self._complete(provider, messages, temperature, max_tokens)
            except UpstreamServiceError as exc:
                self._record(provider, purpose, query_id, started, error=exc)
                last_error = exc
                continue
            self._record(provider, purpose, query_id, started, completion=completion)
            return completion
        raise self._all_failed(last_error)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        purpose: str = "answer",
        query_id: str | None = None,
    ) -> Iterator[str | Completion]:
        """Yields pieces of text, then one Completion holding the whole text and its usage."""
        last_error: UpstreamServiceError | None = None
        for provider in self.providers:
            started = time.perf_counter()
            sent_text = False
            try:
                for item in self._stream(provider, messages):
                    if isinstance(item, Completion):
                        self._record(provider, purpose, query_id, started, completion=item)
                        yield item
                        return
                    sent_text = True
                    yield item
            except UpstreamServiceError as exc:
                self._record(provider, purpose, query_id, started, error=exc)
                if sent_text:
                    raise
                last_error = exc
        raise self._all_failed(last_error)

    def check_providers(self) -> list[ProviderStatus]:
        """Calls `GET /models` on each provider: a cheap check of reachability and the key."""
        statuses: list[ProviderStatus] = []
        for provider in self.providers:
            try:
                response = self.client.get(
                    f"{provider.url}/models",
                    headers=provider.headers,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                statuses.append(
                    ProviderStatus(provider.base_url, provider.model, ok=False, detail=str(exc))
                )
            else:
                statuses.append(ProviderStatus(provider.base_url, provider.model, ok=True))
        return statuses

    def _complete(
        self,
        provider: LLMProvider,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int | None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": provider.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        try:
            response = self.client.post(
                f"{provider.url}/chat/completions",
                headers=provider.headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            text = str(payload["choices"][0]["message"]["content"]).strip()
        except _PAYLOAD_ERRORS as exc:
            raise UpstreamServiceError(f"Answer model request failed: {exc}") from exc
        prompt_tokens, completion_tokens, estimated = _token_counts(
            payload.get("usage"), messages, text
        )
        return Completion(text, provider, prompt_tokens, completion_tokens, estimated)

    def _stream(
        self, provider: LLMProvider, messages: Sequence[Message]
    ) -> Iterator[str | Completion]:
        pieces: list[str] = []
        usage = None
        try:
            with self.client.stream(
                "POST",
                f"{provider.url}/chat/completions",
                headers=provider.headers,
                json={
                    "model": provider.model,
                    "messages": list(messages),
                    "temperature": 0.1,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                timeout=self.timeout_seconds,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue  # blank separators, comments and other SSE fields
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise UpstreamServiceError(
                            f"Answer model stream failed: {chunk['error']}"
                        )
                    # OpenAI sends usage in a final chunk with no choices; Gemini puts it on
                    # the last text chunk. Either way it is the last usage seen.
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    piece = (choices[0].get("delta") or {}).get("content") if choices else None
                    if piece:
                        pieces.append(piece)
                        yield piece
        except _PAYLOAD_ERRORS as exc:
            raise UpstreamServiceError(f"Answer model request failed: {exc}") from exc
        text = "".join(pieces)
        prompt_tokens, completion_tokens, estimated = _token_counts(usage, messages, text)
        yield Completion(text, provider, prompt_tokens, completion_tokens, estimated)

    def _record(
        self,
        provider: LLMProvider,
        purpose: str,
        query_id: str | None,
        started: float,
        *,
        completion: Completion | None = None,
        error: UpstreamServiceError | None = None,
    ) -> None:
        if self.calls is None:
            return
        self.calls.create(
            query_id=query_id,
            purpose=purpose,
            provider=provider.base_url,
            model=provider.model,
            prompt_tokens=completion.prompt_tokens if completion else None,
            completion_tokens=completion.completion_tokens if completion else None,
            tokens_estimated=bool(completion and completion.tokens_estimated),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            ok=completion is not None,
            error=error.message if error else None,
        )

    def _all_failed(self, last_error: UpstreamServiceError | None) -> UpstreamServiceError:
        assert last_error is not None
        if len(self.providers) == 1:
            return last_error
        return UpstreamServiceError(
            f"All {len(self.providers)} LLM providers failed; last error: {last_error.message}"
        )
