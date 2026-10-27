from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Protocol

import httpx

from app.core.exceptions import UpstreamServiceError
from app.services.prompting import build_answer_messages
from app.services.vector_store import SearchHit


class AnswerGenerator(Protocol):
    def generate(self, question: str, hits: list[SearchHit]) -> str: ...

    def stream(self, question: str, hits: list[SearchHit]) -> Iterator[str]: ...


class ExtractiveAnswerGenerator:
    """Offline fallback that returns grounded sentences and source markers."""

    WORD_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
        "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
        "what", "when", "where", "which", "who", "why", "with",
    }

    def __init__(self, min_relevance_score: float = 0.15) -> None:
        self.min_relevance_score = min_relevance_score

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        useful_hits = [hit for hit in hits if hit.score >= self.min_relevance_score]
        if not useful_hits:
            return "I could not find enough relevant information in the indexed documents."

        query_terms = {
            word
            for word in self.WORD_PATTERN.findall(question.lower())
            if word not in self.STOP_WORDS
        }
        candidates: list[tuple[int, float, int, str]] = []
        for source_number, hit in enumerate(useful_hits, start=1):
            for sentence in self.SENTENCE_PATTERN.split(hit.text):
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                sentence_terms = set(self.WORD_PATTERN.findall(sentence.lower()))
                overlap = len(query_terms & sentence_terms)
                candidates.append((overlap, hit.score, source_number, sentence))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected: list[str] = []
        seen: set[str] = set()
        for _, _, source_number, sentence in candidates:
            normalized = sentence.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            selected.append(f"{sentence} [S{source_number}]")
            if len(selected) == 3:
                break

        if not selected:
            top = useful_hits[0]
            return f"{top.text[:700].strip()} [S1]"
        return " ".join(selected)

    def stream(self, question: str, hits: list[SearchHit]) -> Iterator[str]:
        yield self.generate(question, hits)


class OpenAICompatibleAnswerGenerator:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_seconds: float,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "local-key"
        self.timeout_seconds = timeout_seconds
        # Long-lived so connections are reused; tests pass one backed by a mock transport.
        self.client = client or httpx.Client()

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        # Called even with no hits: the prompt then says nothing relevant was found.
        messages = build_answer_messages(question, hits)
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": messages, "temperature": 0.1},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload["choices"][0]["message"]["content"]).strip()
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise UpstreamServiceError(f"Answer model request failed: {exc}") from exc

    def stream(self, question: str, hits: list[SearchHit]) -> Iterator[str]:
        """Yields the answer as the server sends it, from an SSE `"stream": true` response."""
        messages = build_answer_messages(question, hits)
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.1,
                    "stream": True,
                },
                timeout=self.timeout_seconds,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue  # blank separators, comments and other SSE fields
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        return
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise UpstreamServiceError(
                            f"Answer model stream failed: {chunk['error']}"
                        )
                    choices = chunk.get("choices") or []  # empty in usage-only chunks
                    piece = (choices[0].get("delta") or {}).get("content") if choices else None
                    if piece:
                        yield piece
        except (
            httpx.HTTPError, AttributeError, KeyError, IndexError, TypeError, ValueError
        ) as exc:
            raise UpstreamServiceError(f"Answer model request failed: {exc}") from exc
