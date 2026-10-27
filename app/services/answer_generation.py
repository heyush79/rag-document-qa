from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Protocol

from app.services.llm import FallbackLLM
from app.services.prompting import build_answer_messages
from app.services.vector_store import SearchHit


class AnswerGenerator(Protocol):
    """`query_id` is the query being answered, so LLM calls can be logged against it."""

    def generate(
        self, question: str, hits: list[SearchHit], *, query_id: str | None = None
    ) -> str: ...

    def stream(
        self, question: str, hits: list[SearchHit], *, query_id: str | None = None
    ) -> Iterator[str]: ...


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

    def generate(
        self, question: str, hits: list[SearchHit], *, query_id: str | None = None
    ) -> str:
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

    def stream(
        self, question: str, hits: list[SearchHit], *, query_id: str | None = None
    ) -> Iterator[str]:
        yield self.generate(question, hits)


class OpenAICompatibleAnswerGenerator:
    """Asks the configured LLM providers, with the retrieved chunks framed as reference data."""

    def __init__(self, llm: FallbackLLM) -> None:
        self.llm = llm

    def generate(
        self, question: str, hits: list[SearchHit], *, query_id: str | None = None
    ) -> str:
        # Called even with no hits: the prompt then says nothing relevant was found.
        messages = build_answer_messages(question, hits)
        return self.llm.complete(messages, purpose="answer", query_id=query_id).text

    def stream(
        self, question: str, hits: list[SearchHit], *, query_id: str | None = None
    ) -> Iterator[str]:
        messages = build_answer_messages(question, hits)
        for item in self.llm.stream(messages, purpose="answer", query_id=query_id):
            if isinstance(item, str):
                yield item
