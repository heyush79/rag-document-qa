from __future__ import annotations

from collections.abc import Iterator

from app.services.answer_generation import AnswerGenerator
from app.services.vector_store import SearchHit


class AnswerAgent:
    def __init__(self, generator: AnswerGenerator) -> None:
        self.generator = generator

    def run(self, *, question: str, hits: list[SearchHit], query_id: str | None = None) -> str:
        return self.generator.generate(question, hits, query_id=query_id)

    def stream(
        self, *, question: str, hits: list[SearchHit], query_id: str | None = None
    ) -> Iterator[str]:
        return self.generator.stream(question, hits, query_id=query_id)
