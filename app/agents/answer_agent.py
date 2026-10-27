from __future__ import annotations

from collections.abc import Iterator

from app.services.answer_generation import AnswerGenerator
from app.services.vector_store import SearchHit


class AnswerAgent:
    def __init__(self, generator: AnswerGenerator) -> None:
        self.generator = generator

    def run(self, *, question: str, hits: list[SearchHit]) -> str:
        return self.generator.generate(question, hits)

    def stream(self, *, question: str, hits: list[SearchHit]) -> Iterator[str]:
        return self.generator.stream(question, hits)
