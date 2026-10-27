from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from app.agents.answer_agent import AnswerAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.db.repositories import QueryRepository
from app.services.vector_store import SearchHit


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    question: str
    answer: str
    sources: list[SearchHit]
    latency_ms: float


@dataclass(frozen=True)
class QueryStream:
    """A streamed answer. `events` yields pieces of answer text, then the saved QueryResult."""

    sources: list[SearchHit]
    events: Iterator[str | QueryResult]


class RAGOrchestrator:
    """Deterministic agent workflow: retrieve, generate a grounded answer, persist trace data."""

    def __init__(
        self,
        *,
        retrieval_agent: RetrievalAgent,
        answer_agent: AnswerAgent,
        query_repository: QueryRepository,
    ) -> None:
        self.retrieval_agent = retrieval_agent
        self.answer_agent = answer_agent
        self.query_repository = query_repository

    def run(
        self,
        *,
        question: str,
        top_k: int,
        document_id: str | None = None,
    ) -> QueryResult:
        started = time.perf_counter()
        # Chosen up front so LLM calls can be logged against the query before it is saved.
        query_id = str(uuid.uuid4())
        hits = self.retrieval_agent.run(
            question=question,
            top_k=top_k,
            document_id=document_id,
        )
        answer = self.answer_agent.run(question=question, hits=hits, query_id=query_id)
        return self._save(
            query_id=query_id,
            question=question,
            answer=answer,
            top_k=top_k,
            hits=hits,
            started=started,
        )

    def stream(
        self,
        *,
        question: str,
        top_k: int,
        document_id: str | None = None,
    ) -> QueryStream:
        """Retrieves before returning, so retrieval errors raise before anything is streamed."""
        started = time.perf_counter()
        query_id = str(uuid.uuid4())
        hits = self.retrieval_agent.run(
            question=question,
            top_k=top_k,
            document_id=document_id,
        )
        return QueryStream(
            sources=hits,
            events=self._stream_answer(
                query_id=query_id, question=question, top_k=top_k, hits=hits, started=started
            ),
        )

    def _stream_answer(
        self,
        *,
        query_id: str,
        question: str,
        top_k: int,
        hits: list[SearchHit],
        started: float,
    ) -> Iterator[str | QueryResult]:
        pieces: list[str] = []
        for piece in self.answer_agent.stream(question=question, hits=hits, query_id=query_id):
            pieces.append(piece)
            yield piece
        # Only reached when the answer stream completes, so a failed stream saves nothing.
        yield self._save(
            query_id=query_id,
            question=question,
            answer="".join(pieces).strip(),
            top_k=top_k,
            hits=hits,
            started=started,
        )

    def _save(
        self,
        *,
        query_id: str,
        question: str,
        answer: str,
        top_k: int,
        hits: list[SearchHit],
        started: float,
    ) -> QueryResult:
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        self.query_repository.create(
            query_id=query_id,
            question=question.strip(),
            answer=answer,
            top_k=top_k,
            latency_ms=latency_ms,
            sources=[
                {
                    "document_id": hit.document_id,
                    "chunk_id": hit.chunk_id,
                    "score": hit.score,
                }
                for hit in hits
            ],
        )
        return QueryResult(
            query_id=query_id,
            question=question.strip(),
            answer=answer,
            sources=hits,
            latency_ms=latency_ms,
        )
