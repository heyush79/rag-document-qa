from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from app.agents.answer_agent import AnswerAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.db.repositories import ConversationRecord, QueryRepository
from app.services.collection_service import CollectionService
from app.services.conversation_service import ConversationService
from app.services.prompting import Turn
from app.services.query_rewriter import QueryRewriter
from app.services.vector_store import SearchHit


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    question: str
    answer: str
    sources: list[SearchHit]
    latency_ms: float
    # What was searched: the question itself, or its rewrite from the conversation.
    search_query: str
    conversation_id: str | None


@dataclass(frozen=True)
class QueryStream:
    """A streamed answer. `events` yields pieces of answer text, then the saved QueryResult."""

    sources: list[SearchHit]
    events: Iterator[str | QueryResult]
    search_query: str
    conversation_id: str | None


@dataclass(frozen=True)
class _Prepared:
    """Everything known before the answer is generated."""

    query_id: str
    question: str
    search_query: str
    conversation: ConversationRecord | None
    history: list[Turn]
    hits: list[SearchHit]


class RAGOrchestrator:
    """Deterministic agent workflow: retrieve, generate a grounded answer, persist trace data."""

    def __init__(
        self,
        *,
        retrieval_agent: RetrievalAgent,
        answer_agent: AnswerAgent,
        query_repository: QueryRepository,
        collection_service: CollectionService,
        conversation_service: ConversationService,
        query_rewriter: QueryRewriter,
    ) -> None:
        self.retrieval_agent = retrieval_agent
        self.answer_agent = answer_agent
        self.query_repository = query_repository
        self.collection_service = collection_service
        self.conversation_service = conversation_service
        self.query_rewriter = query_rewriter

    def run(
        self,
        *,
        question: str,
        top_k: int,
        document_id: str | None = None,
        collection_id: str | None = None,
        conversation_id: str | None = None,
    ) -> QueryResult:
        started = time.perf_counter()
        prepared = self._prepare(
            question=question,
            top_k=top_k,
            document_id=document_id,
            collection_id=collection_id,
            conversation_id=conversation_id,
        )
        answer = self.answer_agent.run(
            question=prepared.question,
            hits=prepared.hits,
            history=prepared.history,
            query_id=prepared.query_id,
        )
        return self._save(prepared, answer=answer, top_k=top_k, started=started)

    def stream(
        self,
        *,
        question: str,
        top_k: int,
        document_id: str | None = None,
        collection_id: str | None = None,
        conversation_id: str | None = None,
    ) -> QueryStream:
        """Retrieves before returning, so retrieval errors raise before anything is streamed."""
        started = time.perf_counter()
        prepared = self._prepare(
            question=question,
            top_k=top_k,
            document_id=document_id,
            collection_id=collection_id,
            conversation_id=conversation_id,
        )
        return QueryStream(
            sources=prepared.hits,
            events=self._stream_answer(prepared, top_k=top_k, started=started),
            search_query=prepared.search_query,
            conversation_id=prepared.conversation.id if prepared.conversation else None,
        )

    def _prepare(
        self,
        *,
        question: str,
        top_k: int,
        document_id: str | None,
        collection_id: str | None,
        conversation_id: str | None,
    ) -> _Prepared:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty")
        # Chosen up front so LLM calls can be logged against the query before it is saved.
        query_id = str(uuid.uuid4())
        conversation = None
        history: list[Turn] = []
        if conversation_id is not None:
            conversation = self.conversation_service.get(conversation_id)
            if conversation.collection_id is not None:
                if collection_id not in (None, conversation.collection_id):
                    raise ValueError(
                        f"The conversation belongs to collection "
                        f"'{conversation.collection_id}', not '{collection_id}'"
                    )
                collection_id = conversation.collection_id
            history = self.conversation_service.history(conversation.id)
        search_query = self.query_rewriter.rewrite(clean_question, history, query_id=query_id)
        hits = self.retrieval_agent.run(
            question=search_query,
            top_k=top_k,
            document_id=document_id,
            document_ids=self._scope(collection_id),
        )
        return _Prepared(query_id, clean_question, search_query, conversation, history, hits)

    def _scope(self, collection_id: str | None) -> list[str] | None:
        """The document ids a query may search, or None for everything. 404 if unknown."""
        if collection_id is None:
            return None
        return self.collection_service.document_ids(collection_id)

    def _stream_answer(
        self, prepared: _Prepared, *, top_k: int, started: float
    ) -> Iterator[str | QueryResult]:
        pieces: list[str] = []
        for piece in self.answer_agent.stream(
            question=prepared.question,
            hits=prepared.hits,
            history=prepared.history,
            query_id=prepared.query_id,
        ):
            pieces.append(piece)
            yield piece
        # Only reached when the answer stream completes, so a failed stream saves nothing.
        yield self._save(prepared, answer="".join(pieces).strip(), top_k=top_k, started=started)

    def _save(
        self, prepared: _Prepared, *, answer: str, top_k: int, started: float
    ) -> QueryResult:
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        self.query_repository.create(
            query_id=prepared.query_id,
            question=prepared.question,
            answer=answer,
            top_k=top_k,
            latency_ms=latency_ms,
            search_query=prepared.search_query,
            sources=[
                {
                    "document_id": hit.document_id,
                    "chunk_id": hit.chunk_id,
                    "score": hit.score,
                }
                for hit in prepared.hits
            ],
        )
        conversation_id = None
        if prepared.conversation is not None:
            conversation_id = prepared.conversation.id
            self.conversation_service.record_exchange(
                conversation_id=conversation_id,
                question=prepared.question,
                answer=answer,
                query_id=prepared.query_id,
            )
        return QueryResult(
            query_id=prepared.query_id,
            question=prepared.question,
            answer=answer,
            sources=prepared.hits,
            latency_ms=latency_ms,
            search_query=prepared.search_query,
            conversation_id=conversation_id,
        )
