from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.core.exceptions import ConflictError
from app.db.sqlite import SQLiteDatabase


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    title: str
    source: str
    content_hash: str
    chunk_count: int
    status: str
    error_message: str | None
    created_at: str


@dataclass(frozen=True)
class QueryRecord:
    id: str
    question: str
    answer: str
    top_k: int
    latency_ms: float
    created_at: str


class DocumentRepository:
    def __init__(self, db: SQLiteDatabase) -> None:
        self.db = db

    @staticmethod
    def _to_record(row: sqlite3.Row | None) -> DocumentRecord | None:
        return DocumentRecord(**dict(row)) if row else None

    def create_processing(
        self,
        *,
        document_id: str,
        title: str,
        source: str,
        content_hash: str,
    ) -> DocumentRecord:
        created_at = utc_now_iso()
        try:
            with self.db.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO documents(id, title, source, content_hash, status, created_at)
                    VALUES (?, ?, ?, ?, 'processing', ?)
                    """,
                    (document_id, title, source, content_hash, created_at),
                )
        except sqlite3.IntegrityError as exc:
            raise ConflictError("A document with the same content already exists") from exc
        record = self.get(document_id)
        assert record is not None
        return record

    def get(self, document_id: str) -> DocumentRecord | None:
        with self.db.connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._to_record(row)

    def get_by_hash(self, content_hash: str) -> DocumentRecord | None:
        with self.db.connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
            ).fetchone()
        return self._to_record(row)

    def list(self, *, limit: int = 100, offset: int = 0) -> list[DocumentRecord]:
        with self.db.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM documents
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [DocumentRecord(**dict(row)) for row in rows]

    def mark_ready(self, document_id: str, chunk_count: int) -> None:
        with self.db.connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = 'ready', chunk_count = ?, error_message = NULL
                WHERE id = ?
                """,
                (chunk_count, document_id),
            )

    def mark_failed(self, document_id: str, error_message: str) -> None:
        with self.db.connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = 'failed', error_message = ?
                WHERE id = ?
                """,
                (error_message[:1000], document_id),
            )

    def delete(self, document_id: str) -> bool:
        with self.db.connect() as connection:
            cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cursor.rowcount > 0

    def count(self) -> int:
        with self.db.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
        return int(row["count"])


class QueryRepository:
    def __init__(self, db: SQLiteDatabase) -> None:
        self.db = db

    def create(
        self,
        *,
        query_id: str,
        question: str,
        answer: str,
        top_k: int,
        latency_ms: float,
        sources: list[dict[str, Any]],
    ) -> QueryRecord:
        created_at = utc_now_iso()
        with self.db.connect() as connection:
            connection.execute(
                """
                INSERT INTO queries(id, question, answer, top_k, latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (query_id, question, answer, top_k, latency_ms, created_at),
            )
            connection.executemany(
                """
                INSERT INTO query_sources(query_id, rank, document_id, chunk_id, score)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        query_id,
                        rank,
                        source["document_id"],
                        source["chunk_id"],
                        source["score"],
                    )
                    for rank, source in enumerate(sources, start=1)
                ],
            )
        return QueryRecord(
            id=query_id,
            question=question,
            answer=answer,
            top_k=top_k,
            latency_ms=latency_ms,
            created_at=created_at,
        )

    def list_recent(self, limit: int = 20) -> list[QueryRecord]:
        with self.db.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM queries ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [QueryRecord(**dict(row)) for row in rows]

    def count(self) -> int:
        with self.db.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM queries").fetchone()
        return int(row["count"])


@dataclass(frozen=True)
class LLMCallRecord:
    id: int
    query_id: str | None
    purpose: str
    provider: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    tokens_estimated: bool
    latency_ms: float
    ok: bool
    error: str | None
    created_at: str


@dataclass(frozen=True)
class LLMUsageTotals:
    calls: int
    failed_calls: int
    prompt_tokens: int
    completion_tokens: int


class LLMCallRepository:
    def __init__(self, db: SQLiteDatabase) -> None:
        self.db = db

    def create(
        self,
        *,
        query_id: str | None,
        purpose: str,
        provider: str,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        tokens_estimated: bool,
        latency_ms: float,
        ok: bool,
        error: str | None,
    ) -> None:
        with self.db.connect() as connection:
            connection.execute(
                """
                INSERT INTO llm_calls(
                    query_id, purpose, provider, model, prompt_tokens, completion_tokens,
                    tokens_estimated, latency_ms, ok, error, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    purpose,
                    provider,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    int(tokens_estimated),
                    latency_ms,
                    int(ok),
                    error[:1000] if error else None,
                    utc_now_iso(),
                ),
            )

    def list_for_query(self, query_id: str) -> list[LLMCallRecord]:
        with self.db.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM llm_calls WHERE query_id = ? ORDER BY id", (query_id,)
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def list_recent(self, limit: int = 20) -> list[LLMCallRecord]:
        with self.db.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM llm_calls ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._to_record(row) for row in rows]

    def totals(self) -> LLMUsageTotals:
        with self.db.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS calls,
                    COALESCE(SUM(ok = 0), 0) AS failed_calls,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens
                FROM llm_calls
                """
            ).fetchone()
        return LLMUsageTotals(
            calls=int(row["calls"]),
            failed_calls=int(row["failed_calls"]),
            prompt_tokens=int(row["prompt_tokens"]),
            completion_tokens=int(row["completion_tokens"]),
        )

    @staticmethod
    def _to_record(row: sqlite3.Row) -> LLMCallRecord:
        values = dict(row)
        values["tokens_estimated"] = bool(values["tokens_estimated"])
        values["ok"] = bool(values["ok"])
        return LLMCallRecord(**values)


class AuditRepository:
    def __init__(self, db: SQLiteDatabase) -> None:
        self.db = db

    def record(self, event_type: str, entity_id: str | None, payload: dict[str, Any]) -> None:
        with self.db.connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_events(event_type, entity_id, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (event_type, entity_id, json.dumps(payload, sort_keys=True), utc_now_iso()),
            )
