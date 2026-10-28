from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.core.exceptions import ConflictError, NotFoundError
from app.db.sqlite import SQLiteDatabase

DEFAULT_COLLECTION_ID = "default"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    collection_id: str
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
        collection_id: str = DEFAULT_COLLECTION_ID,
    ) -> DocumentRecord:
        created_at = utc_now_iso()
        try:
            with self.db.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO documents(
                        id, collection_id, title, source, content_hash, status, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, 'processing', ?)
                    """,
                    (document_id, collection_id, title, source, content_hash, created_at),
                )
        except sqlite3.IntegrityError as exc:
            if "FOREIGN KEY" in str(exc):
                raise NotFoundError(f"Collection '{collection_id}' was not found") from exc
            raise ConflictError(
                "A document with the same content already exists in this collection"
            ) from exc
        record = self.get(document_id)
        assert record is not None
        return record

    def get(self, document_id: str) -> DocumentRecord | None:
        with self.db.connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._to_record(row)

    def get_by_hash(
        self, content_hash: str, collection_id: str = DEFAULT_COLLECTION_ID
    ) -> DocumentRecord | None:
        with self.db.connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE content_hash = ? AND collection_id = ?",
                (content_hash, collection_id),
            ).fetchone()
        return self._to_record(row)

    def list(
        self, *, limit: int = 100, offset: int = 0, collection_id: str | None = None
    ) -> list[DocumentRecord]:
        with self.db.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM documents
                WHERE (? IS NULL OR collection_id = ?)
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (collection_id, collection_id, limit, offset),
            ).fetchall()
        return [DocumentRecord(**dict(row)) for row in rows]

    def ids_in_collection(self, collection_id: str) -> list[str]:
        with self.db.connect() as connection:
            rows = connection.execute(
                "SELECT id FROM documents WHERE collection_id = ? ORDER BY created_at",
                (collection_id,),
            ).fetchall()
        return [row["id"] for row in rows]

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

    def count(self, collection_id: str | None = None) -> int:
        with self.db.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM documents WHERE (? IS NULL OR collection_id = ?)",
                (collection_id, collection_id),
            ).fetchone()
        return int(row["count"])


@dataclass(frozen=True)
class CollectionRecord:
    id: str
    name: str
    description: str | None
    created_at: str


class CollectionRepository:
    def __init__(self, db: SQLiteDatabase) -> None:
        self.db = db

    def create(
        self, *, collection_id: str, name: str, description: str | None
    ) -> CollectionRecord:
        created_at = utc_now_iso()
        try:
            with self.db.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO collections(id, name, description, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (collection_id, name, description, created_at),
                )
        except sqlite3.IntegrityError as exc:
            raise ConflictError(f"A collection named '{name}' already exists") from exc
        return CollectionRecord(
            id=collection_id, name=name, description=description, created_at=created_at
        )

    def get(self, collection_id: str) -> CollectionRecord | None:
        with self.db.connect() as connection:
            row = connection.execute(
                "SELECT * FROM collections WHERE id = ?", (collection_id,)
            ).fetchone()
        return CollectionRecord(**dict(row)) if row else None

    def list(self) -> list[CollectionRecord]:
        with self.db.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM collections ORDER BY created_at, name"
            ).fetchall()
        return [CollectionRecord(**dict(row)) for row in rows]

    def document_counts(self) -> dict[str, int]:
        with self.db.connect() as connection:
            rows = connection.execute(
                "SELECT collection_id, COUNT(*) AS count FROM documents GROUP BY collection_id"
            ).fetchall()
        return {row["collection_id"]: int(row["count"]) for row in rows}

    def delete(self, collection_id: str) -> bool:
        with self.db.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM collections WHERE id = ?", (collection_id,)
            )
        return cursor.rowcount > 0

    def count(self) -> int:
        with self.db.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM collections").fetchone()
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


@dataclass(frozen=True)
class ApiKeyRecord:
    id: str
    name: str
    prefix: str
    created_at: str
    last_used_at: str | None
    revoked_at: str | None


class ApiKeyRepository:
    """Stores key hashes only; the plaintext key is never written anywhere."""

    COLUMNS = "id, name, prefix, created_at, last_used_at, revoked_at"

    def __init__(self, db: SQLiteDatabase) -> None:
        self.db = db

    def create(self, *, key_id: str, name: str, prefix: str, key_hash: str) -> ApiKeyRecord:
        created_at = utc_now_iso()
        with self.db.connect() as connection:
            connection.execute(
                """
                INSERT INTO api_keys(id, name, prefix, key_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key_id, name, prefix, key_hash, created_at),
            )
        return ApiKeyRecord(
            id=key_id,
            name=name,
            prefix=prefix,
            created_at=created_at,
            last_used_at=None,
            revoked_at=None,
        )

    def get(self, key_id: str) -> ApiKeyRecord | None:
        with self.db.connect() as connection:
            row = connection.execute(
                f"SELECT {self.COLUMNS} FROM api_keys WHERE id = ?", (key_id,)
            ).fetchone()
        return ApiKeyRecord(**dict(row)) if row else None

    def find_active_by_hash(self, key_hash: str) -> ApiKeyRecord | None:
        with self.db.connect() as connection:
            row = connection.execute(
                f"SELECT {self.COLUMNS} FROM api_keys WHERE key_hash = ? AND revoked_at IS NULL",
                (key_hash,),
            ).fetchone()
        return ApiKeyRecord(**dict(row)) if row else None

    def touch(self, key_id: str) -> None:
        with self.db.connect() as connection:
            connection.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE id = ?", (utc_now_iso(), key_id)
            )

    def list(self) -> list[ApiKeyRecord]:
        with self.db.connect() as connection:
            rows = connection.execute(
                f"SELECT {self.COLUMNS} FROM api_keys ORDER BY created_at DESC, id"
            ).fetchall()
        return [ApiKeyRecord(**dict(row)) for row in rows]

    def revoke(self, key_id: str) -> bool:
        """Marks the key revoked. Returns False when there is no such key."""
        with self.db.connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM api_keys WHERE id = ?", (key_id,)
            ).fetchone()
            if not exists:
                return False
            connection.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
                (utc_now_iso(), key_id),
            )
        return True


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
