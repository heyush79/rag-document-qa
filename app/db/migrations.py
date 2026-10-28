from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str


INITIAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK(status IN ('processing', 'ready', 'failed')),
    error_message TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

CREATE TABLE IF NOT EXISTS queries (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS query_sources (
    query_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    document_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY(query_id, rank),
    FOREIGN KEY(query_id) REFERENCES queries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_queries_created_at ON queries(created_at DESC);

CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    entity_id TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

# One row per attempt to call an LLM provider. query_id has no foreign key: a call is recorded
# before its query row exists, and a query that fails keeps its call rows.
LLM_CALLS_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT,
    purpose TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    tokens_estimated INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL,
    ok INTEGER NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_query_id ON llm_calls(query_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_created_at ON llm_calls(created_at DESC);
"""

# Only the SHA-256 hash of a key is stored, plus a short prefix so listings can tell keys apart.
API_KEYS_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    revoked_at TEXT
);
"""

# Collections group documents. Duplicate detection becomes per collection, which means a new
# UNIQUE constraint on documents; SQLite can't change constraints in place, so the table is
# rebuilt and existing rows land in the 'default' collection.
COLLECTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TEXT NOT NULL
);

INSERT OR IGNORE INTO collections(id, name, description, created_at)
VALUES ('default', 'default', 'Documents ingested without a collection',
        strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now'));

CREATE TABLE documents_new (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL DEFAULT 'default' REFERENCES collections(id),
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK(status IN ('processing', 'ready', 'failed')),
    error_message TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(collection_id, content_hash)
);

INSERT INTO documents_new(
    id, collection_id, title, source, content_hash, chunk_count, status, error_message, created_at
)
SELECT id, 'default', title, source, content_hash, chunk_count, status, error_message, created_at
FROM documents;

DROP TABLE documents;
ALTER TABLE documents_new RENAME TO documents;

CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id, created_at DESC);
"""

# Append new migrations with the next version number. Never edit one that has shipped:
# databases that already recorded it will not run it again.
MIGRATIONS: tuple[Migration, ...] = (
    # Idempotent, so databases created before migrations existed upgrade cleanly.
    Migration(1, "initial_schema", INITIAL_SCHEMA),
    Migration(2, "llm_calls", LLM_CALLS_SCHEMA),
    Migration(3, "api_keys", API_KEYS_SCHEMA),
    Migration(4, "collections", COLLECTIONS_SCHEMA),
)

MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
)
"""


def apply_migrations(
    connection: sqlite3.Connection,
    migrations: Sequence[Migration] = MIGRATIONS,
) -> list[int]:
    """Apply pending migrations in version order, each in its own transaction.

    Assumes one process migrates at a time, which holds because the app migrates on startup.
    Returns the versions applied by this call.
    """
    connection.execute(MIGRATIONS_TABLE)
    connection.commit()
    applied = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
    newly_applied: list[int] = []
    for migration in sorted(migrations, key=lambda item: item.version):
        if migration.version in applied:
            continue
        # executescript commits any open transaction first, so BEGIN goes inside the script.
        # On failure the transaction stays open and the caller's rollback undoes all of it.
        connection.executescript("BEGIN;\n" + migration.sql)
        connection.execute(
            "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?, ?, ?)",
            (migration.version, migration.name, datetime.now(UTC).isoformat()),
        )
        connection.commit()
        newly_applied.append(migration.version)
    return newly_applied
