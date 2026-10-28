from __future__ import annotations

import sqlite3

import pytest

from app.db.migrations import INITIAL_SCHEMA, MIGRATIONS, Migration, apply_migrations
from app.db.sqlite import SQLiteDatabase

ALL_VERSIONS = [migration.version for migration in MIGRATIONS]


def applied_versions(db: SQLiteDatabase) -> list[int]:
    with db.connect() as connection:
        rows = connection.execute("SELECT version FROM schema_migrations ORDER BY version")
        return [row["version"] for row in rows]


def table_names(db: SQLiteDatabase) -> set[str]:
    with db.connect() as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        return {row["name"] for row in rows}


def test_initialize_applies_all_migrations(tmp_path) -> None:
    db = SQLiteDatabase(tmp_path / "rag.db")
    db.initialize()
    assert applied_versions(db) == ALL_VERSIONS
    expected = {"documents", "queries", "query_sources", "audit_events", "llm_calls", "api_keys"}
    assert expected <= table_names(db)


def test_initialize_is_idempotent(tmp_path) -> None:
    db = SQLiteDatabase(tmp_path / "rag.db")
    db.initialize()
    db.initialize()
    assert applied_versions(db) == ALL_VERSIONS
    with db.connect() as connection:
        assert apply_migrations(connection) == []


def test_database_created_before_migrations_keeps_its_data(tmp_path) -> None:
    path = tmp_path / "rag.db"
    legacy = sqlite3.connect(path)
    legacy.executescript(INITIAL_SCHEMA)
    legacy.execute(
        """
        INSERT INTO documents(id, title, source, content_hash, status, created_at)
        VALUES ('doc-1', 'Guide', 'test', 'hash-1', 'ready', '2026-01-01T00:00:00+00:00')
        """
    )
    legacy.commit()
    legacy.close()

    db = SQLiteDatabase(path)
    db.initialize()
    assert applied_versions(db) == ALL_VERSIONS
    with db.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1


def test_failed_migration_is_rolled_back(tmp_path) -> None:
    db = SQLiteDatabase(tmp_path / "rag.db")
    db.initialize()
    broken = Migration(
        999,
        "broken",
        "CREATE TABLE half_done (id INTEGER);\nINSERT INTO missing_table VALUES (1);",
    )
    with pytest.raises(sqlite3.OperationalError), db.connect() as connection:
        apply_migrations(connection, [*MIGRATIONS, broken])
    assert "half_done" not in table_names(db)
    assert applied_versions(db) == ALL_VERSIONS
