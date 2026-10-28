from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass

from app.core.exceptions import NotFoundError
from app.db.repositories import (
    DEFAULT_COLLECTION_ID,
    AuditRepository,
    CollectionRepository,
    DocumentRecord,
    DocumentRepository,
)
from app.services.chunking import TextChunker
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


INTERRUPTED_MESSAGE = "interrupted by restart; upload again"


@dataclass(frozen=True)
class IngestionResult:
    document: DocumentRecord
    duplicate: bool


@dataclass(frozen=True)
class Registration:
    """A document row created ahead of indexing, with the text that index() needs."""

    document: DocumentRecord
    duplicate: bool
    text: str


class DocumentService:
    def __init__(
        self,
        *,
        documents: DocumentRepository,
        collections: CollectionRepository,
        audit: AuditRepository,
        chunker: TextChunker,
        vector_store: VectorStore,
    ) -> None:
        self.documents = documents
        self.collections = collections
        self.audit = audit
        self.chunker = chunker
        self.vector_store = vector_store

    def ingest_text(
        self,
        *,
        title: str,
        source: str,
        text: str,
        collection_id: str | None = None,
    ) -> IngestionResult:
        registration = self.register(
            title=title, source=source, text=text, collection_id=collection_id
        )
        if registration.duplicate:
            return IngestionResult(document=registration.document, duplicate=True)
        completed = self.index(registration.document.id, registration.text)
        return IngestionResult(document=completed, duplicate=False)

    def register(
        self,
        *,
        title: str,
        source: str,
        text: str,
        collection_id: str | None = None,
    ) -> Registration:
        """Validates, detects a duplicate, and creates the document as `processing`."""
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Document text cannot be empty")
        collection_id = collection_id or DEFAULT_COLLECTION_ID
        if self.collections.get(collection_id) is None:
            raise NotFoundError(f"Collection '{collection_id}' was not found")
        content_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        # Duplicates are detected per collection: the same text may live in two collections.
        existing = self.documents.get_by_hash(content_hash, collection_id)
        if existing:
            return Registration(document=existing, duplicate=True, text=normalized_text)

        record = self.documents.create_processing(
            document_id=str(uuid.uuid4()),
            title=title.strip(),
            source=source.strip(),
            content_hash=content_hash,
            collection_id=collection_id,
        )
        return Registration(document=record, duplicate=False, text=normalized_text)

    def index(self, document_id: str, text: str) -> DocumentRecord:
        """Chunks, embeds and indexes a registered document, then marks it ready.

        On failure the partial vectors are removed, the document is marked failed with the
        error, and the error is raised again.
        """
        record = self.documents.get(document_id)
        if record is None:
            raise NotFoundError(f"Document '{document_id}' was not found")
        try:
            chunks = self.chunker.split(text)
            self.vector_store.upsert_document(
                document_id=document_id,
                title=record.title,
                source=record.source,
                chunks=chunks,
            )
            self.documents.mark_ready(document_id, len(chunks))
            self.audit.record(
                "document_ingested",
                document_id,
                {
                    "chunk_count": len(chunks),
                    "source": record.source,
                    "collection_id": record.collection_id,
                },
            )
        except Exception as exc:
            logger.exception("Document ingestion failed", extra={"document_id": document_id})
            try:
                self.vector_store.delete_document(document_id)
            finally:
                self.documents.mark_failed(document_id, str(exc))
            raise

        completed = self.documents.get(document_id)
        assert completed is not None
        return completed

    def fail_interrupted(self) -> list[str]:
        """Marks documents still `processing` as failed. Run at startup: their indexing was
        cut off by a restart, and their text was never stored, so they can't be resumed."""
        interrupted = self.documents.list_processing_ids()
        for document_id in interrupted:
            try:
                self.vector_store.delete_document(document_id)
            except Exception as exc:  # partial vectors are harmless; don't block startup
                logger.warning("Could not delete vectors of %s: %s", document_id, exc)
            self.documents.mark_failed(document_id, INTERRUPTED_MESSAGE)
        return interrupted

    def get(self, document_id: str) -> DocumentRecord:
        document = self.documents.get(document_id)
        if not document:
            raise NotFoundError(f"Document '{document_id}' was not found")
        return document

    def list(
        self, *, limit: int, offset: int, collection_id: str | None = None
    ) -> list[DocumentRecord]:
        if collection_id is not None and self.collections.get(collection_id) is None:
            raise NotFoundError(f"Collection '{collection_id}' was not found")
        return self.documents.list(limit=limit, offset=offset, collection_id=collection_id)

    def delete(self, document_id: str) -> None:
        document = self.documents.get(document_id)
        if not document:
            raise NotFoundError(f"Document '{document_id}' was not found")
        self.vector_store.delete_document(document_id)
        self.documents.delete(document_id)
        self.audit.record("document_deleted", document_id, {"source": document.source})
