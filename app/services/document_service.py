from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass

from app.core.exceptions import NotFoundError
from app.db.repositories import AuditRepository, DocumentRecord, DocumentRepository
from app.services.chunking import TextChunker
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    document: DocumentRecord
    duplicate: bool


class DocumentService:
    def __init__(
        self,
        *,
        documents: DocumentRepository,
        audit: AuditRepository,
        chunker: TextChunker,
        vector_store: VectorStore,
    ) -> None:
        self.documents = documents
        self.audit = audit
        self.chunker = chunker
        self.vector_store = vector_store

    def ingest_text(self, *, title: str, source: str, text: str) -> IngestionResult:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Document text cannot be empty")
        content_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        existing = self.documents.get_by_hash(content_hash)
        if existing:
            return IngestionResult(document=existing, duplicate=True)

        document_id = str(uuid.uuid4())
        record = self.documents.create_processing(
            document_id=document_id,
            title=title.strip(),
            source=source.strip(),
            content_hash=content_hash,
        )
        try:
            chunks = self.chunker.split(normalized_text)
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
                {"chunk_count": len(chunks), "source": record.source},
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
        return IngestionResult(document=completed, duplicate=False)

    def get(self, document_id: str) -> DocumentRecord:
        document = self.documents.get(document_id)
        if not document:
            raise NotFoundError(f"Document '{document_id}' was not found")
        return document

    def list(self, *, limit: int, offset: int) -> list[DocumentRecord]:
        return self.documents.list(limit=limit, offset=offset)

    def delete(self, document_id: str) -> None:
        document = self.documents.get(document_id)
        if not document:
            raise NotFoundError(f"Document '{document_id}' was not found")
        self.vector_store.delete_document(document_id)
        self.documents.delete(document_id)
        self.audit.record("document_deleted", document_id, {"source": document.source})
