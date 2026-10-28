from __future__ import annotations

import contextlib

from app.db.repositories import DocumentRecord
from app.services.document_service import DocumentService, IngestionResult, Registration


class IngestionAgent:
    """Coordinates validation, deduplication, chunking, vector indexing and metadata persistence."""

    def __init__(self, document_service: DocumentService) -> None:
        self.document_service = document_service

    def run(
        self, *, title: str, source: str, text: str, collection_id: str | None = None
    ) -> IngestionResult:
        return self.document_service.ingest_text(
            title=title, source=source, text=text, collection_id=collection_id
        )

    def register(
        self, *, title: str, source: str, text: str, collection_id: str | None = None
    ) -> Registration:
        return self.document_service.register(
            title=title, source=source, text=text, collection_id=collection_id
        )

    def index(self, document_id: str, text: str) -> DocumentRecord:
        return self.document_service.index(document_id, text)

    def index_in_background(self, document_id: str, text: str) -> None:
        """For FastAPI background tasks. index() has already logged the error and marked the
        document failed, so nothing is left for the caller to handle."""
        with contextlib.suppress(Exception):
            self.document_service.index(document_id, text)
