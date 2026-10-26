from __future__ import annotations

from app.services.document_service import DocumentService, IngestionResult


class IngestionAgent:
    """Coordinates validation, deduplication, chunking, vector indexing and metadata persistence."""

    def __init__(self, document_service: DocumentService) -> None:
        self.document_service = document_service

    def run(self, *, title: str, source: str, text: str) -> IngestionResult:
        return self.document_service.ingest_text(title=title, source=source, text=text)
