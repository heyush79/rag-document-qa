from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.core.exceptions import ConflictError, NotFoundError
from app.db.repositories import (
    DEFAULT_COLLECTION_ID,
    AuditRepository,
    CollectionRecord,
    CollectionRepository,
    DocumentRepository,
)
from app.services.document_service import DocumentService


@dataclass(frozen=True)
class CollectionSummary:
    collection: CollectionRecord
    document_count: int


class CollectionService:
    """Collections group documents. Filtering stays in SQLite: a query in a collection searches
    the vector store with that collection's document ids."""

    def __init__(
        self,
        *,
        collections: CollectionRepository,
        documents: DocumentRepository,
        document_service: DocumentService,
        audit: AuditRepository,
    ) -> None:
        self.collections = collections
        self.documents = documents
        self.document_service = document_service
        self.audit = audit

    def create(self, *, name: str, description: str | None) -> CollectionSummary:
        record = self.collections.create(
            collection_id=str(uuid.uuid4()),
            name=name.strip(),
            description=description.strip() if description else None,
        )
        self.audit.record("collection_created", record.id, {"name": record.name})
        return CollectionSummary(collection=record, document_count=0)

    def get(self, collection_id: str) -> CollectionSummary:
        record = self.collections.get(collection_id)
        if record is None:
            raise NotFoundError(f"Collection '{collection_id}' was not found")
        return CollectionSummary(
            collection=record, document_count=self.documents.count(collection_id)
        )

    def list(self) -> list[CollectionSummary]:
        counts = self.collections.document_counts()
        return [
            CollectionSummary(collection=record, document_count=counts.get(record.id, 0))
            for record in self.collections.list()
        ]

    def document_ids(self, collection_id: str) -> list[str]:
        if self.collections.get(collection_id) is None:
            raise NotFoundError(f"Collection '{collection_id}' was not found")
        return self.documents.ids_in_collection(collection_id)

    def delete(self, collection_id: str, *, cascade: bool) -> None:
        """Deletes an empty collection; with `cascade`, its documents and their vectors too."""
        summary = self.get(collection_id)
        if collection_id == DEFAULT_COLLECTION_ID:
            raise ConflictError("The default collection cannot be deleted")
        if summary.document_count and not cascade:
            raise ConflictError(
                f"Collection '{collection_id}' holds {summary.document_count} documents; "
                "pass cascade=true to delete them as well"
            )
        for document_id in self.documents.ids_in_collection(collection_id):
            self.document_service.delete(document_id)
        self.collections.delete(collection_id)
        self.audit.record(
            "collection_deleted",
            collection_id,
            {"name": summary.collection.name, "documents_deleted": summary.document_count},
        )
