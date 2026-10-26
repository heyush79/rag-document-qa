from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.exceptions import ConflictError
from app.services.chunking import TextChunk
from app.services.embeddings import Embedder


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    document_title: str
    source: str
    chunk_index: int
    text: str
    score: float
    start_char: int
    end_char: int


class VectorStore(Protocol):
    def upsert_document(
        self,
        *,
        document_id: str,
        title: str,
        source: str,
        chunks: list[TextChunk],
    ) -> None: ...

    def query(
        self,
        *,
        question: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[SearchHit]: ...

    def delete_document(self, document_id: str) -> None: ...

    def count(self) -> int: ...

    def ping(self) -> bool: ...


class ChromaVectorStore:
    def __init__(
        self,
        *,
        embedder: Embedder,
        collection_name: str,
        persist_path: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.embedder = embedder
        if client is None:
            import chromadb

            client = chromadb.PersistentClient(path=persist_path)
        self.client = client
        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"embedding_signature": embedder.signature},
            configuration={"hnsw": {"space": "cosine"}},
        )
        existing_signature = (collection.metadata or {}).get("embedding_signature")
        if existing_signature and existing_signature != embedder.signature:
            raise ConflictError(
                "The Chroma collection was created with a different embedding model. "
                "Use another collection name or clear the persisted Chroma directory."
            )
        self.collection = collection

    def upsert_document(
        self,
        *,
        document_id: str,
        title: str,
        source: str,
        chunks: list[TextChunk],
    ) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)
        ids = [f"{document_id}:{chunk.index}" for chunk in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "document_title": title,
                "source": source,
                "chunk_index": chunk.index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            for chunk in chunks
        ]
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        *,
        question: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        count = self.collection.count()
        if count == 0:
            return []
        query_embedding = self.embedder.embed_query(question)
        where = {"document_id": document_id} if document_id else None
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits: list[SearchHit] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            metadata = metadata or {}
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            hits.append(
                SearchHit(
                    chunk_id=str(chunk_id),
                    document_id=str(metadata["document_id"]),
                    document_title=str(metadata["document_title"]),
                    source=str(metadata["source"]),
                    chunk_index=int(metadata["chunk_index"]),
                    text=str(text),
                    score=round(score, 6),
                    start_char=int(metadata["start_char"]),
                    end_char=int(metadata["end_char"]),
                )
            )
        return hits

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    def count(self) -> int:
        return self.collection.count()

    def ping(self) -> bool:
        self.collection.count()
        return True


@dataclass
class _MemoryRecord:
    chunk_id: str
    document_id: str
    document_title: str
    source: str
    chunk_index: int
    text: str
    embedding: list[float]
    start_char: int
    end_char: int


class InMemoryVectorStore:
    """Small deterministic implementation used by tests; production uses ChromaVectorStore."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._records: dict[str, _MemoryRecord] = {}

    def upsert_document(
        self,
        *,
        document_id: str,
        title: str,
        source: str,
        chunks: list[TextChunk],
    ) -> None:
        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk_id = f"{document_id}:{chunk.index}"
            self._records[chunk_id] = _MemoryRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                document_title=title,
                source=source,
                chunk_index=chunk.index,
                text=chunk.text,
                embedding=embedding,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            )

    def query(
        self,
        *,
        question: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        query_embedding = self.embedder.embed_query(question)
        candidates = [
            record
            for record in self._records.values()
            if document_id is None or record.document_id == document_id
        ]
        scored: list[tuple[float, _MemoryRecord]] = []
        for record in candidates:
            dot = sum(a * b for a, b in zip(query_embedding, record.embedding, strict=True))
            qnorm = math.sqrt(sum(value * value for value in query_embedding))
            rnorm = math.sqrt(sum(value * value for value in record.embedding))
            score = dot / (qnorm * rnorm) if qnorm and rnorm else 0.0
            scored.append((max(0.0, min(1.0, score)), record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchHit(
                chunk_id=record.chunk_id,
                document_id=record.document_id,
                document_title=record.document_title,
                source=record.source,
                chunk_index=record.chunk_index,
                text=record.text,
                score=round(score, 6),
                start_char=record.start_char,
                end_char=record.end_char,
            )
            for score, record in scored[:top_k]
        ]

    def delete_document(self, document_id: str) -> None:
        self._records = {
            chunk_id: record
            for chunk_id, record in self._records.items()
            if record.document_id != document_id
        }

    def count(self) -> int:
        return len(self._records)

    def ping(self) -> bool:
        return True
