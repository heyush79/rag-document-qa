from __future__ import annotations

from app.services.vector_store import SearchHit, VectorStore


class RetrievalAgent:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def run(
        self,
        *,
        question: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty")
        return self.vector_store.query(
            question=clean_question,
            top_k=top_k,
            document_id=document_id,
        )
