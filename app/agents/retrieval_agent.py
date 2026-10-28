from __future__ import annotations

from collections.abc import Sequence

from app.services.tokens import estimate_tokens
from app.services.vector_store import SearchHit, VectorStore


class RetrievalAgent:
    """Over-fetches from the vector store, then keeps the best relevant hits that fit the budget."""

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        min_relevance_score: float,
        token_budget: int,
    ) -> None:
        self.vector_store = vector_store
        self.min_relevance_score = min_relevance_score
        self.token_budget = token_budget

    @staticmethod
    def fetch_size(top_k: int) -> int:
        """How many hits to request so that filtering and trimming still leave top_k to choose."""
        return min(max(top_k * 2, 20), 500)

    def run(
        self,
        *,
        question: str,
        top_k: int,
        document_id: str | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        """`document_ids` is the collection's scope; `document_id` narrows to one document."""
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty")
        if document_ids is not None:
            if document_id is not None and document_id not in document_ids:
                return []  # the document is outside the collection
            if document_id is None and not document_ids:
                return []  # an empty collection
        candidates = self.vector_store.query(
            question=clean_question,
            top_k=self.fetch_size(top_k),
            document_id=document_id,
            document_ids=None if document_id else document_ids,
        )
        kept: list[SearchHit] = []
        used_tokens = 0
        for hit in candidates:
            if len(kept) == top_k:
                break
            if hit.score < self.min_relevance_score:
                continue
            cost = estimate_tokens(f"{hit.document_title}\n{hit.text}")
            if used_tokens + cost > self.token_budget:
                continue
            kept.append(hit)
            used_tokens += cost
        return kept
