from __future__ import annotations

import logging
from collections.abc import Sequence

from app.core.exceptions import UpstreamServiceError
from app.services.llm import FallbackLLM
from app.services.prompting import Turn, build_rewrite_messages

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Turns a follow-up question into a standalone search query, using the conversation.

    "And how many remote days?" retrieves nothing useful on its own; with the earlier turn
    about leave policy it becomes "remote work days per week policy". The rewrite is only for
    retrieval: the model still answers the original question.
    """

    def __init__(self, llm: FallbackLLM | None, *, max_tokens: int = 120) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def rewrite(
        self, question: str, history: Sequence[Turn], *, query_id: str | None = None
    ) -> str:
        if self.llm is None or not history:
            return question
        try:
            completion = self.llm.complete(
                build_rewrite_messages(question, history),
                purpose="rewrite",
                query_id=query_id,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
        except UpstreamServiceError as exc:
            # Searching with the raw question is better than failing the query.
            logger.warning("Query rewrite failed; searching with the question itself: %s", exc)
            return question
        rewritten = completion.text.strip().strip('"').strip()
        return rewritten or question
