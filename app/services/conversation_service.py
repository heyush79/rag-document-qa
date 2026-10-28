from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.core.exceptions import NotFoundError
from app.db.repositories import (
    AuditRepository,
    CollectionRepository,
    ConversationRecord,
    ConversationRepository,
    MessageRecord,
)
from app.services.prompting import Turn
from app.services.tokens import estimate_tokens

TITLE_LENGTH = 80


@dataclass(frozen=True)
class ConversationSummary:
    conversation: ConversationRecord
    message_count: int


class ConversationService:
    """Conversations hold the turns that give a follow-up question its context."""

    def __init__(
        self,
        *,
        conversations: ConversationRepository,
        collections: CollectionRepository,
        audit: AuditRepository,
        history_max_turns: int,
        history_token_limit: int,
    ) -> None:
        self.conversations = conversations
        self.collections = collections
        self.audit = audit
        self.history_max_turns = history_max_turns
        self.history_token_limit = history_token_limit

    def create(self, *, collection_id: str | None, title: str | None) -> ConversationSummary:
        if collection_id is not None and self.collections.get(collection_id) is None:
            raise NotFoundError(f"Collection '{collection_id}' was not found")
        record = self.conversations.create(
            conversation_id=str(uuid.uuid4()),
            collection_id=collection_id,
            title=title.strip() if title and title.strip() else None,
        )
        self.audit.record("conversation_created", record.id, {"collection_id": collection_id})
        return ConversationSummary(conversation=record, message_count=0)

    def get(self, conversation_id: str) -> ConversationRecord:
        record = self.conversations.get(conversation_id)
        if record is None:
            raise NotFoundError(f"Conversation '{conversation_id}' was not found")
        return record

    def get_with_messages(
        self, conversation_id: str
    ) -> tuple[ConversationRecord, list[MessageRecord]]:
        record = self.get(conversation_id)
        return record, self.conversations.messages(conversation_id)

    def list(self, *, limit: int, offset: int) -> list[ConversationSummary]:
        counts = self.conversations.message_counts()
        return [
            ConversationSummary(conversation=record, message_count=counts.get(record.id, 0))
            for record in self.conversations.list(limit=limit, offset=offset)
        ]

    def delete(self, conversation_id: str) -> None:
        if not self.conversations.delete(conversation_id):
            raise NotFoundError(f"Conversation '{conversation_id}' was not found")
        self.audit.record("conversation_deleted", conversation_id, {})

    def history(self, conversation_id: str) -> list[Turn]:
        """The latest turns, newest last: at most `history_max_turns`, and only as many of the
        most recent ones as fit in `history_token_limit`."""
        if self.history_max_turns == 0:
            return []
        messages = self.conversations.last_messages(
            conversation_id, limit=2 * self.history_max_turns
        )
        turns: list[Turn] = []
        pending_question: str | None = None
        for message in messages:
            if message.role == "user":
                pending_question = message.content
            elif pending_question is not None:
                turns.append(Turn(user=pending_question, assistant=message.content))
                pending_question = None
        kept: list[Turn] = []
        used_tokens = 0
        for turn in reversed(turns):
            cost = estimate_tokens(f"{turn.user}\n{turn.assistant}")
            if used_tokens + cost > self.history_token_limit:
                break
            kept.append(turn)
            used_tokens += cost
        kept.reverse()
        return kept

    def record_exchange(
        self, *, conversation_id: str, question: str, answer: str, query_id: str | None
    ) -> None:
        """Saves a question and its answer. An untitled conversation takes its first question."""
        record = self.get(conversation_id)
        self.conversations.add_message(
            conversation_id=conversation_id, role="user", content=question, query_id=query_id
        )
        self.conversations.add_message(
            conversation_id=conversation_id, role="assistant", content=answer, query_id=query_id
        )
        if record.title is None:
            self.conversations.set_title(conversation_id, question[:TITLE_LENGTH])
