from __future__ import annotations

from pydantic import BaseModel, Field


class TextDocumentRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    source: str = Field(default="api", min_length=1, max_length=500)
    text: str = Field(min_length=1)
    # Defaults to the "default" collection.
    collection_id: str | None = None


class DocumentResponse(BaseModel):
    id: str
    collection_id: str
    title: str
    source: str
    chunk_count: int
    status: str
    error_message: str | None
    created_at: str
    duplicate: bool = False


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    count: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_id: str | None = None
    # Searches only this collection's documents; omit to search everything.
    collection_id: str | None = None
    # Continues a conversation: earlier turns give the question its context, and both the
    # question and the answer are saved to it.
    conversation_id: str | None = None


class CollectionCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=1000)


class CollectionResponse(BaseModel):
    id: str
    name: str
    description: str | None
    created_at: str
    document_count: int


class CollectionListResponse(BaseModel):
    items: list[CollectionResponse]
    count: int


class SourceResponse(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    document_title: str
    source: str
    chunk_index: int
    text: str
    score: float
    start_char: int
    end_char: int


class GuardrailWarning(BaseModel):
    kind: str  # email, phone, card, api_key or injection
    location: str  # question, answer or source
    count: int
    detail: str | None = None  # the document title, for a source


class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    sources: list[SourceResponse]
    latency_ms: float
    # What was searched: the question, or its rewrite from the conversation.
    search_query: str
    conversation_id: str | None = None
    # Guardrail matches, when guardrails_mode is flag or redact.
    warnings: list[GuardrailWarning] = Field(default_factory=list)


class QueryHistoryItem(BaseModel):
    id: str
    question: str
    answer: str
    top_k: int
    latency_ms: float
    created_at: str
    search_query: str | None = None


class ConversationCreateRequest(BaseModel):
    # Queries in the conversation search only this collection; omit to search everything.
    collection_id: str | None = None
    # Defaults to the first question asked.
    title: str | None = Field(default=None, max_length=200)


class ConversationResponse(BaseModel):
    id: str
    collection_id: str | None
    title: str | None
    created_at: str
    updated_at: str
    message_count: int


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    query_id: str | None
    created_at: str


class ConversationDetailResponse(ConversationResponse):
    messages: list[MessageResponse]


class ConversationListResponse(BaseModel):
    items: list[ConversationResponse]
    count: int


class LLMUsageStats(BaseModel):
    calls: int
    failed_calls: int
    prompt_tokens: int
    completion_tokens: int


class StatsResponse(BaseModel):
    collection_count: int
    document_count: int
    conversation_count: int
    vector_count: int
    query_count: int
    llm_usage: LLMUsageStats


class ComponentHealth(BaseModel):
    status: str
    detail: str | None = None


class ProviderHealth(ComponentHealth):
    base_url: str
    model: str


class HealthResponse(BaseModel):
    status: str
    sqlite: str
    chroma: str
    embedding_backend: str
    # Only present with ?deep=true.
    embedder: ComponentHealth | None = None
    llm: list[ProviderHealth] | None = None


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    prefix: str
    created_at: str
    last_used_at: str | None
    revoked_at: str | None


class ApiKeyCreatedResponse(ApiKeyResponse):
    # The plaintext key, shown once. Only its hash is stored.
    key: str


class ApiKeyListResponse(BaseModel):
    items: list[ApiKeyResponse]
    count: int


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None
