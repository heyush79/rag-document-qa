from __future__ import annotations

from pydantic import BaseModel, Field


class TextDocumentRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    source: str = Field(default="api", min_length=1, max_length=500)
    text: str = Field(min_length=1)


class DocumentResponse(BaseModel):
    id: str
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


class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    sources: list[SourceResponse]
    latency_ms: float


class QueryHistoryItem(BaseModel):
    id: str
    question: str
    answer: str
    top_k: int
    latency_ms: float
    created_at: str


class StatsResponse(BaseModel):
    document_count: int
    vector_count: int
    query_count: int


class HealthResponse(BaseModel):
    status: str
    sqlite: str
    chroma: str
    embedding_backend: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None
