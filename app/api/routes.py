from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.agents.orchestrator import QueryResult, QueryStream
from app.container import ApplicationContainer
from app.core.exceptions import AppError
from app.schemas import (
    ComponentHealth,
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
    LLMUsageStats,
    ProviderHealth,
    QueryHistoryItem,
    QueryRequest,
    QueryResponse,
    SourceResponse,
    StatsResponse,
    TextDocumentRequest,
)
from app.services.vector_store import SearchHit

logger = logging.getLogger(__name__)

router = APIRouter()


def container_from(request: Request) -> ApplicationContainer:
    return request.app.state.container


def document_response(record, *, duplicate: bool = False) -> DocumentResponse:
    return DocumentResponse(
        id=record.id,
        title=record.title,
        source=record.source,
        chunk_count=record.chunk_count,
        status=record.status,
        error_message=record.error_message,
        created_at=record.created_at,
        duplicate=duplicate,
    )


def source_responses(hits: list[SearchHit]) -> list[SourceResponse]:
    return [
        SourceResponse(
            rank=rank,
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            document_title=hit.document_title,
            source=hit.source,
            chunk_index=hit.chunk_index,
            text=hit.text,
            score=hit.score,
            start_char=hit.start_char,
            end_char=hit.end_char,
        )
        for rank, hit in enumerate(hits, start=1)
    ]


def sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


def sse_events(stream: QueryStream) -> Iterator[str]:
    yield sse(
        {
            "type": "source",
            "sources": [source.model_dump() for source in source_responses(stream.sources)],
        }
    )
    # The status line is already sent, so failures from here on become an error event.
    try:
        for event in stream.events:
            if isinstance(event, QueryResult):
                yield sse(
                    {"type": "end", "query_id": event.query_id, "latency_ms": event.latency_ms}
                )
            else:
                yield sse({"type": "answer", "answer": event})
    except AppError as exc:
        yield sse({"type": "error", "error": exc.code, "message": exc.message})
    except Exception:
        logger.exception("Answer stream failed")
        yield sse(
            {"type": "error", "error": "internal_error", "message": "The answer stream failed"}
        )


def check_embedder(container: ApplicationContainer) -> ComponentHealth:
    try:
        container.embedder.embed_query("health check")
    except Exception as exc:
        logger.warning("Embedder health check failed: %s", exc)
        return ComponentHealth(status="error", detail=str(exc)[:500])
    return ComponentHealth(status="ok")


@router.get(
    "/health",
    response_model=HealthResponse,
    response_model_exclude_none=True,
    tags=["system"],
)
def health(
    request: Request,
    deep: bool = Query(
        default=False,
        description="Also embed a test string and contact each LLM provider. Slower, and "
        "the first call may load the embedding model.",
    ),
) -> HealthResponse:
    container = container_from(request)
    sqlite_ok = container.db.ping()
    chroma_ok = container.vector_store.ping()
    healthy = sqlite_ok and chroma_ok
    embedder = None
    llm = None
    if deep:
        embedder = check_embedder(container)
        providers = container.llm.check_providers() if container.llm else []
        llm = [
            ProviderHealth(
                status="ok" if item.ok else "error",
                detail=item.detail,
                base_url=item.base_url,
                model=item.model,
            )
            for item in providers
        ]
        healthy = healthy and embedder.status == "ok" and all(item.ok for item in providers)
    return HealthResponse(
        status="ok" if healthy else "degraded",
        sqlite="ok" if sqlite_ok else "error",
        chroma="ok" if chroma_ok else "error",
        embedding_backend=container.settings.embedding_backend,
        embedder=embedder,
        llm=llm,
    )


@router.post(
    "/documents/text",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
def ingest_text(
    payload: TextDocumentRequest, request: Request, response: Response
) -> DocumentResponse:
    result = container_from(request).ingestion_agent.run(
        title=payload.title,
        source=payload.source,
        text=payload.text,
    )
    if result.duplicate:
        response.status_code = status.HTTP_200_OK
    return document_response(result.document, duplicate=result.duplicate)


@router.post(
    "/documents/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def upload_document(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
) -> DocumentResponse:
    container = container_from(request)
    content = await file.read(container.settings.max_upload_bytes + 1)
    if len(content) > container.settings.max_upload_bytes:
        raise AppError(
            f"File exceeds the {container.settings.max_upload_mb} MB limit",
            code="file_too_large",
            status_code=413,
        )
    filename = file.filename or "uploaded-document"
    # Parsing and embedding block, so keep them off the event loop.
    text = await run_in_threadpool(container.file_parser.parse, filename, content)
    result = await run_in_threadpool(
        container.ingestion_agent.run,
        title=(title or filename),
        source=f"upload:{filename}",
        text=text,
    )
    if result.duplicate:
        response.status_code = status.HTTP_200_OK
    return document_response(result.document, duplicate=result.duplicate)


@router.get("/documents", response_model=DocumentListResponse, tags=["documents"])
def list_documents(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> DocumentListResponse:
    records = container_from(request).documents.list(limit=limit, offset=offset)
    return DocumentListResponse(
        items=[document_response(record) for record in records],
        count=len(records),
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse, tags=["documents"])
def get_document(document_id: str, request: Request) -> DocumentResponse:
    record = container_from(request).ingestion_agent.document_service.get(document_id)
    return document_response(record)


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["documents"],
)
def delete_document(document_id: str, request: Request) -> Response:
    container_from(request).ingestion_agent.document_service.delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/query", response_model=QueryResponse, tags=["query"])
def query_documents(payload: QueryRequest, request: Request) -> QueryResponse:
    container = container_from(request)
    top_k = payload.top_k or container.settings.default_top_k
    result = container.rag_orchestrator.run(
        question=payload.question,
        top_k=top_k,
        document_id=payload.document_id,
    )
    return QueryResponse(
        query_id=result.query_id,
        question=result.question,
        answer=result.answer,
        latency_ms=result.latency_ms,
        sources=source_responses(result.sources),
    )


@router.post(
    "/query/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Server-Sent Events: one `source` event, then `answer` events, "
            "then `end` or `error`.",
        }
    },
    tags=["query"],
)
def stream_query(payload: QueryRequest, request: Request) -> StreamingResponse:
    container = container_from(request)
    top_k = payload.top_k or container.settings.default_top_k
    # Retrieval runs here, before the response starts, so its errors keep the JSON error shape.
    stream = container.rag_orchestrator.stream(
        question=payload.question,
        top_k=top_k,
        document_id=payload.document_id,
    )
    return StreamingResponse(
        sse_events(stream),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/queries", response_model=list[QueryHistoryItem], tags=["query"])
def query_history(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[QueryHistoryItem]:
    records = container_from(request).queries.list_recent(limit)
    return [QueryHistoryItem(**record.__dict__) for record in records]


@router.get("/stats", response_model=StatsResponse, tags=["system"])
def stats(request: Request) -> StatsResponse:
    container = container_from(request)
    usage = container.llm_calls.totals()
    return StatsResponse(
        document_count=container.documents.count(),
        vector_count=container.vector_store.count(),
        query_count=container.queries.count(),
        llm_usage=LLMUsageStats(
            calls=usage.calls,
            failed_calls=usage.failed_calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        ),
    )
