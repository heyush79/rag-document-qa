from __future__ import annotations

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.container import ApplicationContainer
from app.core.exceptions import AppError
from app.schemas import (
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
    QueryHistoryItem,
    QueryRequest,
    QueryResponse,
    SourceResponse,
    StatsResponse,
    TextDocumentRequest,
)

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


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    container = container_from(request)
    sqlite_ok = container.db.ping()
    chroma_ok = container.vector_store.ping()
    return HealthResponse(
        status="ok" if sqlite_ok and chroma_ok else "degraded",
        sqlite="ok" if sqlite_ok else "error",
        chroma="ok" if chroma_ok else "error",
        embedding_backend=container.settings.embedding_backend,
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
        sources=[
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
            for rank, hit in enumerate(result.sources, start=1)
        ],
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
    return StatsResponse(
        document_count=container.documents.count(),
        vector_count=container.vector_store.count(),
        query_count=container.queries.count(),
    )
