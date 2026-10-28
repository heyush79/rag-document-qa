from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.agents.ingestion_agent import IngestionAgent
from app.agents.orchestrator import QueryResult, QueryStream
from app.api.auth import require_admin_key, require_api_key
from app.container import ApplicationContainer
from app.core.exceptions import AppError, NotFoundError
from app.core.security import display_prefix, generate_api_key, hash_api_key
from app.schemas import (
    ApiKeyCreatedResponse,
    ApiKeyCreateRequest,
    ApiKeyListResponse,
    ApiKeyResponse,
    CollectionCreateRequest,
    CollectionListResponse,
    CollectionResponse,
    ComponentHealth,
    ConversationCreateRequest,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
    LLMUsageStats,
    MessageResponse,
    ProviderHealth,
    QueryHistoryItem,
    QueryRequest,
    QueryResponse,
    SourceResponse,
    StatsResponse,
    TextDocumentRequest,
)
from app.services.document_service import Registration
from app.services.vector_store import SearchHit

logger = logging.getLogger(__name__)

# Three routers: /health needs no key, the API needs an API key when auth is enabled, and key
# management always needs the admin key.
public_router = APIRouter()
router = APIRouter(dependencies=[Depends(require_api_key)])
admin_router = APIRouter(dependencies=[Depends(require_admin_key)], tags=["api-keys"])


def container_from(request: Request) -> ApplicationContainer:
    return request.app.state.container


def document_response(record, *, duplicate: bool = False) -> DocumentResponse:
    return DocumentResponse(
        id=record.id,
        collection_id=record.collection_id,
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
            "search_query": stream.search_query,
            "conversation_id": stream.conversation_id,
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


@public_router.get(
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


BACKGROUND_DESCRIPTION = (
    "Register the document and answer 202 at once; chunking and embedding continue in the "
    "background. Poll GET /documents/{id} until its status is ready or failed."
)
INGEST_RESPONSES = {
    200: {"model": DocumentResponse, "description": "A duplicate: the existing document."},
    202: {
        "model": DocumentResponse,
        "description": "Registered as processing; indexing continues in the background.",
    },
}


def queue_indexing(
    agent: IngestionAgent,
    background_tasks: BackgroundTasks,
    response: Response,
    registration: Registration,
) -> DocumentResponse:
    if registration.duplicate:
        response.status_code = status.HTTP_200_OK
        return document_response(registration.document, duplicate=True)
    # The text stays in memory for the task; it is never written to disk.
    background_tasks.add_task(
        agent.index_in_background, registration.document.id, registration.text
    )
    response.status_code = status.HTTP_202_ACCEPTED
    return document_response(registration.document)


@router.post(
    "/documents/text",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    responses=INGEST_RESPONSES,
    tags=["documents"],
)
def ingest_text(
    payload: TextDocumentRequest,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    background: bool = Query(default=False, description=BACKGROUND_DESCRIPTION),
) -> DocumentResponse:
    agent = container_from(request).ingestion_agent
    if background:
        registration = agent.register(
            title=payload.title,
            source=payload.source,
            text=payload.text,
            collection_id=payload.collection_id,
        )
        return queue_indexing(agent, background_tasks, response, registration)
    result = agent.run(
        title=payload.title,
        source=payload.source,
        text=payload.text,
        collection_id=payload.collection_id,
    )
    if result.duplicate:
        response.status_code = status.HTTP_200_OK
    return document_response(result.document, duplicate=result.duplicate)


@router.post(
    "/documents/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    responses=INGEST_RESPONSES,
    tags=["documents"],
)
async def upload_document(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    collection_id: str | None = Form(default=None),
    background: bool = Query(default=False, description=BACKGROUND_DESCRIPTION),
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
    # Parsing and embedding block, so keep them off the event loop. Parsing always happens
    # here, so an unsupported or broken file gets its 415 straight away.
    text = await run_in_threadpool(container.file_parser.parse, filename, content)
    agent = container.ingestion_agent
    if background:
        registration = await run_in_threadpool(
            agent.register,
            title=(title or filename),
            source=f"upload:{filename}",
            text=text,
            collection_id=collection_id,
        )
        return queue_indexing(agent, background_tasks, response, registration)
    result = await run_in_threadpool(
        agent.run,
        title=(title or filename),
        source=f"upload:{filename}",
        text=text,
        collection_id=collection_id,
    )
    if result.duplicate:
        response.status_code = status.HTTP_200_OK
    return document_response(result.document, duplicate=result.duplicate)


@router.get("/documents", response_model=DocumentListResponse, tags=["documents"])
def list_documents(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    collection_id: str | None = Query(default=None),
) -> DocumentListResponse:
    records = container_from(request).ingestion_agent.document_service.list(
        limit=limit, offset=offset, collection_id=collection_id
    )
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
        collection_id=payload.collection_id,
        conversation_id=payload.conversation_id,
    )
    return QueryResponse(
        query_id=result.query_id,
        question=result.question,
        answer=result.answer,
        latency_ms=result.latency_ms,
        sources=source_responses(result.sources),
        search_query=result.search_query,
        conversation_id=result.conversation_id,
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
        collection_id=payload.collection_id,
        conversation_id=payload.conversation_id,
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
        collection_count=container.collections.collections.count(),
        document_count=container.documents.count(),
        conversation_count=container.conversations.conversations.count(),
        vector_count=container.vector_store.count(),
        query_count=container.queries.count(),
        llm_usage=LLMUsageStats(
            calls=usage.calls,
            failed_calls=usage.failed_calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        ),
    )


def collection_response(summary) -> CollectionResponse:
    return CollectionResponse(**summary.collection.__dict__, document_count=summary.document_count)


@router.post(
    "/collections",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["collections"],
)
def create_collection(payload: CollectionCreateRequest, request: Request) -> CollectionResponse:
    summary = container_from(request).collections.create(
        name=payload.name, description=payload.description
    )
    return collection_response(summary)


@router.get("/collections", response_model=CollectionListResponse, tags=["collections"])
def list_collections(request: Request) -> CollectionListResponse:
    summaries = container_from(request).collections.list()
    return CollectionListResponse(
        items=[collection_response(s) for s in summaries], count=len(summaries)
    )


@router.get(
    "/collections/{collection_id}", response_model=CollectionResponse, tags=["collections"]
)
def get_collection(collection_id: str, request: Request) -> CollectionResponse:
    return collection_response(container_from(request).collections.get(collection_id))


@router.delete(
    "/collections/{collection_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["collections"],
)
def delete_collection(
    collection_id: str,
    request: Request,
    cascade: bool = Query(
        default=False, description="Also delete the collection's documents and their vectors."
    ),
) -> Response:
    container_from(request).collections.delete(collection_id, cascade=cascade)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def conversation_response(summary) -> ConversationResponse:
    return ConversationResponse(
        **summary.conversation.__dict__, message_count=summary.message_count
    )


@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["conversations"],
)
def create_conversation(
    payload: ConversationCreateRequest, request: Request
) -> ConversationResponse:
    summary = container_from(request).conversations.create(
        collection_id=payload.collection_id, title=payload.title
    )
    return conversation_response(summary)


@router.get("/conversations", response_model=ConversationListResponse, tags=["conversations"])
def list_conversations(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> ConversationListResponse:
    summaries = container_from(request).conversations.list(limit=limit, offset=offset)
    return ConversationListResponse(
        items=[conversation_response(s) for s in summaries], count=len(summaries)
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
    tags=["conversations"],
)
def get_conversation(conversation_id: str, request: Request) -> ConversationDetailResponse:
    record, messages = container_from(request).conversations.get_with_messages(conversation_id)
    return ConversationDetailResponse(
        **record.__dict__,
        message_count=len(messages),
        messages=[MessageResponse(**message.__dict__) for message in messages],
    )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["conversations"],
)
def delete_conversation(conversation_id: str, request: Request) -> Response:
    container_from(request).conversations.delete(conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def api_key_response(record) -> ApiKeyResponse:
    return ApiKeyResponse(**record.__dict__)


@admin_router.post(
    "/api-keys", response_model=ApiKeyCreatedResponse, status_code=status.HTTP_201_CREATED
)
def create_api_key(payload: ApiKeyCreateRequest, request: Request) -> ApiKeyCreatedResponse:
    """Creates a key and returns it once; only its hash is stored."""
    key = generate_api_key()
    record = container_from(request).api_keys.create(
        key_id=str(uuid.uuid4()),
        name=payload.name,
        prefix=display_prefix(key),
        key_hash=hash_api_key(key),
    )
    return ApiKeyCreatedResponse(**record.__dict__, key=key)


@admin_router.get("/api-keys", response_model=ApiKeyListResponse)
def list_api_keys(request: Request) -> ApiKeyListResponse:
    records = container_from(request).api_keys.list()
    return ApiKeyListResponse(items=[api_key_response(r) for r in records], count=len(records))


@admin_router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_api_key(key_id: str, request: Request) -> Response:
    """Revokes the key. Revoking an already revoked key is fine; an unknown one is a 404."""
    if not container_from(request).api_keys.revoke(key_id):
        raise NotFoundError(f"API key '{key_id}' was not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
