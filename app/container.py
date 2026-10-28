from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.agents.answer_agent import AnswerAgent
from app.agents.ingestion_agent import IngestionAgent
from app.agents.orchestrator import RAGOrchestrator
from app.agents.retrieval_agent import RetrievalAgent
from app.config import Settings
from app.db.repositories import (
    ApiKeyRepository,
    AuditRepository,
    CollectionRepository,
    DocumentRepository,
    LLMCallRepository,
    QueryRepository,
)
from app.db.sqlite import SQLiteDatabase
from app.services.answer_generation import (
    AnswerGenerator,
    ExtractiveAnswerGenerator,
    OpenAICompatibleAnswerGenerator,
)
from app.services.chunking import TextChunker
from app.services.collection_service import CollectionService
from app.services.document_service import DocumentService
from app.services.embeddings import (
    Embedder,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
)
from app.services.file_parser import FileParser
from app.services.llm import FallbackLLM, LLMProvider
from app.services.vector_store import ChromaVectorStore, VectorStore


@dataclass
class ApplicationContainer:
    settings: Settings
    db: SQLiteDatabase
    documents: DocumentRepository
    queries: QueryRepository
    audit: AuditRepository
    llm_calls: LLMCallRepository
    api_keys: ApiKeyRepository
    collections: CollectionService
    embedder: Embedder
    vector_store: VectorStore
    # None when the extractive answer provider is configured.
    llm: FallbackLLM | None
    file_parser: FileParser
    ingestion_agent: IngestionAgent
    rag_orchestrator: RAGOrchestrator


def build_embedder(settings: Settings, client: httpx.Client) -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder(settings.embedding_dimension)
    if settings.embedding_backend == "openai-compatible":
        return OpenAICompatibleEmbedder(
            base_url=settings.embedding_base_url or "",
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            batch_size=settings.embedding_batch_size,
            timeout_seconds=settings.llm_timeout_seconds,
            client=client,
        )
    return SentenceTransformerEmbedder(settings.embedding_model)


def llm_providers(settings: Settings) -> list[LLMProvider]:
    """The primary provider from the RAG_LLM_* settings, then the configured fallbacks."""
    primary = LLMProvider(
        base_url=settings.llm_base_url or "",
        model=settings.llm_model or "",
        api_key=settings.llm_api_key,
    )
    fallbacks = [
        LLMProvider(base_url=item.base_url, model=item.model, api_key=item.api_key)
        for item in settings.llm_fallbacks
    ]
    return [primary, *fallbacks]


def build_container(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
    answer_generator: AnswerGenerator | None = None,
    http_client: httpx.Client | None = None,
) -> ApplicationContainer:
    settings.ensure_data_directories()
    db = SQLiteDatabase(settings.sqlite_path)
    documents = DocumentRepository(db)
    queries = QueryRepository(db)
    audit = AuditRepository(db)
    llm_calls = LLMCallRepository(db)
    api_keys = ApiKeyRepository(db)
    collection_repository = CollectionRepository(db)
    # Shared by the LLM and embedding clients; tests pass one backed by a mock transport.
    http_client = http_client or httpx.Client()

    embedder = embedder or build_embedder(settings, http_client)
    vector_store = vector_store or ChromaVectorStore(
        embedder=embedder,
        collection_name=settings.chroma_collection,
        persist_path=str(settings.chroma_path),
    )

    chunker = TextChunker(
        chunk_size_words=settings.chunk_size_words,
        overlap_words=settings.chunk_overlap_words,
    )
    document_service = DocumentService(
        documents=documents,
        collections=collection_repository,
        audit=audit,
        chunker=chunker,
        vector_store=vector_store,
    )
    collection_service = CollectionService(
        collections=collection_repository,
        documents=documents,
        document_service=document_service,
        audit=audit,
    )

    llm = None
    if settings.answer_provider == "openai-compatible":
        llm = FallbackLLM(
            llm_providers(settings),
            timeout_seconds=settings.llm_timeout_seconds,
            calls=llm_calls,
            client=http_client,
        )
    if answer_generator is None:
        if llm is not None:
            answer_generator = OpenAICompatibleAnswerGenerator(llm)
        else:
            answer_generator = ExtractiveAnswerGenerator(settings.min_relevance_score)

    retrieval_agent = RetrievalAgent(
        vector_store,
        min_relevance_score=settings.min_relevance_score,
        token_budget=settings.retrieval_token_budget,
    )
    answer_agent = AnswerAgent(answer_generator)
    return ApplicationContainer(
        settings=settings,
        db=db,
        documents=documents,
        queries=queries,
        audit=audit,
        llm_calls=llm_calls,
        api_keys=api_keys,
        collections=collection_service,
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
        file_parser=FileParser(),
        ingestion_agent=IngestionAgent(document_service),
        rag_orchestrator=RAGOrchestrator(
            retrieval_agent=retrieval_agent,
            answer_agent=answer_agent,
            query_repository=queries,
            collection_service=collection_service,
        ),
    )
