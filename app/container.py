from __future__ import annotations

from dataclasses import dataclass

from app.agents.answer_agent import AnswerAgent
from app.agents.ingestion_agent import IngestionAgent
from app.agents.orchestrator import RAGOrchestrator
from app.agents.retrieval_agent import RetrievalAgent
from app.config import Settings
from app.db.repositories import AuditRepository, DocumentRepository, QueryRepository
from app.db.sqlite import SQLiteDatabase
from app.services.answer_generation import (
    AnswerGenerator,
    ExtractiveAnswerGenerator,
    OpenAICompatibleAnswerGenerator,
)
from app.services.chunking import TextChunker
from app.services.document_service import DocumentService
from app.services.embeddings import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from app.services.file_parser import FileParser
from app.services.vector_store import ChromaVectorStore, VectorStore


@dataclass
class ApplicationContainer:
    settings: Settings
    db: SQLiteDatabase
    documents: DocumentRepository
    queries: QueryRepository
    audit: AuditRepository
    vector_store: VectorStore
    file_parser: FileParser
    ingestion_agent: IngestionAgent
    rag_orchestrator: RAGOrchestrator


def build_container(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
    answer_generator: AnswerGenerator | None = None,
) -> ApplicationContainer:
    settings.ensure_data_directories()
    db = SQLiteDatabase(settings.sqlite_path)
    documents = DocumentRepository(db)
    queries = QueryRepository(db)
    audit = AuditRepository(db)

    if embedder is None:
        if settings.embedding_backend == "hashing":
            embedder = HashingEmbedder(settings.embedding_dimension)
        else:
            embedder = SentenceTransformerEmbedder(settings.embedding_model)

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
        audit=audit,
        chunker=chunker,
        vector_store=vector_store,
    )

    if answer_generator is None:
        if settings.answer_provider == "openai-compatible":
            answer_generator = OpenAICompatibleAnswerGenerator(
                base_url=settings.llm_base_url or "",
                model=settings.llm_model or "",
                api_key=settings.llm_api_key,
                timeout_seconds=settings.llm_timeout_seconds,
            )
        else:
            answer_generator = ExtractiveAnswerGenerator(settings.min_relevance_score)

    retrieval_agent = RetrievalAgent(vector_store)
    answer_agent = AnswerAgent(answer_generator)
    return ApplicationContainer(
        settings=settings,
        db=db,
        documents=documents,
        queries=queries,
        audit=audit,
        vector_store=vector_store,
        file_parser=FileParser(),
        ingestion_agent=IngestionAgent(document_service),
        rag_orchestrator=RAGOrchestrator(
            retrieval_agent=retrieval_agent,
            answer_agent=answer_agent,
            query_repository=queries,
        ),
    )
