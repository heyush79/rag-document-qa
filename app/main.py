from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import Settings, get_settings
from app.container import ApplicationContainer, build_container
from app.core.exceptions import AppError
from app.core.logging import configure_logging
from app.services.answer_generation import AnswerGenerator
from app.services.embeddings import Embedder
from app.services.vector_store import VectorStore


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
    answer_generator: AnswerGenerator | None = None,
) -> FastAPI:
    configure_logging()
    resolved_settings = settings or get_settings()
    container: ApplicationContainer = build_container(
        resolved_settings,
        embedder=embedder,
        vector_store=vector_store,
        answer_generator=answer_generator,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        container.db.initialize()
        yield

    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        description="Local document ingestion and retrieval-augmented question answering API.",
        lifespan=lifespan,
    )
    application.state.container = container

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(
            round((time.perf_counter() - started) * 1000, 3)
        )
        return response

    @application.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.code,
                "message": exc.message,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @application.exception_handler(ValueError)
    async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": "validation_error",
                "message": str(exc),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @application.get("/", tags=["system"])
    def root() -> dict[str, str]:
        return {
            "name": resolved_settings.app_name,
            "docs": "/docs",
            "health": f"{resolved_settings.api_prefix}/health",
        }

    application.include_router(router, prefix=resolved_settings.api_prefix)
    return application

