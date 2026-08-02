from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Local RAG Document Q&A"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    sqlite_path: Path = Path("data/rag.db")
    chroma_path: Path = Path("data/chroma")
    chroma_collection: str = "documents"

    embedding_backend: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = Field(default=384, ge=8)

    chunk_size_words: int = Field(default=180, ge=20, le=1000)
    chunk_overlap_words: int = Field(default=30, ge=0, le=500)
    default_top_k: int = Field(default=4, ge=1, le=20)
    min_relevance_score: float = Field(default=0.15, ge=0.0, le=1.0)
    max_upload_mb: int = Field(default=10, ge=1, le=100)

    answer_provider: Literal["extractive", "openai-compatible"] = "extractive"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_timeout_seconds: float = Field(default=60.0, gt=0)

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_cross_field_settings(self) -> "Settings":
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")
        if self.answer_provider == "openai-compatible":
            missing = [
                name
                for name, value in {
                    "llm_base_url": self.llm_base_url,
                    "llm_model": self.llm_model,
                }.items()
                if not value
            ]
            if missing:
                raise ValueError(
                    "OpenAI-compatible answer provider requires: " + ", ".join(missing)
                )
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_data_directories(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
