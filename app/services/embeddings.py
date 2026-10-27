from __future__ import annotations

import hashlib
import importlib.util
import math
import re
from collections.abc import Sequence
from typing import Protocol

import httpx

from app.core.exceptions import UpstreamServiceError


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class Embedder(Protocol):
    @property
    def signature(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Lazy Sentence Transformers adapter so API startup does not load the model eagerly."""

    def __init__(self, model_name: str) -> None:
        if importlib.util.find_spec("sentence_transformers") is None:
            raise RuntimeError(
                "The sentence-transformers embedding backend needs the 'local' extra: "
                "pip install -e '.[local]'. Or set RAG_EMBEDDING_BACKEND to openai-compatible "
                "or hashing."
            )
        self.model_name = model_name
        self._model = None

    @property
    def signature(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _as_list(values) -> list[list[float]]:
        if hasattr(values, "tolist"):
            values = values.tolist()
        return [[float(value) for value in row] for row in values]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._get_model()
        method = getattr(model, "encode_document", model.encode)
        embeddings = method(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._as_list(embeddings)

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        method = getattr(model, "encode_query", model.encode)
        embedding = method(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._as_list(embedding)[0]


class OpenAICompatibleEmbedder:
    """Calls `POST {base_url}/embeddings` in batches and normalizes what comes back."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        batch_size: int = 64,
        timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "local-key"
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.client = client or httpx.Client()

    @property
    def signature(self) -> str:
        return f"openai-compatible:{self.model}"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(list(texts[start : start + self.batch_size])))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = self.client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": batch},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            entries = list(response.json()["data"])
            # The OpenAI format numbers each entry; some servers (Gemini) omit the index and
            # rely on order instead.
            if all("index" in entry for entry in entries):
                entries.sort(key=lambda entry: int(entry["index"]))
            if len(entries) != len(batch):
                raise ValueError(f"expected {len(batch)} embeddings, got {len(entries)}")
            return [normalize([float(v) for v in entry["embedding"]]) for entry in entries]
        except (httpx.HTTPError, AttributeError, KeyError, TypeError, ValueError) as exc:
            raise UpstreamServiceError(f"Embedding request failed: {exc}") from exc


class HashingEmbedder:
    """Deterministic offline embedder used for tests and a zero-download smoke mode."""

    TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension

    @property
    def signature(self) -> str:
        return f"hashing:v1:{self.dimension}"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = self.TOKEN_PATTERN.findall(text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[index] += sign
        return normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
