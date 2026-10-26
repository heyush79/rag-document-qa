from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    @property
    def signature(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Lazy Sentence Transformers adapter so API startup does not load the model eagerly."""

    def __init__(self, model_name: str) -> None:
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
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
