from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    start_char: int
    end_char: int


class TextChunker:
    """Word-window chunker that preserves source offsets for traceable citations."""

    def __init__(self, *, chunk_size_words: int, overlap_words: int) -> None:
        if chunk_size_words <= 0:
            raise ValueError("chunk_size_words must be positive")
        if overlap_words < 0:
            raise ValueError("overlap_words cannot be negative")
        if overlap_words >= chunk_size_words:
            raise ValueError("overlap_words must be smaller than chunk_size_words")
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def split(self, text: str) -> list[TextChunk]:
        if not text or not text.strip():
            raise ValueError("Document text cannot be empty")

        matches = list(re.finditer(r"\S+", text))
        if not matches:
            raise ValueError("Document text cannot be empty")

        chunks: list[TextChunk] = []
        step = self.chunk_size_words - self.overlap_words
        start_word = 0

        while start_word < len(matches):
            end_word = min(start_word + self.chunk_size_words, len(matches))
            start_char = matches[start_word].start()
            end_char = matches[end_word - 1].end()
            chunk_text = text[start_char:end_char].strip()
            chunks.append(
                TextChunk(
                    index=len(chunks),
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                )
            )
            if end_word == len(matches):
                break
            start_word += step

        return chunks
