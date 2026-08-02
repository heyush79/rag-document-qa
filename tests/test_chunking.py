from __future__ import annotations

import pytest

from app.services.chunking import TextChunker


def test_empty_text_is_rejected() -> None:
    chunker = TextChunker(chunk_size_words=5, overlap_words=1)
    with pytest.raises(ValueError, match="cannot be empty"):
        chunker.split("   \n")


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="smaller"):
        TextChunker(chunk_size_words=5, overlap_words=5)


def test_chunker_creates_sliding_overlap() -> None:
    text = "one two three four five six seven eight nine"
    chunks = TextChunker(chunk_size_words=5, overlap_words=2).split(text)
    assert [chunk.text for chunk in chunks] == [
        "one two three four five",
        "four five six seven eight",
        "seven eight nine",
    ]


def test_chunk_offsets_map_back_to_original_text() -> None:
    text = "alpha beta\n\ngamma delta epsilon zeta"
    chunks = TextChunker(chunk_size_words=3, overlap_words=1).split(text)
    assert all(text[chunk.start_char : chunk.end_char].strip() == chunk.text for chunk in chunks)
