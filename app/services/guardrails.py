from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from app.services.vector_store import SearchHit

logger = logging.getLogger(__name__)

Mode = Literal["off", "flag", "redact"]

PLACEHOLDERS = {
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "card": "[REDACTED_CARD]",
    "api_key": "[REDACTED_API_KEY]",
}

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Candidates are filtered to 10 to 15 digits, so dates, times and small numbers don't match.
# A run of digits that continues on either side (such as a 16-digit number that failed the
# card check) is not a phone number either.
_EDGE_BEFORE = r"(?<![\w.])(?<!\d[\s.-])"
_EDGE_AFTER = r"(?![\w.])(?![\s.-]\d)"
PHONE = re.compile(
    _EDGE_BEFORE
    + r"\+?(?:\d[\s.-]?){9,14}\d"
    + _EDGE_AFTER
    + "|"
    + _EDGE_BEFORE
    + r"(?:\+?\d{1,3}[\s.-]?)?\(\d{2,4}\)[\s.-]?(?:\d[\s.-]?){6,10}\d"
    + _EDGE_AFTER
)
# 13 to 19 digits with optional spaces or dashes, confirmed with a Luhn check.
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
API_KEY = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9-]{10,})\b"
)
INJECTION = re.compile(
    r"(?i)\b(?:"
    r"(?:ignore|disregard|forget)\s+(?:all\s+|any\s+|the\s+|your\s+)?"
    r"(?:previous|prior|above|earlier|preceding|system)?\s*(?:instructions|prompts?|rules|guidelines)"
    r"|you\s+are\s+now\b"
    r"|(?:reveal|print|show|repeat|output)\s+(?:your|the)\s+(?:system\s+prompt|instructions|hidden\s+prompt)"
    r"|new\s+instructions?\s*:"
    r"|act\s+as\s+(?:an?\s+)?(?:unrestricted|jailbroken|unfiltered)"
    r"|do\s+anything\s+now"
    r")"
)


@dataclass(frozen=True)
class Match:
    kind: str  # email, phone, card, api_key or injection
    start: int
    end: int
    text: str

    @property
    def redactable(self) -> bool:
        return self.kind in PLACEHOLDERS


@dataclass(frozen=True)
class GuardrailWarning:
    kind: str
    location: str  # question, answer or source
    count: int
    detail: str | None = None


def luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _card_matches(text: str) -> Iterable[Match]:
    for found in CARD.finditer(text):
        digits = re.sub(r"[ -]", "", found.group(0))
        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            yield Match("card", found.start(), found.end(), found.group(0))


def _phone_matches(text: str) -> Iterable[Match]:
    for found in PHONE.finditer(text):
        if 10 <= sum(char.isdigit() for char in found.group(0)) <= 15:
            yield Match("phone", found.start(), found.end(), found.group(0))


def _pattern_matches(kind: str, pattern: re.Pattern[str], text: str) -> Iterable[Match]:
    for found in pattern.finditer(text):
        yield Match(kind, found.start(), found.end(), found.group(0))


def scan(text: str, *, injection_only: bool = False) -> list[Match]:
    """Every match in the text, in order. Earlier kinds win where spans overlap, so a card
    number is a card and not a phone number."""
    finders = [lambda: _pattern_matches("injection", INJECTION, text)]
    if not injection_only:
        finders = [
            lambda: _pattern_matches("api_key", API_KEY, text),
            lambda: _pattern_matches("email", EMAIL, text),
            lambda: _card_matches(text),
            lambda: _phone_matches(text),
            *finders,
        ]
    accepted: list[Match] = []
    for finder in finders:
        for match in finder():
            if not any(match.start < other.end and other.start < match.end for other in accepted):
                accepted.append(match)
    return sorted(accepted, key=lambda match: match.start)


def redact(text: str) -> tuple[str, list[Match]]:
    """The text with emails, phone numbers, card numbers and keys replaced by placeholders.
    Injection phrases are reported but left in place."""
    matches = scan(text)
    redacted = text
    for match in reversed(matches):
        if match.redactable:
            redacted = redacted[: match.start] + PLACEHOLDERS[match.kind] + redacted[match.end :]
    return redacted, matches


def summarize(
    matches: Sequence[Match], location: str, detail: str | None = None
) -> list[GuardrailWarning]:
    counts: dict[str, int] = {}
    for match in matches:
        counts[match.kind] = counts.get(match.kind, 0) + 1
    return [GuardrailWarning(kind, location, count, detail) for kind, count in counts.items()]


def merge(warnings: Sequence[GuardrailWarning]) -> list[GuardrailWarning]:
    """Adds up warnings of the same kind, location and detail, keeping first-seen order."""
    merged: dict[tuple[str, str, str | None], int] = {}
    for warning in warnings:
        key = (warning.kind, warning.location, warning.detail)
        merged[key] = merged.get(key, 0) + warning.count
    return [
        GuardrailWarning(kind, location, count, detail)
        for (kind, location, detail), count in merged.items()
    ]


class SentenceBuffer:
    """Holds streamed text back until a sentence ends, so a match split across two pieces
    is still caught before anything is released."""

    # A sentence ends at ., ! or ? followed by whitespace or the end, or at a line break. The
    # whitespace after it is released with the sentence. "3.5 days" is not a boundary.
    BOUNDARY = re.compile(r"(?:[.!?]+(?=\s|$)|\n)\s*")

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, piece: str) -> str:
        """Returns the complete sentences gathered so far, keeping the rest back."""
        self._pending += piece
        last_end = None
        for found in self.BOUNDARY.finditer(self._pending):
            last_end = found.end()
        if last_end is None:
            return ""
        released, self._pending = self._pending[:last_end], self._pending[last_end:]
        return released

    def flush(self) -> str:
        released, self._pending = self._pending, ""
        return released


class Guardrails:
    """Checks questions, answers and retrieved chunks for personal data, secrets and
    prompt-injection phrases. `flag` reports them; `redact` also replaces the personal data
    and secrets with placeholders. Retrieved chunks are only ever flagged."""

    def __init__(self, mode: Mode = "flag") -> None:
        self.mode: Mode = mode

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def redacting(self) -> bool:
        return self.mode == "redact"

    def check(self, text: str, location: str) -> tuple[str, list[GuardrailWarning]]:
        """The text, redacted when the mode says so, and the warnings for it."""
        if not self.enabled:
            return text, []
        if self.redacting:
            text, matches = redact(text)
        else:
            matches = scan(text)
        warnings = summarize(matches, location)
        for warning in warnings:
            logger.warning(
                "Guardrail: %d %s match(es) in the %s", warning.count, warning.kind, location
            )
        return text, warnings

    def check_sources(self, hits: Sequence[SearchHit]) -> list[GuardrailWarning]:
        """Injection phrases inside retrieved chunks. Flagged, never removed: the chunk text
        is what the stored offsets describe."""
        if not self.enabled:
            return []
        warnings: list[GuardrailWarning] = []
        for hit in hits:
            matches = scan(hit.text, injection_only=True)
            if matches:
                warnings.extend(summarize(matches, "source", detail=hit.document_title))
                logger.warning(
                    "Guardrail: injection phrase in chunk %s of %r",
                    hit.chunk_index,
                    hit.document_title,
                )
        return warnings

    def stream_buffer(self) -> SentenceBuffer | None:
        """A buffer for the stream route in redact mode; None when pieces can pass through."""
        return SentenceBuffer() if self.redacting else None
