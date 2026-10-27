from __future__ import annotations

import math


def estimate_tokens(text: str) -> int:
    """Approximate token count at about four characters per token, typical for English.

    Dependency-free and deterministic. Budgets built on it keep a 10% margin for the error.
    """
    return math.ceil(len(text) / 4)
