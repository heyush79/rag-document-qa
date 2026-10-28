from __future__ import annotations

import hashlib
import hmac
import secrets

API_KEY_PREFIX = "rag_"
# What is shown in listings: "rag_" plus the first 8 random characters.
DISPLAY_PREFIX_LENGTH = 12


def generate_api_key() -> str:
    """A new key: the prefix plus 32 random bytes, URL-safe encoded (43 characters)."""
    return API_KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    # SHA-256 is enough: the key is random, so there is nothing for a slow hash to protect.
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def display_prefix(key: str) -> str:
    return key[:DISPLAY_PREFIX_LENGTH]


def keys_match(presented: str, expected: str) -> bool:
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))
