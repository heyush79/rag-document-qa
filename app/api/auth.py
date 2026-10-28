from __future__ import annotations

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.core.security import hash_api_key, keys_match

# auto_error=False so missing or malformed headers get this app's 401 rather than FastAPI's 403.
bearer = HTTPBearer(auto_error=False, description="An API key, or the admin key")


def presented_key(credentials: HTTPAuthorizationCredentials | None) -> str | None:
    if credentials is None or credentials.scheme.lower() != "bearer":
        return None
    return credentials.credentials.strip() or None


def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> None:
    """Router dependency: with auth enabled, the request must carry a valid key."""
    container = request.app.state.container
    settings = container.settings
    if not settings.auth_enabled:
        return
    key = presented_key(credentials)
    if key is None:
        raise UnauthorizedError("An API key is required: send 'Authorization: Bearer <key>'")
    if settings.admin_api_key and keys_match(key, settings.admin_api_key):
        return
    record = container.api_keys.find_active_by_hash(hash_api_key(key))
    if record is None:
        raise UnauthorizedError("Invalid or revoked API key")
    container.api_keys.touch(record.id)


def require_admin_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> None:
    """Router dependency for key management: always needs the admin key, even with auth off."""
    settings = request.app.state.container.settings
    if not settings.admin_api_key:
        raise ForbiddenError("API key management is off: set RAG_ADMIN_API_KEY to use it")
    key = presented_key(credentials)
    if key is None or not keys_match(key, settings.admin_api_key):
        raise UnauthorizedError("The admin API key is required")
