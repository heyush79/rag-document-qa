from __future__ import annotations


class AppError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int = 400,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.headers = headers


class NotFoundError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="not_found", status_code=404)


class ConflictError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="conflict", status_code=409)


class UnsupportedFileError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="unsupported_file", status_code=415)


class UpstreamServiceError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="upstream_service_error", status_code=502)


class UnauthorizedError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            code="unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )


class ForbiddenError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="forbidden", status_code=403)
