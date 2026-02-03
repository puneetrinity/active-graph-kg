"""Comprehensive error handling for the search/graph pipeline."""
# mypy: ignore-errors

import traceback
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

try:
    from enum import StrEnum  # Python 3.11+
except Exception:  # pragma: no cover - fallback for Python < 3.11

    class StrEnum(str, Enum):  # noqa: UP042
        """Fallback StrEnum for Python < 3.11."""

        pass


class ErrorCode(StrEnum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    SEARCH_ENGINE_ERROR = "SEARCH_ENGINE_ERROR"
    INDEX_BUILD_ERROR = "INDEX_BUILD_ERROR"
    DATA_LOADING_ERROR = "DATA_LOADING_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    STORAGE_ERROR = "STORAGE_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"


class SearchSystemException(Exception):
    """Base exception for system errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        self.request_id = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.error_code.value,
            "message": self.message,
            "details": self.details,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
        }


class ValidationException(SearchSystemException):
    def __init__(self, message: str, field: str | None = None, value: Any = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["invalid_value"] = str(value)[:100]
        super().__init__(message, ErrorCode.VALIDATION_ERROR, details)


class SearchEngineException(SearchSystemException):
    def __init__(self, message: str, query: str | None = None, cause: Exception | None = None):
        details = {}
        if query:
            details["query"] = query[:100]
        super().__init__(message, ErrorCode.SEARCH_ENGINE_ERROR, details, cause)


class IndexBuildException(SearchSystemException):
    def __init__(
        self, message: str, data_source: str | None = None, cause: Exception | None = None
    ):
        details = {}
        if data_source:
            details["data_source"] = data_source
        super().__init__(message, ErrorCode.INDEX_BUILD_ERROR, details, cause)


class EmbeddingException(SearchSystemException):
    def __init__(self, message: str, text: str | None = None, cause: Exception | None = None):
        details: dict[str, Any] = {}
        if text:
            details["text_length"] = len(text)
            details["text_preview"] = text[:50] + "..." if len(text) > 50 else text
        super().__init__(message, ErrorCode.EMBEDDING_ERROR, details, cause)


class ResourceExhaustedException(SearchSystemException):
    def __init__(self, message: str, resource_type: str, current_usage: float | None = None):
        details: dict[str, Any] = {"resource_type": resource_type}
        if current_usage is not None:
            details["current_usage"] = current_usage
        super().__init__(message, ErrorCode.RESOURCE_EXHAUSTED, details)


def safe_execute(func, *args, default_return=None, error_logger=None, **kwargs):
    try:
        return func(*args, **kwargs)
    except SearchSystemException:
        raise
    except Exception as e:
        if error_logger:
            error_logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
        raise SearchSystemException(
            f"Unexpected error in {func.__name__}: {str(e)}",
            ErrorCode.INTERNAL_ERROR,
            {"function": func.__name__, "original_error": str(e)},
            e,
        ) from e


async def safe_execute_async(func, *args, default_return=None, error_logger=None, **kwargs):
    try:
        return await func(*args, **kwargs)
    except SearchSystemException:
        raise
    except Exception as e:
        if error_logger:
            error_logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
        raise SearchSystemException(
            f"Unexpected error in {func.__name__}: {str(e)}",
            ErrorCode.INTERNAL_ERROR,
            {"function": func.__name__, "original_error": str(e)},
            e,
        ) from e


def handle_and_log_error(
    error: Exception, logger, operation: str = "operation"
) -> SearchSystemException:
    if isinstance(error, SearchSystemException):
        logger.error(
            f"Search system error during {operation}",
            extra_fields={
                "error_code": error.error_code.value,
                "request_id": error.request_id,
                "details": error.details,
            },
        )
        return error
    else:
        search_error = SearchSystemException(
            f"Unexpected error during {operation}: {str(error)}",
            ErrorCode.INTERNAL_ERROR,
            {"operation": operation, "original_error": str(error)},
            error,
        )
        logger.error(
            f"Unexpected error during {operation}",
            extra_fields={
                "error_code": search_error.error_code.value,
                "request_id": search_error.request_id,
                "details": search_error.details,
                "traceback": traceback.format_exc(),
            },
        )
        return search_error
