"""Enhanced logger with structured metrics and performance tracking."""

import json
import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Union

# Import metrics from the new package path
try:
    from activekg.common.metrics import PerformanceTimer, metrics
except ImportError:
    # Fallback for when metrics module is not available
    metrics = None
    PerformanceTimer = None


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Add request-scoped context if present
        try:
            ctx = _log_context.get()
            if ctx is not None:
                log_entry.update(ctx)
        except Exception:
            pass

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class MetricsLogger:
    """Enhanced logger with built-in metrics collection."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self._setup_logger()

    def _setup_logger(self):
        """Setup structured logging."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def info(self, message: str, extra_fields: dict[str, Any] | None = None, **kwargs):
        """Log info message with optional structured fields."""
        self._log_with_metrics(logging.INFO, message, extra_fields, **kwargs)

    def error(self, message: str, extra_fields: dict[str, Any] | None = None, **kwargs):
        """Log error message and increment error counter."""
        if metrics:
            metrics.increment_counter("log_errors_total", labels={"level": "error"})
        self._log_with_metrics(logging.ERROR, message, extra_fields, **kwargs)

    def warning(self, message: str, extra_fields: dict[str, Any] | None = None, **kwargs):
        """Log warning message and increment warning counter."""
        if metrics:
            metrics.increment_counter("log_warnings_total", labels={"level": "warning"})
        self._log_with_metrics(logging.WARNING, message, extra_fields, **kwargs)

    def debug(self, message: str, extra_fields: dict[str, Any] | None = None, **kwargs):
        """Log debug message."""
        self._log_with_metrics(logging.DEBUG, message, extra_fields, **kwargs)

    def _log_with_metrics(
        self, level: int, message: str, extra_fields: dict[str, Any] | None = None, **kwargs
    ):
        """Internal method to log with metrics."""
        extra = {"extra_fields": extra_fields or {}}
        extra["extra_fields"].update(kwargs)
        self.logger.log(level, message, extra=extra)
        if metrics:
            metrics.increment_counter(
                "log_messages_total", labels={"level": logging.getLevelName(level).lower()}
            )


# Request-scoped log context (request_id, tenant_id)
_log_context: ContextVar[dict[str, Any] | None] = ContextVar("log_context", default=None)


def set_log_context(request_id: str | None = None, tenant_id: str | None = None) -> None:
    ctx = {}
    if request_id:
        ctx["request_id"] = request_id
    if tenant_id:
        ctx["tenant_id"] = tenant_id
    # Merge with existing, if any
    try:
        cur = _log_context.get()
        if cur is not None:
            cur.update(ctx)
            _log_context.set(cur)
        else:
            _log_context.set(ctx)
    except Exception:
        _log_context.set(ctx)


def clear_log_context() -> None:
    _log_context.set(None)


def get_logger(name: str) -> logging.Logger:
    """Get a basic configured logger instance (backward compatibility)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create a handler to write logs to stdout
    handler = logging.StreamHandler(sys.stdout)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def get_enhanced_logger(name: str) -> MetricsLogger:
    """Get an enhanced logger instance with metrics."""
    return MetricsLogger(name)


def log_performance(operation_name: str, labels: dict[str, str] | None = None):
    """Decorator to log function performance metrics."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if PerformanceTimer:
                with PerformanceTimer(f"{operation_name}_duration_ms", labels):
                    try:
                        if metrics:
                            metrics.increment_counter(
                                f"{operation_name}_calls_total", labels=labels
                            )
                        result = await func(*args, **kwargs)
                        if metrics:
                            metrics.increment_counter(
                                f"{operation_name}_success_total", labels=labels
                            )
                        return result
                    except Exception:
                        if metrics:
                            metrics.increment_counter(
                                f"{operation_name}_errors_total", labels=labels
                            )
                        raise
            else:
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if PerformanceTimer:
                with PerformanceTimer(f"{operation_name}_duration_ms", labels):
                    try:
                        if metrics:
                            metrics.increment_counter(
                                f"{operation_name}_calls_total", labels=labels
                            )
                        result = func(*args, **kwargs)
                        if metrics:
                            metrics.increment_counter(
                                f"{operation_name}_success_total", labels=labels
                            )
                        return result
                    except Exception:
                        if metrics:
                            metrics.increment_counter(
                                f"{operation_name}_errors_total", labels=labels
                            )
                        raise
            else:
                return func(*args, **kwargs)

        import inspect

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator


@contextmanager
def log_operation(logger: Union[logging.Logger, MetricsLogger], operation: str, **context: Any):
    """Context manager for logging operations with timing."""
    start_time = time.time()
    if hasattr(logger, "info"):
        logger.info(f"Starting {operation}", extra_fields=context)
    else:
        logger.info(f"Starting {operation}")

    try:
        yield
        duration = (time.time() - start_time) * 1000
        if hasattr(logger, "info"):
            logger.info(
                f"Completed {operation}",
                extra_fields={**context, "duration_ms": duration, "status": "success"},
            )
        else:
            logger.info(f"Completed {operation} in {duration:.2f}ms")
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        if hasattr(logger, "error"):
            logger.error(
                f"Failed {operation}",
                extra_fields={
                    **context,
                    "duration_ms": duration,
                    "status": "error",
                    "error": str(e),
                },
            )
        else:
            logger.error(f"Failed {operation} after {duration:.2f}ms: {str(e)}")
        raise
