"""Non-PII timings measured inside the executor, not around a fake async call."""

from collections.abc import Callable
from contextvars import ContextVar
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import Request
from fastapi.routing import APIRoute

from activekg.common.logger import get_enhanced_logger

logger = get_enhanced_logger(__name__)
_timing_id: ContextVar[str | None] = ContextVar("sourcing_timing_id", default=None)


class TimingRoute(APIRoute):
    """Only route templates, generated correlation IDs, durations and status."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def timed(request: Request):
            token = _timing_id.set(uuid4().hex)
            start = perf_counter()
            status = None
            try:
                response = await handler(request)
                status = response.status_code
                return response
            finally:
                logger.info(
                    "sourcing_request_timing",
                    extra_fields={
                        "timing_id": _timing_id.get(),
                        "route": self.path,
                        "status": status,
                        "elapsed_ms": round(max(0, perf_counter() - start) * 1000, 2),
                    },
                )
                _timing_id.reset(token)

        return timed


def timed_work(operation: str, queued_at: float, work: Callable[[], Any]) -> Any:
    if operation not in {"sourced_candidate_ingest", "privacy_eligibility_batch"}:
        raise ValueError("unsupported_timing_operation")
    started = perf_counter()
    outcome = "error"
    try:
        value = work()
        outcome = "ok"
        return value
    finally:
        finished = perf_counter()
        logger.info(
            "sourcing_work_timing",
            extra_fields={
                "operation": operation,
                "timing_id": _timing_id.get(),
                "outcome": outcome,
                "executor_wait_ms": round(max(0, started - queued_at) * 1000, 2),
                "work_ms": round(max(0, finished - started) * 1000, 2),
                "total_ms": round(max(0, finished - queued_at) * 1000, 2),
            },
        )
