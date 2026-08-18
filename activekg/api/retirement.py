from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

SEMANTIC_TRIGGERS_UNAVAILABLE_DETAIL = {
    "code": "MEMORY_SEMANTIC_TRIGGERS_UNAVAILABLE",
    "message": "Semantic triggers are not available.",
}

semantic_triggers_router = APIRouter()


def semantic_triggers_unavailable_response() -> JSONResponse:
    return JSONResponse(
        status_code=410,
        content={"detail": SEMANTIC_TRIGGERS_UNAVAILABLE_DETAIL},
        headers={"Cache-Control": "no-store"},
    )


@semantic_triggers_router.post(
    "/triggers",
    response_model=None,
    status_code=410,
    response_description="Semantic triggers unavailable",
)
def register_trigger_pattern() -> JSONResponse:
    """Return the launch-time semantic-trigger quarantine response."""
    return semantic_triggers_unavailable_response()


@semantic_triggers_router.get(
    "/triggers",
    response_model=None,
    status_code=410,
    response_description="Semantic triggers unavailable",
)
def list_trigger_patterns() -> JSONResponse:
    """Return the launch-time semantic-trigger quarantine response."""
    return semantic_triggers_unavailable_response()


@semantic_triggers_router.delete(
    "/triggers/{name}",
    response_model=None,
    status_code=410,
    response_description="Semantic triggers unavailable",
)
def delete_trigger_pattern(name: str) -> JSONResponse:
    """Return the launch-time semantic-trigger quarantine response."""
    del name
    return semantic_triggers_unavailable_response()
