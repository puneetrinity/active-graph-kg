from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

SEMANTIC_TRIGGERS_UNAVAILABLE_DETAIL = {
    "code": "MEMORY_SEMANTIC_TRIGGERS_UNAVAILABLE",
    "message": "Semantic triggers are not available.",
}

MEMORY_CONNECTORS_UNAVAILABLE_DETAIL = {
    "code": "MEMORY_CONNECTORS_UNAVAILABLE",
    "message": "Connectors are not available.",
}

MEMORY_GROUNDED_QA_UNAVAILABLE_DETAIL = {
    "code": "MEMORY_GROUNDED_QA_UNAVAILABLE",
    "message": "Grounded Q&A and search explanation are not available.",
}

semantic_triggers_router = APIRouter()
connector_retirement_router = APIRouter(tags=["connectors-unavailable"])
grounded_qa_retirement_router = APIRouter(tags=["grounded-qa-unavailable"])


def semantic_triggers_unavailable_response() -> JSONResponse:
    return JSONResponse(
        status_code=410,
        content={"detail": SEMANTIC_TRIGGERS_UNAVAILABLE_DETAIL},
        headers={"Cache-Control": "no-store"},
    )


def connectors_unavailable_response() -> JSONResponse:
    return JSONResponse(
        status_code=410,
        content={"detail": MEMORY_CONNECTORS_UNAVAILABLE_DETAIL},
        headers={"Cache-Control": "no-store"},
    )


def grounded_qa_unavailable_response() -> JSONResponse:
    return JSONResponse(
        status_code=410,
        content={"detail": MEMORY_GROUNDED_QA_UNAVAILABLE_DETAIL},
        headers={"Cache-Control": "no-store"},
    )


@grounded_qa_retirement_router.post(
    "/ask",
    response_model=None,
    status_code=410,
    response_description="Grounded Q&A unavailable",
)
async def ask_unavailable() -> JSONResponse:
    return grounded_qa_unavailable_response()


@grounded_qa_retirement_router.post(
    "/ask/stream",
    response_model=None,
    status_code=410,
    response_description="Grounded Q&A unavailable",
)
async def ask_stream_unavailable() -> JSONResponse:
    return grounded_qa_unavailable_response()


@grounded_qa_retirement_router.post(
    "/debug/search_explain",
    response_model=None,
    status_code=410,
    response_description="Search explanation unavailable",
)
async def search_explain_unavailable() -> JSONResponse:
    return grounded_qa_unavailable_response()


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


@connector_retirement_router.post(
    "/_admin/connectors/s3/register",
    response_model=None,
    status_code=410,
    response_description="Connectors unavailable",
)
async def register_s3_connector() -> JSONResponse:
    return connectors_unavailable_response()


@connector_retirement_router.post(
    "/_admin/connectors/gcs/register",
    response_model=None,
    status_code=410,
    response_description="Connectors unavailable",
)
async def register_gcs_connector() -> JSONResponse:
    return connectors_unavailable_response()


@connector_retirement_router.post(
    "/_admin/connectors/drive/register",
    response_model=None,
    status_code=410,
    response_description="Connectors unavailable",
)
async def register_drive_connector() -> JSONResponse:
    return connectors_unavailable_response()


@connector_retirement_router.get(
    "/_admin/connectors/",
    response_model=None,
    status_code=410,
    response_description="Connectors unavailable",
)
async def list_connectors() -> JSONResponse:
    return connectors_unavailable_response()


# Register static paths before the dynamic provider path.
@connector_retirement_router.get(
    "/_admin/connectors/drive/cursor",
    response_model=None,
    status_code=410,
    response_description="Connectors unavailable",
)
async def get_drive_connector_cursor() -> JSONResponse:
    return connectors_unavailable_response()


@connector_retirement_router.get(
    "/_admin/connectors/cache/health",
    response_model=None,
    status_code=410,
    response_description="Connectors unavailable",
)
async def get_connector_cache_health() -> JSONResponse:
    return connectors_unavailable_response()


@connector_retirement_router.post(
    "/_admin/connectors/rotate_keys",
    response_model=None,
    status_code=410,
    response_description="Connectors unavailable",
)
async def rotate_connector_keys() -> JSONResponse:
    return connectors_unavailable_response()


@connector_retirement_router.get(
    "/_admin/connectors/{provider}",
    response_model=None,
    status_code=410,
    response_description="Connectors unavailable",
)
async def get_connector() -> JSONResponse:
    return connectors_unavailable_response()


@connector_retirement_router.post(
    "/_admin/connectors/{provider}/enable",
    response_model=None,
    status_code=410,
    response_description="Connectors unavailable",
)
async def enable_connector() -> JSONResponse:
    return connectors_unavailable_response()


@connector_retirement_router.post(
    "/_admin/connectors/{provider}/disable",
    response_model=None,
    status_code=410,
    response_description="Connectors unavailable",
)
async def disable_connector() -> JSONResponse:
    return connectors_unavailable_response()


@connector_retirement_router.post(
    "/_admin/connectors/{provider}/backfill",
    response_model=None,
    status_code=410,
    response_description="Connectors unavailable",
)
async def backfill_connector() -> JSONResponse:
    return connectors_unavailable_response()


@connector_retirement_router.post(
    "/_admin/connectors/{provider}/ingest",
    response_model=None,
    status_code=410,
    response_description="Connectors unavailable",
)
async def ingest_connector() -> JSONResponse:
    return connectors_unavailable_response()


@connector_retirement_router.get(
    "/_admin/connectors/{provider}/queue-status",
    response_model=None,
    status_code=410,
    response_description="Connectors unavailable",
)
async def get_connector_queue_status() -> JSONResponse:
    return connectors_unavailable_response()


@connector_retirement_router.post(
    "/_webhooks/s3",
    response_model=None,
    status_code=410,
    response_description="Connectors unavailable",
)
async def receive_s3_connector_webhook() -> JSONResponse:
    return connectors_unavailable_response()


@connector_retirement_router.get(
    "/_webhooks/s3/health",
    response_model=None,
    status_code=410,
    response_description="Connectors unavailable",
)
async def get_s3_connector_webhook_health() -> JSONResponse:
    return connectors_unavailable_response()
