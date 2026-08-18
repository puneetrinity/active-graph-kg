from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import Callable

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from activekg.api.retirement import (
    connector_retirement_router,
    connectors_unavailable_response,
)
from activekg.connectors.worker import start_worker
from activekg.refresh.scheduler import RefreshScheduler

CONNECTOR_REGISTRATIONS = {
    ("POST", "/_admin/connectors/s3/register"),
    ("POST", "/_admin/connectors/gcs/register"),
    ("POST", "/_admin/connectors/drive/register"),
    ("GET", "/_admin/connectors/"),
    ("GET", "/_admin/connectors/{provider}"),
    ("POST", "/_admin/connectors/{provider}/enable"),
    ("POST", "/_admin/connectors/{provider}/disable"),
    ("POST", "/_admin/connectors/{provider}/backfill"),
    ("GET", "/_admin/connectors/drive/cursor"),
    ("POST", "/_admin/connectors/{provider}/ingest"),
    ("GET", "/_admin/connectors/{provider}/queue-status"),
    ("GET", "/_admin/connectors/cache/health"),
    ("POST", "/_admin/connectors/rotate_keys"),
    ("POST", "/_webhooks/s3"),
    ("GET", "/_webhooks/s3/health"),
}

EXPECTED_BODY = (
    b'{"detail":{"code":"MEMORY_CONNECTORS_UNAVAILABLE","message":"Connectors are not available."}}'
)


async def _asgi_request(app: FastAPI, method: str, path: str) -> tuple[int, dict[str, str], bytes]:
    messages: list[dict[str, object]] = []
    request_sent = False

    async def receive() -> dict[str, object]:
        nonlocal request_sent
        if request_sent:
            return {"type": "http.disconnect"}
        request_sent = True
        return {
            "type": "http.request",
            "body": b'{"not":"a supported request"',
            "more_body": False,
        }

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"tenant_id=ignored",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", b"999999999"),
                (b"authorization", b"Bearer ignored"),
            ],
            "client": ("test", 1),
            "server": ("test", 80),
            "root_path": "",
        },
        receive,
        send,
    )

    start = next(message for message in messages if message["type"] == "http.response.start")
    status = int(start["status"])  # type: ignore[arg-type]
    headers = {
        key.decode(): value.decode()
        for key, value in start["headers"]  # type: ignore[union-attr]
    }
    body = b"".join(
        message.get("body", b"")  # type: ignore[arg-type]
        for message in messages
        if message["type"] == "http.response.body"
    )
    return status, headers, body


def _assert_unavailable(response: JSONResponse) -> None:
    assert response.status_code == 410
    assert response.headers["cache-control"] == "no-store"
    assert bytes(response.body) == EXPECTED_BODY


def test_exactly_fifteen_dependency_free_connector_methods_are_registered() -> None:
    routes = [route for route in connector_retirement_router.routes if isinstance(route, APIRoute)]
    registrations = {(method, route.path) for route in routes for method in route.methods}

    assert registrations == CONNECTOR_REGISTRATIONS
    assert len(routes) == 15
    assert all(route.status_code == 410 for route in routes)
    assert all(not route.dependencies for route in routes)
    assert all(not route.dependant.body_params for route in routes)


def test_retirement_helper_has_stable_no_store_contract() -> None:
    _assert_unavailable(connectors_unavailable_response())


def test_connector_openapi_has_only_no_body_410_operations() -> None:
    isolated_app = FastAPI()
    isolated_app.include_router(connector_retirement_router)
    schema = isolated_app.openapi()

    operations = {
        (method.upper(), path): operation
        for path, path_item in schema["paths"].items()
        for method, operation in path_item.items()
    }
    assert set(operations) == CONNECTOR_REGISTRATIONS
    for operation in operations.values():
        assert "requestBody" not in operation
        assert set(operation["responses"]) == {"410"}


@pytest.mark.parametrize(
    ("method", "path"),
    sorted(
        {
            ("POST", "/_admin/connectors/s3/register"),
            ("POST", "/_admin/connectors/gcs/register"),
            ("POST", "/_admin/connectors/drive/register"),
            ("GET", "/_admin/connectors/"),
            ("GET", "/_admin/connectors/s3"),
            ("POST", "/_admin/connectors/s3/enable"),
            ("POST", "/_admin/connectors/s3/disable"),
            ("POST", "/_admin/connectors/s3/backfill"),
            ("GET", "/_admin/connectors/drive/cursor"),
            ("POST", "/_admin/connectors/s3/ingest"),
            ("GET", "/_admin/connectors/s3/queue-status"),
            ("GET", "/_admin/connectors/cache/health"),
            ("POST", "/_admin/connectors/rotate_keys"),
            ("POST", "/_webhooks/s3"),
            ("GET", "/_webhooks/s3/health"),
        }
    ),
)
def test_all_connector_requests_return_same_contract(method: str, path: str) -> None:
    isolated_app = FastAPI()
    isolated_app.include_router(connector_retirement_router)

    status, headers, body = asyncio.run(_asgi_request(isolated_app, method, path))

    assert status == 410
    assert headers["cache-control"] == "no-store"
    assert body == EXPECTED_BODY


def test_static_drive_cursor_path_is_not_captured_by_provider_route() -> None:
    routes = [route for route in connector_retirement_router.routes if isinstance(route, APIRoute)]
    static_index = next(
        index
        for index, route in enumerate(routes)
        if route.path == "/_admin/connectors/drive/cursor"
    )
    dynamic_index = next(
        index for index, route in enumerate(routes) if route.path == "/_admin/connectors/{provider}"
    )

    assert static_index < dynamic_index


def test_api_has_no_connector_control_plane_startup_or_live_router_wiring() -> None:
    source = Path("activekg/api/main.py").read_text()

    forbidden = (
        "connectors_admin_router",
        "connectors_webhook_router",
        "get_subscriber_health",
        "RotateKeysRequest",
        "RUN_GCS_POLLER",
        "get_encryption",
        "start_subscriber",
        "Connector config cache warmup",
        "connector_cache_health",
        "connector_rotate_keys",
    )
    assert all(token not in source for token in forbidden)
    assert "app.include_router(connector_retirement_router)" in source

    tree = ast.parse(source)
    scheduler_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RefreshScheduler"
    ]
    assert len(scheduler_calls) == 2
    assert all(
        not any(keyword.arg == "gcs_poller_enabled" for keyword in call.keywords)
        for call in scheduler_calls
    )


def test_resume_upload_keeps_internal_extract_and_chunk_helpers() -> None:
    source = Path("activekg/api/main.py").read_text()

    assert "from activekg.connectors.chunker import create_chunk_nodes" in source
    assert "from activekg.connectors.extract import extract_text" in source


class _RecordedScheduler:
    def __init__(self) -> None:
        self.job_ids: list[str] = []
        self.started = False

    def add_job(self, _callable: Callable[..., object], *_args: object, **kwargs: object) -> None:
        self.job_ids.append(str(kwargs["id"]))

    def start(self) -> None:
        self.started = True


def test_scheduler_registers_refresh_and_purge_but_no_connector_jobs() -> None:
    refresh_scheduler = RefreshScheduler(object(), object(), trigger_engine=None)
    recorded = _RecordedScheduler()
    refresh_scheduler.scheduler = recorded  # type: ignore[assignment]

    refresh_scheduler.start()

    assert recorded.started
    assert recorded.job_ids == ["refresh_cycle", "purge_deleted_cycle"]

    scheduler_source = Path("activekg/refresh/scheduler.py").read_text()
    assert 'id="drive_poller"' not in scheduler_source
    assert 'id="gcs_poller"' not in scheduler_source
    assert "def run_drive_poller(" in scheduler_source
    assert "def run_gcs_poller(" in scheduler_source


def test_connector_worker_cli_fails_before_dependency_construction(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        start_worker()

    assert exc_info.value.code == 1
    assert capsys.readouterr().err.strip() == (
        "MEMORY_CONNECTORS_UNAVAILABLE: Connectors are not available."
    )
