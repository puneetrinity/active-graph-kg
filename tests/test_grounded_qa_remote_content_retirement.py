from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError

from activekg.api.retirement import (
    grounded_qa_retirement_router,
    grounded_qa_unavailable_response,
)
from activekg.common.validation import NodeBatchCreate, NodeCreate
from activekg.graph.models import Node
from activekg.graph.repository import GraphRepository

REGISTRATIONS = {
    ("POST", "/ask"),
    ("POST", "/ask/stream"),
    ("POST", "/debug/search_explain"),
}
EXPECTED_BODY = (
    b'{"detail":{"code":"MEMORY_GROUNDED_QA_UNAVAILABLE",'
    b'"message":"Grounded Q&A and search explanation are not available."}}'
)


async def _request(path: str, body: bytes, content_length: bytes | None = None):
    app = FastAPI()
    app.include_router(grounded_qa_retirement_router)
    messages: list[dict[str, object]] = []
    sent = False

    async def receive() -> dict[str, object]:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    headers = [(b"content-type", b"application/json")]
    if content_length is not None:
        headers.append((b"content-length", content_length))
    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": headers,
            "client": ("test", 1),
            "server": ("test", 80),
            "root_path": "",
        },
        receive,
        send,
    )
    start = next(message for message in messages if message["type"] == "http.response.start")
    response_headers = {
        key.decode(): value.decode()
        for key, value in start["headers"]  # type: ignore[union-attr]
    }
    response_body = b"".join(
        message.get("body", b"")  # type: ignore[arg-type]
        for message in messages
        if message["type"] == "http.response.body"
    )
    return int(start["status"]), response_headers, response_body  # type: ignore[arg-type]


def test_exactly_three_dependency_and_body_free_tombstones() -> None:
    routes = [
        route for route in grounded_qa_retirement_router.routes if isinstance(route, APIRoute)
    ]
    registrations = {(method, route.path) for route in routes for method in route.methods}

    assert registrations == REGISTRATIONS
    assert len(routes) == 3
    assert all(route.status_code == 410 for route in routes)
    assert all(not route.dependencies for route in routes)
    assert all(not route.dependant.body_params for route in routes)


def test_tombstone_helper_is_stable_and_not_cacheable() -> None:
    response: JSONResponse = grounded_qa_unavailable_response()
    assert response.status_code == 410
    assert response.headers["cache-control"] == "no-store"
    assert bytes(response.body) == EXPECTED_BODY


@pytest.mark.parametrize("path", ["/ask", "/ask/stream", "/debug/search_explain"])
@pytest.mark.parametrize(
    ("body", "content_length"),
    [
        (b'{"question":"ignored"}', None),
        (b'{"malformed":', None),
        (b"", None),
        (b'{"ignored":true}', b"999999999"),
    ],
)
def test_every_body_shape_returns_the_same_no_work_contract(
    path: str, body: bytes, content_length: bytes | None
) -> None:
    status, headers, response_body = asyncio.run(_request(path, body, content_length))
    assert status == 410
    assert headers["cache-control"] == "no-store"
    assert response_body == EXPECTED_BODY


def test_openapi_has_bodyless_410_operations_only() -> None:
    app = FastAPI()
    app.include_router(grounded_qa_retirement_router)
    schema = app.openapi()
    operations = {
        (method.upper(), path): operation
        for path, path_item in schema["paths"].items()
        for method, operation in path_item.items()
    }
    assert set(operations) == REGISTRATIONS
    assert all("requestBody" not in operation for operation in operations.values())
    assert all(set(operation["responses"]) == {"410"} for operation in operations.values())


def test_full_app_preserves_tombstone_for_oversized_declared_body(monkeypatch) -> None:
    monkeypatch.setenv("ACTIVEKG_TEST_NO_DB", "true")
    monkeypatch.setenv("JWT_ENABLED", "false")
    from activekg.api.main import app

    assert len(app.routes) == 70

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for path in ("/ask", "/ask/stream", "/debug/search_explain"):
                response = await client.post(
                    path,
                    content=b'{"malformed":',
                    headers={"content-type": "application/json", "content-length": "999999999"},
                )
                assert response.status_code == 410
                assert response.headers["cache-control"] == "no-store"
                assert response.content == EXPECTED_BODY

    asyncio.run(exercise())


def test_api_import_ignores_stale_q_and_a_model_configuration() -> None:
    env = os.environ.copy()
    env.update(
        {
            "ACTIVEKG_TEST_NO_DB": "true",
            "JWT_ENABLED": "false",
            "LLM_BACKEND": "invalid-closed-product-backend",
            "LLM_MODEL": "retired-model",
            "ASK_FAST_BACKEND": "invalid-closed-product-backend",
            "ASK_FAST_MODEL": "retired-fast-model",
            "ASK_FALLBACK_BACKEND": "invalid-closed-product-backend",
            "ASK_FALLBACK_MODEL": "retired-fallback-model",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; from activekg.api.main import app, health; "
                "h=health(); assert len(app.routes)==70; "
                "assert json.loads(h.body)=={'status':'alive','service':'activekg-api'}; "
                "print('Q_AND_A_WIRING_INERT')"
            ),
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Q_AND_A_WIRING_INERT" in result.stdout


@pytest.mark.parametrize("value", ["", "https://example.com/a", "s3://bucket/key", "file:///tmp/a"])
def test_node_writes_reject_every_non_null_payload_reference(value: str) -> None:
    payload = {"classes": ["Document"], "props": {"text": "inline"}, "payload_ref": value}
    with pytest.raises(ValidationError):
        NodeCreate.model_validate(payload)
    with pytest.raises(ValidationError):
        NodeBatchCreate.model_validate({"nodes": [payload]})


def test_node_writes_accept_omitted_or_null_payload_reference() -> None:
    omitted = NodeCreate.model_validate({"classes": ["Document"], "props": {"text": "inline"}})
    explicit_null = NodeCreate.model_validate(
        {"classes": ["Document"], "props": {"text": "inline"}, "payload_ref": None}
    )
    assert omitted.payload_ref is None
    assert explicit_null.payload_ref is None


def test_historical_payload_reference_is_inert_and_inline_order_is_preserved() -> None:
    repo = object.__new__(GraphRepository)
    node = Node(
        id="node-1",
        classes=["Document"],
        props={
            "description": "description",
            "content": "content",
            "resume_text": "resume",
            "text": "text",
        },
        payload_ref="https://127.0.0.1/private",
    )
    assert repo.load_payload_text(node) == "text"

    inert = Node(id="node-2", classes=["Document"], props={}, payload_ref="file:///tmp/private")
    assert repo.load_payload_text(inert) == ""
    assert not hasattr(repo, "_load_from_url")
    assert not hasattr(repo, "_load_from_file")
    assert not hasattr(repo, "_load_from_s3")


def test_bounded_upload_still_parses_to_inline_content(monkeypatch) -> None:
    monkeypatch.setenv("ACTIVEKG_TEST_NO_DB", "true")
    monkeypatch.setenv("JWT_ENABLED", "false")
    from activekg.api import main
    from activekg.connectors import chunker, extract

    class UploadRepo:
        pass

    class MemoryUpload:
        filename = "resume.txt"
        content_type = "text/plain"

        async def read(self) -> bytes:
            return b"synthetic resume"

    observed: dict[str, object] = {}

    def fake_extract(data: bytes, content_type: str) -> str:
        observed["data"] = data
        observed["content_type"] = content_type
        return "bounded inline resume text"

    def fake_chunks(**kwargs):
        observed["chunk_args"] = kwargs
        return ["chunk-1"]

    monkeypatch.setattr(main, "repo", UploadRepo())
    monkeypatch.setattr(main, "AUTO_EMBED_ON_CREATE", False)
    monkeypatch.setattr(extract, "extract_text", fake_extract)
    monkeypatch.setattr(chunker, "create_chunk_nodes", fake_chunks)

    result = asyncio.run(
        main.upload_files(
            files=[MemoryUpload()],  # type: ignore[list-item]
            tenant_id="synthetic-tenant",
            classes="Document,Resume",
            _rl=None,
            claims=None,
        )
    )

    assert result == {
        "uploaded": 1,
        "skipped": 0,
        "chunks_created": 1,
        "embeddings_queued": 0,
        "files": [{"filename": "resume.txt", "chunks": 1, "status": "ok"}],
    }
    assert observed["data"] == b"synthetic resume"
    assert observed["content_type"] == "text/plain"
    chunk_args = observed["chunk_args"]
    assert isinstance(chunk_args, dict)
    assert chunk_args["text"] == "bounded inline resume text"
    assert chunk_args["tenant_id"] == "synthetic-tenant"


def test_api_source_has_no_q_and_a_provider_or_live_handler_wiring() -> None:
    source = Path("activekg/api/main.py").read_text()
    forbidden = (
        "AskRequest",
        "LLMProvider",
        "build_strict_citation_prompt",
        "filter_context_by_similarity",
        '@app.post("/ask"',
        '@app.post("/ask/stream"',
        '@app.post("/debug/search_explain"',
    )
    assert all(token not in source for token in forbidden)
    assert "app.include_router(grounded_qa_retirement_router)" in source
    assert '@app.post("/search"' in source
