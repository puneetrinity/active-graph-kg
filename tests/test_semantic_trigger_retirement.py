from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable

import pytest
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from activekg.api.retirement import (
    delete_trigger_pattern,
    list_trigger_patterns,
    register_trigger_pattern,
    semantic_triggers_router,
    semantic_triggers_unavailable_response,
)


@pytest.mark.parametrize(
    ("handler", "args"),
    [
        (register_trigger_pattern, ()),
        (list_trigger_patterns, ()),
        (delete_trigger_pattern, ("ignored",)),
    ],
)
def test_semantic_trigger_routes_are_stable_no_store_tombstones(
    handler: Callable[..., JSONResponse], args: tuple[object, ...]
) -> None:
    response = handler(*args)

    assert response.status_code == 410
    assert response.headers["cache-control"] == "no-store"
    assert bytes(response.body) == (
        b'{"detail":{"code":"MEMORY_SEMANTIC_TRIGGERS_UNAVAILABLE",'
        b'"message":"Semantic triggers are not available."}}'
    )


def test_only_the_three_compatibility_methods_are_registered() -> None:
    routes = [route for route in semantic_triggers_router.routes if isinstance(route, APIRoute)]
    registrations = {
        (method, route.path) for route in routes for method in getattr(route, "methods", [])
    }

    assert registrations == {
        ("POST", "/triggers"),
        ("GET", "/triggers"),
        ("DELETE", "/triggers/{name}"),
    }
    assert len(routes) == 3
    assert all(route.status_code == 410 for route in routes)
    assert all(not route.dependencies for route in routes)
    assert all(not route.dependant.body_params for route in routes)


def test_retirement_helper_has_the_same_stable_contract() -> None:
    response = semantic_triggers_unavailable_response()

    assert response.status_code == 410
    assert response.headers["cache-control"] == "no-store"
    assert bytes(response.body) == (
        b'{"detail":{"code":"MEMORY_SEMANTIC_TRIGGERS_UNAVAILABLE",'
        b'"message":"Semantic triggers are not available."}}'
    )


def test_api_has_no_trigger_engine_runtime_or_demo_wiring() -> None:
    source = Path("activekg/api/main.py").read_text()

    assert "from activekg.triggers" not in source
    assert "PatternStore(" not in source
    assert "TriggerEngine(" not in source
    assert "trigger_engine=trigger_engine" not in source
    assert "trigger_engine=None" in source
    assert "fetch('/triggers" not in source
    assert "app.include_router(semantic_triggers_router)" in source

    tree = ast.parse(source)
    scheduler_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RefreshScheduler"
    ]
    assert len(scheduler_calls) == 2
    for call in scheduler_calls:
        trigger_keyword = next(
            (keyword for keyword in call.keywords if keyword.arg == "trigger_engine"), None
        )
        assert trigger_keyword is not None
        assert isinstance(trigger_keyword.value, ast.Constant)
        assert trigger_keyword.value.value is None


def test_events_reader_remains_registered_independently() -> None:
    source = Path("activekg/api/main.py").read_text()

    assert '@app.get("/events", response_model=None)' in source
    assert "def list_events(" in source
