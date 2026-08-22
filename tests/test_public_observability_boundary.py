from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute

os.environ.setdefault("ACTIVEKG_TEST_NO_DB", "true")
os.environ.setdefault("JWT_ENABLED", "false")

from activekg.api import main, operational
from activekg.api.operational import (
    MetricsBoundary,
    OperationalBusy,
    OperationalPayloadTooLarge,
    ReadinessCoordinator,
    ReadinessResult,
    bounded_readiness_check,
    filter_json_metrics,
)
from activekg.api.retirement import public_observability_retirement_router
from activekg.common.control_plane import (
    ControlPlaneUnauthorized,
    ControlPlaneUnavailable,
    verify_control_plane_authorization,
)
from activekg.extraction.worker import WorkerHealthState, worker_health_response

TOKEN = "synthetic-control-plane-token-for-tests-only"


def test_control_plane_verifier_fails_closed_and_compares_exact_bearer() -> None:
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ControlPlaneUnavailable):
            verify_control_plane_authorization(None)

    with patch.dict(os.environ, {"ACTIVEKG_CONTROL_PLANE_TOKEN": TOKEN}, clear=False):
        for value in (None, "", TOKEN, f"Basic {TOKEN}", "Bearer wrong"):
            with pytest.raises(ControlPlaneUnauthorized):
                verify_control_plane_authorization(value)
        verify_control_plane_authorization(f"Bearer {TOKEN}")


def test_route_registration_count_and_public_retirement_contract() -> None:
    routes = [route for route in main.app.routes if isinstance(route, APIRoute)]
    registrations = {(method, route.path) for route in routes for method in route.methods}
    assert len(routes) == 75
    assert {
        ("GET", "/openapi.json"),
        ("GET", "/docs"),
        ("GET", "/docs/oauth2-redirect"),
        ("GET", "/redoc"),
        ("GET", "/demo"),
    } <= registrations

    retired = [
        route
        for route in public_observability_retirement_router.routes
        if isinstance(route, APIRoute)
    ]
    assert len(retired) == 5
    assert all(route.status_code == 410 for route in retired)
    assert all(not route.dependencies for route in retired)
    assert all(not route.dependant.body_params for route in retired)

    debug_intent = next(route for route in routes if route.path == "/debug/intent")
    dependency_calls = {
        getattr(dependency.call, "__name__", "")
        for dependency in debug_intent.dependant.dependencies
    }
    assert "get_jwt_claims" in dependency_calls
    assert "dep" in dependency_calls


@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("/demo", "MEMORY_DEMO_UNAVAILABLE"),
        ("/openapi.json", "MEMORY_API_DOCS_UNAVAILABLE"),
        ("/docs", "MEMORY_API_DOCS_UNAVAILABLE"),
        ("/docs/oauth2-redirect", "MEMORY_API_DOCS_UNAVAILABLE"),
        ("/redoc", "MEMORY_API_DOCS_UNAVAILABLE"),
    ],
)
def test_demo_and_docs_are_dependency_free_no_store_tombstones(path: str, code: str) -> None:
    route = next(
        route
        for route in public_observability_retirement_router.routes
        if isinstance(route, APIRoute) and route.path == path
    )
    response = route.endpoint()
    assert response.status_code == 410
    assert response.headers["cache-control"] == "no-store"
    assert json.loads(response.body)["detail"]["code"] == code


def test_api_health_is_public_constant_cost_and_no_store() -> None:
    response = main.health()
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert bytes(response.body) == b'{"status":"alive","service":"activekg-api"}'


def test_operational_auth_runs_before_readiness_or_metrics_work() -> None:
    routes = {route.path: route for route in main.app.routes if isinstance(route, APIRoute)}
    for path in ("/readyz", "/metrics", "/prometheus"):
        calls = {dependency.call for dependency in routes[path].dependant.dependencies}
        assert main._require_control_plane in calls

    with patch.dict(os.environ, {"ACTIVEKG_CONTROL_PLANE_TOKEN": TOKEN}, clear=False):
        with pytest.raises(HTTPException) as error:
            main._require_control_plane(None)
    assert error.value.status_code == 401
    assert error.value.headers["Cache-Control"] == "no-store"


def test_missing_operational_config_returns_503_before_work() -> None:
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(HTTPException) as error:
            main._require_control_plane(f"Bearer {TOKEN}")
    assert error.value.status_code == 503
    assert error.value.headers["Cache-Control"] == "no-store"
    assert error.value.detail["code"] == "CONTROL_PLANE_AUTH_UNAVAILABLE"


def test_readyz_uses_bounded_result_after_auth() -> None:
    coordinator = ReadinessCoordinator()
    with (
        patch.dict(os.environ, {"ACTIVEKG_CONTROL_PLANE_TOKEN": TOKEN}, clear=False),
        patch.object(main, "_readiness_coordinator", coordinator),
        patch.object(
            main,
            "bounded_readiness_check",
            return_value=ReadinessResult(False, ("database_unavailable",)),
        ),
        patch.object(main, "verification_key_problems", return_value=[]),
    ):
        main._require_control_plane(f"Bearer {TOKEN}")
        response = main.readyz(None)
    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert json.loads(response.body) == {
        "status": "not_ready",
        "reasons": ["database_unavailable"],
    }


def test_authenticated_no_cache_directive_forces_a_fresh_readiness_snapshot() -> None:
    class RecordingCoordinator:
        force_refresh: bool | None = None

        def run(self, _check, *, force_refresh: bool = False) -> ReadinessResult:
            self.force_refresh = force_refresh
            return ReadinessResult(True)

    coordinator = RecordingCoordinator()
    with patch.object(main, "_readiness_coordinator", coordinator):
        response = main.readyz(None, "max-age=0, no-cache")

    assert response.status_code == 200
    assert coordinator.force_refresh is True


def test_readiness_is_single_flight_and_uses_at_most_eight_catalog_statements() -> None:
    class FakeCursor:
        def __init__(self) -> None:
            self.statements: list[str] = []
            self.last = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, statement: str, _params=None) -> None:
            self.last = " ".join(statement.split())
            self.statements.append(self.last)

        def fetchone(self):
            if "to_regclass" in self.last:
                return (
                    "schema_migrations",
                    "activekg_schema_control.target_identity",
                    "activekg_schema_control.release_attempts",
                    "candidate_privacy_directive_events",
                    "candidate_privacy_directives",
                    "candidate_privacy_identity_tokens",
                )
            if "FROM activekg_schema_control.target_identity" in self.last:
                return (
                    1,
                    "memory",
                    "production",
                    "11111111-1111-4111-8111-111111111111",
                    4,
                    0,
                    "success",
                )
            if "FROM pg_roles r" in self.last:
                return (False, False, False, False, False)
            raise AssertionError(self.last)

        def fetchall(self):
            if "FROM schema_migrations" in self.last:
                base = Path("db/migrations")
                return [
                    (name, hashlib.sha256((base / name).read_bytes()).hexdigest())
                    for name in operational.MIGRATIONS
                ]
            if "SELECT 'index'" in self.last:
                rows = [
                    ("index", name, {"valid": True})
                    for name in operational._REQUIRED_INDEXES | operational._PRIVACY_INDEXES
                ]
                for name in operational._REQUIRED_FUNCTIONS:
                    details = {}
                    if name == operational._APPEND_ONLY_FUNCTION:
                        details = {
                            "arguments": 0,
                            "returns_trigger": True,
                            "language": "plpgsql",
                            "security_definer": False,
                            "source": operational._APPEND_ONLY_FUNCTION_BODY,
                            "owned_by_runtime": False,
                        }
                    rows.append(("function", name, details))
                for name in operational._PRIVACY_FUNCTIONS:
                    if name == operational._PRIVACY_APPEND_ONLY_FUNCTION:
                        details = {
                            "arguments": 0,
                            "returns_trigger": True,
                            "language": "plpgsql",
                            "security_definer": False,
                            "source": operational._PRIVACY_APPEND_ONLY_FUNCTION_BODY,
                            "owned_by_runtime": False,
                        }
                    else:
                        details = {
                            "arguments": operational._PRIVACY_FUNCTION_ARGUMENTS[name],
                            "returns_trigger": False,
                            "language": "sql",
                            "security_definer": True,
                            "config": ["search_path=pg_catalog, public"],
                            "source": "synthetic",
                            "owned_by_runtime": False,
                        }
                    rows.append(("function", name, details))
                for table, names in operational._REQUIRED_CONSTRAINTS_BY_TABLE.items():
                    for name in names:
                        details = {
                            "type": "c",
                            "delete_action": " ",
                            "validated": True,
                            "definition": "CHECK (true)",
                        }
                        expected_check = operational._EXPECTED_CHECK_DEFINITIONS.get((table, name))
                        expected_structural = operational._EXPECTED_STRUCTURAL_DEFINITIONS.get(
                            (table, name)
                        )
                        if expected_check is not None:
                            details["definition"] = expected_check
                        if expected_structural is not None:
                            details["type"] = expected_structural[0]
                            details["definition"] = expected_structural[1]
                        rows.append(("constraint", f"{table}.{name}", details))
                rows.extend(
                    (
                        "trigger",
                        f"{table}.{name}",
                        {"function": function, "type": trigger_type, "enabled": "O"},
                    )
                    for table, name, function, trigger_type in operational._SUPPRESSION_TRIGGERS
                )
                rows.extend(
                    (
                        "trigger",
                        f"{table}.{name}",
                        {"function": function, "type": trigger_type, "enabled": "O"},
                    )
                    for table, name, function, trigger_type in operational._PRIVACY_TRIGGERS
                )
                rows.append(
                    (
                        "sequence",
                        operational._SUPPRESSION_SEQUENCE,
                        {"kind": "S", "usage": True, "owned_by_runtime": False},
                    )
                )
                rows.append(
                    (
                        "sequence",
                        operational._PRIVACY_SEQUENCE,
                        {"kind": "S", "usage": False, "owned_by_runtime": False},
                    )
                )
                rows.extend(("public_column", name, {}) for name in operational._PUBLIC_COLUMNS)
                rows.append(
                    (
                        "privilege",
                        "suppression_tables",
                        {
                            "receipt_select": True,
                            "receipt_insert": True,
                            "receipt_update": False,
                            "receipt_delete": False,
                            "receipt_truncate": False,
                            "tombstone_delete": False,
                            "tombstone_truncate": False,
                            "person_delete": False,
                            "person_truncate": False,
                        },
                    )
                )
                rows.append(
                    (
                        "privacy_privilege",
                        "runtime",
                        {
                            "events_select": True,
                            "events_write": False,
                            "directives_select": True,
                            "directives_write": False,
                            "tokens_select": False,
                            "tokens_write": False,
                        },
                    )
                )
                rows.extend(
                    ("privacy_function_privilege", name, {"execute": True})
                    for name in operational._PRIVACY_FUNCTION_ARGUMENTS
                )
                return rows
            if "FROM pg_class c" in self.last:
                return [
                    (name, True, name == "candidate_contact_evidence", "owner", "runtime")
                    for name in operational._CANDIDATE_TABLES
                    + operational._SHARED_TABLES
                    + operational._PRIVACY_TABLES
                ]
            if "FROM pg_policies" in self.last:
                tenant_expression = (
                    "((tenant_id = current_setting('app.current_tenant_id', true)) "
                    "AND (tenant_id <> '__quarantine__'))"
                )
                rows = []
                for name in operational._CANDIDATE_TABLES:
                    rows.extend(
                        [
                            (
                                name,
                                f"tenant_isolation_{name}",
                                "PERMISSIVE",
                                "{public}",
                                "ALL",
                                tenant_expression,
                                tenant_expression,
                            ),
                            (
                                name,
                                f"admin_all_{name}",
                                "PERMISSIVE",
                                "{admin_role}",
                                "ALL",
                                "true",
                                "true",
                            ),
                        ]
                    )
                rows.extend(
                    [
                        (
                            "candidate_privacy_directive_events",
                            "candidate_privacy_events_runtime_read",
                            "PERMISSIVE",
                            "{public}",
                            "SELECT",
                            "true",
                            None,
                        ),
                        (
                            "candidate_privacy_directives",
                            "candidate_privacy_directives_runtime_read",
                            "PERMISSIVE",
                            "{public}",
                            "SELECT",
                            "true",
                            None,
                        ),
                    ]
                )
                return rows
            raise AssertionError(self.last)

    cursor = FakeCursor()

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return cursor

    class FakePool:
        timeout: float | None = None

        def connection(self, *, timeout: float):
            self.timeout = timeout
            return FakeConnection()

    class FakeRepository:
        pool = FakePool()

    with patch.dict(
        os.environ,
        {
            "ACTIVEKG_SCHEMA_TARGET_ID": "11111111-1111-4111-8111-111111111111",
            "ACTIVEKG_SCHEMA_ENVIRONMENT": "production",
        },
        clear=False,
    ):
        result = bounded_readiness_check(
            FakeRepository(),
            unsafe_search_configuration=False,
            jwt_enabled=True,
            jwt_problems=[],
        )
    assert result == ReadinessResult(True)
    assert FakeRepository.pool.timeout == 0.25
    assert len(cursor.statements) == 8
    assert not any("from candidates" in statement.lower() for statement in cursor.statements)

    cached = ReadinessCoordinator()
    checks = 0

    def changing_check() -> ReadinessResult:
        nonlocal checks
        checks += 1
        return ReadinessResult(checks == 1)

    assert cached.run(changing_check).ready
    assert cached.run(changing_check).ready
    assert checks == 1
    assert not cached.run(changing_check, force_refresh=True).ready
    assert checks == 2

    coordinator = ReadinessCoordinator()
    coordinator._lock.acquire()
    try:
        with pytest.raises(OperationalBusy):
            coordinator.run(lambda: ReadinessResult(True))
    finally:
        coordinator._lock.release()


def test_metrics_remove_sensitive_labels_and_share_fixed_size_limit() -> None:
    snapshot = {
        "counters": {
            "safe[mode=hybrid]": 1,
            "private[tenant_id=tenant-a]": 2,
            "private_org[organization_id=org-a]": 3,
        },
        "history": [
            {"value": 1, "labels": {"mode": "hybrid"}},
            {"value": 2, "labels": {"org_id": "org-a"}},
        ],
    }
    assert filter_json_metrics(snapshot) == {
        "counters": {"safe[mode=hybrid]": 1},
        "history": [{"value": 1, "labels": {"mode": "hybrid"}}],
    }

    boundary = MetricsBoundary()
    prometheus = boundary.prometheus_bytes(
        b"# HELP safe safe\n"
        b'safe_total{mode="hybrid"} 1\n'
        b'private_total{tenant_id="tenant-a"} 2\n'
        b'private_org_total{organization="org-a"} 3\n'
    )
    assert b"# HELP safe safe" in prometheus
    assert b"safe_total" in prometheus
    assert b"tenant-a" not in prometheus
    assert b"org-a" not in prometheus

    with pytest.raises(OperationalPayloadTooLarge):
        MetricsBoundary(max_bytes=4).json_bytes({"safe": 1})


def test_authenticated_metrics_are_no_store_and_label_filtered() -> None:
    with (
        patch.dict(os.environ, {"ACTIVEKG_CONTROL_PLANE_TOKEN": TOKEN}, clear=False),
        patch.object(
            main.metrics,
            "get_all_metrics",
            return_value={"counters": {"safe[mode=x]": 1, "private[tenant_id=t]": 2}},
        ),
        patch.object(
            main,
            "generate_latest",
            return_value=b'safe_total 1\nprivate_total{org_id="o"} 2\n',
        ),
    ):
        main._require_control_plane(f"Bearer {TOKEN}")
        json_response = main.get_metrics(None)
        prom_response = main.prometheus_metrics(None)

    assert json_response.status_code == prom_response.status_code == 200
    assert (
        json_response.headers["cache-control"]
        == prom_response.headers["cache-control"]
        == "no-store"
    )
    assert b"tenant_id" not in json_response.body
    assert b"org_id" not in prom_response.body
    assert b"safe_total" in prom_response.body


def test_worker_state_is_truthful_and_provider_heals() -> None:
    state = WorkerHealthState(1.0)
    ready, components = state.snapshot()
    assert not ready
    assert components["loop"] == "stale"

    state.provider_configured()
    state.loop_cycle_success()
    state.database_success()
    assert state.snapshot()[0]

    state.provider_failure()
    assert state.snapshot()[0]
    assert state.snapshot()[1]["provider"] == "degraded"
    state.provider_failure()
    assert not state.snapshot()[0]
    assert state.snapshot()[1]["provider"] == "error"
    state.provider_success()
    assert state.snapshot()[0]


def test_worker_health_is_public_and_readyz_is_private_in_memory_only() -> None:
    state = WorkerHealthState(1.0)
    state.provider_configured()
    state.loop_cycle_success()
    state.database_success()
    with patch.dict(os.environ, {"ACTIVEKG_CONTROL_PLANE_TOKEN": TOKEN}, clear=False):
        health = worker_health_response("/health", None, state)
        unauthenticated = worker_health_response("/readyz", None, state)
        authenticated = worker_health_response("/readyz", f"Bearer {TOKEN}", state)

    assert health[0] == 200
    assert health[1] == b'{"status":"alive","service":"extraction-worker"}'
    assert unauthenticated[0] == 401
    assert authenticated[0] == 200
    assert json.loads(authenticated[1])["components"] == {
        "loop": "ready",
        "redis": "ready",
        "database": "ready",
        "provider": "configured",
    }


def test_railway_healthcheck_uses_public_liveness() -> None:
    config = json.loads(Path("railway.json").read_text())
    assert config["deploy"]["healthcheckPath"] == "/health"
