from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from activekg.api import auth
from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.api.candidate_privacy import (
    changes as changes_endpoint,
)
from activekg.api.candidate_privacy import router, set_repository
from activekg.api.candidate_privacy import (
    snapshot as snapshot_endpoint,
)
from activekg.privacy.models import (
    CandidatePrivacyAction,
    CandidatePrivacyDecision,
    CandidatePrivacyScope,
    CandidatePrivacyState,
    DirectiveRecord,
)
from activekg.privacy.repository import CandidatePrivacyConflict

REQUEST_ID = UUID("11111111-1111-4111-8111-111111111111")
DIRECTIVE_ID = UUID("22222222-2222-4222-8222-222222222222")
EVIDENCE_ID = UUID("33333333-3333-4333-8333-333333333333")
CANARY = "raw.person+private@example.test"


class FakeRepository:
    def __init__(self) -> None:
        self.create_identifiers: list[tuple[str, str]] = []

    @staticmethod
    def _record() -> DirectiveRecord:
        return DirectiveRecord(
            directive_id=DIRECTIVE_ID,
            action=CandidatePrivacyAction.REQUEST_ERASURE,
            scope=CandidatePrivacyScope.ACTIVE_PROFILE,
            state=CandidatePrivacyState.ACTIVE_QUARANTINE,
            version=3,
            effective_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
            decision=CandidatePrivacyDecision.BLOCK_ALL,
        )

    def create_directive(self, **kwargs):
        self.create_identifiers = [
            (item.identifier_type, item.normalized) for item in kwargs["identifiers"]
        ]
        return kwargs["request_id"], self._record()

    def transition_directive(self, **kwargs):
        return kwargs["request_id"], self._record()

    def evaluate(self, **_kwargs):
        return CandidatePrivacyDecision.BLOCK_GLOBAL

    def changes(self, **_kwargs):
        return [
            {
                "cursor": 4,
                "event_id": "44444444-4444-4444-8444-444444444444",
                "directive_id": str(DIRECTIVE_ID),
                "action": "request_erasure",
                "scope": "active_profile",
                "state": "active_quarantine",
                "version": 3,
                "effective_at": "2026-08-22T00:00:00+00:00",
            }
        ]

    def snapshot(self, **_kwargs):
        return 4, [
            {
                "directive_id": str(DIRECTIVE_ID),
                "action": "request_erasure",
                "scope": "active_profile",
                "state": "active_quarantine",
                "version": 3,
                "effective_at": "2026-08-22T00:00:00+00:00",
            }
        ]


def _env(*, intake: bool) -> dict[str, str]:
    return {
        "CANDIDATE_PRIVACY_HMAC_KEY_V1": base64.b64encode(b"k" * 32).decode(),
        "CANDIDATE_PRIVACY_HMAC_ACTIVE_VERSION": "1",
        "CANDIDATE_PRIVACY_INTAKE_ENABLED": str(intake).lower(),
        "CANDIDATE_PRIVACY_FLOW_ISSUER": "flow",
        "CANDIDATE_PRIVACY_FLOW_ACTOR_ID": "flow-service",
        "CANDIDATE_PRIVACY_SIGNAL_ISSUER": "signal",
        "CANDIDATE_PRIVACY_SIGNAL_ACTOR_ID": "signal-service",
    }


def _claims(issuer: str, actor: str, scope: str) -> JWTClaims:
    return JWTClaims(
        tenant_id="system",
        actor_id=actor,
        actor_type="service",
        scopes=[scope],
        issuer=issuer,
    )


def _app(claims: JWTClaims | None, repository: FakeRepository) -> FastAPI:
    app = FastAPI()
    app.include_router(router)

    async def claims_override():
        return claims

    app.dependency_overrides[get_jwt_claims] = claims_override
    set_repository(repository)
    return app


def _request(app: FastAPI, method: str, path: str, **kwargs) -> httpx.Response:
    async def execute() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(execute())


def _create_body() -> dict[str, object]:
    return {
        "request_id": str(REQUEST_ID),
        "action": "request_erasure",
        "authority_type": "verified_candidate",
        "evidence_ref": str(EVIDENCE_ID),
        "reason_code": "candidate_erasure_request",
        "identifiers": [{"identifier_type": "email", "value": CANARY}],
    }


def test_unauthorized_malformed_and_oversized_body_fail_before_parsing() -> None:
    repo = FakeRepository()
    denied = _claims("flow", "wrong-service", "candidate-privacy:write")
    with (
        patch.dict("os.environ", _env(intake=True), clear=True),
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "JWT_ISSUER", "flow"),
        patch.object(auth, "SIGNAL_JWT_ISSUER", "signal"),
    ):
        response = _request(
            _app(denied, repo),
            "POST",
            "/candidate-privacy/directives",
            content=(CANARY + "x" * 70_000),
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 403
    assert response.json() == {"detail": "candidate_privacy_service_auth_denied"}
    assert CANARY not in response.text


def test_disabled_intake_refuses_before_body_dependent_behavior() -> None:
    repo = FakeRepository()
    writer = _claims("flow", "flow-service", "candidate-privacy:write")
    with (
        patch.dict("os.environ", _env(intake=False), clear=True),
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "JWT_ISSUER", "flow"),
        patch.object(auth, "SIGNAL_JWT_ISSUER", "signal"),
    ):
        response = _request(
            _app(writer, repo),
            "POST",
            "/candidate-privacy/directives",
            content=CANARY,
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 503
    assert response.json() == {"detail": "candidate_privacy_intake_disabled"}
    assert CANARY not in response.text


def test_authorized_validation_errors_never_echo_submitted_identity() -> None:
    repo = FakeRepository()
    writer = _claims("flow", "flow-service", "candidate-privacy:write")
    with (
        patch.dict("os.environ", _env(intake=True), clear=True),
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "JWT_ISSUER", "flow"),
        patch.object(auth, "SIGNAL_JWT_ISSUER", "signal"),
    ):
        malformed = _request(
            _app(writer, repo),
            "POST",
            "/candidate-privacy/directives",
            content='{"identifiers":[{"identifier_type":"email","value":"' + CANARY + '"}]',
            headers={"content-type": "application/json"},
        )
        oversized = _request(
            _app(writer, repo),
            "POST",
            "/candidate-privacy/directives",
            content=CANARY + "x" * 70_000,
            headers={"content-type": "application/json"},
        )
    assert malformed.status_code == 422
    assert malformed.json() == {"detail": "candidate_privacy_request_invalid"}
    assert oversized.status_code == 413
    assert oversized.json() == {"detail": "candidate_privacy_request_too_large"}
    assert CANARY not in malformed.text + oversized.text


def test_hard_purge_transition_is_unrepresentable() -> None:
    repo = FakeRepository()
    writer = _claims("flow", "flow-service", "candidate-privacy:write")
    body = {
        "request_id": str(REQUEST_ID),
        "expected_version": 3,
        "transition": "hard_purge_eligible",
        "evidence_ref": str(EVIDENCE_ID),
        "reason_code": "operator_correction",
    }
    with (
        patch.dict("os.environ", _env(intake=True), clear=True),
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "JWT_ISSUER", "flow"),
        patch.object(auth, "SIGNAL_JWT_ISSUER", "signal"),
    ):
        response = _request(
            _app(writer, repo),
            "POST",
            f"/candidate-privacy/directives/{DIRECTIVE_ID}/transitions",
            json=body,
        )
    assert response.status_code == 422
    assert response.json() == {"detail": "candidate_privacy_request_invalid"}


def test_write_response_is_minimal_and_never_echoes_identity_or_evidence() -> None:
    repo = FakeRepository()
    writer = _claims("flow", "flow-service", "candidate-privacy:write")
    with (
        patch.dict("os.environ", _env(intake=True), clear=True),
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "JWT_ISSUER", "flow"),
        patch.object(auth, "SIGNAL_JWT_ISSUER", "signal"),
    ):
        response = _request(
            _app(writer, repo), "POST", "/candidate-privacy/directives", json=_create_body()
        )
    assert response.status_code == 200
    assert set(response.json()) == {
        "request_id",
        "directive_id",
        "action",
        "scope",
        "state",
        "version",
        "effective_at",
        "decision",
    }
    assert response.json()["decision"] == "block_all"
    assert CANARY not in response.text
    assert str(EVIDENCE_ID) not in response.text
    assert repo.create_identifiers == [("email", CANARY)]


def test_read_routes_accept_signal_and_expose_only_bounded_authority_fields() -> None:
    repo = FakeRepository()
    reader = _claims("signal", "signal-service", "candidate-privacy:read")
    body = {
        "subjects": [
            {
                "request_ref": str(REQUEST_ID),
                "identifiers": [{"identifier_type": "email", "value": CANARY}],
            }
        ]
    }
    with (
        patch.dict("os.environ", _env(intake=False), clear=True),
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "JWT_ISSUER", "flow"),
        patch.object(auth, "SIGNAL_JWT_ISSUER", "signal"),
    ):
        app = _app(reader, repo)
        eligibility = _request(app, "POST", "/candidate-privacy/eligibility/batch", json=body)
        changes = changes_endpoint(reader, after_cursor=0, limit=10)
        snapshot = snapshot_endpoint(
            reader, after_directive_id=None, high_water_cursor=None, limit=10
        )
    assert eligibility.status_code == 200
    assert eligibility.json() == {
        "results": [{"request_ref": str(REQUEST_ID), "decision": "block_global"}],
        "count": 1,
    }
    combined = eligibility.text + str(changes) + str(snapshot)
    assert CANARY not in combined
    assert str(EVIDENCE_ID) not in combined


def test_snapshot_rejects_a_future_high_water_cursor_without_leaking_details() -> None:
    class ConflictRepository(FakeRepository):
        def snapshot(self, **_kwargs):
            raise CandidatePrivacyConflict("internal cursor detail")

    reader = _claims("signal", "signal-service", "candidate-privacy:read")
    with (
        patch.dict("os.environ", _env(intake=False), clear=True),
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "JWT_ISSUER", "flow"),
        patch.object(auth, "SIGNAL_JWT_ISSUER", "signal"),
    ):
        set_repository(ConflictRepository())
        with pytest.raises(HTTPException) as captured:
            snapshot_endpoint(
                reader,
                after_directive_id=None,
                high_water_cursor=999999,
                limit=100,
            )

    assert captured.value.status_code == 409
    assert captured.value.detail == "candidate_privacy_snapshot_conflict"
    assert "internal cursor detail" not in str(captured.value.detail)


def test_router_registers_exactly_five_internal_routes() -> None:
    paths = {(route.path, method) for route in router.routes for method in route.methods or set()}
    assert paths == {
        ("/candidate-privacy/directives", "POST"),
        ("/candidate-privacy/directives/{directive_id}/transitions", "POST"),
        ("/candidate-privacy/eligibility/batch", "POST"),
        ("/candidate-privacy/changes", "GET"),
        ("/candidate-privacy/snapshot", "GET"),
    }


def test_full_application_registers_exactly_75_routes() -> None:
    script = """
from fastapi.routing import APIRoute
from activekg.api.main import app
routes = [route for route in app.routes if isinstance(route, APIRoute)]
privacy = sorted((method, route.path) for route in routes for method in route.methods or set()
                 if route.path.startswith('/candidate-privacy/'))
print(len(routes))
print(repr(privacy))
"""
    env = {
        **os.environ,
        "ACTIVEKG_TEST_NO_DB": "true",
        "JWT_ENABLED": "false",
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.fspath(Path(__file__).resolve().parents[1]),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    lines = result.stdout.strip().splitlines()
    assert lines[-2] == "75"
    assert lines[-1] == repr(
        sorted(
            {
                ("POST", "/candidate-privacy/directives"),
                ("POST", "/candidate-privacy/directives/{directive_id}/transitions"),
                ("POST", "/candidate-privacy/eligibility/batch"),
                ("GET", "/candidate-privacy/changes"),
                ("GET", "/candidate-privacy/snapshot"),
            }
        )
    )
