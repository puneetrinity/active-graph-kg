"""Fake cache only: no database, provider or production credentials."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from activekg.api import auth
from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.rc_skills.contracts import ReadResponse
from activekg.rc_skills.repository import RcRepository, RcUnavailable
from activekg.rc_skills.router import router

ID = "10000000-0000-4000-8000-000000000001"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("RC_SKILLS_ENABLED", "true")
    monkeypatch.setenv("ACTIVEKG_DSN", "synthetic-never-connected")
    monkeypatch.setattr(auth, "JWT_ENABLED", True)
    monkeypatch.setattr(auth, "SIGNAL_JWT_ISSUER", "signal")
    app = FastAPI()
    app.include_router(router)
    state = {
        "claims": JWTClaims(
            "org_1", "signal-service", "service", ["rc-skills:read"], issuer="signal"
        )
    }
    app.dependency_overrides[get_jwt_claims] = lambda: state["claims"]
    calls = []

    def read(self, service, command):
        calls.append((service, command))
        return ReadResponse.model_validate(
            {
                "version": 1,
                "candidates": [
                    {"globalCandidateId": value, "state": "miss", "observation": None}
                    for value in command.candidate_ids
                ],
            }
        )

    monkeypatch.setattr(RcRepository, "read", read)
    with TestClient(app) as http:
        yield http, state, calls


def test_authorized_minimal_response(client):
    http, _, calls = client
    response = http.post("/internal/rc-skills/read", json={"version": 1, "candidateIds": [ID]})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json() == {
        "version": 1,
        "candidates": [{"globalCandidateId": ID, "state": "miss", "observation": None}],
    }
    assert calls[0][0] == "signal-service"


@pytest.mark.parametrize(
    "change",
    [
        {"issuer": "flow"},
        {"actor_type": "user"},
        {"actor_id": "other-service"},
        {"tenant_id": "global"},
        {"tenant_id": "org_0"},
        {"scopes": ["kg:write"]},
        {"scopes": ["rc-skills:write"]},
        {"scopes": ["rc-skills:read", "kg:write"]},
    ],
)
def test_exact_service_authority_before_body(client, change):
    http, state, calls = client
    for name, value in change.items():
        setattr(state["claims"], name, value)
    response = http.post("/internal/rc-skills/read", content="invalid-json")
    assert response.status_code == 403
    assert calls == []


def test_no_token_is_not_service_authority(client):
    http, state, calls = client
    state["claims"] = None
    assert http.post("/internal/rc-skills/read", json={}).status_code == 401
    assert calls == []


@pytest.mark.parametrize("gate", ["false", "invalid"])
def test_flag_off_never_opens_cache_or_parses_body(client, monkeypatch, gate):
    http, _, calls = client
    monkeypatch.setenv("RC_SKILLS_ENABLED", gate)
    assert http.post("/internal/rc-skills/read", content="invalid-json").status_code == 503
    assert calls == []


@pytest.mark.parametrize(
    "body",
    [
        {"version": True, "candidateIds": []},
        {"version": 1, "candidateIds": [ID], "tenantId": "org_2"},
        {"version": 1, "candidateIds": [ID, ID]},
        {"version": 1, "candidateIds": ["bad"]},
    ],
)
def test_closed_bounded_wire(client, body):
    http, _, calls = client
    assert http.post("/internal/rc-skills/read", json=body).status_code == 422
    assert calls == []


def test_body_bound_before_json(client):
    http, _, calls = client
    assert http.post("/internal/rc-skills/read", content=" " * 65537).status_code == 413
    assert calls == []


@pytest.mark.parametrize(
    "error,status",
    [
        ("temporarily_unavailable", 503),
        ("unavailable", 404),
        ("restricted", 403),
        ("conflict", 409),
        ("denied", 403),
        ("unsafe-secret", 503),
    ],
)
def test_permanent_errors_not_retryable_and_details_do_not_escape(
    client, monkeypatch, error, status
):
    http, _, _ = client

    def fail(*_args):
        raise RcUnavailable(error)

    monkeypatch.setattr(RcRepository, "read", fail)
    response = http.post("/internal/rc-skills/read", json={"version": 1, "candidateIds": []})
    assert response.status_code == status
    assert "unsafe-secret" not in response.text


def test_read_scope_cannot_finish(client):
    http, _, calls = client
    assert http.post("/internal/rc-skills/finish", json={}).status_code == 403
    assert calls == []
