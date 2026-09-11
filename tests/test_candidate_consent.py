from __future__ import annotations

import copy
import json

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from activekg.api import candidate_consent as receiver
from activekg.api.auth import JWTClaims, get_jwt_claims


def vector(with_resume: bool = False) -> dict:
    payload = {
        "schema_version": 1,
        "subject_id": "11111111-1111-4111-8111-111111111111",
        "event_id": "22222222-2222-4222-8222-222222222222",
        "version": 42,
        "action": "grant",
        "purpose": "platform_professional_matching",
        "purpose_version": 1,
        "copy_version": 1,
        "copy_sha256": receiver.COPY_SHA256,
        "captured_at": "2026-09-10T01:02:03.004Z",
        "source": {
            "source_id": "33333333-3333-4333-8333-333333333333",
            "source_version": 42,
            "profile": {
                "display_name": "Zoë 🦋",
                "headline": "Ingénieure",
                "location": "Paris",
                "skills": ["TypeScript", "Café"],
                "linkedin": None,
            },
            "resume": None,
        },
        "proof": {
            "verified_email": "consent@fixture.invalid",
            "privacy_subject": [{"identifier_type": "email", "value": "consent@fixture.invalid"}],
        },
        "idempotency_key": "0" * 64,
    }
    if with_resume:
        payload["source"]["resume"] = {
            "reference_id": "44444444-4444-4444-8444-444444444444",
            "resume_version_id": "55555555-5555-4555-8555-555555555555",
            "organization_id": 7,
            "application_id": 8,
            "job_id": 9,
            "content_sha256": "a" * 64,
            "byte_count": 123,
            "media_type": "application/pdf",
            "source_observed_at": "2026-09-01T02:03:04.005Z",
        }
        payload["proof"]["privacy_subject"].append(
            {"identifier_type": "vantahire_application_id", "value": "8"}
        )
    parsed = receiver.ConsentCommand.model_validate(payload)
    payload["idempotency_key"] = receiver.command_identity(parsed)[1]
    return payload


@pytest.mark.parametrize(
    ("resume", "digest", "key"),
    [
        (
            False,
            "5238f0df4cd9be9a2ef98de2cbd02b445babc15c2773bbb1e5b5ca14c7f155ea",
            "c25cf69df708ffd180faea6ec1bb4b4fe4e96f33187627b0a7602bf42fe2b5ab",
        ),
        (
            True,
            "d312454ae6c6fa71914d0259f348de838cdeb23995083146f04704eeaeedbe72",
            "630be69bd974c44e21a67bc9da2d254b4a8c2194c58a20d9338add1e6015e4c8",
        ),
    ],
)
def test_same_flow_unicode_null_resume_bytes(resume, digest, key):
    payload = receiver.ConsentCommand.model_validate(vector(resume))
    assert (
        receiver.COPY_SHA256 == "487f78065f2b6fd493814d9a1c5a760ce0550b0a1846f2a72d7d6b58d333f094"
    )
    assert receiver.command_identity(payload) == (digest, key)
    expected = [
        1,
        payload.subject_id,
        payload.event_id,
        42,
        "grant",
        "platform_professional_matching",
        1,
        receiver.COPY_SHA256,
        [
            "33333333-3333-4333-8333-333333333333",
            42,
            "Zoë 🦋",
            "Ingénieure",
            "Paris",
            ["TypeScript", "Café"],
            None,
            [
                "44444444-4444-4444-8444-444444444444",
                "55555555-5555-4555-8555-555555555555",
                7,
                8,
                9,
                "a" * 64,
                123,
                "application/pdf",
                "2026-09-01T02:03:04.005Z",
            ]
            if resume
            else None,
        ],
        "2026-09-10T01:02:03.004Z",
    ]
    assert receiver.canonical_bytes(payload) == json.dumps(
        expected, ensure_ascii=False, separators=(",", ":")
    )


@pytest.mark.parametrize("value", ["\ud800", "\udfff", "name\nline", "name\x00", "name\x85"])
def test_bad_scalar_text(value):
    payload = vector()
    payload["source"]["profile"]["display_name"] = value
    with pytest.raises(ValidationError):
        receiver.ConsentCommand.model_validate(payload)


@pytest.mark.parametrize(
    "delta",
    [
        {"skills": ["Java", " Java "]},
        {"email": "no@fixture.invalid"},
        {"linkedin": "https://example.invalid/in/person"},
        {"headline": "x" * 301},
        {"skills": [None]},
    ],
)
def test_invalid_profile(delta):
    payload = vector()
    payload["source"]["profile"].update(delta)
    with pytest.raises(ValidationError):
        receiver.ConsentCommand.model_validate(payload)


def test_transient_proof_not_persisted_in_digest():
    payload = vector()
    changed = copy.deepcopy(payload)
    changed["proof"] = {
        "verified_email": "another@fixture.invalid",
        "privacy_subject": [{"identifier_type": "email", "value": "another@fixture.invalid"}],
    }
    assert receiver.command_identity(
        receiver.ConsentCommand.model_validate(payload)
    ) == receiver.command_identity(receiver.ConsentCommand.model_validate(changed))
    assert "fixture.invalid" not in receiver.canonical_bytes(
        receiver.ConsentCommand.model_validate(payload)
    )


def test_unchecked_or_mismatched_proof_rejected():
    payload = vector(True)
    payload["proof"]["privacy_subject"][-1]["value"] = "9"
    with pytest.raises(ValidationError):
        receiver.ConsentCommand.model_validate(payload)
    payload = vector()
    payload["source"]["source_version"] = 43
    with pytest.raises(ValidationError):
        receiver.ConsentCommand.model_validate(payload)


@pytest.mark.parametrize(
    "delta,status",
    [
        ({"issuer": "signal"}, 403),
        ({"actor_id": "another-service"}, 403),
        ({"actor_type": "user"}, 403),
        ({"scopes": ["kg:write"]}, 403),
        ({"scopes": ["candidate-consent:write", "kg:write"]}, 403),
        ({"tenant_id": "org_7"}, 403),
        ({}, 200),
    ],
)
def test_exact_service_gate(monkeypatch, delta, status):
    monkeypatch.setattr(receiver.auth, "JWT_ENABLED", True)
    monkeypatch.setattr(receiver.auth, "JWT_AUDIENCE", "activekg")
    monkeypatch.setenv("CANDIDATE_CONSENT_INTAKE_ENABLED", "true")
    args = {
        "issuer": "vantahire",
        "actor_id": "vantahire-backend",
        "actor_type": "service",
        "scopes": ["candidate-consent:write"],
        "tenant_id": "candidate_11111111-1111-4111-8111-111111111111",
        **delta,
    }
    app = FastAPI()
    app.include_router(receiver.router)
    app.dependency_overrides[get_jwt_claims] = lambda: JWTClaims(**args)
    calls = []
    monkeypatch.setattr(receiver, "_store", lambda p, c: calls.append(True) or {"accepted": True})
    with TestClient(app) as client:
        response = client.post("/candidate-consent/events", json=vector())
    assert response.status_code == status
    assert len(calls) == (status == 200)


def test_connect_failure_returns_bounded_retryable_response(monkeypatch):
    monkeypatch.setattr(receiver.auth, "JWT_ENABLED", True)
    monkeypatch.setattr(receiver.auth, "JWT_AUDIENCE", "activekg")
    monkeypatch.setenv("CANDIDATE_CONSENT_INTAKE_ENABLED", "true")
    claims = JWTClaims(
        issuer="vantahire",
        actor_id="vantahire-backend",
        actor_type="service",
        scopes=["candidate-consent:write"],
        tenant_id="candidate_11111111-1111-4111-8111-111111111111",
    )
    app = FastAPI()
    app.include_router(receiver.router)
    app.dependency_overrides[get_jwt_claims] = lambda: claims

    def unavailable():
        raise psycopg.OperationalError("private-database-detail-sentinel")

    monkeypatch.setattr(receiver, "_connect", unavailable)
    with TestClient(app) as client:
        response = client.post("/candidate-consent/events", json=vector())
    assert response.status_code == 503
    assert response.json() == {"detail": "candidate_consent_temporarily_unavailable"}
    assert "sentinel" not in response.text
