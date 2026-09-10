from __future__ import annotations

import asyncio
import hashlib
import json
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi import FastAPI

from activekg.api import auth
from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.api.organization_candidates import (
    OrganizationCandidateIntakeRequest,
    compute_idempotency_key,
    router,
)


def _claims(**changes: object) -> JWTClaims:
    values: dict[str, object] = {
        "tenant_id": "org_42",
        "actor_id": "vantahire-backend",
        "actor_type": "service",
        "scopes": ["organization-candidate:write"],
        "issuer": "vantahire",
    }
    values.update(changes)
    return JWTClaims(**values)


def _body(**changes: object) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "reference_id": str(uuid4()),
        "application_id": 42,
        "job_id": 7,
        "resume_version_id": str(uuid4()),
        "resume_version": 1,
        "origin": "candidate_applied",
        "source_kind": "direct_upload",
        "source_resume_id": None,
        "source_observed_at": "2026-09-01T10:00:00Z",
        "content_sha256": "a" * 64,
        "byte_count": 1024,
        "media_type": "application/pdf",
        "extracted_text_sha256": "b" * 64,
        "captured_at": "2026-09-01T10:00:00Z",
        "idempotency_key": "0" * 64,
        "privacy_subject": [
            {"identifier_type": "vantahire_application_id", "value": "42"},
            {"identifier_type": "email", "value": "private.person@example.test"},
        ],
    }
    body.update(changes)
    payload = OrganizationCandidateIntakeRequest.model_validate_json(json.dumps(body))
    body["idempotency_key"] = compute_idempotency_key(payload, "org_42")
    return body


def _app(claims: JWTClaims | None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)

    async def claims_override() -> JWTClaims | None:
        return claims

    app.dependency_overrides[get_jwt_claims] = claims_override
    return app


def _validate_body(body: dict[str, object]) -> OrganizationCandidateIntakeRequest:
    return OrganizationCandidateIntakeRequest.model_validate_json(json.dumps(body))


def _request(app: FastAPI, **kwargs: object) -> httpx.Response:
    async def execute() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post("/organization-candidates/intake", **kwargs)

    return asyncio.run(execute())


def test_exact_service_authority_and_flag_precede_body_parsing() -> None:
    canary = "private.person@example.test"
    denied = (
        _claims(actor_id="other"),
        _claims(actor_type="user"),
        _claims(scopes=["organization-candidate:write", "kg:write"]),
        _claims(tenant_id="platform"),
        _claims(issuer="signal"),
    )
    for claims in denied:
        with (
            patch.object(auth, "JWT_ENABLED", True),
            patch.object(auth, "JWT_ISSUER", "vantahire"),
            patch.dict("os.environ", {"ORGANIZATION_CANDIDATE_INTAKE_ENABLED": "true"}),
        ):
            response = _request(_app(claims), content=canary + "{" * 70_000)
        assert response.status_code == 403
        assert canary not in response.text

    with (
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "JWT_ISSUER", "vantahire"),
        patch.dict("os.environ", {"ORGANIZATION_CANDIDATE_INTAKE_ENABLED": "false"}),
    ):
        response = _request(_app(_claims()), content=canary)
    assert response.status_code == 503
    assert canary not in response.text


def test_valid_request_passes_only_pii_minimized_model(monkeypatch) -> None:
    captured: list[OrganizationCandidateIntakeRequest] = []

    def store(payload: OrganizationCandidateIntakeRequest, claims: JWTClaims):
        captured.append(payload)
        assert claims.tenant_id == "org_42"
        return {"delivery_status": "recorded"}

    monkeypatch.setattr("activekg.api.organization_candidates._store", store)
    with (
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "JWT_ISSUER", "vantahire"),
        patch.dict("os.environ", {"ORGANIZATION_CANDIDATE_INTAKE_ENABLED": "true"}),
    ):
        response = _request(_app(_claims()), json=_body())
    assert response.status_code == 200
    assert response.json() == {"delivery_status": "recorded"}
    persistent = captured[0].model_dump(exclude={"privacy_subject"})
    assert "email" not in str(persistent).lower()
    assert "phone" not in str(persistent).lower()


def test_source_binding_unknown_fields_and_idempotency_are_strict() -> None:
    direct = _body()
    saved = _body(
        source_kind="saved_resume",
        source_resume_id=9,
        privacy_subject=[
            {"identifier_type": "vantahire_application_id", "value": "42"},
            {"identifier_type": "vantahire_resume_id", "value": "9"},
        ],
    )
    assert _validate_body(direct)
    assert _validate_body(saved)
    for bad in (
        {**direct, "unknown": True},
        {**direct, "source_kind": "saved_resume"},
        {**direct, "byte_count": 5 * 1024 * 1024 + 1},
        {**direct, "media_type": "text/plain"},
        {
            **direct,
            "privacy_subject": [{"identifier_type": "vantahire_application_id", "value": "41"}],
        },
    ):
        try:
            _validate_body(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid organization candidate body was accepted")
    assert len(str(direct["idempotency_key"])) == hashlib.sha256().digest_size * 2


def test_idempotency_vector_matches_flow_contract() -> None:
    body = _body(
        reference_id="11111111-1111-4111-8111-111111111111",
        application_id=42,
        job_id=7,
        resume_version_id="22222222-2222-4222-8222-222222222222",
        resume_version=1,
        origin="candidate_applied",
    )
    assert body["idempotency_key"] == (
        "b058485eb5868da8e7c7cd41968b193387f1d42236b48c68cbbc9c4ee81382ee"
    )
