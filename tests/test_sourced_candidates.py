from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI

from activekg.api import auth
from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.api.sourced_candidates import (
    SourcedCandidateIngestRequest,
    _assert_no_private_evidence,
    compute_idempotency_key,
    router,
)


def _claims(
    *,
    tenant: str = "org_7",
    actor: str = "signal-service",
    actor_type: str = "service",
    scopes: list[str] | None = None,
    issuer: str = "signal",
) -> JWTClaims:
    return JWTClaims(
        tenant_id=tenant,
        actor_id=actor,
        actor_type=actor_type,
        scopes=scopes if scopes is not None else ["candidate-source:write"],
        issuer=issuer,
    )


def _body(**changes: object) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "provider_namespace": "crustdata",
        "record_type": "person",
        "adapter_version": "crustdata_person_v1",
        "provider_record_id": "123456",
        "linkedin_url": "https://linkedin.com/in/alice-example",
        "acquisition_receipt_id": "receipt-7",
        "acquisition_generation": 2,
        "acquisition_slot": "exact",
        "acquired_at": "2026-09-01T10:00:00Z",
        "provider_observed_at": "2026-09-01T09:59:00Z",
        "normalized_profile": {
            "display_name": "Alice Example",
            "headline": "Backend Engineer",
            "languages": ["English"],
            "skills": ["Python", "PostgreSQL"],
            "current_employment": [],
            "past_employment": [],
            "education": [],
            "certifications": [],
            "honors": [],
            "public_profiles": [
                {
                    "platform": "linkedin",
                    "profile_url": "https://linkedin.com/in/alice-example",
                }
            ],
        },
    }
    body.update(changes)
    components = (
        "v1",
        str(body["provider_namespace"]),
        str(body["record_type"]),
        str(body["provider_record_id"]),
        str(body["acquisition_receipt_id"]),
        str(body["acquisition_generation"]),
        str(body["acquisition_slot"]),
    )
    body["idempotency_key"] = hashlib.sha256("\0".join(components).encode()).hexdigest()
    return body


def _app(claims: JWTClaims | None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)

    async def claims_override() -> JWTClaims | None:
        return claims

    app.dependency_overrides[get_jwt_claims] = claims_override
    return app


def _request(app: FastAPI, **kwargs: object) -> httpx.Response:
    async def execute() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post("/sourced-candidates/ingest", **kwargs)

    return asyncio.run(execute())


def test_exact_authority_and_mode_are_required_before_body_parsing() -> None:
    canary = "private.person@example.test"
    cases = [
        _claims(actor="other-service"),
        _claims(actor_type="user"),
        _claims(scopes=["kg:write"]),
        _claims(scopes=["candidate-source:write", "kg:write"]),
        _claims(tenant="platform"),
        _claims(issuer="vantahire"),
    ]
    for claims in cases:
        with (
            patch.object(auth, "JWT_ENABLED", True),
            patch.object(auth, "SIGNAL_JWT_ISSUER", "signal"),
            patch.dict("os.environ", {"SOURCED_CANDIDATE_INGEST_MODE": "dual"}),
        ):
            response = _request(
                _app(claims),
                content=canary + "{" * 300_000,
                headers={"content-type": "application/json"},
            )
        assert response.status_code == 403
        assert canary not in response.text

    with (
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "SIGNAL_JWT_ISSUER", "signal"),
        patch.dict("os.environ", {"SOURCED_CANDIDATE_INGEST_MODE": "off"}),
    ):
        response = _request(
            _app(_claims()),
            content=canary,
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 503
    assert response.json() == {"detail": "sourced_candidate_ingest_disabled"}
    assert canary not in response.text


def test_valid_request_reaches_only_the_closed_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[SourcedCandidateIngestRequest] = []

    def store(payload: SourcedCandidateIngestRequest, claims: JWTClaims) -> dict[str, object]:
        captured.append(payload)
        assert claims.actor_id == "signal-service"
        return {"delivery_status": "recorded"}

    monkeypatch.setattr("activekg.api.sourced_candidates._store", store)
    with (
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "SIGNAL_JWT_ISSUER", "signal"),
        patch.dict("os.environ", {"SOURCED_CANDIDATE_INGEST_MODE": "canonical_only"}),
    ):
        response = _request(_app(_claims()), json=_body())
    assert response.status_code == 200
    assert response.json() == {"delivery_status": "recorded"}
    assert len(captured) == 1
    assert compute_idempotency_key(captured[0]) == captured[0].idempotency_key


@pytest.mark.parametrize(
    ("change", "status"),
    [
        ({"provider_namespace": "other"}, 422),
        ({"adapter_version": "future"}, 422),
        ({"provider_record_id": "0"}, 422),
        ({"linkedin_url": "https://linkedin.com/company/example"}, 422),
        ({"unexpected": "field"}, 422),
    ],
)
def test_contract_refuses_unknown_provider_identity_and_fields(
    change: dict[str, object], status: int
) -> None:
    with (
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "SIGNAL_JWT_ISSUER", "signal"),
        patch.dict("os.environ", {"SOURCED_CANDIDATE_INGEST_MODE": "dual"}),
    ):
        response = _request(_app(_claims()), json=_body(**change))
    assert response.status_code == status


def test_private_fields_values_and_profile_bounds_are_refused() -> None:
    for profile in (
        {**_body()["normalized_profile"], "email": "private@example.test"},
        {**_body()["normalized_profile"], "headline": "private@example.test"},
        {**_body()["normalized_profile"], "skills": ["x"] * 257},
    ):
        with pytest.raises(ValueError):
            SourcedCandidateIngestRequest.model_validate_json(
                json.dumps(_body(normalized_profile=profile))
            )
    with pytest.raises(ValueError):
        _assert_no_private_evidence({"summary": "+1 (415) 555-0123"})
    with pytest.raises(ValueError):
        _assert_no_private_evidence({"headline": "Call 9876543210"})
    _assert_no_private_evidence(
        {"public_profiles": [{"profile_url": "https://linkedin.com/in/123456789"}]}
    )


def test_body_and_profile_size_caps_fail_closed() -> None:
    with (
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "SIGNAL_JWT_ISSUER", "signal"),
        patch.dict("os.environ", {"SOURCED_CANDIDATE_INGEST_MODE": "dual"}),
    ):
        response = _request(
            _app(_claims()),
            content=b"{" + b"x" * (256 * 1024),
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 413

    body = _body()
    profile = dict(body["normalized_profile"])
    profile["professional_summary"] = "x" * 8000
    body["normalized_profile"] = profile
    body["provider_observed_at"] = datetime.now(timezone.utc).isoformat()
    assert SourcedCandidateIngestRequest.model_validate_json(json.dumps(body))
