from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from activekg.api import auth
from activekg.api import organization_decision_events as receiver
from activekg.api.auth import JWTClaims, get_jwt_claims


def _payload(**changes):
    value = {
        "event_id": "11111111-1111-4111-8111-111111111111",
        "delivery_sequence": 10,
        "source_event_sequence": 4,
        "organization_id": 73,
        "payload_schema_version": 1,
        "source_system": "flow",
        "subject_type": "application",
        "subject_id": 901,
        "job_id": 44,
        "action_code": "application_stage_moved",
        "taxonomy_version": 1,
        "rubric_id": None,
        "rubric_version": None,
        "rubric_approval_mode": None,
        "jd_digest_version": 2,
        "recommendation_action": "advance",
        "reason_code": None,
        "before_state": {"stage_id": None},
        "after_state": {"stage_id": 8},
        "occurred_at": "2026-09-03T12:00:00Z",
    }
    value.update(changes)
    return value


def _claims(**changes) -> JWTClaims:
    values = {
        "tenant_id": "org_73",
        "actor_id": "flow-service",
        "actor_type": "service",
        "scopes": ["decision-history:write"],
        "issuer": "flow",
    }
    values.update(changes)
    return JWTClaims(**values)


def _app(claims: JWTClaims) -> FastAPI:
    app = FastAPI()
    app.include_router(receiver.router)

    async def override() -> JWTClaims:
        return claims

    app.dependency_overrides[get_jwt_claims] = override
    return app


def _request(app: FastAPI, **kwargs) -> httpx.Response:
    async def execute() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post("/organization-decision-events/ingest", **kwargs)

    return asyncio.run(execute())


def test_strict_wire_model_and_rubric_reference_contract() -> None:
    parsed = receiver.OrganizationDecisionEvent.model_validate_json(json.dumps(_payload()))
    assert parsed.event_id == UUID(_payload()["event_id"])
    assert parsed.occurred_at == datetime(2026, 9, 3, 12, tzinfo=timezone.utc)

    for bad in (
        _payload(email="candidate@example.test"),
        _payload(before_state={"stage_id": None, "name": "private"}),
        _payload(after_state={"stage_id": None}),
        _payload(before_state={"stage_id": 8}),
        _payload(rubric_id="22222222-2222-4222-8222-222222222222"),
    ):
        with pytest.raises(ValidationError):
            receiver.OrganizationDecisionEvent.model_validate_json(json.dumps(bad))


@pytest.mark.parametrize(
    "claims",
    [
        _claims(issuer="other"),
        _claims(actor_type="user"),
        _claims(actor_id="other"),
        _claims(scopes=["decision-history:read"]),
    ],
)
def test_writer_authority_is_exact(claims: JWTClaims) -> None:
    with (
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "JWT_ISSUER", "flow"),
        patch.dict("os.environ", {"ORG_DECISION_INBOX_FLOW_ACTOR_ID": "flow-service"}),
        pytest.raises(HTTPException) as exc,
    ):
        asyncio.run(receiver.require_decision_history_writer(claims))
    assert exc.value.status_code == 403
    assert exc.value.detail == "decision_inbox_service_auth_denied"


def test_disabled_or_missing_auth_fails_before_store() -> None:
    with patch.object(auth, "JWT_ENABLED", False), pytest.raises(HTTPException) as exc:
        asyncio.run(receiver.require_decision_history_writer(None))
    assert exc.value.status_code == 401

    with (
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "JWT_ISSUER", "flow"),
        patch.dict(
            "os.environ",
            {
                "ORG_DECISION_INBOX_ENABLED": "false",
                "ORG_DECISION_INBOX_FLOW_ACTOR_ID": "flow-service",
            },
            clear=False,
        ),
        patch.object(receiver, "_store") as store,
    ):
        response = _request(_app(_claims()), json=_payload())
    assert response.status_code == 503
    assert response.json()["detail"] == "decision_inbox_disabled"
    store.assert_not_called()


def test_enabled_http_contract_tenant_and_body_limits() -> None:
    environment = {
        "ORG_DECISION_INBOX_ENABLED": "true",
        "ORG_DECISION_INBOX_FLOW_ACTOR_ID": "flow-service",
    }
    with (
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "JWT_ISSUER", "flow"),
        patch.dict("os.environ", environment, clear=False),
        patch.object(receiver, "_store", return_value="inserted") as store,
    ):
        app = _app(_claims())
        response = _request(app, json=_payload())
        assert response.status_code == 200
        assert response.json() == {
            "event_id": _payload()["event_id"],
            "delivery_sequence": 10,
            "status": "inserted",
        }
        store.assert_called_once()

        wrong = _app(_claims(tenant_id="org_74"))
        assert _request(wrong, json=_payload()).status_code == 403
        unknown = _request(app, json=_payload(candidate_name="private"))
        assert unknown.status_code == 422
        oversized = _request(
            app,
            content=b"{" + b"x" * (64 * 1024) + b"}",
            headers={"content-type": "application/json"},
        )
        assert oversized.status_code == 413


def test_store_rolls_back_and_closes_on_database_failure() -> None:
    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, *_args):
            raise RuntimeError("classified only")

    class Connection:
        def __init__(self):
            self.rolled_back = False
            self.closed = False

        def cursor(self):
            return Cursor()

        def rollback(self):
            self.rolled_back = True

        def close(self):
            self.closed = True

    connection = Connection()
    payload = receiver.OrganizationDecisionEvent.model_validate_json(json.dumps(_payload()))
    with (
        patch.object(receiver, "_connect", return_value=connection),
        pytest.raises(HTTPException) as exc,
    ):
        receiver._store(payload, "org_73")
    assert exc.value.status_code == 503
    assert connection.rolled_back is True
    assert connection.closed is True
