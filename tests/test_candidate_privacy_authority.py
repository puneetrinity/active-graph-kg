from __future__ import annotations

import asyncio
import base64
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from activekg.api import auth
from activekg.api.auth import JWTClaims
from activekg.api.candidate_privacy import (
    _require_service_claims,
    require_candidate_privacy_read,
    require_candidate_privacy_write,
)


def _env() -> dict[str, str]:
    return {
        "CANDIDATE_PRIVACY_HMAC_KEY_V1": base64.b64encode(b"k" * 32).decode(),
        "CANDIDATE_PRIVACY_HMAC_ACTIVE_VERSION": "1",
        "CANDIDATE_PRIVACY_INTAKE_ENABLED": "false",
        "CANDIDATE_PRIVACY_FLOW_ISSUER": "flow",
        "CANDIDATE_PRIVACY_FLOW_ACTOR_ID": "flow-service",
        "CANDIDATE_PRIVACY_SIGNAL_ISSUER": "signal",
        "CANDIDATE_PRIVACY_SIGNAL_ACTOR_ID": "signal-service",
    }


def _claims(issuer: str, actor: str, scope: str, *, actor_type: str = "service") -> JWTClaims:
    return JWTClaims(
        tenant_id="system",
        actor_id=actor,
        actor_type=actor_type,
        scopes=[scope],
        issuer=issuer,
    )


def test_write_is_flow_service_only_and_read_accepts_flow_or_signal() -> None:
    with (
        patch.dict("os.environ", _env(), clear=True),
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "JWT_ISSUER", "flow"),
        patch.object(auth, "SIGNAL_JWT_ISSUER", "signal"),
    ):
        flow_writer = _claims("flow", "flow-service", "candidate-privacy:write")
        assert _require_service_claims(flow_writer, write=True) is flow_writer
        for issuer, actor in (("flow", "flow-service"), ("signal", "signal-service")):
            reader = _claims(issuer, actor, "candidate-privacy:read")
            assert _require_service_claims(reader, write=False) is reader


@pytest.mark.parametrize(
    "claims",
    [
        _claims("signal", "signal-service", "candidate-privacy:write"),
        _claims("flow", "wrong", "candidate-privacy:write"),
        _claims("wrong", "flow-service", "candidate-privacy:write"),
        _claims("flow", "flow-service", "candidate-privacy:read"),
        _claims("flow", "flow-service", "candidate-privacy:write", actor_type="user"),
    ],
)
def test_wrong_authority_or_scope_is_denied(claims: JWTClaims) -> None:
    with (
        patch.dict("os.environ", _env(), clear=True),
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "JWT_ISSUER", "flow"),
        patch.object(auth, "SIGNAL_JWT_ISSUER", "signal"),
    ):
        with pytest.raises(HTTPException) as exc:
            _require_service_claims(claims, write=True)
    assert exc.value.status_code == 403
    assert exc.value.detail == "candidate_privacy_service_auth_denied"


def test_auth_disabled_fails_closed_for_both_dependencies() -> None:
    with (
        patch.dict("os.environ", _env(), clear=True),
        patch.object(auth, "JWT_ENABLED", False),
        patch.object(auth, "JWT_ISSUER", "flow"),
        patch.object(auth, "SIGNAL_JWT_ISSUER", "signal"),
    ):
        for dependency in (require_candidate_privacy_read, require_candidate_privacy_write):
            with pytest.raises(HTTPException) as exc:
                asyncio.run(dependency(None))
            assert exc.value.status_code == 401
            assert exc.value.detail == "candidate_privacy_service_auth_required"


def test_authority_issuers_cannot_be_swapped_away_from_verified_jwt_roles() -> None:
    swapped = {
        **_env(),
        "CANDIDATE_PRIVACY_FLOW_ISSUER": "signal",
        "CANDIDATE_PRIVACY_SIGNAL_ISSUER": "flow",
    }
    with (
        patch.dict("os.environ", swapped, clear=True),
        patch.object(auth, "JWT_ENABLED", True),
        patch.object(auth, "JWT_ISSUER", "flow"),
        patch.object(auth, "SIGNAL_JWT_ISSUER", "signal"),
    ):
        with pytest.raises(HTTPException) as exc:
            _require_service_claims(
                _claims("signal", "signal-service", "candidate-privacy:write"),
                write=True,
            )
    assert exc.value.status_code == 503
    assert exc.value.detail == "candidate_privacy_configuration_invalid"
