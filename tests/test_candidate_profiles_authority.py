import time

import pytest
from fastapi import HTTPException

from activekg.api import auth
from activekg.api.auth import JWTClaims
from activekg.candidate_profiles.authority import require_authority

SUBJECT = "22222222-2222-4222-8222-222222222222"
SCOPE = "candidate-profile:foundation-write"


@pytest.fixture
def claims(monkeypatch):
    monkeypatch.setattr(auth, "JWT_ENABLED", True)
    monkeypatch.setattr(auth, "JWT_ALGORITHM", "RS256")
    monkeypatch.setattr(auth, "JWT_AUDIENCE", "activekg")
    now = int(time.time())
    return JWTClaims(
        f"candidate_{SUBJECT}",
        "vantahire-backend",
        "service",
        [SCOPE],
        exp=now + 60,
        issuer="vantahire",
        iat=now,
    )


def test_exact_service_authority(claims):
    assert require_authority(claims, SCOPE) == SUBJECT


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("issuer", "signal"),
        ("actor_id", "other"),
        ("actor_type", "user"),
        ("tenant_id", "org_32"),
        ("scopes", [SCOPE, "*"]),
        ("scopes", ["candidate-consent:write"]),
        ("iat", None),
        ("iat", True),
        ("exp", None),
        ("exp", True),
    ],
)
def test_wrong_authority(claims, field, value):
    setattr(claims, field, value)
    with pytest.raises(HTTPException) as error:
        require_authority(claims, SCOPE)
    assert error.value.status_code == 403


def test_lifetime_and_clock(claims):
    claims.exp = claims.iat + 61
    with pytest.raises(HTTPException):
        require_authority(claims, SCOPE)
    claims.iat += 100
    claims.exp = claims.iat + 60
    with pytest.raises(HTTPException):
        require_authority(claims, SCOPE)


def test_disabled_auth_is_not_a_bypass(claims, monkeypatch):
    monkeypatch.setattr(auth, "JWT_ENABLED", False)
    with pytest.raises(HTTPException) as error:
        require_authority(claims, SCOPE)
    assert error.value.status_code == 401


def test_legacy_claim_constructor_is_unchanged():
    assert JWTClaims("old", "old").iat is None
