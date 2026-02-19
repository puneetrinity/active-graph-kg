"""Unit tests for JWT scope normalization and require_scope enforcement.

These tests exercise auth.py directly (no running server needed).

Run with:
    pytest tests/test_jwt_scopes.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt as pyjwt
import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Shared HS256 key for test JWTs
# ---------------------------------------------------------------------------
TEST_SECRET = "test-secret-key-minimum-32-chars!!"
TEST_ALGORITHM = "HS256"
TEST_AUDIENCE = "activekg"
TEST_ISSUER = "vantahire"


def _make_token(
    *,
    tenant_id: str = "t_test",
    sub: str = "vantahire-backend",
    scopes=None,
    scope=None,
    aud: str = TEST_AUDIENCE,
    iss: str = TEST_ISSUER,
    exp_delta: timedelta | None = None,
) -> str:
    """Helper to mint an HS256 test JWT."""
    now = datetime.now(timezone.utc)
    payload: dict = {
        "sub": sub,
        "tenant_id": tenant_id,
        "actor_type": "service",
        "aud": aud,
        "iss": iss,
        "iat": now,
        "nbf": now,
        "exp": now + (exp_delta or timedelta(hours=1)),
    }
    if scopes is not None:
        payload["scopes"] = scopes
    if scope is not None:
        payload["scope"] = scope
    return pyjwt.encode(payload, TEST_SECRET, algorithm=TEST_ALGORITHM)


# ---------------------------------------------------------------------------
# Patch env before importing auth module
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _patch_auth_env(monkeypatch):
    """Configure auth module globals for HS256 test mode."""
    monkeypatch.setenv("JWT_SECRET_KEY", TEST_SECRET)
    monkeypatch.setenv("JWT_ALGORITHM", TEST_ALGORITHM)
    monkeypatch.setenv("JWT_AUDIENCE", TEST_AUDIENCE)
    monkeypatch.setenv("JWT_ISSUER", TEST_ISSUER)
    monkeypatch.setenv("JWT_ENABLED", "true")

    import activekg.api.auth as auth_mod

    monkeypatch.setattr(auth_mod, "JWT_SECRET_KEY", TEST_SECRET)
    monkeypatch.setattr(auth_mod, "JWT_PUBLIC_KEY", None)
    monkeypatch.setattr(auth_mod, "JWT_ALGORITHM", TEST_ALGORITHM)
    monkeypatch.setattr(auth_mod, "JWT_AUDIENCE", TEST_AUDIENCE)
    monkeypatch.setattr(auth_mod, "JWT_ISSUER", TEST_ISSUER)
    monkeypatch.setattr(auth_mod, "JWT_ENABLED", True)
    monkeypatch.setattr(auth_mod, "JWT_LEEWAY_SECONDS", 30)


# ===================================================================
# 1. Scope normalization tests (verify_jwt)
# ===================================================================


class TestScopeNormalization:
    """verify_jwt must always return JWTClaims.scopes as list[str]."""

    def test_scopes_as_list(self):
        from activekg.api.auth import verify_jwt

        token = _make_token(scopes=["search:read", "kg:write"])
        claims = verify_jwt(token)
        assert claims.scopes == ["search:read", "kg:write"]

    def test_scopes_as_space_delimited_string(self):
        from activekg.api.auth import verify_jwt

        token = _make_token(scopes="search:read kg:write")
        claims = verify_jwt(token)
        assert claims.scopes == ["search:read", "kg:write"]

    def test_scope_fallback_field(self):
        """'scope' (singular, OAuth2 style) should be used when 'scopes' absent."""
        from activekg.api.auth import verify_jwt

        token = _make_token(scope="ask:read kg:write")
        claims = verify_jwt(token)
        assert claims.scopes == ["ask:read", "kg:write"]

    def test_scopes_takes_priority_over_scope(self):
        """When both 'scopes' and 'scope' are present, 'scopes' wins."""
        from activekg.api.auth import verify_jwt

        # Build token with both fields
        now = datetime.now(timezone.utc)
        payload = {
            "sub": "svc",
            "tenant_id": "t1",
            "aud": TEST_AUDIENCE,
            "iss": TEST_ISSUER,
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(hours=1),
            "scopes": ["search:read"],
            "scope": "kg:write admin:refresh",
        }
        token = pyjwt.encode(payload, TEST_SECRET, algorithm=TEST_ALGORITHM)
        claims = verify_jwt(token)
        assert claims.scopes == ["search:read"]

    def test_scopes_list_filters_non_strings(self):
        from activekg.api.auth import verify_jwt

        now = datetime.now(timezone.utc)
        payload = {
            "sub": "svc",
            "tenant_id": "t1",
            "aud": TEST_AUDIENCE,
            "iss": TEST_ISSUER,
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(hours=1),
            "scopes": ["search:read", 42, None, "kg:write", True],
        }
        token = pyjwt.encode(payload, TEST_SECRET, algorithm=TEST_ALGORITHM)
        claims = verify_jwt(token)
        assert claims.scopes == ["search:read", "kg:write"]

    def test_invalid_scopes_type_yields_empty(self):
        from activekg.api.auth import verify_jwt

        now = datetime.now(timezone.utc)
        payload = {
            "sub": "svc",
            "tenant_id": "t1",
            "aud": TEST_AUDIENCE,
            "iss": TEST_ISSUER,
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(hours=1),
            "scopes": 12345,
        }
        token = pyjwt.encode(payload, TEST_SECRET, algorithm=TEST_ALGORITHM)
        claims = verify_jwt(token)
        assert claims.scopes == []

    def test_empty_scopes_yields_empty_list(self):
        from activekg.api.auth import verify_jwt

        token = _make_token(scopes=[])
        claims = verify_jwt(token)
        assert claims.scopes == []

    def test_no_scopes_at_all_yields_empty(self):
        from activekg.api.auth import verify_jwt

        token = _make_token()  # no scopes/scope field
        claims = verify_jwt(token)
        assert claims.scopes == []


# ===================================================================
# 2. require_scope enforcement via test app
# ===================================================================


@pytest.fixture()
def scope_app():
    """Build a small FastAPI app wired with require_scope for testing."""
    from activekg.api.auth import get_jwt_claims, require_scope

    app = FastAPI()

    @app.post("/search", dependencies=[Depends(require_scope("search:read"))])
    async def fake_search(claims=Depends(get_jwt_claims)):  # noqa: B008
        return {"ok": True, "tenant": claims.tenant_id if claims else None}

    @app.post("/ask", dependencies=[Depends(require_scope("ask:read"))])
    async def fake_ask(claims=Depends(get_jwt_claims)):  # noqa: B008
        return {"ok": True}

    @app.post("/nodes", dependencies=[Depends(require_scope("kg:write"))])
    async def fake_nodes(claims=Depends(get_jwt_claims)):  # noqa: B008
        return {"ok": True}

    return TestClient(app)


class TestRequireScope:
    def test_correct_scope_allowed(self, scope_app):
        token = _make_token(scopes=["search:read"])
        r = scope_app.post("/search", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_missing_scope_returns_403(self, scope_app):
        token = _make_token(scopes=["kg:write"])
        r = scope_app.post("/search", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 403
        assert "search:read" in r.json()["detail"]

    def test_ask_scope_enforced(self, scope_app):
        token = _make_token(scopes=["search:read"])
        r = scope_app.post("/ask", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 403
        assert "ask:read" in r.json()["detail"]

    def test_write_scope_enforced(self, scope_app):
        token = _make_token(scopes=["search:read", "ask:read"])
        r = scope_app.post("/nodes", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 403
        assert "kg:write" in r.json()["detail"]

    def test_write_scope_accepted(self, scope_app):
        token = _make_token(scopes=["kg:write"])
        r = scope_app.post("/nodes", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_space_delimited_scopes_work(self, scope_app):
        token = _make_token(scopes="search:read ask:read kg:write")
        r = scope_app.post("/search", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        r2 = scope_app.post("/ask", headers={"Authorization": f"Bearer {token}"})
        assert r2.status_code == 200
        r3 = scope_app.post("/nodes", headers={"Authorization": f"Bearer {token}"})
        assert r3.status_code == 200

    def test_scope_fallback_works_for_enforcement(self, scope_app):
        token = _make_token(scope="search:read")
        r = scope_app.post("/search", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_no_token_returns_401(self, scope_app):
        r = scope_app.post("/search")
        assert r.status_code == 401


# ===================================================================
# 3. Issuer/audience validation
# ===================================================================


class TestIssuerAudienceValidation:
    def test_vantahire_issuer_accepted(self):
        from activekg.api.auth import verify_jwt

        token = _make_token(iss="vantahire", aud="activekg")
        claims = verify_jwt(token)
        assert claims.tenant_id == "t_test"

    def test_wrong_issuer_rejected(self):
        from activekg.api.auth import verify_jwt

        token = _make_token(iss="evil-corp")
        with pytest.raises(HTTPException) as exc_info:
            verify_jwt(token)
        assert exc_info.value.status_code == 401
        assert "issuer" in exc_info.value.detail.lower()

    def test_wrong_audience_rejected(self):
        from activekg.api.auth import verify_jwt

        token = _make_token(aud="wrong-audience")
        with pytest.raises(HTTPException) as exc_info:
            verify_jwt(token)
        assert exc_info.value.status_code == 401
        assert "audience" in exc_info.value.detail.lower()

    def test_expired_token_rejected(self):
        from activekg.api.auth import verify_jwt

        token = _make_token(exp_delta=timedelta(hours=-1))
        with pytest.raises(HTTPException) as exc_info:
            verify_jwt(token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_missing_tenant_id_rejected(self):
        from activekg.api.auth import verify_jwt

        now = datetime.now(timezone.utc)
        payload = {
            "sub": "svc",
            "aud": TEST_AUDIENCE,
            "iss": TEST_ISSUER,
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(hours=1),
        }
        token = pyjwt.encode(payload, TEST_SECRET, algorithm=TEST_ALGORITHM)
        with pytest.raises(HTTPException) as exc_info:
            verify_jwt(token)
        assert exc_info.value.status_code == 401
        assert "tenant_id" in exc_info.value.detail

    def test_missing_sub_rejected(self):
        from activekg.api.auth import verify_jwt

        now = datetime.now(timezone.utc)
        payload = {
            "tenant_id": "t1",
            "aud": TEST_AUDIENCE,
            "iss": TEST_ISSUER,
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(hours=1),
        }
        token = pyjwt.encode(payload, TEST_SECRET, algorithm=TEST_ALGORITHM)
        with pytest.raises(HTTPException) as exc_info:
            verify_jwt(token)
        assert exc_info.value.status_code == 401
        assert "sub" in exc_info.value.detail.lower() or "actor_id" in exc_info.value.detail.lower()
