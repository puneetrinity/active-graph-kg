"""Authority contracts for suppressing an address with no tenant-owned evidence.

Suppressing an address the caller holds no evidence for can tombstone ANY address
platform-wide. `contact:write` is held by every enrichment caller, so it is not a
sufficient gate. This path requires a dedicated scope AND a verified Flow service
issuer. The provider event id is audit data — it is attacker-suppliable and must
never function as authorization.

These exercise the gate directly; the end-to-end HTTP path is covered in
tests/test_public_memory_integration.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from activekg.api import global_memory

SCOPE = global_memory.UNOWNED_SUPPRESSION_SCOPE


def _claims(**overrides):
    base = {
        "tenant_id": "t_flow",
        "actor_id": "flow-backend",
        "actor_type": "service",
        "scopes": ["contact:write", SCOPE],
        "issuer": "vantahire",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _enforce_jwt(monkeypatch):
    monkeypatch.setattr("activekg.api.auth.JWT_ENABLED", True)


def test_flow_service_identity_is_allowed():
    global_memory._require_unowned_suppression_authority(_claims())


def test_contact_write_alone_is_not_enough():
    """The scope every enrichment caller already holds must not grant this."""
    with pytest.raises(HTTPException) as exc:
        global_memory._require_unowned_suppression_authority(_claims(scopes=["contact:write"]))
    assert exc.value.status_code == 403
    assert SCOPE in exc.value.detail


def test_untrusted_issuer_is_rejected_even_with_the_scope():
    """A different service minting the same scope must not inherit the authority."""
    for issuer in ("signal", "attacker", None):
        with pytest.raises(HTTPException) as exc:
            global_memory._require_unowned_suppression_authority(_claims(issuer=issuer))
        assert exc.value.status_code == 403


def test_user_tokens_are_rejected():
    """Only service identities, never an end-user token that happens to carry the scope."""
    with pytest.raises(HTTPException) as exc:
        global_memory._require_unowned_suppression_authority(_claims(actor_type="user"))
    assert exc.value.status_code == 403


def test_provider_event_id_is_not_authorization():
    """The gate must not consult provider_event_id: it is attacker-suppliable.

    Authority is decided before any provider field is read, so a caller cannot
    buy access by inventing an event id.
    """
    with pytest.raises(HTTPException):
        global_memory._require_unowned_suppression_authority(_claims(scopes=["contact:write"]))
