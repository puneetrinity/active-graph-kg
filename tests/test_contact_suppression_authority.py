"""Authority contracts for platform-wide contact suppression."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from activekg.api import global_memory

SCOPE = global_memory.CONTACT_SUPPRESSION_SCOPE


def _claims(**overrides):
    base = {
        "tenant_id": "t_flow",
        "actor_id": global_memory.CONTACT_SUPPRESSION_ACTOR_ID,
        "actor_type": "service",
        "scopes": ["contact:write", SCOPE],
        "issuer": global_memory.CONTACT_SUPPRESSION_ISSUER,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_flow_service_identity_is_allowed():
    global_memory._require_contact_suppression_authority(_claims())


def test_contact_write_alone_is_not_enough():
    with pytest.raises(HTTPException) as exc:
        global_memory._require_contact_suppression_authority(_claims(scopes=["contact:write"]))
    assert exc.value.status_code == 403
    assert SCOPE in exc.value.detail


@pytest.mark.parametrize("issuer", ["signal", "attacker", None])
def test_untrusted_issuer_is_rejected_even_with_scope(issuer):
    with pytest.raises(HTTPException) as exc:
        global_memory._require_contact_suppression_authority(_claims(issuer=issuer))
    assert exc.value.status_code == 403


def test_user_tokens_are_rejected():
    with pytest.raises(HTTPException) as exc:
        global_memory._require_contact_suppression_authority(_claims(actor_type="user"))
    assert exc.value.status_code == 403


@pytest.mark.parametrize("actor_id", ["flow-backend", "attacker", None])
def test_untrusted_service_subject_is_rejected(actor_id):
    with pytest.raises(HTTPException) as exc:
        global_memory._require_contact_suppression_authority(_claims(actor_id=actor_id))
    assert exc.value.status_code == 403


def test_provider_event_id_is_not_authorization():
    """A fabricated event hash cannot replace the dedicated authority."""
    with pytest.raises(HTTPException):
        global_memory._require_contact_suppression_authority(_claims(scopes=["contact:write"]))


@pytest.mark.parametrize("status", ["hard_bounce", "complaint"])
def test_provider_evidence_cannot_write_terminal_suppression(status):
    with pytest.raises(ValidationError):
        global_memory.ContactEvidenceRecord(
            global_candidate_id="00000000-0000-0000-0000-000000000001",
            email="person@example.com",
            provider="fullenrich",
            status=status,
        )


@pytest.mark.parametrize("event_id", ["brevo-event", "A" * 64, "0" * 63])
def test_suppression_requires_lowercase_sha256_event_id(event_id):
    with pytest.raises(ValidationError):
        global_memory.ContactSuppressionRecord(
            email_hash="1" * 64,
            reason="hard_bounce",
            provider_event_id=event_id,
        )


@pytest.mark.parametrize("email_hash", ["person@example.com", "A" * 64, "0" * 63])
def test_suppression_accepts_only_lowercase_sha256_email_hash(email_hash):
    with pytest.raises(ValidationError):
        global_memory.ContactSuppressionRecord(
            email_hash=email_hash,
            reason="hard_bounce",
            provider_event_id="0" * 64,
        )


def test_suppression_payload_does_not_accept_a_raw_email():
    with pytest.raises(ValidationError):
        global_memory.ContactSuppressionRecord(
            email="person@example.com",
            email_hash="1" * 64,
            reason="hard_bounce",
            provider_event_id="0" * 64,
        )
