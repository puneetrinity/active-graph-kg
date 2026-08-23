from __future__ import annotations

import base64
import hashlib
from unittest.mock import patch

import pytest

from activekg.privacy.config import (
    CandidatePrivacyConfigurationError,
    load_candidate_privacy_config,
)
from activekg.privacy.identity import (
    CandidatePrivacyIdentityError,
    identity_token,
    normalize_privacy_identifier,
)
from activekg.privacy.models import (
    CandidatePrivacyAction,
    CandidatePrivacyDecision,
    CandidatePrivacyState,
    decision_for,
)


def test_normalization_and_hmac_vectors_are_stable() -> None:
    key = bytes(range(32))
    vectors = {
        ("email", " Test.User+tag@Example.COM "): (
            "test.user+tag@example.com",
            "6ecca1557b186c1a654d826ba138e67ce29c4f96513daf089eb2e613211591dc",
        ),
        ("phone", " +1 (415) 555-0123 "): (
            "+14155550123",
            "6f5c4f50f74efa1b75edee1b010f48de8b9bb4502f31c5eac1e81c29819cfdb0",
        ),
        ("linkedin_url", "https://www.linkedin.com/in/Test-User/?trk=abc"): (
            "https://linkedin.com/in/test-user",
            "f83d0bb744d05fa0795259b2227e1ef8b830932246dba9fa7a0516d81ca4cf00",
        ),
        ("github_url", "https://github.com/TestUser/"): (
            "https://github.com/testuser",
            "394ae563a821d58bb7383e4b3baaed3e49467143b9149dd618940e177342cbf1",
        ),
    }
    for (identifier_type, raw), (normalized, token) in vectors.items():
        identifier = normalize_privacy_identifier(identifier_type, raw)
        assert identifier.normalized == normalized
        assert identity_token(key, identifier).hex() == token


def test_email_lookup_digests_cover_canonical_and_legacy_applicant_hashes() -> None:
    identifier = normalize_privacy_identifier("email", " First.Last@GoogleMail.com ")
    assert identifier.normalized == "firstlast@gmail.com"
    assert identifier.lookup_digests == (
        hashlib.sha256(b"firstlast@gmail.com").digest(),
        hashlib.sha256(b"first.last@googlemail.com").digest(),
    )
    ordinary = normalize_privacy_identifier("email", "person@example.test")
    assert ordinary.lookup_digests == (hashlib.sha256(b"person@example.test").digest(),)


@pytest.mark.parametrize(
    ("identifier_type", "value"),
    [
        ("other", "opaque"),
        ("email", "not-an-email"),
        ("phone", "123"),
        ("linkedin_url", "https://linkedin.com/company/ealana"),
        ("github_url", "https://example.com/person"),
    ],
)
def test_privacy_identifier_allowlist_and_validation_are_strict(
    identifier_type: str, value: str
) -> None:
    with pytest.raises(CandidatePrivacyIdentityError, match="identifier"):
        normalize_privacy_identifier(identifier_type, value)


def test_api_key_ring_is_dual_read_single_write() -> None:
    older = base64.b64encode(b"o" * 32).decode()
    active = base64.b64encode(b"n" * 32).decode()
    env = {
        "CANDIDATE_PRIVACY_HMAC_KEY_V1": older,
        "CANDIDATE_PRIVACY_HMAC_KEY_V2": active,
        "CANDIDATE_PRIVACY_HMAC_ACTIVE_VERSION": "2",
        "CANDIDATE_PRIVACY_INTAKE_ENABLED": "false",
        "CANDIDATE_PRIVACY_FLOW_ISSUER": "flow",
        "CANDIDATE_PRIVACY_FLOW_ACTOR_ID": "flow-service",
        "CANDIDATE_PRIVACY_SIGNAL_ISSUER": "signal",
        "CANDIDATE_PRIVACY_SIGNAL_ACTOR_ID": "signal-service",
    }
    with patch.dict("os.environ", env, clear=True):
        config = load_candidate_privacy_config(require_hmac=True)
    assert config.active_key_version == 2
    assert config.keys == {1: b"o" * 32, 2: b"n" * 32}
    assert config.intake_enabled is False


def test_key_ring_rejects_missing_old_key_or_worker_key_leak() -> None:
    encoded = base64.b64encode(b"k" * 32).decode()
    with (
        patch.dict(
            "os.environ",
            {
                "CANDIDATE_PRIVACY_HMAC_KEY_V2": encoded,
                "CANDIDATE_PRIVACY_HMAC_ACTIVE_VERSION": "1",
            },
            clear=True,
        ),
        pytest.raises(CandidatePrivacyConfigurationError),
    ):
        load_candidate_privacy_config(require_hmac=True)
    with (
        patch.dict(
            "os.environ",
            {
                "CANDIDATE_PRIVACY_HMAC_KEY_V1": encoded,
                "CANDIDATE_PRIVACY_HMAC_ACTIVE_VERSION": "1",
            },
            clear=True,
        ),
        pytest.raises(CandidatePrivacyConfigurationError, match="worker"),
    ):
        load_candidate_privacy_config(require_hmac=False)


def test_decision_semantics_are_conservative_and_reversible() -> None:
    assert (
        decision_for(
            CandidatePrivacyState.ACTIVE_QUARANTINE,
            CandidatePrivacyAction.WITHDRAW_GLOBAL_MATCHING,
        )
        is CandidatePrivacyDecision.BLOCK_GLOBAL
    )
    assert (
        decision_for(
            CandidatePrivacyState.ACTIVE_QUARANTINE,
            CandidatePrivacyAction.REQUEST_ERASURE,
        )
        is CandidatePrivacyDecision.BLOCK_ALL
    )
    assert (
        decision_for(
            CandidatePrivacyState.NEEDS_REVIEW,
            CandidatePrivacyAction.WITHDRAW_GLOBAL_MATCHING,
        )
        is CandidatePrivacyDecision.REVIEW
    )
    assert (
        decision_for(
            CandidatePrivacyState.HARD_PURGE_ELIGIBLE,
            CandidatePrivacyAction.REQUEST_ERASURE,
        )
        is CandidatePrivacyDecision.REVIEW
    )
    assert (
        decision_for(
            CandidatePrivacyState.RELEASED,
            CandidatePrivacyAction.REQUEST_ERASURE,
        )
        is CandidatePrivacyDecision.ALLOW
    )
