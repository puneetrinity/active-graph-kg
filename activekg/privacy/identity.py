"""Strict normalization and non-reversible privacy identity tokens."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from activekg.graph.candidate_identifiers import normalize_identifier

PRIVACY_IDENTIFIER_TYPES = frozenset(
    {
        "email",
        "phone",
        "linkedin_url",
        "github_url",
        "signal_candidate_id",
        "vantahire_application_id",
        "vantahire_resume_id",
    }
)
_DOMAIN = b"ealana:candidate-privacy:v1"


class CandidatePrivacyIdentityError(ValueError):
    """An identifier cannot safely participate in privacy matching."""


@dataclass(frozen=True)
class NormalizedIdentifier:
    identifier_type: str
    normalized: str
    lookup_alias_digests: tuple[bytes, ...] = ()

    @property
    def lookup_digest(self) -> bytes:
        return hashlib.sha256(self.normalized.encode("utf-8")).digest()

    @property
    def lookup_digests(self) -> tuple[bytes, ...]:
        """Transient canonical plus legacy lookup aliases, without identity text."""
        canonical = self.lookup_digest
        return tuple(dict.fromkeys((canonical, *self.lookup_alias_digests)))


def normalize_privacy_identifier(identifier_type: str, value: str) -> NormalizedIdentifier:
    if identifier_type not in PRIVACY_IDENTIFIER_TYPES:
        raise CandidatePrivacyIdentityError("identifier type is not permitted")
    try:
        normalized = normalize_identifier(identifier_type, value)
    except Exception as exc:
        raise CandidatePrivacyIdentityError("identifier is invalid") from exc
    aliases: tuple[bytes, ...] = ()
    if identifier_type == "email":
        # Applicant projection historically hashed lower(trim(raw_email)) while
        # the canonical identifier normalizer folds Gmail dots/googlemail.  A
        # transient second digest lets the resolver associate those existing
        # rows without storing raw identity or weakening the HMAC receipt.
        legacy = value.strip().lower()
        if legacy.startswith("mailto:"):
            legacy = legacy[len("mailto:") :]
        legacy_digest = hashlib.sha256(legacy.encode("utf-8")).digest()
        if legacy_digest != hashlib.sha256(normalized.encode("utf-8")).digest():
            aliases = (legacy_digest,)
    return NormalizedIdentifier(
        identifier_type=identifier_type,
        normalized=normalized,
        lookup_alias_digests=aliases,
    )


def identity_token(key: bytes, identifier: NormalizedIdentifier) -> bytes:
    message = (
        _DOMAIN + identifier.identifier_type.encode("utf-8") + identifier.normalized.encode("utf-8")
    )
    return hmac.new(key, message, hashlib.sha256).digest()
