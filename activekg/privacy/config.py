"""Strict, secret-safe candidate privacy configuration."""

from __future__ import annotations

import base64
import binascii
import os
import re
from dataclasses import dataclass

_KEY_NAME = re.compile(r"^CANDIDATE_PRIVACY_HMAC_KEY_V([1-9][0-9]*)$")


class CandidatePrivacyConfigurationError(RuntimeError):
    """Privacy configuration is absent, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class CandidatePrivacyConfig:
    active_key_version: int | None
    keys: dict[int, bytes]
    intake_enabled: bool
    flow_issuer: str
    flow_actor_id: str
    signal_issuer: str
    signal_actor_id: str

    def require_hmac(self) -> tuple[int, dict[int, bytes]]:
        if self.active_key_version is None or not self.keys:
            raise CandidatePrivacyConfigurationError("candidate privacy HMAC ring is unavailable")
        return self.active_key_version, self.keys


def _strict_bool(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default).strip().lower()
    if value not in {"true", "false"}:
        raise CandidatePrivacyConfigurationError(f"{name} must be true or false")
    return value == "true"


def _decode_key(value: str) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CandidatePrivacyConfigurationError("candidate privacy HMAC key is malformed") from exc
    if len(decoded) < 32:
        raise CandidatePrivacyConfigurationError("candidate privacy HMAC key is too short")
    return decoded


def load_candidate_privacy_config(*, require_hmac: bool) -> CandidatePrivacyConfig:
    keys: dict[int, bytes] = {}
    for name, value in os.environ.items():
        match = _KEY_NAME.fullmatch(name)
        if match:
            keys[int(match.group(1))] = _decode_key(value)

    raw_active = os.getenv("CANDIDATE_PRIVACY_HMAC_ACTIVE_VERSION", "").strip()
    active: int | None = None
    if raw_active:
        try:
            active = int(raw_active)
        except ValueError as exc:
            raise CandidatePrivacyConfigurationError(
                "candidate privacy active key version is malformed"
            ) from exc
        if active <= 0:
            raise CandidatePrivacyConfigurationError(
                "candidate privacy active key version is malformed"
            )

    if require_hmac:
        if active is None or active not in keys:
            raise CandidatePrivacyConfigurationError("candidate privacy active HMAC key is absent")
        if active != max(keys):
            raise CandidatePrivacyConfigurationError(
                "candidate privacy active HMAC version is not the highest configured version"
            )
    elif active is not None or keys:
        raise CandidatePrivacyConfigurationError(
            "candidate privacy worker must not receive the HMAC key ring"
        )

    return CandidatePrivacyConfig(
        active_key_version=active,
        keys=keys,
        intake_enabled=_strict_bool("CANDIDATE_PRIVACY_INTAKE_ENABLED"),
        flow_issuer=os.getenv("CANDIDATE_PRIVACY_FLOW_ISSUER", "").strip(),
        flow_actor_id=os.getenv("CANDIDATE_PRIVACY_FLOW_ACTOR_ID", "").strip(),
        signal_issuer=os.getenv("CANDIDATE_PRIVACY_SIGNAL_ISSUER", "").strip(),
        signal_actor_id=os.getenv("CANDIDATE_PRIVACY_SIGNAL_ACTOR_ID", "").strip(),
    )


def candidate_privacy_configuration_problems(
    *,
    require_hmac: bool,
    trusted_flow_issuer: str | None = None,
    trusted_signal_issuer: str | None = None,
) -> list[str]:
    try:
        config = load_candidate_privacy_config(require_hmac=require_hmac)
    except CandidatePrivacyConfigurationError:
        return ["candidate_privacy_configuration_invalid"]
    problems: list[str] = []
    if not config.flow_issuer or not config.flow_actor_id:
        problems.append("candidate_privacy_flow_authority_missing")
    if not config.signal_issuer or not config.signal_actor_id:
        problems.append("candidate_privacy_signal_authority_missing")
    if config.flow_issuer == config.signal_issuer:
        problems.append("candidate_privacy_issuers_not_distinct")
    if trusted_flow_issuer is not None and config.flow_issuer != trusted_flow_issuer:
        problems.append("candidate_privacy_flow_issuer_untrusted")
    if trusted_signal_issuer is not None and config.signal_issuer != trusted_signal_issuer:
        problems.append("candidate_privacy_signal_issuer_untrusted")
    return problems


def candidate_privacy_key_versions() -> set[int]:
    """Return configured versions only; callers never expose key material."""
    return set(load_candidate_privacy_config(require_hmac=True).keys)


def candidate_privacy_key_versions_for_readiness() -> set[int] | None:
    try:
        return candidate_privacy_key_versions()
    except CandidatePrivacyConfigurationError:
        return None
