"""Identifier normalization for canonical candidate identity.

Each upstream source (VantaHire, Signal, scraped profiles, etc.) presents the
same identifier in slightly different shapes — query strings on LinkedIn URLs,
``mailto:`` prefixes, inconsistent phone formats, trailing slashes, casing.
Before we can dedupe candidates across sources we must reduce every identifier
to a single canonical string.

The rules here are intentionally strict: we would rather fail a match than
silently collapse two different people into one canonical candidate.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse

IDENTIFIER_TYPES: frozenset[str] = frozenset(
    {
        "signal_candidate_id",
        "vantahire_application_id",
        "vantahire_resume_id",
        "linkedin_url",
        "github_url",
        "medium_url",
        "email",
        "phone",
        "website_url",
        "twitter_url",
        "stackoverflow_url",
        "portfolio_url",
        "other",
    }
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_DIGIT_RE = re.compile(r"[^\d+]")


class IdentifierNormalizationError(ValueError):
    """Raised when an identifier value cannot be normalized to a canonical form."""


def _clean(value: str) -> str:
    return _WHITESPACE_RE.sub("", value).strip()


def _normalize_opaque_id(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise IdentifierNormalizationError("identifier value is empty")
    return cleaned


def _normalize_email(value: str) -> str:
    cleaned = _clean(value).lower()
    if cleaned.startswith("mailto:"):
        cleaned = cleaned[len("mailto:") :]
    if not _EMAIL_RE.match(cleaned):
        raise IdentifierNormalizationError(f"invalid email: {value!r}")
    local, _, domain = cleaned.partition("@")
    # Gmail treats dots in the local part as insignificant; fold them for
    # dedupe. Plus-aliases stay attached so recruiters keep the distinction.
    if domain in {"gmail.com", "googlemail.com"}:
        local = local.replace(".", "")
        domain = "gmail.com"
    return f"{local}@{domain}"


def _normalize_phone(value: str) -> str:
    cleaned = _NON_DIGIT_RE.sub("", value)
    if cleaned.startswith("00"):
        cleaned = "+" + cleaned[2:]
    if not cleaned:
        raise IdentifierNormalizationError(f"invalid phone: {value!r}")
    digits = cleaned.lstrip("+")
    if len(digits) < 7 or len(digits) > 15:
        raise IdentifierNormalizationError(f"phone has invalid length: {value!r}")
    return "+" + digits if cleaned.startswith("+") else digits


def _normalize_url_host_path(value: str, *, expected_host_suffix: str | None = None) -> str:
    cleaned = _clean(value)
    if not cleaned:
        raise IdentifierNormalizationError("url is empty")
    if "://" not in cleaned:
        cleaned = "https://" + cleaned
    parsed = urlparse(cleaned)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        raise IdentifierNormalizationError(f"url missing host: {value!r}")
    if expected_host_suffix and not host.endswith(expected_host_suffix):
        raise IdentifierNormalizationError(
            f"url host {host!r} does not match expected {expected_host_suffix!r}"
        )
    path = parsed.path.rstrip("/")
    return urlunparse(("https", host, path, "", "", ""))


def _normalize_linkedin(value: str) -> str:
    url = _normalize_url_host_path(value, expected_host_suffix="linkedin.com")
    parsed = urlparse(url)
    # Only /in/<handle> and /pub/<handle> are canonical candidate profile paths.
    # Everything else (feed, company pages, posts) is not a stable identity.
    segments = [s for s in parsed.path.split("/") if s]
    if len(segments) < 2 or segments[0] not in {"in", "pub"}:
        raise IdentifierNormalizationError(f"not a linkedin profile url: {value!r}")
    handle = segments[1].lower()
    return f"https://linkedin.com/in/{handle}"


def _normalize_github(value: str) -> str:
    url = _normalize_url_host_path(value, expected_host_suffix="github.com")
    parsed = urlparse(url)
    segments = [s for s in parsed.path.split("/") if s]
    if not segments:
        raise IdentifierNormalizationError(f"not a github profile url: {value!r}")
    handle = segments[0].lower()
    return f"https://github.com/{handle}"


def _normalize_medium(value: str) -> str:
    cleaned = _clean(value)
    if not cleaned:
        raise IdentifierNormalizationError("medium url is empty")
    if "://" not in cleaned:
        cleaned = "https://" + cleaned
    parsed = urlparse(cleaned)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host.endswith("medium.com"):
        raise IdentifierNormalizationError(f"not a medium url: {value!r}")
    # medium supports https://medium.com/@handle and https://handle.medium.com
    if host == "medium.com":
        segments = [s for s in parsed.path.split("/") if s]
        if not segments or not segments[0].startswith("@"):
            raise IdentifierNormalizationError(f"not a medium profile url: {value!r}")
        handle = segments[0].lower()
        return f"https://medium.com/{handle}"
    subdomain = host[: -len(".medium.com")]
    if not subdomain:
        raise IdentifierNormalizationError(f"not a medium profile url: {value!r}")
    return f"https://medium.com/@{subdomain.lower()}"


def _normalize_generic_url(value: str) -> str:
    return _normalize_url_host_path(value)


_NORMALIZERS = {
    "signal_candidate_id": _normalize_opaque_id,
    "vantahire_application_id": _normalize_opaque_id,
    "vantahire_resume_id": _normalize_opaque_id,
    "email": _normalize_email,
    "phone": _normalize_phone,
    "linkedin_url": _normalize_linkedin,
    "github_url": _normalize_github,
    "medium_url": _normalize_medium,
    "website_url": _normalize_generic_url,
    "twitter_url": _normalize_generic_url,
    "stackoverflow_url": _normalize_generic_url,
    "portfolio_url": _normalize_generic_url,
    "other": _normalize_opaque_id,
}


def normalize_identifier(identifier_type: str, value: str) -> str:
    """Return the canonical form of an identifier, or raise.

    Callers should always persist the return value of this function in
    ``candidate_identifiers.value_normalized`` and keep the original in
    ``value_raw``.
    """
    if not isinstance(identifier_type, str) or identifier_type not in IDENTIFIER_TYPES:
        raise IdentifierNormalizationError(f"unknown identifier_type: {identifier_type!r}")
    if not isinstance(value, str):
        raise IdentifierNormalizationError("identifier value must be a string")
    normalizer = _NORMALIZERS[identifier_type]
    return normalizer(value)
