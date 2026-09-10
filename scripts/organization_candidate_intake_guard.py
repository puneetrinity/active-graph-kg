#!/usr/bin/env python3
"""Static fail-closed guard for Wave 4B organization-private intake."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FROZEN_HASHES = {
    "activekg/privacy/config.py": "97c6bc67fafcf4953c85e7b86579fae4b537f0c264550f378cad4ea159fa9546",
    "activekg/privacy/identity.py": "3c5bc791e51d0e8b5bd01964ab0f7ddd52a53420b44e0684f7a52c2ac16fa65c",
    "activekg/privacy/models.py": "078caf25e85a7e07aa32dad47c7ed86fe7bd9b75f3aeb3852527efad4e99522f",
    "activekg/privacy/repository.py": "faf4b08c16f10517e4173228702fd5be6b7388dc5192dd1e0544a83a8d270cdc",
    "db/migrations/023_candidate_privacy_directives.sql": "de179e695497c96321de2990b590b6e93702220b0071b488da18b0beffd94e1e",
    "db/migrations/025_approved_provider_candidate_ingest.sql": "2f75201a30493f099953c62aa077c43a8dbb5745b14fc84fb06b76190570b2fe",
}


class GuardError(RuntimeError):
    pass


def _read(root: Path, relative: str) -> str:
    try:
        return (root / relative).read_text(encoding="utf-8")
    except OSError as exc:
        raise GuardError(f"required file missing: {relative}") from exc


def _require(source: str, tokens: tuple[str, ...], code: str) -> None:
    if any(token not in source for token in tokens):
        raise GuardError(code)


def validate(root: Path = ROOT) -> None:
    for relative, expected in FROZEN_HASHES.items():
        try:
            actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        except OSError as exc:
            raise GuardError(f"frozen authority missing: {relative}") from exc
        if actual != expected:
            raise GuardError(f"frozen authority drifted: {relative}")

    migration = _read(root, "db/migrations/026_organization_private_candidate_intake.sql")
    receiver = _read(root, "activekg/api/organization_candidates.py")
    main = _read(root, "activekg/api/main.py")
    manifest = _read(root, "activekg/common/migration_manifest.py")

    _require(
        migration,
        (
            "scope IN ('shared', 'organization_private')",
            "CREATE TABLE organization_candidate_references",
            "CREATE TABLE organization_candidate_resume_evidence",
            "CREATE TABLE organization_candidate_ingest_receipts",
            "FORCE ROW LEVEL SECURITY",
            "organization_candidate_evidence_append_only",
            "ERRCODE = '55000'",
            "ON DELETE RESTRICT",
            "REVOKE ALL ON FUNCTION organization_candidate_evidence_append_only() FROM PUBLIC",
        ),
        "private intake migration authority is incomplete",
    )
    if migration.count("BEFORE UPDATE OR DELETE") != 3 or migration.count("BEFORE TRUNCATE") != 3:
        raise GuardError("private evidence append-only trigger census drifted")
    if re.search(
        r"^\s*(?:UPDATE\s+candidates|DELETE\s+FROM\s+candidates|INSERT\s+INTO\s+global_)",
        migration,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        raise GuardError("private intake migration performs a forbidden data mutation")

    _require(
        receiver,
        (
            "claims.issuer != auth.JWT_ISSUER",
            'claims.actor_type != "service"',
            'claims.actor_id != "vantahire-backend"',
            'claims.scopes != ["organization-candidate:write"]',
            "require_allowed(decision, global_use=False)",
            "candidate_privacy_match(%s::jsonb,NULL,NULL,NULL)",
            "pg_advisory_xact_lock",
            "conn.rollback()",
            "conn.commit()",
            "'organization_private'",
            "organization_candidate_idempotency_conflict",
        ),
        "private intake receiver authority is incomplete",
    )
    privacy_index = receiver.index("_require_private_privacy(")
    candidate_insert = receiver.index("INSERT INTO candidates", privacy_index)
    if privacy_index > candidate_insert:
        raise GuardError("private privacy proof moved after candidate persistence")
    persistent_digest = receiver[receiver.index("def _canonical_persistent_input") :]
    if 'exclude={"idempotency_key", "privacy_subject"}' not in persistent_digest:
        raise GuardError("transient privacy subject entered persistent digest")
    for forbidden in (
        "INSERT INTO candidate_identifiers",
        "INSERT INTO candidate_source_records",
        "INSERT INTO global_candidates",
        "logger.info(",
        "payload.model_dump()",
    ):
        if forbidden in receiver:
            raise GuardError("private intake receiver gained forbidden persistence or logging")

    _require(
        main,
        (
            "app.include_router(organization_candidates_router)",
            "organization_candidate_intake_enabled=",
        ),
        "private intake route or readiness registration is incomplete",
    )
    _require(
        manifest,
        (
            '"026_organization_private_candidate_intake.sql"',
            "len(MIGRATIONS) != 26",
        ),
        "private intake migration manifest is incomplete",
    )


def main() -> int:
    try:
        validate()
        print("organization-candidate-intake-guard: OK")
        return 0
    except GuardError as exc:
        print(f"organization-candidate-intake-guard: REFUSED ({exc})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
