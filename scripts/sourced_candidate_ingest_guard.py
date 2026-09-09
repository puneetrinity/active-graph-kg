#!/usr/bin/env python3
"""Static fail-closed guard for Wave 4A approved-provider ingestion."""

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
    "db/migrations/024_organization_decision_event_inbox.sql": "a39bedef181f6152a5ecad1f5167afd9a266ea08be74686fe37686538665dacf",
    "db/migrations/026_organization_private_candidate_intake.sql": "154f14ad9eff1bf6f4b86c4944769356c9324cd5bb0a5becaa675179857eb2e6",
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


def _function(source: str, name: str, next_name: str | None = None) -> str:
    marker = f"def {name}("
    start = source.find(marker)
    if start < 0:
        marker = f"async def {name}("
        start = source.find(marker)
    if start < 0:
        raise GuardError(f"required function missing: {name}")
    if next_name is None:
        return source[start:]
    end_candidates = [
        position
        for prefix in ("\ndef ", "\nasync def ", "\n@router.")
        if (position := source.find(prefix + next_name, start + len(marker))) >= 0
    ]
    return source[start : min(end_candidates)] if end_candidates else source[start:]


def validate(root: Path = ROOT) -> None:
    for relative, expected in FROZEN_HASHES.items():
        try:
            actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        except OSError as exc:
            raise GuardError(f"frozen file missing: {relative}") from exc
        if actual != expected:
            raise GuardError(f"frozen authority drifted: {relative}")

    migration = _read(root, "db/migrations/025_approved_provider_candidate_ingest.sql")
    receiver = _read(root, "activekg/api/sourced_candidates.py")
    main = _read(root, "activekg/api/main.py")
    global_memory = _read(root, "activekg/api/global_memory.py")
    manifest = _read(root, "activekg/common/migration_manifest.py")

    _require(
        migration,
        (
            "CREATE TABLE global_candidate_source_identities",
            "CREATE TABLE global_candidate_source_observations",
            "CREATE TABLE global_candidate_ingest_receipts",
            "approved_provider_candidate_evidence_append_only",
            "ERRCODE = '55000'",
            "ON DELETE RESTRICT",
            "REVOKE ALL ON global_candidate_source_identities FROM PUBLIC",
            "REVOKE ALL ON FUNCTION approved_provider_candidate_evidence_append_only() FROM PUBLIC",
        ),
        "migration authority is incomplete",
    )
    if "CREATE SEQUENCE" in migration.upper() or re.search(
        r"^\s*(?:DELETE\s+FROM|UPDATE\s+global_candidates|TRUNCATE\s+global_candidates)",
        migration,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        raise GuardError("migration performs a forbidden product mutation")
    if re.search(r"^\s*(tenant_id|organization_id|job_id|rank|query)\s+", migration, re.MULTILINE):
        raise GuardError("global source evidence gained tenant/job/rank state")
    if migration.count("BEFORE UPDATE OR DELETE") != 3 or migration.count("BEFORE TRUNCATE") != 3:
        raise GuardError("append-only trigger census drifted")

    _require(
        receiver,
        (
            'Literal["crustdata"]',
            'Literal["person"]',
            'Literal["crustdata_person_v1"]',
            'claims.scopes != ["candidate-source:write"]',
            'claims.actor_id != "signal-service"',
            'claims.actor_type != "service"',
            "claims.issuer != auth.SIGNAL_JWT_ISSUER",
            "decision_for(row[1], row[0])",
            "require_allowed(decision, global_use=True)",
            "candidate_privacy_match(%s::jsonb,%s,NULL,NULL)",
            "pg_advisory_xact_lock",
            "sourced_candidate_idempotency_conflict",
            'resolution = "conflict_review_required"',
            "conn.rollback()",
            "conn.commit()",
        ),
        "receiver authority or atomic resolver is incomplete",
    )
    store = _function(receiver, "_store", "ingest_sourced_candidate")
    if store.index("_require_privacy_for_candidate(") > store.index(
        "INSERT INTO global_candidate_source_observations"
    ):
        raise GuardError("privacy proof moved after the evidence write")
    if any(
        token in receiver
        for token in (
            'logger.info("Sourced',
            "payload.model_dump()",
            "claims.tenant_id,\n                    payload",
        )
    ):
        raise GuardError("receiver gained unsafe candidate logging or tenant evidence")

    route = _function(receiver, "ingest_sourced_candidate")
    _require(
        route,
        (
            'if sourced_candidate_ingest_mode() == "off"',
            "payload = await _parse_body(request)",
            "return _store(payload, claims)",
        ),
        "new route lost its mode/body/store boundary",
    )

    old_route = _function(main, "resolve_candidate_from_signal")
    retirement_gate = 'sourced_candidate_ingest_mode() == "canonical_only"'
    if retirement_gate not in old_route:
        raise GuardError("old Signal writer lost its canonical-only retirement gate")
    if old_route.index(retirement_gate) > old_route.index("payload.source_record_type"):
        raise GuardError("old Signal writer performs body-dependent work before retirement")
    for name in ("upsert_global_candidate", "create_provenance", "upsert_access"):
        block = _function(global_memory, name)
        if block.index("raise HTTPException(status_code=410") > block.index("_require_enabled()"):
            raise GuardError(f"retired global writer performs work before 410: {name}")

    _require(
        main,
        (
            "app.include_router(sourced_candidates_router)",
            '"SOURCED_CANDIDATE_INGEST_MODE", "off"',
        ),
        "route registration/readiness mode is incomplete",
    )
    _require(
        manifest,
        (
            '"025_approved_provider_candidate_ingest.sql"',
            '"026_organization_private_candidate_intake.sql"',
            "len(MIGRATIONS) != 26",
        ),
        "migration manifest did not advance exactly once",
    )


def main() -> int:
    try:
        validate()
        print("sourced-candidate-ingest-guard: OK")
        return 0
    except GuardError as exc:
        print(f"sourced-candidate-ingest-guard: REFUSED ({exc})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
