#!/usr/bin/env python3
"""Static consent authority boundaries; executable proofs live in the test matrices."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = {
    "activekg/api/organization_candidates.py": "311df4f1385eb1c4756cf5f61f850c50f831b1dad16411e226402e7c07cba50a",
    "activekg/api/sourced_candidates.py": "2d3342ff555c29f006df4a3aec9d607202f5f093023097b5f359740e74863d2c",
    "activekg/api/global_memory.py": "2e4a5971278dce4f091cbf1ceefa86f75b268cfa5e3b2a8fe22808974aa3ce58",
    "activekg/api/auth.py": "f4005bf2df5818d27fcae77cd3659d860170bdb0f4b75b8643b76876ee323071",
    "activekg/privacy/config.py": "97c6bc67fafcf4953c85e7b86579fae4b537f0c264550f378cad4ea159fa9546",
    "activekg/privacy/identity.py": "3c5bc791e51d0e8b5bd01964ab0f7ddd52a53420b44e0684f7a52c2ac16fa65c",
    "activekg/privacy/models.py": "078caf25e85a7e07aa32dad47c7ed86fe7bd9b75f3aeb3852527efad4e99522f",
    "activekg/privacy/repository.py": "faf4b08c16f10517e4173228702fd5be6b7388dc5192dd1e0544a83a8d270cdc",
    "activekg/embedding/worker.py": "9219f79df280e8a9c9394efd56b9ac81959edeef9dd6636f7fa8d94bf30628d1",
    "activekg/extraction/worker.py": "474f29f68818736c79764bc0cde97f780b35603202caf512c031e89bb05390f5",
    "db/migrations/023_candidate_privacy_directives.sql": "de179e695497c96321de2990b590b6e93702220b0071b488da18b0beffd94e1e",
    "db/migrations/024_organization_decision_event_inbox.sql": "a39bedef181f6152a5ecad1f5167afd9a266ea08be74686fe37686538665dacf",
    "db/migrations/025_approved_provider_candidate_ingest.sql": "2f75201a30493f099953c62aa077c43a8dbb5745b14fc84fb06b76190570b2fe",
    "db/migrations/026_organization_private_candidate_intake.sql": "154f14ad9eff1bf6f4b86c4944769356c9324cd5bb0a5becaa675179857eb2e6",
}


class GuardError(RuntimeError):
    pass


def _read(root: Path, path: str) -> str:
    try:
        return (root / path).read_text(encoding="utf-8")
    except OSError as exc:
        raise GuardError(f"missing consent authority: {path}") from exc


def _require(source: str, tokens: tuple[str, ...], code: str) -> None:
    if any(token not in source for token in tokens):
        raise GuardError(code)


def validate(root: Path = ROOT) -> None:
    for path, expected in FROZEN.items():
        if hashlib.sha256(_read(root, path).encode()).hexdigest() != expected:
            raise GuardError("frozen consent dependency drifted")
    receiver = _read(root, "activekg/api/candidate_consent.py")
    migration = _read(root, "db/migrations/027_candidate_consent.sql")
    readiness = _read(root, "activekg/api/operational.py")
    main = _read(root, "activekg/api/main.py")
    try:
        tree = ast.parse(_read(root, "activekg/common/migration_manifest.py"))
        declaration = next(
            n
            for n in tree.body
            if isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.target.id == "MIGRATIONS"
        )
        ordered = list(ast.literal_eval(declaration.value))
        caller = json.loads(_read(root, "scripts/schema_control_callers.json"))
        if (
            len(ordered) != 27
            or len(set(ordered)) != 27
            or ordered[-1] != "027_candidate_consent.sql"
        ):
            raise ValueError("tail")
        if ordered != caller["migration_manifest"]:
            raise ValueError("authority")
        if hashlib.sha256(migration.encode()).hexdigest() != caller["migration_files"][ordered[-1]]:
            raise ValueError("migration pin")
    except (ValueError, TypeError, KeyError, StopIteration, SyntaxError) as exc:
        raise GuardError("consent manifest mismatch") from exc
    _require(
        receiver,
        (
            'claims.issuer != "vantahire"',
            'auth.JWT_AUDIENCE != "activekg"',
            'claims.actor_id != "vantahire-backend"',
            'claims.actor_type != "service"',
            'claims.scopes != ["candidate-consent:write"]',
            "not _TENANT.fullmatch(claims.tenant_id)",
            'claims.tenant_id != f"candidate_{payload.subject_id}"',
            "Depends(require_consent_writer)",
            'ConfigDict(extra="forbid", strict=True)',
            "from typing_extensions import Self",
            "if len(body) > 65536:",
            "if not candidate_consent_intake_enabled():",
        ),
        "consent receiver authority missing",
    )
    _require(
        receiver,
        (
            '"candidate-consent:v1"',
            "payload.source.ordered() if payload.source else None",
            "values != expected",
            "self.source.source_version != self.version",
            "require_allowed(decision, global_use=True)",
            "_require_global(cur, tokens, None)",
            "_require_global(cur, tokens, global_id)",
            "candidate-privacy-token:",
            "candidate-privacy-global:",
            "candidate_consent_replay_conflict",
            "candidate_consent_idempotency_invalid",
            "identity_review_required",
            "conn.commit()",
            "conn.rollback()",
            "SET LOCAL lock_timeout",
            "SET LOCAL statement_timeout",
            "time.monotonic() - started > 3",
        ),
        "consent ordering/privacy protocol missing",
    )
    blocks = {
        n.name: ast.get_source_segment(receiver, n)
        for n in ast.parse(receiver).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    store = blocks["_store"]
    if store.count("conn.rollback()") != 3 or store.count("conn.commit()") != 1:
        raise GuardError("consent transaction/replay rollback census changed")
    if max(
        store.index("_require_global(cur, tokens, None)"),
        store.index("_require_global(cur, tokens, global_id)"),
    ) > store.index("INSERT INTO global_candidates"):
        raise GuardError("consent privacy fence after persistence")
    resume = blocks["_check_resume"]
    _require(
        resume,
        (
            "WHERE v.resume_version_id=%s::uuid",
            "v.reference_id=%s::uuid",
            "v.tenant_id=%s",
            "AS matches",
            "(tenant,)",
            "candidate_consent_resume_unavailable",
        ),
        "consent resume key/context lost",
    )
    statements = re.findall(r"\bcur\.execute\(", resume)
    if len(statements) != 3:
        raise GuardError("consent organization-context query count changed")
    if "VALUES(%s,'consent_pending',NULL,'{}'::jsonb)" not in store:
        raise GuardError("consent shell became a publication")
    _require(
        blocks["_resolve_identity"],
        ("email_hash=%s", "if bound:", "all_ids and not email_ids"),
        "consent email-only binding lost",
    )
    for forbidden in (
        r"INSERT\s+INTO\s+(?:candidate_source_records|candidate_identifiers|candidate_provenance|nodes|edges)",
        r"UPDATE\s+global_candidates",
        r"\b(?:print|logger\.\w+|logging\.\w+)\s*\(",
        r"\b(?:enqueue_embedding_job|enqueue_extraction_job)\s*\(",
    ):
        if re.search(forbidden, receiver, re.I):
            raise GuardError("consent gained forbidden indexing/persistence/output")
    _require(
        migration,
        (
            "CREATE TABLE candidate_consent_state",
            "CREATE TABLE candidate_consent_sources",
            "CREATE TABLE candidate_consent_receipts",
            "ON DELETE RESTRICT",
            "ERRCODE='55000'",
            "SET row_security=off",
            "OLD.global_candidate_id IS NOT NULL",
            "consent_active_source_fk",
        ),
        "consent schema authority missing",
    )
    if (
        migration.count("FORCE ROW LEVEL SECURITY") != 3
        or migration.count("BEFORE UPDATE OR DELETE") != 2
        or migration.count("BEFORE TRUNCATE") != 2
    ):
        raise GuardError("consent RLS/append-only census mismatch")
    if re.search(
        r"DISABLE\s+TRIGGER|NO\s+FORCE\s+ROW\s+LEVEL|^\s*(?:INSERT|UPDATE|DELETE)\s+",
        migration,
        re.I | re.M,
    ):
        raise GuardError("consent migration gained bypass or backfill")
    _require(
        readiness,
        (
            "CANDIDATE_CONSENT_CATALOG_SQL",
            "candidate_consent_intake_disabled",
            "relforcerowsecurity",
            "candidate_consent_binding_immutable",
        ),
        "consent readiness authority missing",
    )
    _require(
        main,
        (
            "app.include_router(candidate_consent_router)",
            "candidate_consent_intake_enabled=candidate_consent_intake_enabled()",
        ),
        "consent registration missing",
    )
    _require(
        migration,
        (
            "CONSTRAINT consent_source_profile_shape",
            "CONSTRAINT consent_source_resume_shape",
            "jsonb_typeof(resume->'organization_id')='number'",
            "IS TRUE)",
            "5242880",
            "2147483647",
        ),
        "consent scalar constraint missing",
    )
    constraints = re.findall(
        r"\('(candidate_consent_[a-z]+)','([a-z_0-9]+)','[cfpu]','[0-9a-f]{32}'\)", readiness
    )
    if (
        len(constraints) != 44
        or len(set(constraints)) != 44
        or any(len(name.encode()) > 63 for _, name in constraints)
    ):
        raise GuardError("consent named constraint census invalid")
    _require(
        readiness,
        (
            "md5(pg_get_constraintdef(c.oid))=expected.definition_hash",
            "c.convalidated",
            "NOT c.condeferrable",
        ),
        "consent constraint definition check missing",
    )


if __name__ == "__main__":
    try:
        validate()
    except GuardError as exc:
        print(f"candidate-consent-guard: REFUSED ({exc})", file=sys.stderr)
        raise SystemExit(1) from None
    print("candidate-consent-guard: OK")
