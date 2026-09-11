from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts import candidate_consent_guard as guard

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    *guard.FROZEN,
    "activekg/api/candidate_consent.py",
    "activekg/api/main.py",
    "activekg/api/operational.py",
    "db/migrations/027_candidate_consent.sql",
    "activekg/common/migration_manifest.py",
    "scripts/schema_control_callers.json",
)


def _copy(tmp_path):
    root = tmp_path / "consent"
    for name in FILES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return root


def test_checked_in_guard():
    guard.validate()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('claims.issuer != "vantahire"', 'claims.issuer != "any"'),
        ('claims.actor_type != "service"', 'claims.actor_type != "user"'),
        ('claims.actor_id != "vantahire-backend"', 'claims.actor_id != "any"'),
        ('claims.scopes != ["candidate-consent:write"]', "False"),
        ("not _TENANT.fullmatch(claims.tenant_id)", "False"),
        ('claims.tenant_id != f"candidate_{payload.subject_id}"', "False"),
        ("Depends(require_consent_writer)", "Depends(get_jwt_claims)"),
        ('extra="forbid"', 'extra="ignore"'),
        ("values != expected", "False"),
        (
            "require_allowed(decision, global_use=True)",
            "require_allowed(decision, global_use=False)",
        ),
        ("_require_global(cur, tokens, global_id)", "_require_global(cur, [], None)"),
        ("WHERE v.resume_version_id=%s::uuid", "WHERE TRUE"),
        ("v.reference_id=%s::uuid", "TRUE"),
        ("(tenant,)", '("org_7",)'),
        ("conn.commit()", "pass"),
        ("conn.rollback()", "pass"),
        ("'consent_pending'", "'ready'"),
        ("all_ids and not email_ids", "False"),
        ("if len(body) > 65536:", "if False:"),
        ("if not candidate_consent_intake_enabled():", "if False:"),
        ("time.monotonic() - started > 3", "False"),
    ],
)
def test_receiver_mutation_refused(tmp_path, old, new):
    root = _copy(tmp_path)
    path = root / "activekg/api/candidate_consent.py"
    original = path.read_text()
    assert old in original
    path.write_text(original.replace(old, new, 1))
    with pytest.raises(guard.GuardError):
        guard.validate(root)


@pytest.mark.parametrize(
    "old",
    [
        "FORCE ROW LEVEL SECURITY",
        "BEFORE UPDATE OR DELETE",
        "BEFORE TRUNCATE",
        "SET row_security=off",
        "ON DELETE RESTRICT",
        "jsonb_typeof(resume->'organization_id')='number'",
        "IS TRUE)",
    ],
)
def test_schema_semantic_mutations_survive_repin(tmp_path, old):
    root = _copy(tmp_path)
    path = root / "db/migrations/027_candidate_consent.sql"
    assert old in path.read_text()
    path.write_text(path.read_text().replace(old, "REMOVED"))
    caller = root / "scripts/schema_control_callers.json"
    value = json.loads(caller.read_text())
    value["migration_files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    caller.write_text(json.dumps(value))
    with pytest.raises(guard.GuardError):
        guard.validate(root)


def test_drifted_frozen_privacy_refused(tmp_path):
    root = _copy(tmp_path)
    path = root / "activekg/privacy/repository.py"
    path.write_text(path.read_text() + "\n# drift\n")
    with pytest.raises(guard.GuardError, match="frozen"):
        guard.validate(root)


def test_missing_manifest_authority_refused(tmp_path):
    root = _copy(tmp_path)
    path = root / "activekg/common/migration_manifest.py"
    path.write_text(path.read_text().replace('"027_candidate_consent.sql",', ""))
    with pytest.raises(guard.GuardError, match="manifest"):
        guard.validate(root)


def test_privacy_cannot_move_after_shell_insert(tmp_path):
    root = _copy(tmp_path)
    path = root / "activekg/api/candidate_consent.py"
    source = path.read_text()
    source = source.replace("                _require_global(cur, tokens, global_id)\n", "")
    source = source.replace(
        "        conn.commit()",
        "        _require_global(cur, tokens, global_id)\n        conn.commit()",
    )
    path.write_text(source)
    with pytest.raises(guard.GuardError, match="after persistence"):
        guard.validate(root)
