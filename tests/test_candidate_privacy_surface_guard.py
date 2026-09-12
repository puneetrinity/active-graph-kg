from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

import pytest

from scripts import candidate_privacy_surface_guard as guard

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _temporary_mutation(relative: str, old: bytes, new: bytes):
    path = ROOT / relative
    original = path.read_bytes()
    original_hash = hashlib.sha256(original).hexdigest()
    assert old in original
    try:
        path.write_bytes(original.replace(old, new, 1))
        yield
    finally:
        path.write_bytes(original)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash


def test_checked_in_surface_manifest_is_complete() -> None:
    guard.validate()


def test_guard_rejects_duplicate_manifest_keys_and_restores_exact_bytes() -> None:
    old = b'  "version": 1,\n'
    new = old + b'  "version": 1,\n'
    with _temporary_mutation("activekg/privacy/surfaces.json", old, new):
        with pytest.raises(guard.GuardError, match="duplicate key"):
            guard.validate()
    guard.validate()


def test_guard_rejects_unfenced_search_and_restores_exact_bytes() -> None:
    old = (
        b"                    \"(candidate_privacy_node_decision(id) = 'allow' OR (\"\n"
        b"                    \"candidate_privacy_node_decision(id) = 'block_global' \"\n"
        b'                    "AND tenant_id IS NOT NULL AND tenant_id IS NOT DISTINCT FROM %s))"\n'
    )
    with _temporary_mutation("activekg/graph/repository.py", old, b'                    "true"\n'):
        with pytest.raises(guard.GuardError, match="enforcement anchor"):
            guard.validate()
    guard.validate()


def test_guard_rejects_non_materialized_ann_underfill_fallback() -> None:
    with _temporary_mutation(
        "activekg/graph/repository.py",
        b"WITH privacy_filtered_nodes AS MATERIALIZED (",
        b"WITH privacy_filtered_nodes AS NOT MATERIALIZED (",
    ):
        with pytest.raises(guard.GuardError, match="exact-rescan"):
            guard.validate()
    guard.validate()


def test_guard_rejects_per_subject_batch_regression() -> None:
    with _temporary_mutation(
        "activekg/privacy/repository.py",
        b"CROSS JOIN LATERAL candidate_privacy_resolve_subject(",
        b"CROSS JOIN LATERAL candidate_privacy_resolve_subject /* mutation */ (",
    ):
        with pytest.raises(guard.GuardError, match="three-query set-based boundary"):
            guard.validate()
    guard.validate()


def test_guard_rejects_an_unclassified_writer_and_restores_exact_bytes() -> None:
    path = ROOT / "activekg/graph/repository.py"
    original = path.read_bytes()
    original_hash = hashlib.sha256(original).hexdigest()
    try:
        path.write_bytes(
            original
            + b"\n\ndef _candidate_privacy_guard_mutation(cur):\n"
            + b'    cur.execute("INSERT INTO candidates (candidate_id) VALUES (gen_random_uuid())")\n'
        )
        with pytest.raises(guard.GuardError, match="census drifted"):
            guard.validate()
    finally:
        path.write_bytes(original)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
    guard.validate()


def test_guard_rejects_removed_stale_job_recheck_and_restores_exact_bytes() -> None:
    path = "activekg/embedding/worker.py"
    original = (ROOT / path).read_bytes()
    marker = b"self.privacy_repository.node_decision"
    first = original.find(marker)
    second = original.find(marker, first + len(marker))
    assert first >= 0 and second > first
    mutated = (
        original[:second]
        + b"self.privacy_repository.canonical_decision"
        + original[second + len(marker) :]
    )
    original_hash = hashlib.sha256(original).hexdigest()
    try:
        (ROOT / path).write_bytes(mutated)
        with pytest.raises(guard.GuardError, match="stale-job"):
            guard.validate()
    finally:
        (ROOT / path).write_bytes(original)
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == original_hash
    guard.validate()


def test_guard_rejects_raw_output_primitive_and_restores_exact_bytes() -> None:
    old = b'"""Strict normalization and non-reversible privacy identity tokens."""'
    new = old + b"\nprint('privacy-output-regression')"
    with _temporary_mutation("activekg/privacy/identity.py", old, new):
        with pytest.raises(guard.GuardError, match="raw-output"):
            guard.validate()
    guard.validate()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            b"decision_for(row[1], row[0])",
            b"decision_for(row[0], row[1])",
        ),
        (
            b"require_allowed(decision, global_use=True)",
            b"require_allowed(decision, global_use=False)",
        ),
    ],
)
def test_guard_rejects_approved_provider_privacy_authority_mutation(
    old: bytes,
    new: bytes,
) -> None:
    with _temporary_mutation("activekg/api/sourced_candidates.py", old, new):
        with pytest.raises(guard.GuardError, match="privacy/source authority"):
            guard.validate()
    guard.validate()


def test_guard_rejects_approved_provider_route_authority_mutation() -> None:
    with _temporary_mutation(
        "activekg/api/sourced_candidates.py",
        b"Depends(require_sourced_candidate_writer)",
        b"Depends(get_jwt_claims)",
    ):
        with pytest.raises(guard.GuardError, match="route authority"):
            guard.validate()
    guard.validate()


def test_guard_rejects_private_intake_privacy_authority_mutation() -> None:
    with _temporary_mutation(
        "activekg/api/organization_candidates.py",
        b"require_allowed(decision, global_use=False)",
        b"require_allowed(decision, global_use=True)",
    ):
        with pytest.raises(guard.GuardError, match="enforcement anchor"):
            guard.validate()
    guard.validate()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (b"Depends(require_consent_writer)", b"Depends(get_jwt_claims)"),
        (b"global_use=True", b"global_use=False"),
        (b"_require_global(cur, tokens, None)", b"pass # removed global proof"),
    ],
)
def test_guard_rejects_consent_authority_mutation(old: bytes, new: bytes) -> None:
    with _temporary_mutation("activekg/api/candidate_consent.py", old, new):
        with pytest.raises(guard.GuardError):
            guard.validate()
    guard.validate()
