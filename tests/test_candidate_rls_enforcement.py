"""Functional RLS enforcement tests for the candidate identity tables.

These run against a real database with TWO roles:

* an owner/migration role (``ACTIVEKG_RLS_TEST_OWNER_DSN``) used for fixture
  setup/teardown — owners bypass ordinary RLS, so it is NOT what we assert on
* the restricted runtime role (``ACTIVEKG_RLS_TEST_RUNTIME_DSN``) provisioned
  NOSUPERUSER/NOBYPASSRLS without table ownership — every assertion about
  isolation runs as this role, because it is what the deployed API uses

Covered: tenant A/B read isolation, write isolation via WITH CHECK, missing
tenant context (no GUC → no rows), NULL-tenant rejection, and cross-tenant
child references (composite FK). Skipped entirely when the two DSNs are not
configured (e.g. the no-DB unit job).
"""

import os
import uuid

import psycopg
import pytest

OWNER_DSN = os.getenv("ACTIVEKG_RLS_TEST_OWNER_DSN")
RUNTIME_DSN = os.getenv("ACTIVEKG_RLS_TEST_RUNTIME_DSN")

pytestmark = pytest.mark.skipif(
    not (OWNER_DSN and RUNTIME_DSN),
    reason="ACTIVEKG_RLS_TEST_OWNER_DSN / ACTIVEKG_RLS_TEST_RUNTIME_DSN not configured",
)

TENANT_A = f"rls_test_a_{uuid.uuid4().hex[:8]}"
TENANT_B = f"rls_test_b_{uuid.uuid4().hex[:8]}"


def _set_tenant(cur: psycopg.Cursor, tenant_id: str | None) -> None:
    cur.execute(
        "SELECT set_config('app.current_tenant_id', %s, true)",
        ("" if tenant_id is None else tenant_id,),
    )


@pytest.fixture(scope="module")
def seeded_candidates():
    """Create one candidate per tenant as the owner role; clean up after."""
    cand_a = str(uuid.uuid4())
    cand_b = str(uuid.uuid4())
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO candidates (candidate_id, tenant_id, display_name) "
                "VALUES (%s, %s, %s), (%s, %s, %s)",
                (cand_a, TENANT_A, "Person A", cand_b, TENANT_B, "Person B"),
            )
    yield {"a": cand_a, "b": cand_b}
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM candidates WHERE tenant_id IN (%s, %s)",
                (TENANT_A, TENANT_B),
            )


@pytest.fixture()
def runtime_conn():
    with psycopg.connect(RUNTIME_DSN) as conn:
        yield conn


def test_runtime_role_posture(runtime_conn):
    """The runtime role must be subject to RLS: no superuser, no bypass, no ownership."""
    with runtime_conn.cursor() as cur:
        cur.execute("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
        is_super, bypasses = cur.fetchone()
        assert not is_super, "runtime role is superuser — RLS is nominal"
        assert not bypasses, "runtime role has BYPASSRLS — RLS is nominal"
        cur.execute(
            "SELECT count(*) FROM pg_tables WHERE tablename LIKE 'candidate%' "
            "AND tableowner = current_user"
        )
        assert cur.fetchone()[0] == 0, "runtime role owns candidate tables — RLS is nominal"


def test_tenant_a_sees_only_tenant_a(runtime_conn, seeded_candidates):
    with runtime_conn.transaction():
        with runtime_conn.cursor() as cur:
            _set_tenant(cur, TENANT_A)
            cur.execute(
                "SELECT candidate_id, tenant_id FROM candidates WHERE tenant_id IN (%s, %s)",
                (TENANT_A, TENANT_B),
            )
            rows = cur.fetchall()
    assert [str(r[0]) for r in rows] == [seeded_candidates["a"]]
    assert all(r[1] == TENANT_A for r in rows)


def test_missing_tenant_context_sees_nothing(runtime_conn, seeded_candidates):
    with runtime_conn.transaction():
        with runtime_conn.cursor() as cur:
            # No GUC installed at all in this transaction.
            cur.execute(
                "SELECT count(*) FROM candidates WHERE tenant_id IN (%s, %s)",
                (TENANT_A, TENANT_B),
            )
            assert cur.fetchone()[0] == 0


def test_write_into_other_tenant_rejected(runtime_conn, seeded_candidates):
    """WITH CHECK must reject rows whose tenant differs from the session tenant."""
    with pytest.raises(psycopg.errors.Error) as excinfo:
        with runtime_conn.transaction():
            with runtime_conn.cursor() as cur:
                _set_tenant(cur, TENANT_A)
                cur.execute(
                    "INSERT INTO candidates (candidate_id, tenant_id, display_name) "
                    "VALUES (%s, %s, %s)",
                    (str(uuid.uuid4()), TENANT_B, "Smuggled"),
                )
    # 42501 = insufficient_privilege ("new row violates row-level security policy")
    assert excinfo.value.sqlstate == "42501"


def test_write_own_tenant_allowed_and_isolated(runtime_conn, seeded_candidates):
    new_id = str(uuid.uuid4())
    with runtime_conn.transaction():
        with runtime_conn.cursor() as cur:
            _set_tenant(cur, TENANT_A)
            cur.execute(
                "INSERT INTO candidates (candidate_id, tenant_id, display_name) "
                "VALUES (%s, %s, %s)",
                (new_id, TENANT_A, "Legit A"),
            )
    # Tenant B must not see it.
    with runtime_conn.transaction():
        with runtime_conn.cursor() as cur:
            _set_tenant(cur, TENANT_B)
            cur.execute("SELECT count(*) FROM candidates WHERE candidate_id = %s", (new_id,))
            assert cur.fetchone()[0] == 0
    # Cleanup as tenant A.
    with runtime_conn.transaction():
        with runtime_conn.cursor() as cur:
            _set_tenant(cur, TENANT_A)
            cur.execute("DELETE FROM candidates WHERE candidate_id = %s", (new_id,))


def test_null_tenant_rejected(runtime_conn):
    with pytest.raises(psycopg.errors.Error) as excinfo:
        with runtime_conn.transaction():
            with runtime_conn.cursor() as cur:
                _set_tenant(cur, TENANT_A)
                cur.execute(
                    "INSERT INTO candidates (candidate_id, tenant_id, display_name) "
                    "VALUES (%s, NULL, %s)",
                    (str(uuid.uuid4()), "No Tenant"),
                )
    # Either the NOT NULL constraint (23502) or the RLS WITH CHECK (42501)
    # stops it — both are acceptable; the row must never land.
    assert excinfo.value.sqlstate in {"23502", "42501"}


def test_cross_tenant_child_reference_rejected(seeded_candidates):
    """Composite FK: a tenant-B identifier can never point at a tenant-A candidate.

    Run as the OWNER role on purpose — owners bypass RLS, so this proves the
    schema itself (not just policies) blocks cross-tenant references.
    """
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                cur.execute(
                    "INSERT INTO candidate_identifiers "
                    "(candidate_id, tenant_id, identifier_type, value_normalized) "
                    "VALUES (%s, %s, 'email', %s)",
                    (seeded_candidates["a"], TENANT_B, f"x_{uuid.uuid4().hex[:8]}@example.com"),
                )


def test_repository_conn_enforces_tenant_context(seeded_candidates):
    """Exercise CandidateRepository._conn() itself (not a manually set GUC).

    The repository must install the tenant GUC per call: same-tenant reads
    work, cross-tenant and missing-tenant reads return nothing — all through
    the restricted runtime role the deployed API uses.
    """
    pytest.importorskip("numpy")  # transitively required by activekg.graph.models
    from activekg.graph.candidate_repository import CandidateRepository

    repo = CandidateRepository(RUNTIME_DSN)
    try:
        assert repo.get_candidate(seeded_candidates["a"], tenant_id=TENANT_A) is not None
        assert repo.get_candidate(seeded_candidates["a"], tenant_id=TENANT_B) is None
        assert repo.get_candidate(seeded_candidates["a"], tenant_id=None) is None
    finally:
        repo.close()


@pytest.fixture(scope="module")
def seeded_children(seeded_candidates):
    """Attach one identifier and one source record per tenant (owner role)."""
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO candidate_identifiers "
                "(candidate_id, tenant_id, identifier_type, value_normalized) "
                "VALUES (%s, %s, 'email', %s), (%s, %s, 'email', %s)",
                (
                    seeded_candidates["a"],
                    TENANT_A,
                    f"a_{TENANT_A}@example.com",
                    seeded_candidates["b"],
                    TENANT_B,
                    f"b_{TENANT_B}@example.com",
                ),
            )
            cur.execute(
                "INSERT INTO candidate_source_records "
                "(candidate_id, tenant_id, source, source_record_type, source_record_id) "
                "VALUES (%s, %s, 'signal', 'profile', %s), (%s, %s, 'signal', 'profile', %s)",
                (
                    seeded_candidates["a"],
                    TENANT_A,
                    f"rec_{TENANT_A}",
                    seeded_candidates["b"],
                    TENANT_B,
                    f"rec_{TENANT_B}",
                ),
            )
    yield seeded_candidates
    # Parent-row cleanup cascades via the composite FK.


def test_identifier_isolation(runtime_conn, seeded_children):
    with runtime_conn.transaction():
        with runtime_conn.cursor() as cur:
            _set_tenant(cur, TENANT_A)
            cur.execute(
                "SELECT tenant_id FROM candidate_identifiers WHERE tenant_id IN (%s, %s)",
                (TENANT_A, TENANT_B),
            )
            rows = cur.fetchall()
    assert rows and all(r[0] == TENANT_A for r in rows)


def test_source_record_isolation(runtime_conn, seeded_children):
    with runtime_conn.transaction():
        with runtime_conn.cursor() as cur:
            _set_tenant(cur, TENANT_B)
            cur.execute(
                "SELECT tenant_id FROM candidate_source_records WHERE tenant_id IN (%s, %s)",
                (TENANT_A, TENANT_B),
            )
            rows = cur.fetchall()
    assert rows and all(r[0] == TENANT_B for r in rows)


def test_cross_tenant_source_record_rejected(seeded_children):
    """Composite FK on candidate_source_records, proven schema-level (owner role)."""
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                cur.execute(
                    "INSERT INTO candidate_source_records "
                    "(candidate_id, tenant_id, source, source_record_type, source_record_id) "
                    "VALUES (%s, %s, 'signal', 'profile', %s)",
                    (seeded_children["a"], TENANT_B, f"x_{uuid.uuid4().hex[:8]}"),
                )


def test_blank_tenant_rejected_by_constraint(seeded_candidates):
    """Migration 018: blank tenant_id can never be stored, even by the owner."""
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    "INSERT INTO candidates (candidate_id, tenant_id, display_name) "
                    "VALUES (%s, '', %s)",
                    (str(uuid.uuid4()), "Blank Tenant"),
                )
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    "INSERT INTO candidates (candidate_id, tenant_id, display_name) "
                    "VALUES (%s, '   ', %s)",
                    (str(uuid.uuid4()), "Whitespace Tenant"),
                )
