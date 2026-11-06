"""Opt-in against retained, disposable full-schema release fixture only."""

import os
from types import SimpleNamespace

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from scripts import spark_private_job as job

DSN = os.getenv("SPARK_PRIVATE_JOB_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="disposable private-job database not supplied")


@pytest.fixture
def db():
    fields = conninfo_to_dict(DSN)
    assert fields.get("host") in {"127.0.0.1", "localhost", "::1"}
    assert fields["dbname"].endswith("_test")
    with psycopg.connect(DSN) as conn:
        with conn.transaction(force_rollback=True):
            yield conn


def test_real_permission_catalogue(db):
    with db.cursor() as cur:
        job.permissions(cur, "activekg_app")


def test_unlisted_worker_function_refused(db):
    with db.cursor() as cur:
        cur.execute(
            "CREATE FUNCTION public.private_job_probe() RETURNS int LANGUAGE sql AS 'SELECT 1'"
        )
        cur.execute(
            "GRANT EXECUTE ON FUNCTION public.private_job_probe() TO activekg_network_worker"
        )
        with pytest.raises(ValueError):
            job.permissions(cur, "activekg_app")


def test_worker_schema_write_refused(db):
    with db.cursor() as cur:
        cur.execute("GRANT CREATE ON SCHEMA public TO activekg_network_worker")
        with pytest.raises(ValueError):
            job.permissions(cur, "activekg_app")


def test_real_registration_and_replay(db):
    with db.cursor() as cur:
        job.consent(cur, register=True)
        job.consent(cur, register=True)
        job.consent(cur)


def test_real_disabled_copy_refused(db):
    with db.cursor() as cur:
        job.consent(cur, register=True)
        cur.execute(
            "UPDATE public.candidate_consent_copy_catalog SET approved=false WHERE copy_version=2"
        )
        with pytest.raises(ValueError):
            job.consent(cur, register=True)


def test_readback_readonly_ledger(db):
    # The tested entrypoint owns its transaction; use a new read-only connection.
    with db.cursor() as cur:
        cur.execute("SELECT target_id::text FROM activekg_schema_control.target_identity")
        target = cur.fetchone()[0]
    with psycopg.connect(DSN) as conn:
        result = job.database_job(
            "readback",
            SimpleNamespace(target_id=target, environment="development"),
            conn,
            phase="pre",
        )
        assert result == (45, "045_candidate_network_recovery.sql")
