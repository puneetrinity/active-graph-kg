"""Offline refusal/registration tests; no production, provider or Railway calls."""

import pytest

from scripts import spark_private_job as job


def test_private_dsn_minimal_and_public_refusal():
    host = "memory.railway.internal"
    url = f"postgresql://owner:secret@{host}/memory"
    assert job.private_dsn(url, host)["sslmode"] == "disable"
    for bad in (
        "postgresql://owner:secret@public.example/memory",
        url + "?hostaddr=1.2.3.4",
        url + "?options=-csearch_path=evil",
        url + "?service=other",
        url + "?sslmode=prefer",
    ):
        with pytest.raises(ValueError):
            job.private_dsn(bad, host)


def test_arming_and_railway_binding():
    env = {
        "SPARK_PRIVATE_JOB_APPLY": "readback",
        "RAILWAY_PROJECT_ID": job.PROJECT,
        "RAILWAY_ENVIRONMENT_ID": job.ENVIRONMENT,
        "RAILWAY_DEPLOYMENT_ID": "deployment",
        "SPARK_PRIVATE_JOB_SERVICE_ID": "12345678-1234-1234-1234-123456789abc",
        "RAILWAY_SERVICE_ID": "12345678-1234-1234-1234-123456789abc",
        "ACTIVEKG_SCHEMA_ENVIRONMENT": "production",
        "SPARK_MEMORY_PRIVATE_HOST": "db.railway.internal",
        "ACTIVEKG_MIGRATE_DSN": "postgresql://owner:secret@db.railway.internal/db",
    }
    assert job.preflight("readback", env)
    for key in (
        "SPARK_PRIVATE_JOB_APPLY",
        "RAILWAY_PROJECT_ID",
        "RAILWAY_ENVIRONMENT_ID",
        "RAILWAY_SERVICE_ID",
        "RAILWAY_DEPLOYMENT_ID",
    ):
        with pytest.raises(ValueError):
            job.preflight("readback", {**env, key: ""})
    for key in ("ACTIVEKG_MIGRATION_APPLY", "ACTIVEKG_SCHEMA_FRESH_INIT", "NODE_OPTIONS"):
        with pytest.raises(ValueError):
            job.preflight("readback", {**env, key: "1"})


class Cursor:
    def __init__(self, row, owner=True):
        self.row, self.owner, self.queries = row, owner, []

    def execute(self, sql, args=None):
        self.queries.append(sql)
        if sql.startswith("INSERT"):
            self.row = (job.COPY_HASH, True)

    def fetchone(self):
        if "pg_get_userbyid" in self.queries[-1]:
            return (self.owner,)
        return self.row


def test_registration_insert_exact_replay_and_no_update():
    cur = Cursor(None)
    job.consent(cur, True)
    job.consent(cur, True)
    assert sum(q.startswith("INSERT") for q in cur.queries) == 1
    assert not any(q.startswith(("UPDATE", "DELETE")) for q in cur.queries)


@pytest.mark.parametrize("row", [(job.COPY_HASH, False), ("wrong", True)])
def test_registration_conflict_refuses_without_insert(row):
    cur = Cursor(row)
    with pytest.raises(ValueError):
        job.consent(cur, True)
    assert not any(q.startswith("INSERT") for q in cur.queries)


def test_registration_requires_actual_catalog_owner():
    with pytest.raises(ValueError):
        job.consent(Cursor(None, False), True)


def test_failure_never_prints_exception_or_secret(monkeypatch, capsys):
    monkeypatch.setattr(job.sys, "argv", ["private-job", "readback"])

    def fail(*args):
        raise RuntimeError("postgresql://owner:private-password@host/db candidate-person")

    monkeypatch.setattr(job, "preflight", fail)
    assert job.main() == 1
    assert capsys.readouterr().out == "SPARK_PRIVATE_JOB FAIL\n"
