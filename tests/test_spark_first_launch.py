"""Offline first-launch refusals; never contacts a database or provider."""

import pytest

from scripts import spark_first_launch as launch
from scripts import spark_private_job as job


def env():
    return {
        "CANDIDATE_NETWORK_RECOVERY_EPOCH": "first_launch_test_20261106",
        "SPARK_RECOVERY_RUN_ID": "12345678-1234-1234-1234-123456789abc",
        "FLOW_SCHEMA_TARGET_ID": "22345678-1234-1234-1234-123456789abc",
        "SPARK_FIRST_LAUNCH_CONFIRM": "never_activated_no_restore",
    }


def test_requires_explicit_first_launch_not_restore():
    assert launch.launch_input(env())
    for key, value in (
        ("SPARK_FIRST_LAUNCH_CONFIRM", ""),
        ("SPARK_RECOVERY_RESTORED", "memory"),
        ("SPARK_RECOVERY_RUN_ID", "bad"),
        ("CANDIDATE_NETWORK_RECOVERY_EPOCH", "short"),
        ("FLOW_SCHEMA_TARGET_ID", ""),
    ):
        with pytest.raises(ValueError):
            launch.launch_input({**env(), key: value})


def tls_env():
    return {
        "FLOW_RECOVERY_DATABASE_URL": "postgresql://owner:secret@proxy.example:5433/flow",
        "FLOW_RECOVERY_HOST": "proxy.example",
        "FLOW_RECOVERY_TLS_SERVERNAME": "database.example",
    }


def resolve(*args, **kwargs):
    return [(None, None, None, None, ("192.0.2.1", 5433))]


def test_tls_verifies_pinned_name_not_proxy_and_never_downgrades():
    settings = launch.flow_tls_settings(tls_env(), "/test-ca.pem", resolve)
    assert settings["host"] == "database.example"
    assert settings["hostaddr"] == "192.0.2.1"
    assert settings["sslmode"] == "verify-full"
    assert settings["sslrootcert"] == "/test-ca.pem"
    for suffix in (
        "?sslmode=disable",
        "?sslmode=require",
        "?hostaddr=192.0.2.2",
        "?options=-csearch_path=bad",
    ):
        with pytest.raises(ValueError):
            launch.flow_tls_settings(
                {
                    **tls_env(),
                    "FLOW_RECOVERY_DATABASE_URL": tls_env()["FLOW_RECOVERY_DATABASE_URL"] + suffix,
                },
                "/test-ca.pem",
                resolve,
            )
    with pytest.raises(ValueError):
        launch.flow_tls_settings(
            {**tls_env(), "FLOW_RECOVERY_HOST": "other.example"}, "/test-ca.pem", resolve
        )


def test_first_launch_failure_output_is_fixed(monkeypatch, capsys):
    monkeypatch.setattr(job.sys, "argv", ["private-job", "first-launch"])

    def fail(*args):
        raise RuntimeError("secret owner credential and candidate data")

    monkeypatch.setattr(job, "preflight", fail)
    assert job.main() == 1
    assert capsys.readouterr().out == "SPARK_PRIVATE_JOB FAIL\n"


def test_empty_refuses_any_row_in_every_listed_table():
    class Cursor:
        def execute(self, query):
            pass

        def fetchone(self):
            return (True,)

    for table in (*launch.FLOW_TABLES, *launch.MEMORY_TABLES):
        with pytest.raises(ValueError):
            launch.empty(Cursor(), (table,))
