"""Opt-in full-schema proof. Creates and retains isolated local clones only."""

import os
import uuid
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from scripts import spark_first_launch as launch
from scripts import spark_private_job as job

ADMIN = os.getenv("SPARK_FIRST_LAUNCH_TEST_ADMIN")
pytestmark = pytest.mark.skipif(not ADMIN, reason="local two-database templates not supplied")


@pytest.fixture
def estate():
    fields = conninfo_to_dict(ADMIN)
    assert fields["host"] in {"127.0.0.1", "localhost", "::1"}
    templates = [
        os.environ["SPARK_FIRST_LAUNCH_MEMORY_TEMPLATE"],
        os.environ["SPARK_FIRST_LAUNCH_FLOW_TEMPLATE"],
    ]
    assert all(t.endswith("_test") or t.startswith("flow_schema_control_test_") for t in templates)
    names = ["spark_launch_" + uuid.uuid4().hex[:12] + "_test" for _ in templates]
    with psycopg.connect(ADMIN, autocommit=True) as admin:
        for name, template in zip(names, templates, strict=True):
            admin.execute(
                sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
                    sql.Identifier(name), sql.Identifier(template)
                )
            )
    dsns = [{**fields, "dbname": name} for name in names]
    target = str(uuid.uuid4())
    for index, (dsn, tables) in enumerate(
        zip(dsns, [launch.MEMORY_TABLES, launch.FLOW_TABLES], strict=True)
    ):
        # Clean only newly created clones; source databases remain untouched.
        with psycopg.connect(**dsn) as conn:
            conn.execute("SET LOCAL session_replication_role=replica")
            extras = (
                [
                    "candidate_consent_state",
                    "candidate_consent_receipts",
                    "candidate_network_recovery_attestations",
                ]
                if index == 0
                else ["candidate_consent_events", "candidate_consent_outbox"]
            )
            conn.execute(
                sql.SQL("TRUNCATE {} CASCADE").format(
                    sql.SQL(",").join(sql.Identifier("public", t) for t in (*tables, *extras))
                )
            )
            if index == 0:
                with conn.cursor() as cur:
                    job.consent(cur, register=True)
            else:
                conn.execute(
                    sql.SQL("ALTER TABLE public.candidate_product_accounts OWNER TO {}").format(
                        sql.Identifier(fields["user"])
                    )
                )
                conn.execute("UPDATE schema_control.identity SET target_id=%s", (target,))
    yield dsns, target


@pytest.fixture
def pair(estate):
    dsns, target = estate
    with psycopg.connect(**dsns[0]) as memory, psycopg.connect(**dsns[1]) as flow:
        with memory.transaction(force_rollback=True), flow.transaction(force_rollback=True):
            row = memory.execute(
                "SELECT target_id::text,environment FROM activekg_schema_control.target_identity"
            ).fetchone()
            config = SimpleNamespace(target_id=row[0], environment=row[1])
            env = {
                "CANDIDATE_NETWORK_RECOVERY_EPOCH": "first_launch_test_20261106",
                "SPARK_RECOVERY_RUN_ID": str(uuid.uuid4()),
                "FLOW_SCHEMA_TARGET_ID": target,
                "SPARK_FIRST_LAUNCH_CONFIRM": "never_activated_no_restore",
            }
            yield memory, flow, config, env


def test_empty_opens_and_exact_replay_only(pair):
    memory, flow, config, env = pair
    launch.first_launch(*pair)
    launch.first_launch(*pair)
    assert memory.execute(
        "SELECT count(*) FROM candidate_network_recovery_attestations"
    ).fetchone() == (1,)
    with pytest.raises(ValueError):
        launch.first_launch(
            memory, flow, config, {**env, "SPARK_RECOVERY_RUN_ID": str(uuid.uuid4())}
        )


def test_nonempty_memory_refuses_even_on_replay(pair):
    memory = pair[0]
    launch.first_launch(*pair)
    memory.execute(
        "INSERT INTO candidate_owned_subjects(subject_id,flow_binding_id) VALUES(%s,%s)",
        (uuid.uuid4(), uuid.uuid4()),
    )
    with pytest.raises(ValueError):
        launch.first_launch(*pair)


def test_nonempty_flow_refuses(pair):
    memory, flow, config, env = pair
    flow.execute("SET LOCAL session_replication_role=replica")
    flow.execute(
        "INSERT INTO candidate_product_accounts(user_id,subject_id,binding_id) VALUES(987654,%s,%s)",
        (uuid.uuid4(), uuid.uuid4()),
    )
    flow.execute("SET LOCAL session_replication_role=origin")
    with pytest.raises(ValueError):
        launch.first_launch(*pair)
    assert memory.execute(
        "SELECT count(*) FROM candidate_network_recovery_attestations"
    ).fetchone() == (0,)


def test_wrong_flow_target_refuses(pair):
    memory, flow, config, env = pair
    with pytest.raises(ValueError):
        launch.first_launch(
            memory, flow, config, {**env, "FLOW_SCHEMA_TARGET_ID": str(uuid.uuid4())}
        )


def test_flow_writes_locked_until_memory_attested(pair, estate, monkeypatch):
    original = launch.empty
    checked = []

    def observed(cur, tables):
        original(cur, tables)
        if tables == launch.FLOW_TABLES:
            with psycopg.connect(**estate[0][1]) as writer:
                writer.execute("SET LOCAL lock_timeout='100ms'")
                with pytest.raises(psycopg.errors.LockNotAvailable):
                    writer.execute("LOCK TABLE candidate_product_accounts IN ROW EXCLUSIVE MODE")
            checked.append(True)

    monkeypatch.setattr(launch, "empty", observed)
    launch.first_launch(*pair)
    assert checked == [True]


def test_runtime_cannot_prove_unfiltered_emptiness(pair):
    memory = pair[0]
    with memory.transaction():
        memory.execute("SET LOCAL ROLE activekg_app")
        with memory.cursor() as cur, pytest.raises(ValueError):
            launch.owner_and_unfiltered(cur, "candidate_owned_subjects")


@pytest.mark.parametrize("schema", [1, 2])
def test_legacy_consent_preserved_but_v2_refused(pair, schema):
    memory = pair[0]
    subject = uuid.uuid4()
    memory.execute(
        "INSERT INTO candidate_consent_state(subject_id,tenant_id,effective_schema_version) VALUES(%s,%s,%s)",
        (subject, "candidate_" + str(subject), schema),
    )
    if schema == 2:
        with pytest.raises(ValueError):
            launch.first_launch(*pair)
    else:
        launch.first_launch(*pair)
    assert memory.execute(
        "SELECT effective_schema_version FROM candidate_consent_state WHERE subject_id=%s",
        (subject,),
    ).fetchone() == (schema,)


def test_bad_flow_ledger_refused(pair):
    pair[1].execute("SET LOCAL session_replication_role=replica")
    pair[1].execute(
        "UPDATE schema_control.applied SET checksum=repeat('0',64) WHERE version='0030'"
    )
    with pytest.raises(ValueError):
        launch.first_launch(*pair)


def test_attestation_commits_and_survives_connection_restart(estate):
    dsns, target = estate
    env = {
        "CANDIDATE_NETWORK_RECOVERY_EPOCH": "first_launch_test_20261106",
        "SPARK_RECOVERY_RUN_ID": str(uuid.uuid4()),
        "FLOW_SCHEMA_TARGET_ID": target,
        "SPARK_FIRST_LAUNCH_CONFIRM": "never_activated_no_restore",
    }
    with (
        psycopg.connect(**dsns[0], autocommit=True) as memory,
        psycopg.connect(**dsns[1], autocommit=True) as flow,
    ):
        row = memory.execute(
            "SELECT target_id::text,environment FROM activekg_schema_control.target_identity"
        ).fetchone()
        config = SimpleNamespace(target_id=row[0], environment=row[1])
        launch.first_launch(memory, flow, config, env)
    with psycopg.connect(**dsns[0]) as memory, psycopg.connect(**dsns[1]) as flow:
        launch.first_launch(memory, flow, config, env)
        assert memory.execute(
            "SELECT candidate_network_recovery_open(%s)", (env["CANDIDATE_NETWORK_RECOVERY_EPOCH"],)
        ).fetchone() == (True,)
