from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from unittest.mock import patch

import psycopg
import pytest

from activekg.api import organization_decision_events as receiver

OWNER_DSN = os.getenv("ACTIVEKG_DECISION_INBOX_TEST_OWNER_DSN")
RUNTIME_DSN = os.getenv("ACTIVEKG_DECISION_INBOX_TEST_RUNTIME_DSN")

pytestmark = pytest.mark.skipif(
    not OWNER_DSN or not RUNTIME_DSN,
    reason="disposable organization-decision inbox PostgreSQL DSNs are not configured",
)


def _payload(
    event_id: str,
    *,
    organization_id: int,
    delivery_sequence: int,
    source_event_sequence: int,
    after_stage: int,
    reason_code: str | None = None,
) -> receiver.OrganizationDecisionEvent:
    return receiver.OrganizationDecisionEvent.model_validate_json(
        json.dumps(
            {
                "event_id": event_id,
                "delivery_sequence": delivery_sequence,
                "source_event_sequence": source_event_sequence,
                "organization_id": organization_id,
                "payload_schema_version": 1,
                "source_system": "flow",
                "subject_type": "application",
                "subject_id": 900000 + organization_id,
                "job_id": 800000 + organization_id,
                "action_code": "application_stage_moved",
                "taxonomy_version": 1,
                "rubric_id": None,
                "rubric_version": None,
                "rubric_approval_mode": None,
                "jd_digest_version": None,
                "recommendation_action": None,
                "reason_code": reason_code,
                "before_state": {"stage_id": None},
                "after_state": {"stage_id": after_stage},
                "occurred_at": datetime(2026, 9, 3, 12, tzinfo=timezone.utc).isoformat(),
            }
        )
    )


def _counts() -> tuple[int, int, int]:
    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT (SELECT count(*) FROM global_candidates),"
            "(SELECT count(*) FROM feedback_events),(SELECT count(*) FROM nodes)"
        )
        return tuple(int(value) for value in cur.fetchone())


def _cleanup() -> None:
    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM organization_decision_stream_state "
            "WHERE tenant_id IN ('org_930001','org_930002')"
        )
        cur.execute(
            "DELETE FROM organization_decision_event_inbox WHERE organization_id IN (930001,930002)"
        )


def test_force_rls_acl_idempotence_ordering_and_atomicity() -> None:
    event_a = "31000000-0000-4000-8000-000000000001"
    event_b = "31000000-0000-4000-8000-000000000002"
    event_gap = "31000000-0000-4000-8000-000000000003"
    event_regression = "31000000-0000-4000-8000-000000000004"
    _cleanup()
    untouched_before = _counts()
    try:
        with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT relname,relrowsecurity,relforcerowsecurity "
                "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname='public' AND relname IN "
                "('organization_decision_event_inbox','organization_decision_stream_state')"
            )
            assert {row[0]: row[1:] for row in cur.fetchall()} == {
                "organization_decision_event_inbox": (True, True),
                "organization_decision_stream_state": (True, True),
            }

        with psycopg.connect(RUNTIME_DSN) as conn, conn.cursor() as cur:
            for relation, expected in (
                ("organization_decision_event_inbox", (True, True, False, False, False)),
                ("organization_decision_stream_state", (True, True, True, False, False)),
            ):
                cur.execute(
                    "SELECT has_table_privilege(current_user,%s,'SELECT'),"
                    "has_table_privilege(current_user,%s,'INSERT'),"
                    "has_table_privilege(current_user,%s,'UPDATE'),"
                    "has_table_privilege(current_user,%s,'DELETE'),"
                    "has_table_privilege(current_user,%s,'TRUNCATE')",
                    (relation,) * 5,
                )
                assert cur.fetchone() == expected
            cur.execute("SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user")
            assert cur.fetchone() == (False, False)

        first = _payload(
            event_a,
            organization_id=930001,
            delivery_sequence=1001,
            source_event_sequence=1,
            after_stage=11,
        )
        with patch.dict("os.environ", {"ACTIVEKG_DSN": RUNTIME_DSN}, clear=False):
            assert receiver._store(first, "org_930001") == "inserted"
            with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT received_at,updated_at FROM organization_decision_event_inbox i "
                    "JOIN organization_decision_stream_state s USING (tenant_id) "
                    "WHERE i.event_id=%s",
                    (event_a,),
                )
                timestamps = cur.fetchone()
            assert receiver._store(first, "org_930001") == "replayed"
            with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT received_at,updated_at FROM organization_decision_event_inbox i "
                    "JOIN organization_decision_stream_state s USING (tenant_id) "
                    "WHERE i.event_id=%s",
                    (event_a,),
                )
                assert cur.fetchone() == timestamps

            changed = _payload(
                event_a,
                organization_id=930001,
                delivery_sequence=1001,
                source_event_sequence=1,
                after_stage=11,
                reason_code="other",
            )
            with pytest.raises(Exception) as conflict:
                receiver._store(changed, "org_930001")
            assert getattr(conflict.value, "status_code", None) == 409

            second_tenant = _payload(
                event_b,
                organization_id=930002,
                delivery_sequence=1002,
                source_event_sequence=2,
                after_stage=12,
            )
            assert receiver._store(second_tenant, "org_930002") == "inserted"
            gap = _payload(
                event_gap,
                organization_id=930001,
                delivery_sequence=1005,
                source_event_sequence=5,
                after_stage=13,
            )
            assert receiver._store(gap, "org_930001") == "inserted"
            regression = _payload(
                event_regression,
                organization_id=930001,
                delivery_sequence=1006,
                source_event_sequence=4,
                after_stage=14,
            )
            with pytest.raises(Exception) as sequence_conflict:
                receiver._store(regression, "org_930001")
            assert getattr(sequence_conflict.value, "status_code", None) == 409

        with psycopg.connect(RUNTIME_DSN) as conn, conn.cursor() as cur:
            cur.execute("SELECT set_config('app.current_tenant_id','org_930001',true)")
            cur.execute(
                "SELECT count(*) FROM organization_decision_event_inbox WHERE tenant_id='org_930002'"
            )
            assert cur.fetchone()[0] == 0
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                cur.execute("DELETE FROM organization_decision_event_inbox")
            conn.rollback()

        with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                "CREATE OR REPLACE FUNCTION decision_inbox_test_fail() RETURNS trigger "
                "LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'test'; END $$"
            )
            cur.execute(
                "CREATE TRIGGER decision_inbox_test_fail BEFORE INSERT OR UPDATE "
                "ON organization_decision_stream_state FOR EACH ROW "
                "EXECUTE FUNCTION decision_inbox_test_fail()"
            )
        failed_event = _payload(
            "31000000-0000-4000-8000-000000000005",
            organization_id=930002,
            delivery_sequence=1010,
            source_event_sequence=6,
            after_stage=15,
        )
        with patch.dict("os.environ", {"ACTIVEKG_DSN": RUNTIME_DSN}, clear=False):
            with pytest.raises(Exception) as unavailable:
                receiver._store(failed_event, "org_930002")
            assert getattr(unavailable.value, "status_code", None) == 503
        with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM organization_decision_event_inbox WHERE event_id=%s",
                (failed_event.event_id,),
            )
            assert cur.fetchone()[0] == 0
    finally:
        with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                "DROP TRIGGER IF EXISTS decision_inbox_test_fail "
                "ON organization_decision_stream_state"
            )
            cur.execute("DROP FUNCTION IF EXISTS decision_inbox_test_fail()")
        _cleanup()
    assert _counts() == untouched_before
