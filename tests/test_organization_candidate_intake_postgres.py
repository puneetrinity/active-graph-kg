from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

import psycopg
import pytest

from activekg.api import organization_candidates as receiver
from activekg.api.auth import JWTClaims

OWNER_DSN = os.getenv("ACTIVEKG_ORGANIZATION_CANDIDATE_TEST_OWNER_DSN")
RUNTIME_DSN = os.getenv("ACTIVEKG_ORGANIZATION_CANDIDATE_TEST_RUNTIME_DSN")

pytestmark = pytest.mark.skipif(
    not OWNER_DSN or not RUNTIME_DSN,
    reason="disposable organization-candidate PostgreSQL DSNs are not configured",
)


def _claims(tenant: str = "org_480001") -> JWTClaims:
    return JWTClaims(
        tenant_id=tenant,
        actor_id="vantahire-backend",
        actor_type="service",
        scopes=["organization-candidate:write"],
        issuer="vantahire",
    )


def _payload(application_id: int = 480001) -> receiver.OrganizationCandidateIntakeRequest:
    body: dict[str, object] = {
        "schema_version": 1,
        "reference_id": str(uuid4()),
        "application_id": application_id,
        "job_id": 4800,
        "resume_version_id": str(uuid4()),
        "resume_version": 1,
        "origin": "candidate_applied",
        "source_kind": "direct_upload",
        "source_resume_id": None,
        "source_observed_at": datetime.now(timezone.utc).isoformat(),
        "content_sha256": "a" * 64,
        "byte_count": 2048,
        "media_type": "application/pdf",
        "extracted_text_sha256": "b" * 64,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "idempotency_key": "0" * 64,
        "privacy_subject": [
            {"identifier_type": "vantahire_application_id", "value": str(application_id)},
            {"identifier_type": "email", "value": "private.fixture@example.test"},
        ],
    }
    provisional = receiver.OrganizationCandidateIntakeRequest.model_validate_json(json.dumps(body))
    body["idempotency_key"] = receiver.compute_idempotency_key(provisional, "org_480001")
    return receiver.OrganizationCandidateIntakeRequest.model_validate_json(json.dumps(body))


def _counts() -> tuple[int, int, int, int]:
    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT "
            "(SELECT count(*) FROM candidates WHERE tenant_id='org_480001'),"
            "(SELECT count(*) FROM organization_candidate_references),"
            "(SELECT count(*) FROM organization_candidate_resume_evidence),"
            "(SELECT count(*) FROM organization_candidate_ingest_receipts)"
        )
        return tuple(int(value) for value in cur.fetchone())


def test_private_intake_replay_conflict_rls_acl_and_append_only() -> None:
    assert OWNER_DSN and RUNTIME_DSN
    env = {
        "ACTIVEKG_DSN": RUNTIME_DSN,
        "CANDIDATE_PRIVACY_HMAC_KEY_V1": base64.b64encode(b"k" * 32).decode(),
        "CANDIDATE_PRIVACY_HMAC_ACTIVE_VERSION": "1",
    }
    with psycopg.connect(RUNTIME_DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user")
        assert cur.fetchone() == (False, False)
        for table in (
            "organization_candidate_references",
            "organization_candidate_resume_evidence",
            "organization_candidate_ingest_receipts",
        ):
            cur.execute(
                "SELECT has_table_privilege(current_user,%s,'SELECT'),"
                "has_table_privilege(current_user,%s,'INSERT'),"
                "has_table_privilege(current_user,%s,'UPDATE'),"
                "has_table_privilege(current_user,%s,'DELETE'),"
                "has_table_privilege(current_user,%s,'TRUNCATE'),"
                "has_table_privilege(current_user,%s,'REFERENCES'),"
                "has_table_privilege(current_user,%s,'TRIGGER')",
                (table,) * 7,
            )
            assert cur.fetchone() == (True, True, False, False, False, False, False)

    payload = _payload()
    before = _counts()
    with patch.dict(os.environ, env, clear=False):
        created = receiver._store(payload, _claims())
        assert created["delivery_status"] == "recorded"
        after = _counts()
        assert tuple(after[index] - before[index] for index in range(4)) == (1, 1, 1, 1)
        replay = receiver._store(payload, _claims())
        assert replay == {**created, "delivery_status": "replayed", "resolution": "replayed"}
        assert _counts() == after

        changed = payload.model_copy(update={"byte_count": payload.byte_count + 1})
        with pytest.raises(Exception) as conflict:
            receiver._store(changed, _claims())
        assert getattr(conflict.value, "status_code", None) == 409
        assert _counts() == after

    candidate_id = created["candidate_id"]
    with psycopg.connect(RUNTIME_DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT set_config('app.current_tenant_id','org_480001',true)")
        cur.execute(
            "SELECT scope,display_name,primary_email,primary_phone,global_candidate_id "
            "FROM candidates WHERE candidate_id=%s",
            (candidate_id,),
        )
        assert cur.fetchone() == ("organization_private", None, None, None, None)
        cur.execute("SELECT set_config('app.current_tenant_id','org_480002',true)")
        cur.execute(
            "SELECT count(*) FROM organization_candidate_references WHERE reference_id=%s",
            (payload.reference_id,),
        )
        assert cur.fetchone()[0] == 0

    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        for table in (
            "organization_candidate_references",
            "organization_candidate_resume_evidence",
            "organization_candidate_ingest_receipts",
        ):
            with pytest.raises(psycopg.Error) as mutation:
                cur.execute(f"UPDATE {table} SET created_at=created_at")
            assert mutation.value.sqlstate == "55000"
            conn.rollback()
        with pytest.raises(psycopg.Error) as truncation:
            cur.execute(
                "TRUNCATE organization_candidate_ingest_receipts, "
                "organization_candidate_resume_evidence, "
                "organization_candidate_references"
            )
        assert truncation.value.sqlstate == "55000"
        conn.rollback()
