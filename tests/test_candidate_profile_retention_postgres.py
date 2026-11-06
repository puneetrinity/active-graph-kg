"""Erasure and expiry regressions against migration 041; synthetic data only."""

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from activekg.candidate_profiles.profile_contracts import canonical
from tests import test_candidate_profile_drafts_postgres as storage_tests
from tests.test_candidate_profile_drafts_postgres import (
    FACTS,
    PREFS,
    apply,
    command,
    identity,
    read,
)

# Reuse the existing disposable S1 + S2 setup, not replacement hand-written tables.
database = storage_tests.database
db = storage_tests.db

pytestmark = storage_tests.pytestmark


def erase(db, who, request=None):
    request = request or str(uuid4())
    return db.execute(
        "SELECT candidate_profile_erase(%s,%s,%s,'candidate') result", (*who, request)
    ).fetchone()["result"]


def expire(db, who, command_id, request):
    digest = hashlib.sha256(canonical(request).encode()).hexdigest()
    routine = (
        "candidate_profile_expire_v3"
        if request["schemaVersion"] == 3
        else "candidate_profile_expire_command"
    )
    return db.execute(
        sql.SQL("SELECT {}(%s,%s,%s,%s,%s) result").format(sql.Identifier(routine)),
        (*who, command_id, request["idempotencyKey"], digest),
    ).fetchone()["result"]


def v3_request():
    return {
        **command(
            "save_preference_section",
            {"section": "employment_types", "answered": True, "value": ["permanent"]},
        ),
        "schemaVersion": 3,
        "expectedPreferenceDraftRevision": 0,
    }


def apply_versioned(db, who, request, cid):
    if request["schemaVersion"] == 2:
        return apply(db, who, request, cid)
    wire = canonical(request)
    return db.execute(
        "SELECT candidate_preference_draft_apply(%s,%s,%s,%s,%s) result",
        (*who, cid, hashlib.sha256(wire.encode()).hexdigest(), wire),
    ).fetchone()["result"]


@pytest.mark.parametrize("saved", [False, True])
def test_v3_expiry_recovers_receipt_or_blocks_late_save(db, saved):
    from activekg.candidate_profiles.spark_contracts import receipt as validate_receipt

    who, request, cid = identity(db), v3_request(), str(uuid4())
    original = apply_versioned(db, who, request, cid) if saved else None
    result = expire(db, who, cid, request)
    validate_receipt(result)
    assert result["schemaVersion"] == 3
    assert result["status"] == ("applied" if saved else "rejected")
    assert result["preferenceDraftRevision"] == (1 if saved else 0)
    assert result["reason"] == (None if saved else "command_expired")
    if saved:
        assert result == original
    assert expire(db, who, cid, request) == result
    assert apply_versioned(db, who, request, cid) == result
    with pytest.raises(psycopg.Error, match="not_found"):
        expire(db, identity(db), cid, request)
    with pytest.raises(psycopg.Error, match="idempotency_conflict"):
        expire(db, who, cid, v3_request())
    erase(db, who)
    with pytest.raises(psycopg.Error, match="candidate_privacy_restricted"):
        expire(db, who, cid, request)


def test_expiry_returns_a_lost_applied_receipt_instead_of_claiming_failure(db):
    who = identity(db)
    request = command("save_draft", {"profile": FACTS})
    cid = str(uuid4())
    applied = apply(db, who, request, cid)
    assert expire(db, who, cid, request) == applied
    assert read(db, who)["draft"] == FACTS


def test_expiry_tombstone_blocks_late_delivery_but_not_new_commands(db):
    who = identity(db)
    request = command("save_draft", {"profile": FACTS})
    cid = str(uuid4())
    expired = expire(db, who, cid, request)
    assert expired["status"] == "rejected" and expired["reason"] == "command_expired"
    assert expire(db, who, cid, request) == expired
    assert apply(db, who, request, cid) == expired
    assert read(db, who)["draft"] is None
    assert apply(db, who, command("save_draft", {"profile": FACTS}))["status"] == "applied"


def test_expiry_checks_identity_key_and_hash_and_never_restores_erased_hashes(db):
    one, two = identity(db), identity(db)
    request = command("save_draft", {"profile": FACTS})
    cid = str(uuid4())
    expire(db, one, cid, request)
    with pytest.raises(psycopg.Error, match="not_found"):
        expire(db, two, cid, request)
    with pytest.raises(psycopg.Error, match="idempotency_conflict"):
        expire(db, one, cid, command("save_draft", {"profile": FACTS}))
    erase(db, one)
    with pytest.raises(psycopg.Error, match="candidate_privacy_restricted"):
        expire(db, one, cid, request)
    assert (
        db.execute(
            "SELECT wire_hash FROM candidate_profile_receipts WHERE command_id=%s", (cid,)
        ).fetchone()["wire_hash"]
        is None
    )


def test_erasure_blanks_all_manual_payloads_hashes_and_prevents_replay(db):
    who = identity(db)
    original = command("save_draft", {"profile": FACTS})
    cid = str(uuid4())
    apply(db, who, original, cid)
    apply(db, who, command("save_preferences", {"preferences": PREFS}))
    apply(
        db,
        who,
        command(
            "confirm",
            {
                "draftHash": hashlib.sha256(canonical(FACTS).encode()).hexdigest(),
                "consentVersion": 0,
            },
            1,
            0,
            1,
        ),
    )
    before = read(db, who)
    request = str(uuid4())
    assert erase(db, who, request) == {"requestId": request, "status": "erased"}
    # Rehearse an older payload-table restore while the erasure ledger remains.
    # Reads stay fenced; reapplying the same approved erasure scrubs it again.
    db.execute(
        "UPDATE candidate_profile_drafts SET facts=%s::jsonb,document_sha256=%s WHERE subject_id=%s",
        (canonical(FACTS), "a" * 64, who[0]),
    )
    assert erase(db, who, request)["status"] == "erased"
    for table, columns in (
        ("candidate_profile_drafts", ["facts", "document_sha256"]),
        ("candidate_profile_versions", ["facts", "document_sha256"]),
        ("candidate_preference_versions", ["preferences"]),
    ):
        rows = db.execute(
            sql.SQL("SELECT {} FROM {} WHERE subject_id=%s").format(
                sql.SQL(",").join(map(sql.Identifier, columns)), sql.Identifier(table)
            ),
            (who[0],),
        ).fetchall()
        assert rows and all(all(v is None for v in row.values()) for row in rows)
    receipts = db.execute(
        "SELECT wire_hash,result FROM candidate_profile_receipts WHERE subject_id=%s", (who[0],)
    ).fetchall()
    assert len(receipts) == 3
    assert all(r["wire_hash"] is None and "commandHash" not in r["result"] for r in receipts)
    head = db.execute(
        "SELECT * FROM candidate_profile_heads WHERE subject_id=%s", (who[0],)
    ).fetchone()
    assert head["restricted"] is True
    assert (head["draft_version"], head["profile_version"], head["preference_version"]) == (
        before["draftVersion"],
        before["profileVersion"],
        before["preferenceVersion"],
    )
    with pytest.raises(psycopg.Error, match="candidate_privacy_restricted"):
        apply(db, who, original, cid)
    with pytest.raises(psycopg.Error, match="candidate_privacy_restricted"):
        apply(db, who, command("save_draft", {"profile": FACTS}, 1, 1, 1))
    with pytest.raises(psycopg.Error, match="candidate_privacy_restricted"):
        read(db, who)
    assert (
        db.execute(
            "SELECT count(*) n FROM candidate_profile_erasure_audit WHERE request_id=%s", (request,)
        ).fetchone()["n"]
        == 1
    )


def test_erasure_before_first_save_blocks_delayed_first_write(db):
    who = identity(db)
    erase(db, who)
    with pytest.raises(psycopg.Error, match="candidate_privacy_restricted"):
        apply(db, who, command("save_draft", {"profile": FACTS}))


def test_wrong_binding_or_request_owner_never_erases_another_subject(db):
    one, two = identity(db), identity(db)
    for who in (one, two):
        apply(db, who, command("save_draft", {"profile": FACTS}))
    with pytest.raises(psycopg.Error, match="not_found"):
        erase(db, (one[0], two[1]))
    assert read(db, one)["draft"] == FACTS
    request = str(uuid4())
    erase(db, one, request)
    with pytest.raises(psycopg.Error, match="not_found"):
        erase(db, two, request)
    assert read(db, two)["draft"] == FACTS


def test_ordinary_runtime_cannot_erase_or_read_the_audit(db):
    who = identity(db)
    role = "s2_erase_test_" + uuid4().hex
    with db.transaction():
        db.execute(
            sql.SQL("CREATE ROLE {} NOLOGIN NOSUPERUSER NOBYPASSRLS").format(sql.Identifier(role))
        )
        db.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(role)))
        for query, args in (
            ("SELECT candidate_profile_erase(%s,%s,%s,'candidate')", (*who, str(uuid4()))),
            ("SELECT * FROM candidate_profile_erasure_audit", ()),
            (
                "SELECT candidate_profile_expire_v3(%s,%s,%s,%s,%s)",
                (*who, str(uuid4()), str(uuid4()), "a" * 64),
            ),
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege), db.transaction():
                db.execute(query, args)
        db.execute("RESET ROLE")
        db.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))


@pytest.mark.parametrize("erase_first", [False, True])
@pytest.mark.parametrize("operation", ["erase", "expire", "expire_v3"])
def test_real_concurrent_save_and_retention_share_the_same_fence(db, erase_first, operation):
    who = identity(db)
    request = (
        v3_request() if operation == "expire_v3" else command("save_draft", {"profile": FACTS})
    )
    cid = str(uuid4())
    started = Queue()

    def retain(connection):
        return (
            erase(connection, who)
            if operation == "erase"
            else expire(connection, who, cid, request)
        )

    def competing_operation():
        with psycopg.connect(db.info.dsn, autocommit=True, row_factory=dict_row) as other:
            started.put(other.info.backend_pid)
            try:
                if erase_first:
                    return apply_versioned(other, who, request, cid)
                return retain(other)
            except psycopg.Error as error:
                return {"refused": error.diag.message_primary}

    with ThreadPoolExecutor(max_workers=1) as executor:
        with db.transaction():
            if erase_first:
                retain(db)
            else:
                apply_versioned(db, who, request, cid)
            pending = executor.submit(competing_operation)
            pid = started.get(timeout=2)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                # Clear the per-transaction activity snapshot on each poll.
                db.execute("SELECT pg_stat_clear_snapshot()")
                state = db.execute(
                    "SELECT wait_event_type FROM pg_stat_activity WHERE pid=%s", (pid,)
                ).fetchone()
                if state and state["wait_event_type"] == "Lock":
                    break
                time.sleep(0.01)
            else:
                pytest.fail("competing connection did not wait on the erasure/save fence")
            assert not pending.done()
        outcome = pending.result(timeout=5)
    if operation == "erase":
        assert outcome == (
            {"refused": "candidate_privacy_restricted"}
            if erase_first
            else {"requestId": outcome["requestId"], "status": "erased"}
        )
    else:
        assert outcome["status"] == ("rejected" if erase_first else "applied")
        assert outcome["reason"] == ("command_expired" if erase_first else None)
    assert db.execute(
        "SELECT count(*) n FROM candidate_profile_drafts WHERE subject_id=%s AND facts IS NOT NULL",
        (who[0],),
    ).fetchone()["n"] == (1 if operation == "expire" and not erase_first else 0)
