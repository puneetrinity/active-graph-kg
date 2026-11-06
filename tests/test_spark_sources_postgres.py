"""Private source ingestion against real 041 and restricted runtime permissions."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import psycopg
import pytest

from activekg.candidate_profiles.spark_contracts import BUNDLED_PURPOSES, old
from tests import test_spark_postgres as storage_tests
from tests.test_spark_sources import cases, fingerprint, fixture, lookup_fixture, rehash

spark_db = storage_tests.spark_db
spark_runtime_dsn = storage_tests.spark_runtime_dsn


def test_interpretation_storage_is_private_idempotent_and_consent_fenced(
    spark_db, source, spark_runtime_dsn
):
    from tests.test_spark_contracts import interpretation_transfer

    value = interpretation_transfer()
    value.update(subjectId=source["subjectId"], bindingId=source["bindingId"])
    proposal = value["proposal"]
    proposal["expiresAt"] = (
        (datetime.now(UTC) + timedelta(days=1)).isoformat().replace("+00:00", "Z")
    )
    proposal["proposalHash"] = fingerprint(
        {key: item for key, item in proposal.items() if key != "proposalHash"}
    )

    def deliver():
        return spark_db.execute(
            "SELECT candidate_profile_interpretation_receive(%s,%s,%s,%s,%s)",
            (
                value["subjectId"],
                value["bindingId"],
                value["commandId"],
                fingerprint(value),
                old.canonical(value),
            ),
        ).fetchone()[0]

    receipt = deliver()
    assert receipt["outcome"] == "stored"
    assert deliver() == receipt
    view = spark_db.execute(
        "SELECT candidate_profile_v3_storage_read(%s,%s)",
        (value["subjectId"], value["bindingId"]),
    ).fetchone()[0]
    assert view["proposals"][0]["proposalId"] == proposal["proposalId"]
    assert set(view["proposals"][0]) == {
        "proposalId",
        "proposalHash",
        "kind",
        "sourceId",
        "sourceVersion",
        "expiresAt",
    }
    assert view["hasMoreProposals"] is False
    assert (
        spark_db.execute(
            "SELECT candidate_profile_interpretation_read(%s,%s,%s)",
            (value["subjectId"], value["bindingId"], proposal["proposalId"]),
        ).fetchone()[0]
        == proposal
    )
    assert spark_db.execute(
        "SELECT draft_version,profile_version,preference_version FROM candidate_profile_heads WHERE subject_id=%s",
        (value["subjectId"],),
    ).fetchone() == (0, 0, 0)
    with psycopg.connect(spark_runtime_dsn, autocommit=True) as runtime:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            runtime.execute("SELECT * FROM candidate_profile_interpretations")
        assert (
            runtime.execute(
                "SELECT candidate_profile_interpretation_read(%s,%s,%s)",
                (value["subjectId"], value["bindingId"], proposal["proposalId"]),
            ).fetchone()[0]
            == proposal
        )
    changed = dict(value, commandId=str(uuid4()))
    changed["proposal"] = dict(proposal, expectedDraftVersion=1)
    changed["proposal"]["proposalHash"] = fingerprint(
        {key: item for key, item in changed["proposal"].items() if key != "proposalHash"}
    )
    stale = spark_db.execute(
        "SELECT candidate_profile_interpretation_receive(%s,%s,%s,%s,%s)",
        (
            changed["subjectId"],
            changed["bindingId"],
            changed["commandId"],
            fingerprint(changed),
            old.canonical(changed),
        ),
    ).fetchone()[0]
    assert stale["reason"] == "stale_edit"
    with pytest.raises(psycopg.errors.UniqueViolation):
        spark_db.execute(
            "SELECT candidate_profile_interpretation_receive(%s,%s,%s,%s,%s)",
            (
                value["subjectId"],
                value["bindingId"],
                value["commandId"],
                fingerprint(dict(value, consentRevision=2)),
                old.canonical(dict(value, consentRevision=2)),
            ),
        )
    accept = {
        "schemaVersion": 3,
        "idempotencyKey": str(uuid4()),
        "action": "accept_interpretation",
        "expectedDraftVersion": 0,
        "expectedProfileVersion": 0,
        "expectedPreferenceVersion": 0,
        "expectedPreferenceDraftRevision": 0,
        "payload": {"proposalId": proposal["proposalId"], "proposalHash": proposal["proposalHash"]},
    }
    command_id = str(uuid4())
    args = (
        value["subjectId"],
        value["bindingId"],
        command_id,
        fingerprint(accept),
        old.canonical(accept),
    )
    applied = spark_db.execute(
        "SELECT candidate_profile_interpretation_apply(%s,%s,%s,%s,%s)",
        args,
    ).fetchone()[0]
    assert applied["status"] == "applied"
    assert applied["preferenceDraftRevision"] == 1
    assert applied["publicationActive"] is False
    assert (
        spark_db.execute(
            "SELECT candidate_profile_v3_storage_read(%s,%s)",
            (value["subjectId"], value["bindingId"]),
        ).fetchone()[0]["proposals"]
        == []
    )
    assert (
        spark_db.execute(
            "SELECT candidate_profile_interpretation_apply(%s,%s,%s,%s,%s)",
            args,
        ).fetchone()[0]
        == applied
    )
    assert spark_db.execute(
        "SELECT sections->'arrangement' FROM candidate_preference_drafts WHERE subject_id=%s",
        (value["subjectId"],),
    ).fetchone()[0] == {
        "answered": True,
        "value": {"workArrangements": ["onsite"], "remoteRegions": None},
    }
    consent(spark_db, value, "withdraw", 2)
    value["commandId"] = str(uuid4())
    rejected = deliver()
    assert rejected["outcome"] == "rejected"
    assert rejected["reason"] == "consent_changed"
    spark_db.execute(
        "UPDATE candidate_profile_heads SET restricted=true WHERE subject_id=%s",
        (value["subjectId"],),
    )
    assert spark_db.execute(
        "SELECT payload,proposal_hash,state FROM candidate_profile_interpretations WHERE proposal_id=%s",
        (proposal["proposalId"],),
    ).fetchone() == (None, None, "erased")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        deliver()


@pytest.fixture
def source(spark_db):
    value = fixture()
    value.update(subjectId=str(uuid4()), bindingId=str(uuid4()), commandId=str(uuid4()))
    value["proposal"]["proposalId"] = str(uuid4())
    value["proposal"]["expiresAt"] = (
        (datetime.now(UTC) + timedelta(days=1)).isoformat().replace("+00:00", "Z")
    )
    spark_db.execute(
        "SELECT candidate_owned_ensure(%s,%s,%s,1,%s)",
        (value["subjectId"], value["bindingId"], str(uuid4()), "a" * 64),
    )
    spark_db.execute(
        "INSERT INTO candidate_consent_copy_catalog VALUES(1,%s,true) ON CONFLICT DO NOTHING",
        ("a" * 64,),
    )
    consent(spark_db, value, "grant", 1)
    return rehash(value)


@pytest.mark.parametrize("withdraw", [False, True])
@pytest.mark.parametrize("whole_lookup", [False, True])
def test_interpretation_accepts_owned_source_correction_only_with_current_consent(
    spark_db, source, withdraw, whole_lookup
):
    from tests.test_spark_contracts import interpretation_transfer

    if whole_lookup:
        template = source["proposal"]["facts"][0]
        source["proposal"]["facts"] = [
            dict(template, factId=str(uuid4()), kind="displayName", value="Priya Rao"),
            dict(template, factId=str(uuid4()), kind="skill", value="Go"),
            dict(
                template,
                factId=str(uuid4()),
                kind="employment",
                value={
                    "id": str(uuid4()),
                    "employer": "Freshworks",
                    "title": "Engineer",
                    "description": "Built services",
                    "current": False,
                    "start": {"value": "2019", "precision": "year"},
                    "end": {"value": "2022", "precision": "year"},
                },
            ),
        ]
        source = lookup_fixture(source)
    assert receive(spark_db, source)["outcome"] == "stored"
    value = interpretation_transfer()
    value.update(
        subjectId=source["subjectId"],
        bindingId=source["bindingId"],
        use="source_correction",
        sourceProposalId=source["proposal"]["proposalId"],
        sourceVersion=1,
    )
    p = value["proposal"]
    fact = dict(source["proposal"]["facts"][0], value="Golang")
    if whole_lookup:
        original = source["proposal"]["facts"][2]
        fact = dict(
            original, value={**original["value"], "end": {"value": "2021", "precision": "year"}}
        )
    p.update(
        changes=[{"kind": "source_correction", "fact": fact}],
        expiresAt=source["proposal"]["expiresAt"],
    )
    p["proposalHash"] = fingerprint({k: v for k, v in p.items() if k != "proposalHash"})
    assert (
        spark_db.execute(
            "SELECT candidate_profile_interpretation_receive(%s,%s,%s,%s,%s)",
            (
                value["subjectId"],
                value["bindingId"],
                value["commandId"],
                fingerprint(value),
                old.canonical(value),
            ),
        ).fetchone()[0]["outcome"]
        == "stored"
    )
    if withdraw:
        consent(spark_db, value, "withdraw", 2)
    command = source_command(source)
    command.update(
        action="accept_interpretation",
        payload={"proposalId": p["proposalId"], "proposalHash": p["proposalHash"]},
    )
    args = (
        value["subjectId"],
        value["bindingId"],
        str(uuid4()),
        fingerprint(command),
        old.canonical(command),
    )
    result = spark_db.execute(
        "SELECT candidate_profile_interpretation_apply(%s,%s,%s,%s,%s)", args
    ).fetchone()[0]
    assert result["status"] == ("rejected" if withdraw else "applied")
    assert result["reason"] == ("consent_changed" if withdraw else None)
    assert result["draftVersion"] == (0 if withdraw else 1)
    assert (
        spark_db.execute(
            "SELECT candidate_profile_interpretation_apply(%s,%s,%s,%s,%s)", args
        ).fetchone()[0]
        == result
    )
    if not withdraw and not whole_lookup:
        assert spark_db.execute(
            "SELECT facts->'skills' FROM candidate_profile_drafts WHERE subject_id=%s AND version=1",
            (value["subjectId"],),
        ).fetchone()[0] == ["Golang"]
    if whole_lookup:
        if withdraw:
            assert (
                spark_db.execute(
                    "SELECT count(*) FROM candidate_profile_drafts WHERE subject_id=%s",
                    (value["subjectId"],),
                ).fetchone()[0]
                == 0
            )
        else:
            draft = spark_db.execute(
                "SELECT facts FROM candidate_profile_drafts WHERE subject_id=%s AND version=1",
                (value["subjectId"],),
            ).fetchone()[0]
            assert draft["displayName"] == "Priya Rao"
            assert draft["skills"] == ["Go"]
            assert len(draft["employment"]) == 1
            job = draft["employment"][0]
            assert {k: v for k, v in job.items() if k != "id"} == {
                k: v for k, v in fact["value"].items() if k != "id"
            }
            state, acceptance = spark_db.execute(
                "SELECT state,acceptance FROM candidate_profile_proposals WHERE proposal_id=%s",
                (source["proposal"]["proposalId"],),
            ).fetchone()
            assert state == "accepted"
            assert set(acceptance["selection"]["selectedFactIds"]) == {
                f["factId"] for f in source["proposal"]["facts"]
            }
            assert len(acceptance["selection"]["corrections"]) == 1
            assert (
                spark_db.execute(
                    "SELECT count(*) FROM candidate_profile_drafts WHERE subject_id=%s",
                    (value["subjectId"],),
                ).fetchone()[0]
                == 1
            )


def consent(db, value, action, version):
    event = str(uuid4())
    command = {
        "schemaVersion": 2,
        "subjectId": value["subjectId"],
        "bindingId": value["bindingId"],
        "eventId": event,
        "version": version,
        "action": action,
        "copyVersion": 1 if action == "grant" else None,
        "copyHash": "a" * 64 if action == "grant" else None,
        "purposes": BUNDLED_PURPOSES if action == "grant" else [],
    }
    return db.execute(
        "SELECT candidate_consent_v2_project(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (
            value["subjectId"],
            value["bindingId"],
            event,
            version,
            action,
            1 if action == "grant" else None,
            "a" * 64 if action == "grant" else None,
            BUNDLED_PURPOSES if action == "grant" else [],
            fingerprint(command),
        ),
    ).fetchone()[0]


def receive(db, value, **overrides):
    args = {
        "subject": value["subjectId"],
        "binding": value["bindingId"],
        "command": value["commandId"],
        "digest": fingerprint(value),
        "wire": old.canonical(value),
    }
    args.update(overrides)
    return db.execute(
        "SELECT candidate_profile_source_receive(%s,%s,%s,%s,%s)", tuple(args.values())
    ).fetchone()[0]


def source_command(source, action="accept_source"):
    p = source["proposal"]
    payload = {
        "proposalId": p["proposalId"],
        "proposalHash": p["proposalHash"],
        "sourceVersion": p["sourceVersion"],
    }
    if action == "accept_source":
        payload.update(selectedFactIds=[p["facts"][0]["factId"]], replaceFactIds=[], corrections=[])
    else:
        payload["reason"] = "not_needed"
    return {
        "schemaVersion": 3,
        "idempotencyKey": str(uuid4()),
        "action": action,
        "expectedDraftVersion": 0,
        "expectedProfileVersion": 0,
        "expectedPreferenceVersion": 0,
        "expectedPreferenceDraftRevision": 0,
        "payload": payload,
    }


def apply_source(db, source, command, command_id):
    return db.execute(
        "SELECT candidate_profile_source_apply(%s,%s,%s,%s,%s)",
        (
            source["subjectId"],
            source["bindingId"],
            command_id,
            fingerprint(command),
            old.canonical(command),
        ),
    ).fetchone()[0]


def test_photo_accept_read_remove_without_publication(spark_db, spark_runtime_dsn, source):
    url = "https://media.licdn.com/dms/image/synthetic?e=1234567890"
    fact = source["proposal"]["facts"][0]
    fact.update(kind="photo", value={"url": url})
    source = lookup_fixture(source)
    receive(spark_db, source)
    command = source_command(source)
    with psycopg.connect(spark_runtime_dsn, autocommit=True) as runtime:
        result = apply_source(runtime, source, command, str(uuid4()))
        assert result["status"] == "applied" and result["publicationActive"] is False
        draft = spark_db.execute(
            "SELECT facts FROM candidate_profile_drafts WHERE subject_id=%s AND version=1",
            (source["subjectId"],),
        ).fetchone()[0]
        assert old.profile(draft)["photoUrl"] == url
        draft["photoUrl"] = None
        save = {
            "schemaVersion": 2,
            "idempotencyKey": str(uuid4()),
            "action": "save_draft",
            "expectedDraftVersion": 1,
            "expectedProfileVersion": 0,
            "expectedPreferenceVersion": 0,
            "payload": {"profile": draft},
        }
        from psycopg.types.json import Jsonb

        receipt = runtime.execute(
            "SELECT candidate_profile_apply(%s,%s,%s,%s,%s)",
            (
                source["subjectId"],
                source["bindingId"],
                str(uuid4()),
                fingerprint(save),
                Jsonb(old.canonical(save)),
            ),
        ).fetchone()[0]
        assert receipt["status"] == "applied" and receipt["publicationActive"] is False
    assert (
        spark_db.execute(
            "SELECT facts->'photoUrl' FROM candidate_profile_drafts WHERE subject_id=%s AND version=2",
            (source["subjectId"],),
        ).fetchone()[0]
        is None
    )


def test_photo_erasure_clears_proposal_and_profile(spark_db, source):
    for value in [
        {"other": "https://media.licdn.com/a"},
        {"url": None},
        {"url": "https://evil.test/a"},
    ]:
        from psycopg.types.json import Jsonb

        fact = {**source["proposal"]["facts"][0], "kind": "photo", "value": value}
        assert (
            spark_db.execute(
                "SELECT candidate_profile_source_fact_valid(%s)", (Jsonb(fact),)
            ).fetchone()[0]
            is False
        )
    source["proposal"]["facts"][0].update(
        kind="photo", value={"url": "https://media.licdn.com/dms/image/private"}
    )
    source = lookup_fixture(source)
    receive(spark_db, source)
    assert (
        apply_source(spark_db, source, source_command(source), str(uuid4()))["status"] == "applied"
    )
    spark_db.execute(
        "SELECT candidate_profile_erase(%s,%s,%s,%s)",
        (source["subjectId"], source["bindingId"], str(uuid4()), "candidate"),
    )
    assert (
        spark_db.execute(
            "SELECT facts FROM candidate_profile_drafts WHERE subject_id=%s", (source["subjectId"],)
        ).fetchone()[0]
        is None
    )
    assert (
        spark_db.execute(
            "SELECT payload FROM candidate_profile_proposals WHERE proposal_id=%s",
            (source["proposal"]["proposalId"],),
        ).fetchone()[0]
        is None
    )


def test_read_accept_and_replay_without_publication(spark_db, spark_runtime_dsn, source):
    receive(spark_db, source)
    command, command_id = source_command(source), str(uuid4())
    with psycopg.connect(spark_runtime_dsn, autocommit=True) as runtime:
        assert (
            runtime.execute(
                "SELECT candidate_profile_source_read(%s,%s,%s,%s)",
                (
                    source["subjectId"],
                    source["bindingId"],
                    source["sourceId"],
                    source["proposal"]["proposalId"],
                ),
            ).fetchone()[0]
            == source["proposal"]
        )
        result = apply_source(runtime, source, command, command_id)
        assert result["status"] == "applied" and result["draftVersion"] == 1
        assert result["profileVersion"] == 0 and result["publicationActive"] is False
        assert apply_source(runtime, source, command, command_id) == result
    assert spark_db.execute(
        "SELECT facts->'skills' FROM candidate_profile_drafts WHERE subject_id=%s AND version=1",
        (source["subjectId"],),
    ).fetchone()[0] == ["Go"]
    state, audit = spark_db.execute(
        "SELECT state,acceptance FROM candidate_profile_proposals WHERE proposal_id=%s",
        (source["proposal"]["proposalId"],),
    ).fetchone()
    assert (
        state == "accepted"
        and audit["appliedEvidence"][0]["factId"] == source["proposal"]["facts"][0]["factId"]
    )


def test_lookup_stores_and_replays_exact_receipt_then_accepts_without_publication(spark_db, source):
    source = lookup_fixture(source)
    first = receive(spark_db, source)
    assert first["outcome"] == "stored" and first["publicationActive"] is False
    assert receive(spark_db, source) == first
    assert spark_db.execute(
        "SELECT payload->>'sourceKind' FROM candidate_profile_proposals WHERE proposal_id=%s",
        (source["proposal"]["proposalId"],),
    ).fetchone() == ("professional_lookup",)
    command_id, command = str(uuid4()), source_command(source)
    accepted = apply_source(spark_db, source, command, command_id)
    assert accepted["status"] == "applied" and accepted["profileVersion"] == 0
    assert accepted["publicationActive"] is False
    assert apply_source(spark_db, source, command, command_id) == accepted


def test_lookup_late_write_after_withdrawal_is_rejected(spark_db, source):
    source = lookup_fixture(source)
    consent(spark_db, source, "withdraw", 2)
    assert receive(spark_db, source)["reason"] == "consent_changed"
    assert spark_db.execute(
        "SELECT count(*) FROM candidate_profile_proposals WHERE proposal_id=%s",
        (source["proposal"]["proposalId"],),
    ).fetchone() == (0,)


def test_lookup_sql_refuses_fabricated_normalized_evidence(spark_db, source):
    source = lookup_fixture(source)
    source["proposal"]["facts"][0]["value"] = "invented"
    rehash(source)
    with pytest.raises(psycopg.errors.InvalidParameterValue):
        receive(spark_db, source)


@pytest.mark.parametrize(
    "change,reason",
    [("withdraw", "consent_changed"), ("selection", "source_conflict"), ("stale", "stale_edit")],
)
def test_accept_refusal_changes_no_profile(spark_db, source, change, reason):
    receive(spark_db, source)
    command = source_command(source)
    if change == "withdraw":
        consent(spark_db, source, "withdraw", 2)
    elif change == "selection":
        command["payload"]["selectedFactIds"] = [str(uuid4())]
    else:
        command["expectedDraftVersion"] = 1
    result = apply_source(spark_db, source, command, str(uuid4()))
    assert result["status"] == "rejected" and result["reason"] == reason
    assert result["draftVersion"] == 0 and result["changed"] is False


def test_reject_after_withdrawal_does_not_import(spark_db, source):
    receive(spark_db, source)
    consent(spark_db, source, "withdraw", 2)
    result = apply_source(spark_db, source, source_command(source, "reject_source"), str(uuid4()))
    assert result["status"] == "applied" and result["changed"] is False
    assert result["draftVersion"] == 0


def test_owner_reader_rechecks_privacy_and_rejects_foreign_proposal(
    spark_db, spark_runtime_dsn, source, monkeypatch
):
    from fastapi import HTTPException
    from psycopg.rows import dict_row

    from activekg.candidate_profiles import source_repository

    receive(spark_db, source)
    monkeypatch.setattr(
        source_repository,
        "_connect",
        lambda: psycopg.connect(spark_runtime_dsn, row_factory=dict_row),
    )
    args = (
        source["subjectId"],
        source["bindingId"],
        source["sourceId"],
        source["proposal"]["proposalId"],
    )
    assert source_repository.read_source(*args) == source["proposal"]
    with pytest.raises(HTTPException) as missing:
        source_repository.read_source(*args[:3], str(uuid4()))
    assert missing.value.status_code == 404
    checks = []
    original = source_repository.check_owner_privacy

    def restrict(cur, subject):
        original(cur, subject)
        checks.append(subject)
        if len(checks) == 2:
            raise HTTPException(451, "candidate_privacy_restricted")

    monkeypatch.setattr(source_repository, "check_owner_privacy", restrict)
    with pytest.raises(HTTPException) as restricted:
        source_repository.read_source(*args)
    assert restricted.value.status_code == 451


def test_accept_is_atomic_on_a_later_bad_selection(spark_db, source):
    receive(spark_db, source)
    command = source_command(source)
    command["payload"]["selectedFactIds"].append(str(uuid4()))
    result = apply_source(spark_db, source, command, str(uuid4()))
    assert result["reason"] == "source_conflict" and result["draftVersion"] == 0
    assert (
        spark_db.execute(
            "SELECT count(*) FROM candidate_profile_drafts WHERE subject_id=%s",
            (source["subjectId"],),
        ).fetchone()[0]
        == 0
    )


def test_concurrent_acceptance_writes_one_draft(spark_db, spark_runtime_dsn, source):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    receive(spark_db, source)
    barrier = Barrier(2)

    def run():
        with psycopg.connect(spark_runtime_dsn, autocommit=True) as db:
            barrier.wait(timeout=5)
            return apply_source(db, source, source_command(source), str(uuid4()))

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(run), pool.submit(run)
        results = [a.result(timeout=10), b.result(timeout=10)]
    assert sorted(r["status"] for r in results) == ["applied", "rejected"]
    assert (
        spark_db.execute(
            "SELECT count(*) FROM candidate_profile_drafts WHERE subject_id=%s",
            (source["subjectId"],),
        ).fetchone()[0]
        == 1
    )


@pytest.mark.parametrize("replace", [False, True])
def test_scalar_replacement_requires_explicit_selection(spark_db, source, replace):
    from psycopg.types.json import Jsonb

    spark_db.execute(
        "INSERT INTO candidate_profile_heads(subject_id,binding_id) VALUES(%s,%s)",
        (source["subjectId"], source["bindingId"]),
    )

    profile = {
        "displayName": "Candidate-written name",
        "headline": "",
        "linkedinUrl": None,
        "currentLocation": None,
        "skills": [],
        "employment": [],
        "education": [],
        "projects": [],
    }
    spark_db.execute(
        "INSERT INTO candidate_profile_drafts(subject_id,version,facts,document_sha256,command_id) VALUES(%s,1,%s,%s,%s)",
        (source["subjectId"], Jsonb(profile), fingerprint(profile), str(uuid4())),
    )
    spark_db.execute(
        "UPDATE candidate_profile_heads SET draft_version=1 WHERE subject_id=%s",
        (source["subjectId"],),
    )
    source["proposal"]["originalDraftVersion"] = 1
    source["proposal"]["facts"][0]["kind"] = "displayName"
    rehash(source)
    receive(spark_db, source)
    command = source_command(source)
    command["expectedDraftVersion"] = 1
    if replace:
        command["payload"]["replaceFactIds"] = command["payload"]["selectedFactIds"][:]
    result = apply_source(spark_db, source, command, str(uuid4()))
    assert result["status"] == ("applied" if replace else "rejected")
    assert result["draftVersion"] == (2 if replace else 1)
    assert spark_db.execute(
        "SELECT facts->>'displayName' FROM candidate_profile_drafts WHERE subject_id=%s AND version=%s",
        (source["subjectId"], result["draftVersion"]),
    ).fetchone()[0] == ("Go" if replace else "Candidate-written name")


def test_store_replay_and_no_canonical_write(spark_db, spark_runtime_dsn, source):
    with psycopg.connect(spark_runtime_dsn, autocommit=True) as runtime:
        result = receive(runtime, source)
        assert result["outcome"] == "stored"
        assert result == receive(runtime, source)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            runtime.execute("SELECT * FROM candidate_profile_proposals")
    assert spark_db.execute(
        "SELECT draft_version,profile_version,preference_version FROM candidate_profile_heads WHERE subject_id=%s",
        (source["subjectId"],),
    ).fetchone() == (0, 0, 0)
    payload = spark_db.execute(
        "SELECT payload FROM candidate_profile_proposals WHERE proposal_id=%s",
        (source["proposal"]["proposalId"],),
    ).fetchone()[0]
    assert payload == source["proposal"] and "text" not in payload
    assert result["canonicalProfileChanged"] is False and result["publicationActive"] is False


def test_withdrawal_blocks_new_but_replays_original(spark_db, source):
    result = receive(spark_db, source)
    consent(spark_db, source, "withdraw", 2)
    assert receive(spark_db, source) == result
    source["commandId"] = str(uuid4())
    assert receive(spark_db, source)["reason"] == "consent_changed"


@pytest.mark.parametrize(
    "change,reason", [("stale", "stale_edit"), ("expired", "proposal_expired")]
)
def test_current_version_and_expiry(spark_db, source, change, reason):
    if change == "stale":
        source["proposal"]["originalDraftVersion"] = 1
    else:
        source["proposal"]["expiresAt"] = "2020-01-01T00:00:00Z"
    assert receive(spark_db, rehash(source))["reason"] == reason


@pytest.mark.parametrize("name,value,valid", cases(), ids=[v[0] for v in cases()])
def test_sql_matches_transfer_contract_vectors(spark_db, source, name, value, valid):
    import copy

    value = copy.deepcopy(value)
    value.update(subjectId=source["subjectId"], bindingId=source["bindingId"])
    if old.UUID.fullmatch(value["commandId"]):
        value["commandId"] = str(uuid4())
    # Bind fixture identities without repairing deliberately malformed payloads.
    original_hash = fingerprint({k: v for k, v in value["proposal"].items() if k != "proposalHash"})
    good_hash = value["proposal"]["proposalHash"] == original_hash
    value["proposal"]["proposalId"] = str(uuid4())
    if good_hash:
        value["proposal"]["proposalHash"] = fingerprint(
            {k: v for k, v in value["proposal"].items() if k != "proposalHash"}
        )
    if valid:
        result = receive(spark_db, value)
        assert result["outcome"] in ("stored", "rejected")
        if result["outcome"] == "rejected":
            assert result["reason"] == "proposal_expired"
    else:
        with pytest.raises(psycopg.errors.InvalidParameterValue):
            receive(
                spark_db,
                value,
                command=value["commandId"]
                if old.UUID.fullmatch(value["commandId"])
                else str(uuid4()),
            )


@pytest.mark.parametrize("change", ["hash", "quote", "surrogate", "extra"])
def test_sql_revalidates_wire(spark_db, source, change):
    if change == "hash":
        source["textHash"] = "0" * 64
    elif change == "quote":
        source["proposal"]["facts"][0]["evidence"][0]["quote"] = "invented"
        rehash(source)
    elif change == "surrogate":
        source["proposal"]["facts"][0]["evidence"][0]["locator"] = "text:1:2"
        rehash(source)
    else:
        source["extra"] = True
    with pytest.raises(psycopg.errors.InvalidParameterValue):
        receive(spark_db, source)


def test_erasure_scrubs_proposals_and_fences_late_delivery(spark_db, source):
    receive(spark_db, source)
    spark_db.execute(
        "SELECT candidate_profile_erase(%s,%s,%s,%s)",
        (source["subjectId"], source["bindingId"], str(uuid4()), "candidate"),
    )
    assert spark_db.execute(
        "SELECT payload,proposal_hash,state FROM candidate_profile_proposals WHERE proposal_id=%s",
        (source["proposal"]["proposalId"],),
    ).fetchone() == (None, None, "erased")
    assert spark_db.execute(
        "SELECT content_hash,text_hash FROM candidate_profile_sources WHERE subject_id=%s",
        (source["subjectId"],),
    ).fetchone() == (None, None)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        receive(spark_db, source)


def test_changed_command_conflicts(spark_db, source):
    receive(spark_db, source)
    source["policyVersion"] = "changed"
    with pytest.raises(psycopg.errors.UniqueViolation):
        receive(spark_db, source)


def test_concurrent_delivery_stores_once(spark_db, spark_runtime_dsn, source):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2)

    def run():
        with psycopg.connect(spark_runtime_dsn, autocommit=True) as conn:
            barrier.wait(timeout=5)
            return receive(conn, source)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(run), pool.submit(run)
        assert a.result(timeout=10) == b.result(timeout=10)
    for table in ("candidate_profile_source_receipts", "candidate_profile_proposals"):
        from psycopg import sql

        assert (
            spark_db.execute(
                sql.SQL("SELECT count(*) FROM {} WHERE subject_id=%s").format(
                    sql.Identifier(table)
                ),
                (source["subjectId"],),
            ).fetchone()[0]
            == 1
        )


def test_privacy_changed_before_reply_rolls_back_storage(
    spark_db, spark_runtime_dsn, source, monkeypatch
):
    from fastapi import HTTPException
    from psycopg.rows import dict_row

    from activekg.candidate_profiles import source_repository

    checks = []
    original = source_repository.check_owner_privacy

    def restrict_after_save(cur, subject):
        original(cur, subject)
        checks.append(subject)
        if len(checks) == 2:
            raise HTTPException(451, "candidate_privacy_restricted")

    monkeypatch.setattr(
        source_repository,
        "_connect",
        lambda: psycopg.connect(spark_runtime_dsn, row_factory=dict_row),
    )
    monkeypatch.setattr(source_repository, "check_owner_privacy", restrict_after_save)
    with pytest.raises(HTTPException) as error:
        source_repository.receive_source(
            source["subjectId"],
            source["bindingId"],
            source["commandId"],
            fingerprint(source),
            old.canonical(source).encode(),
        )
    assert error.value.status_code == 451 and len(checks) == 2
    assert (
        spark_db.execute(
            "SELECT count(*) FROM candidate_profile_proposals WHERE subject_id=%s",
            (source["subjectId"],),
        ).fetchone()[0]
        == 0
    )


def test_missing_receive_grant_is_not_a_privacy_refusal(
    spark_db, spark_runtime_dsn, source, monkeypatch
):
    from fastapi import HTTPException
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict
    from psycopg.rows import dict_row

    from activekg.candidate_profiles import source_repository

    role = conninfo_to_dict(spark_runtime_dsn)["user"]
    routine = "candidate_profile_source_receive(uuid,uuid,uuid,text,text)"
    spark_db.execute(
        sql.SQL("REVOKE EXECUTE ON FUNCTION " + routine + " FROM {}").format(sql.Identifier(role))
    )
    monkeypatch.setattr(
        source_repository,
        "_connect",
        lambda: psycopg.connect(spark_runtime_dsn, row_factory=dict_row),
    )
    try:
        with pytest.raises(HTTPException) as error:
            source_repository.receive_source(
                source["subjectId"],
                source["bindingId"],
                source["commandId"],
                fingerprint(source),
                old.canonical(source).encode(),
            )
        assert error.value.status_code == 503
    finally:
        spark_db.execute(
            sql.SQL("GRANT EXECUTE ON FUNCTION " + routine + " TO {}").format(sql.Identifier(role))
        )


def test_nul_is_refused_before_database_connection(source, monkeypatch):
    from fastapi import HTTPException

    from activekg.candidate_profiles import source_repository

    source["proposal"]["reviewNotes"] = ["bad\x00note"]
    rehash(source)

    def unexpected():
        pytest.fail("Malformed source must be refused before connecting")

    monkeypatch.setattr(source_repository, "_connect", unexpected)
    with pytest.raises(HTTPException) as error:
        source_repository.receive_source(
            source["subjectId"],
            source["bindingId"],
            source["commandId"],
            fingerprint(source),
            old.canonical(source).encode(),
        )
    assert error.value.status_code == 422


@pytest.mark.parametrize("kind", ["resume", "lookup"])
def test_signed_receiver_replay_and_authority(
    spark_db, spark_runtime_dsn, source, monkeypatch, kind
):
    import time

    import jwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from psycopg.rows import dict_row

    from activekg.api import auth, candidate_profiles
    from activekg.candidate_profiles import source_repository

    if kind == "lookup":
        source = lookup_fixture(source)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    for name, value in {
        "JWT_ENABLED": True,
        "JWT_ALGORITHM": "RS256",
        "JWT_AUDIENCE": "activekg",
        "JWT_ISSUER": "vantahire",
        "JWT_PUBLIC_KEY": public,
        "JWT_LEEWAY_SECONDS": 0,
    }.items():
        monkeypatch.setattr(auth, name, value)
    monkeypatch.setattr(
        source_repository,
        "_connect",
        lambda: psycopg.connect(spark_runtime_dsn, row_factory=dict_row),
    )
    app = FastAPI()
    app.include_router(candidate_profiles.router)
    now = int(time.time())
    claims = {
        "iss": "vantahire",
        "aud": "activekg",
        "sub": "vantahire-backend",
        "actor_type": "service",
        "tenant_id": "candidate_" + source["subjectId"],
        "scopes": "candidate-profile:owner-write",
        "iat": now,
        "exp": now + 60,
    }
    headers = {
        "content-type": "application/json",
        "x-command-sha256": fingerprint(source),
        "x-candidate-binding-id": source["bindingId"],
        "x-candidate-command-id": source["commandId"],
    }
    with TestClient(app) as client:

        def send(changes=None, header_changes=None):
            return client.post(
                "/candidate-profiles/sources/" + kind,
                content=old.canonical(source),
                headers={
                    **headers,
                    "authorization": "Bearer "
                    + jwt.encode({**claims, **(changes or {})}, key, algorithm="RS256"),
                    **(header_changes or {}),
                },
            )

        assert send().status_code == 404
        for flag in [
            "CANDIDATE_FOUNDATION_ENABLED",
            "CANDIDATE_PROFILE_ENABLED",
            "CANDIDATE_SPARK_ENABLED",
            "CANDIDATE_CONSENT_V2_ENABLED",
            "CANDIDATE_SOURCE_IMPORT_ENABLED",
        ]:
            monkeypatch.setenv(flag, "true")
        for change in [
            {"tenant_id": "candidate_" + str(uuid4())},
            {"scopes": "candidate-profile:owner-read"},
            {"sub": "wrong-service"},
        ]:
            assert send(change).status_code == 403
        assert send(header_changes={"x-candidate-binding-id": str(uuid4())}).status_code == 403
        assert send(header_changes={"x-command-sha256": "0" * 64}).status_code == 422
        first = send()
        assert first.status_code == 200, first.text
        assert first.headers["cache-control"] == "no-store"
        assert send().json() == first.json()
        consent(spark_db, source, "withdraw", 2)
        assert send().json() == first.json()
