"""Synthetic storage proof using committed-path migrations 027, 040 and 041.

No production DSNs. Databases are retained for independent inspection.
"""

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from activekg.candidate_profiles.profile_contracts import canonical, parse_profile_command

URL = os.getenv("S2_PROFILE_TEST_ADMIN_URL")
pytestmark = pytest.mark.skipif(not URL, reason="explicit disposable PostgreSQL required")


@pytest.fixture(scope="module")
def database():
    parsed = urlparse(URL)
    assert (
        parsed.hostname == "127.0.0.1"
        and parsed.path == "/postgres"
        and os.getenv("S2_PROTOTYPE_DISPOSABLE") == "1"
    )
    name = "s2_profile_test_storage_" + uuid4().hex[:10]
    with psycopg.connect(URL, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(name)))
    target = URL.rsplit("/", 1)[0] + "/" + name
    with psycopg.connect(target, autocommit=True) as conn:
        # Only the prerequisite global identity table is stubbed; consent and
        # candidate storage use actual migration files, never injected SQL paths.
        # Full historical release rehearsal is a separate suite.
        conn.execute("CREATE TABLE public.global_candidates(id uuid PRIMARY KEY)")
        root = Path(__file__).resolve().parents[1] / "db/migrations"
        for filename in (
            "027_candidate_consent.sql",
            "040_candidate_owned_foundation.sql",
            "041_candidate_profile_drafts.sql",
        ):
            conn.execute((root / filename).read_text())
    print("Retained S2 storage fixture", name)
    return target


@pytest.fixture
def db(database):
    with psycopg.connect(database, autocommit=True, row_factory=dict_row) as conn:
        yield conn


def identity(db):
    subject, binding = str(uuid4()), str(uuid4())
    db.execute(
        "SELECT candidate_owned_ensure(%s,%s,%s,1,%s)", (subject, binding, str(uuid4()), "a" * 64)
    )
    return subject, binding


FACTS = {
    "displayName": "José 🌍",
    "headline": "Engineer",
    "linkedinUrl": None,
    "currentLocation": {"city": "Bengaluru", "countryCode": "IN", "displayCountry": "India"},
    "skills": ["Python"],
    "employment": [],
    "education": [],
    "projects": [],
}
PREFS = {
    "desiredRoles": ["Engineer"],
    "preferredSeniority": [],
    "workArrangements": ["remote"],
    "locations": {"mode": "anywhere"},
    "remoteRegions": {"mode": "worldwide"},
    "relocation": {"mode": "discuss", "destinations": []},
    "employmentTypes": ["permanent"],
    "availability": {"mode": "unknown"},
    "compensation": {"mode": "open_to_discussion"},
    "strengths": {
        "roles": "firm",
        "location": "flexible",
        "workArrangement": "flexible",
        "employmentType": "firm",
        "compensation": "flexible",
    },
}


def command(action, payload, d=0, p=0, f=0):
    return {
        "schemaVersion": 2,
        "idempotencyKey": str(uuid4()),
        "action": action,
        "payload": payload,
        "expectedDraftVersion": d,
        "expectedProfileVersion": p,
        "expectedPreferenceVersion": f,
    }


def apply(db, who, request, cid=None):
    _, wire, digest = parse_profile_command(canonical(request).encode())
    return db.execute(
        "SELECT candidate_profile_apply(%s,%s,%s,%s,%s) result",
        (*who, cid or str(uuid4()), digest, Jsonb(wire)),
    ).fetchone()["result"]


def read(db, who):
    return db.execute("SELECT candidate_profile_read(%s,%s) result", who).fetchone()["result"]


def test_real_anchor_and_empty_owner_read(db):
    who = identity(db)
    assert read(db, who)["draftVersion"] == 0
    with pytest.raises(psycopg.Error, match="not_found"):
        read(db, (who[0], str(uuid4())))


def test_canonical_hash_matches_python(db):
    value = {"z": ["\n", None, True], "a": FACTS}
    encoded = db.execute("SELECT candidate_profile_canonical(%s) c", (Jsonb(value),)).fetchone()[
        "c"
    ]
    assert encoded == canonical(value)


def test_save_replay_conflict_and_separate_preference_writer(db):
    who = identity(db)
    req = command("save_draft", {"profile": FACTS})
    cid = str(uuid4())
    first = apply(db, who, req, cid)
    assert first["draftVersion"] == 1 and first["profileVersion"] == 0
    assert apply(db, who, req, cid) == first
    with pytest.raises(psycopg.Error, match="idempotency_conflict"):
        apply(db, who, command("save_draft", {"profile": FACTS}, 1), cid)
    assert (
        apply(db, who, command("save_preferences", {"preferences": PREFS}))["preferenceVersion"]
        == 1
    )
    assert apply(db, who, command("save_draft", {"profile": FACTS}))["reason"] == "stale_edit"
    assert read(db, who)["draft"] == FACTS


def test_confirm_requires_exact_displayed_hash_and_never_publishes(db):
    who = identity(db)
    apply(db, who, command("save_draft", {"profile": FACTS}))
    apply(db, who, command("save_preferences", {"preferences": PREFS}))
    payload = {"draftHash": "0" * 64, "consentVersion": 0}
    assert apply(db, who, command("confirm", payload, 1, 0, 1))["reason"] == "confirmation_conflict"
    payload["draftHash"] = hashlib.sha256(canonical(FACTS).encode()).hexdigest()
    good = apply(db, who, command("confirm", payload, 1, 0, 1))
    assert (
        good["status"] == "applied"
        and good["profileVersion"] == 1
        and good["preferenceVersion"] == 1
    )
    again = apply(db, who, command("confirm", payload, 1, 1, 1))
    assert again["profileVersion"] == 1 and again["publicationActive"] is False


def test_foreign_receipt_does_not_disclose_existence(db):
    one, two = identity(db), identity(db)
    cid = str(uuid4())
    req = command("save_draft", {"profile": FACTS})
    apply(db, one, req, cid)
    with pytest.raises(psycopg.Error, match="not_found"):
        apply(db, two, req, cid)


def test_compacts_old_payloads_and_replays_without_original_content(db):
    who = identity(db)
    cid = str(uuid4())
    req = command("save_draft", {"profile": FACTS})
    original = apply(db, who, req, cid)
    changed = {**FACTS, "headline": "New"}
    apply(db, who, command("save_draft", {"profile": changed}, 1))
    row = db.execute(
        "SELECT facts,document_sha256 FROM candidate_profile_drafts WHERE subject_id=%s AND version=1",
        (who[0],),
    ).fetchone()
    assert row == {"facts": None, "document_sha256": None}
    assert apply(db, who, req, cid) == original
    assert read(db, who)["draft"] == changed


def test_concurrent_drafts_only_one_wins(database, db):
    from concurrent.futures import ThreadPoolExecutor

    who = identity(db)

    def write():
        with psycopg.connect(database, autocommit=True, row_factory=dict_row) as other:
            return apply(other, who, command("save_draft", {"profile": FACTS}))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: write(), range(2)))
    assert sorted(r["status"] for r in results) == ["applied", "rejected"]
    assert read(db, who)["draftVersion"] == 1


def test_restriction_blocks_old_replay(db):
    who = identity(db)
    cid = str(uuid4())
    req = command("save_draft", {"profile": FACTS})
    apply(db, who, req, cid)
    db.execute("UPDATE candidate_profile_heads SET restricted=true WHERE subject_id=%s", (who[0],))
    with pytest.raises(psycopg.Error, match="candidate_privacy_restricted"):
        apply(db, who, req, cid)


def test_runtime_has_routines_not_tables(db):
    who = identity(db)
    role = "s2_runtime_" + uuid4().hex
    with db.transaction():
        db.execute(
            sql.SQL("CREATE ROLE {} NOLOGIN NOSUPERUSER NOBYPASSRLS").format(sql.Identifier(role))
        )
        from psycopg.rows import tuple_row

        from activekg.candidate_profiles.profile_repository import profile_catalog_evidence
        from scripts.init_railway_db import _harden_candidate_profile_runtime_privileges

        with db.cursor(row_factory=tuple_row) as cur:
            _harden_candidate_profile_runtime_privileges(cur, role)
            digest, permissions = profile_catalog_evidence(cur, role)
        assert permissions is True
        print("S2 Memory catalog", digest)
        db.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(role)))
        assert apply(db, who, command("save_draft", {"profile": FACTS}))["status"] == "applied"
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with db.transaction():
                db.execute("SELECT * FROM candidate_profile_drafts")
        db.execute("RESET ROLE")
        # Roll back the test role, grants and edits together. No second hand-kept
        # routine inventory is needed for cleanup.
        raise psycopg.Rollback()
