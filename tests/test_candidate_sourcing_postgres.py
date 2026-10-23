"""Synthetic intake and real restricted-role sourcing SQL; no model calls."""

import asyncio
import json
import time
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import httpx
import jwt
import psycopg
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from psycopg.conninfo import conninfo_to_dict
from psycopg.types.json import Jsonb

from activekg.api import candidate_sourcing as api
from activekg.candidate_index import sourcing
from activekg.candidate_index.repository import INDEX_CATALOG_SHA256, catalog_evidence
from activekg.privacy.models import CandidatePrivacyAction
from tests import test_candidate_index_postgres as base

pytestmark = base.pytestmark
target = base.target
isolated_coordination = base.isolated_coordination


def test_public_applicant_membership_does_not_require_published_resume(target):
    repo, scope, _, _ = base.source()
    public_id, absent = uuid4(), uuid4()
    target.execute("INSERT INTO global_candidates(id) VALUES (%s)", (public_id,))
    target.execute("SELECT set_config('app.current_tenant_id',%s,false)", (scope,))
    target.execute(
        "UPDATE candidates SET global_candidate_id=%s WHERE tenant_id=%s", (public_id, scope)
    )

    def lookup(tenant, ids):
        with repo.transaction(tenant) as cur:
            cur.execute("SET LOCAL transaction_read_only=on")
            cur.execute("SELECT * FROM candidate_index_public_applicants(%s)", (Jsonb(ids),))
            return [row[0] for row in cur.fetchall()]

    assert lookup(scope, [str(public_id), str(absent)]) == [
        {"publicId": str(public_id), "isApplicant": True},
        {"publicId": str(absent), "isApplicant": False},
    ]
    assert lookup("org_2147483647", [str(public_id)]) == [
        {"publicId": str(public_id), "isApplicant": False}
    ]
    for ids in [[str(public_id)] * 2, [], ["00000000-0000-0000-0000-000000000000"]]:
        with pytest.raises(psycopg.errors.InvalidParameterValue):
            lookup(scope, ids)


def test_v3_unsupported_job_evidence_is_unknown_and_still_saves(target, monkeypatch):
    """One unsupported date/title no longer rejects the resume; the database accepts the result."""
    from activekg.candidate_index import resume_history_policy
    from activekg.candidate_index.processing import resume_extraction

    text = "Python. Backend engineer, March 2019 to June 2024. Intern, 2017 to 2018."
    monkeypatch.setattr(base, "TEXT", text)
    monkeypatch.setattr(
        resume_history_policy, "token_fits", lambda policy: lambda t: len(t) <= 1200
    )
    policy = replace(base.POLICY, **resume_history_policy.policy_fields())
    repo, scope, _, _ = base.source(policy=policy)
    lease = repo.reserve(repo.claim("extract")[0], policy)
    raw = {
        "skills_raw": ["Python"],
        "skills_normalized": [],
        "confidence": 0.9,
        "employment_history": [
            {
                "title": "Backend engineer",
                "employment_type": None,
                "start": "2019-01",  # contradicts its verbatim quote: unknown
                "end": "Jun 2024",  # resume wording instead of YYYY-MM: read from the quote
                "ongoing": False,
                "start_quote": "March 2019",
                "end_quote": "June 2024",
            },
            {
                "title": "Invented title",  # not in the resume: unknown
                "employment_type": None,
                "start": "2017",
                "end": "2018",
                "ongoing": False,
                "start_quote": "2017",
                "end_quote": "2018",
            },
        ],
    }
    result = resume_extraction(raw, text, policy, policy.primary_model_id, {})
    assert repo.complete_extract(
        lease,
        namespace=result.namespace,
        evidence=result.evidence,
        confidence=result.confidence,
        model_id=policy.primary_model_id,
        chunks=result.chunks,
        professional_text=result.professional_text,
    )
    embed = repo.reserve(repo.claim("embed")[0], policy)
    assert repo.complete_embed(embed, [base.VECTOR for _ in embed["extraction"]["chunks"]])
    rows = read(repo, scope)
    assert rows[0]["evidence"]["employment"] == [
        {"title": "Backend engineer", "end": "2024-06", "ongoing": False},
        {"start": "2017", "end": "2018", "ongoing": False},
    ]


def test_v3_history_roundtrip_and_no_quotes_in_search(target, monkeypatch):
    from activekg.candidate_index import resume_history_policy
    from activekg.candidate_index.processing import resume_extraction

    text = "Python. Backend engineer, March 2019 to June 2024."
    monkeypatch.setattr(base, "TEXT", text)
    monkeypatch.setattr(
        resume_history_policy, "token_fits", lambda policy: lambda t: len(t) <= 1200
    )
    policy = replace(base.POLICY, **resume_history_policy.policy_fields())
    repo, scope, _, _ = base.source(policy=policy)
    lease = repo.reserve(repo.claim("extract")[0], policy)
    raw = {
        "skills_raw": ["Python"],
        "skills_normalized": [],
        "confidence": 0.9,
        "employment_history": [
            {
                "title": "Backend engineer",
                "employment_type": None,
                "start": "2019-03",
                "end": "2024-06",
                "ongoing": False,
                "start_quote": "March 2019",
                "end_quote": "June 2024",
            }
        ],
    }
    result = resume_extraction(raw, text, policy, policy.primary_model_id, {})
    assert repo.complete_extract(
        lease,
        namespace=result.namespace,
        evidence=result.evidence,
        confidence=result.confidence,
        model_id=policy.primary_model_id,
        chunks=result.chunks,
        professional_text=result.professional_text,
    )
    embed = repo.reserve(repo.claim("embed")[0], policy)
    assert repo.complete_embed(embed, [base.VECTOR for _ in embed["extraction"]["chunks"]])
    rows = read(repo, scope)
    assert rows[0]["evidence"]["employment"] == [
        {"title": "Backend engineer", "start": "2019-03", "end": "2024-06", "ongoing": False}
    ]
    assert "quote" not in json.dumps(rows)


@pytest.mark.parametrize(
    "start,quote,expected",
    [
        ("2019-03", "March 2019", True),
        ("2019-04", "March 2019", False),
        ("2019-03", "2019", False),
        ("2019", "2019", True),
        ("2019-03-04", "4 March 2019", True),
        ("2019-03-04", "04/03/2019", False),
    ],
)
def test_sql_rejects_invented_date_precision(target, start, quote, expected):
    history = [
        {
            "title": None,
            "employment_type": None,
            "start": start,
            "start_quote": quote,
            "end": None,
            "end_quote": None,
            "ongoing": False,
        }
    ]
    assert (
        target.execute(
            "SELECT candidate_index_history_valid(%s,%s)", (Jsonb(history), quote)
        ).fetchone()[0]
        is expected
    )


def read(repo, scope, job=189, limit=100):
    with repo.transaction(scope) as cur:
        cur.execute("SET LOCAL transaction_read_only=on")
        cur.execute(
            "SELECT * FROM candidate_index_read_sourcing(%s::vector,%s,%s,%s,%s)",
            (
                json.dumps(base.VECTOR),
                base.POLICY.embedding_model_id,
                base.POLICY.embedding_artifact_revision,
                limit,
                job,
            ),
        )
        return [r[0] for r in cur.fetchall()]


def revalidate(repo, scope, identity, job=189):
    with repo.transaction(scope) as cur:
        cur.execute("SET LOCAL transaction_read_only=on")
        cur.execute(
            "SELECT * FROM candidate_index_revalidate_sourcing(%s,%s)", (job, Jsonb([identity]))
        )
        return cur.fetchone()[0]


def test_scoped_read_has_identity_and_only_structured_facts(target):
    repo, scope, accepted, _ = base.source()
    base.publish(repo)
    rows = read(repo, scope)
    assert len(rows) == 1
    row = rows[0]
    assert row["identity"]["generationId"] == accepted["generation_id"]
    assert row["evidence"]["skills"] == ["python"]
    assert row["evidence"]["scope"] == scope
    assert row["linkedPublicId"] is None
    assert set(row) == {"identity", "evidence", "linkedPublicId", "distance"}
    assert not any(
        key in json.dumps(row) for key in ("fixture.invalid", "professional_text", "match_tokens")
    )
    assert revalidate(repo, scope, row["identity"])["decision"] == "allow"
    assert read(repo, "org_2147483647") == []
    assert revalidate(repo, "org_2147483647", row["identity"])["decision"] == "not_found"


def test_same_job_excluded_and_revalidated(target):
    repo, scope, _, _ = base.source()
    base.publish(repo)
    identity = read(repo, scope)[0]["identity"]
    assert read(repo, scope, job=identity["applicationJobId"]) == []
    assert (
        revalidate(repo, scope, identity, job=identity["applicationJobId"])["decision"]
        == "same_job"
    )


def test_changed_source_and_generation_not_replaced(target):
    repo, scope, _, _ = base.source()
    base.publish(repo)
    identity = read(repo, scope)[0]["identity"]
    for key, value in [
        ("sourceHash", "f" * 64),
        ("applicationId", 2147483647),
        ("generationId", "11111111-1111-4111-8111-111111111111"),
    ]:
        changed = {**identity, key: value}
        assert revalidate(repo, scope, changed) == {
            "identity": changed,
            "decision": "source_changed",
        }


def test_erasure_hides_read_and_blocks_existing_tuple(target):
    repo, scope, _, args = base.source()
    base.publish(repo)
    identity = read(repo, scope)[0]["identity"]
    base.directive({**args, "scope": scope}, CandidatePrivacyAction.REQUEST_ERASURE)
    assert read(repo, scope) == []
    assert revalidate(repo, scope, identity)["decision"] == "privacy_restricted"


def test_bounds_and_runtime_catalogue(target):
    repo, scope, _, _ = base.source()
    with target.cursor() as cur:
        assert catalog_evidence(cur, conninfo_to_dict(base.RUNTIME)["user"]) == (
            INDEX_CATALOG_SHA256,
            True,
        )
    for limit in (0, 101):
        with pytest.raises(psycopg.errors.InvalidParameterValue):
            read(repo, scope, limit=limit)
    for items in ([], [{}], [{}] * 101):
        with pytest.raises(psycopg.errors.InvalidParameterValue), repo.transaction(scope) as cur:
            cur.execute(
                "SELECT * FROM candidate_index_revalidate_sourcing(189,%s)", (Jsonb(items),)
            )


def test_old_reader_unchanged(target):
    repo, scope, accepted, _ = base.source()
    base.publish(repo)
    with repo.transaction(scope) as cur:
        cur.execute(
            "SELECT * FROM candidate_index_read_private(%s::vector,%s,%s,100,NULL)",
            (
                json.dumps(base.VECTOR),
                base.POLICY.embedding_model_id,
                base.POLICY.embedding_artifact_revision,
            ),
        )
        old = cur.fetchone()[0]
    new = read(repo, scope)[0]
    assert old["generation_id"] == new["identity"]["generationId"] == accepted["generation_id"]
    assert old["score"] == 1 - new["distance"]
    assert "highlights" in old and "highlights" not in new


def test_same_person_application_excludes_other_resume_before_limit(target, monkeypatch):
    original = base._body
    job = [7]
    monkeypatch.setattr(base, "_body", lambda **kwargs: original(job_id=job[0], **kwargs))
    repo, scope, _, _ = base.source()
    base.publish(repo)
    first = read(repo, scope)[0]["identity"]
    job[0] = 189
    base.source(scope)
    base.publish(repo)
    # Explicit synthetic verified link, never inferred from names/email/text.
    global_id = uuid4()
    target.execute("INSERT INTO global_candidates(id) VALUES (%s)", (global_id,))
    target.execute("SELECT set_config('app.current_tenant_id',%s,false)", (scope,))
    target.execute(
        "UPDATE candidates SET global_candidate_id=%s WHERE tenant_id=%s", (global_id, scope)
    )
    assert read(repo, scope, job=189, limit=1) == []
    assert revalidate(repo, scope, first, job=189)["decision"] == "same_job"
    # Third unrelated applicant to another job must still fill the one-slot read.
    job[0] = 8
    base.source(scope)
    base.publish(repo)
    results = read(repo, scope, job=189, limit=1)
    assert len(results) == 1 and results[0]["identity"]["applicationJobId"] == 8


def test_linked_duplicates_return_one_authoritative_person(target):
    repo, scope, _, _ = base.source()
    base.publish(repo)
    base.source(scope)
    base.publish(repo)
    assert len(read(repo, scope)) == 2
    global_id = uuid4()
    target.execute("INSERT INTO global_candidates(id) VALUES (%s)", (global_id,))
    target.execute("SELECT set_config('app.current_tenant_id',%s,false)", (scope,))
    target.execute(
        "UPDATE candidates SET global_candidate_id=%s WHERE tenant_id=%s", (global_id, scope)
    )
    results = read(repo, scope)
    assert len(results) == 1 and results[0]["linkedPublicId"] == str(global_id)


def test_privacy_lock_contention_is_not_empty_success(target):
    repo, scope, _, _ = base.source()
    base.publish(repo)
    candidate = read(repo, scope)[0]["identity"]["candidateId"]
    started = time.monotonic()
    with target.transaction():
        target.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
            ("candidate-privacy-candidate:" + scope + ":" + candidate,),
        )
        with pytest.raises(psycopg.errors.LockNotAvailable):
            read(repo, scope)
    assert time.monotonic() - started < 4
    assert len(read(repo, scope)) == 1


def test_real_http_auth_embedding_boundary_sql_and_revalidation(target, monkeypatch):
    repo, scope, _, _ = base.source()
    base.publish(repo)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    monkeypatch.setattr(api.auth, "JWT_ENABLED", True)
    monkeypatch.setattr(api.auth, "SIGNAL_JWT_PUBLIC_KEY", public)
    monkeypatch.setattr(api, "enabled", lambda _: True)
    monkeypatch.setattr(api, "configuration", lambda: SimpleNamespace(policy=base.POLICY))
    calls = []

    async def embedding(*args, **kwargs):
        calls.append(args)
        return base.VECTOR

    monkeypatch.setattr(sourcing, "embed_query_once", embedding)
    app = FastAPI()
    app.include_router(api.router)
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": "signal",
            "aud": "activekg",
            "sub": "signal-service",
            "actor_type": "service",
            "tenant_id": scope,
            "scopes": [api.SOURCING_SCOPE],
            "request_id": "synthetic-test",
            "iat": now,
            "exp": now + 300,
            "jti": str(uuid4()),
        },
        key,
        algorithm="RS256",
    )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"authorization": "Bearer " + token},
        ) as client:
            body = {
                "version": 1,
                "requestId": "synthetic-test",
                "externalJobId": 189,
                "flowRunId": str(uuid4()),
                "artifactHash": "a" * 64,
                "query": "Python",
                "limit": 100,
            }
            response = await client.post("/organization-candidates/sourcing/search", json=body)
            assert response.status_code == 200, response.text
            results = response.json()["results"]
            assert len(results) == 1 and results[0]["evidence"]["skills"] == ["python"]
            verify = {
                "version": 1,
                "requestId": "synthetic-test",
                "externalJobId": 189,
                "items": [results[0]["identity"]],
            }
            response = await client.post(
                "/organization-candidates/sourcing/revalidate", json=verify
            )
            assert response.status_code == 200, response.text
            assert response.json()["results"][0]["decision"] == "allow"

    asyncio.run(run())
    assert len(calls) == 1
