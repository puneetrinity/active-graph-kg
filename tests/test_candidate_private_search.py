"""Database-backed contracts for tenant-private sourcing recall."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from types import SimpleNamespace

import psycopg
import pytest

OWNER_DSN = os.getenv("ACTIVEKG_RLS_TEST_OWNER_DSN")
RUNTIME_DSN = os.getenv("ACTIVEKG_RLS_TEST_RUNTIME_DSN")

pytestmark = pytest.mark.skipif(
    not (OWNER_DSN and RUNTIME_DSN),
    reason="private-search integration DSNs are not configured",
)

TENANT_A = f"private_search_a_{uuid.uuid4().hex[:8]}"
TENANT_B = f"private_search_b_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def private_candidates():
    from activekg.graph.candidate_repository import CandidateRepository

    ids = {
        "a_application": str(uuid.uuid4()),
        "a_upload": str(uuid.uuid4()),
        "a_signal_only": str(uuid.uuid4()),
        "b_application": str(uuid.uuid4()),
        "a_dual_node": str(uuid.uuid4()),
        "a_node_only": str(uuid.uuid4()),
        "a_node_without_linkedin": str(uuid.uuid4()),
        "b_node_only": str(uuid.uuid4()),
    }
    global_id = str(uuid.uuid4())
    ids["a_dual_global"] = global_id
    a_node_only_global_id = str(uuid.uuid4())
    a_node_without_linkedin_global_id = str(uuid.uuid4())
    malformed_provenance_global_id = str(uuid.uuid4())
    b_node_only_global_id = str(uuid.uuid4())
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO global_candidates (id, linkedin_id, name)
                VALUES
                    (%s, %s, 'PRIVATE_GLOBAL_NAME'),
                    (%s, %s, 'PRIVATE_NODE_ONLY_A'),
                    (%s, %s, 'PRIVATE_NODE_NO_LINKEDIN_A'),
                    (%s, %s, 'PRIVATE_MALFORMED_PROVENANCE_A'),
                    (%s, %s, 'PRIVATE_NODE_ONLY_B')
                """,
                (
                    global_id,
                    f"private-global-{uuid.uuid4().hex[:8]}",
                    a_node_only_global_id,
                    f"private-node-a-{uuid.uuid4().hex[:8]}",
                    a_node_without_linkedin_global_id,
                    f"private-node-no-linkedin-a-{uuid.uuid4().hex[:8]}",
                    malformed_provenance_global_id,
                    f"private-malformed-a-{uuid.uuid4().hex[:8]}",
                    b_node_only_global_id,
                    f"private-node-b-{uuid.uuid4().hex[:8]}",
                ),
            )
            cur.execute(
                """
                INSERT INTO candidates
                    (candidate_id, tenant_id, display_name, primary_email,
                     primary_phone, props, profile, headline, location_raw,
                     skills, seniority_level, linkedin_url, linkedin_id,
                     global_candidate_id)
                VALUES
                    (%s, %s, 'Applicant A', 'private-a@example.test', '+910000000001',
                     %s::jsonb, %s::jsonb, 'Python backend engineer', 'Bengaluru',
                     ARRAY[]::text[], 'senior', %s, %s, %s),
                    (%s, %s, 'Upload A', 'private-upload@example.test', '+910000000002',
                     '{}'::jsonb, %s::jsonb, 'Engineering leader', NULL,
                     ARRAY[]::text[], NULL, NULL, NULL, NULL),
                    (%s, %s, 'Signal-only A', NULL, NULL, '{}'::jsonb, '{}'::jsonb,
                     'Python engineer', 'Bengaluru', ARRAY['python'], 'senior',
                     %s, %s, NULL),
                    (%s, %s, 'Applicant B', 'private-b@example.test', '+910000000003',
                     %s::jsonb, %s::jsonb, 'Python backend engineer', 'Bengaluru',
                     ARRAY['python'], 'senior', %s, %s, NULL)
                """,
                (
                    ids["a_application"],
                    TENANT_A,
                    '{"skills":["Python","Django",{"email":"NESTED_PRIVATE"}]}',
                    '{"email":"RAW_PROFILE_PRIVATE_A","resume":"PRIVATE_RESUME_A"}',
                    "https://www.linkedin.com/in/private-app-a",
                    "private-app-a",
                    global_id,
                    ids["a_upload"],
                    TENANT_A,
                    '{"email":"RAW_PROFILE_PRIVATE_UPLOAD","notes":"PRIVATE_NOTES"}',
                    ids["a_signal_only"],
                    TENANT_A,
                    "https://www.linkedin.com/in/private-signal-a",
                    "private-signal-a",
                    ids["b_application"],
                    TENANT_B,
                    '{"skills":["Python"]}',
                    '{"email":"RAW_PROFILE_PRIVATE_B"}',
                    "https://www.linkedin.com/in/private-app-b",
                    "private-app-b",
                ),
            )
            source_rows = [
                (
                    ids["a_application"],
                    TENANT_A,
                    "vantahire",
                    "application",
                    f"application-{uuid.uuid4()}",
                ),
                (
                    ids["a_upload"],
                    TENANT_A,
                    "vantahire",
                    "resume",
                    f"upload-{uuid.uuid4()}",
                ),
                (
                    ids["a_signal_only"],
                    TENANT_A,
                    "signal",
                    "sourced_candidate",
                    f"signal-{uuid.uuid4()}",
                ),
                (
                    ids["b_application"],
                    TENANT_B,
                    "vantahire",
                    "application",
                    f"application-{uuid.uuid4()}",
                ),
            ]
            cur.executemany(
                """
                INSERT INTO candidate_source_records
                    (candidate_id, tenant_id, source, source_record_type,
                     source_record_id, payload)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                """,
                [(*row, '{"private":"SOURCE_PAYLOAD_SENTINEL"}') for row in source_rows],
            )
            cur.execute(
                """
                INSERT INTO candidate_identifiers
                    (candidate_id, tenant_id, identifier_type,
                     value_raw, value_normalized)
                VALUES (%s, %s, 'linkedin_url', %s, %s)
                """,
                (
                    ids["a_upload"],
                    TENANT_A,
                    "https://www.linkedin.com/in/private-upload-a",
                    "https://linkedin.com/in/private-upload-a",
                ),
            )
            cur.execute(
                """
                INSERT INTO nodes (id, tenant_id, classes, props, metadata)
                VALUES
                    (%s, %s, ARRAY['Document','Resume'], %s::jsonb, %s::jsonb),
                    (%s, %s, ARRAY['Document','Resume'], %s::jsonb, %s::jsonb),
                    (%s, %s, ARRAY['Document','Resume'], %s::jsonb, %s::jsonb),
                    (%s, %s, ARRAY['Document','Resume'], %s::jsonb, %s::jsonb)
                """,
                (
                    ids["a_dual_node"],
                    TENANT_A,
                    (
                        '{"resume_text":"PRIVATE_RESUME_NODE_DUAL",'
                        '"linkedin_url":"https://www.linkedin.com/in/private-app-a",'
                        '"current_title":"Senior Python Engineer",'
                        '"skills_normalized":["Python","FastAPI",{"email":"NODE_NESTED_PRIVATE"}],'
                        '"seniority":"senior","location":{"city":"Bengaluru"}}'
                    ),
                    (
                        '{"provenance_type":"platform_applicant",'
                        '"applicant_name":"Applicant A from node",'
                        '"applicant_email":"node-private-a@example.test",'
                        '"job_id":"PRIVATE_JOB_A"}'
                    ),
                    ids["a_node_only"],
                    TENANT_A,
                    (
                        '{"resume_text":"PRIVATE_RESUME_NODE_ONLY_A",'
                        '"linkedin_url":"https://www.linkedin.com/in/private-node-only-a",'
                        '"primary_titles":["Product Manager"],'
                        '"skills_raw":["Roadmapping"],'
                        '"seniority":"manager","location":{"raw":"Mumbai, India"}}'
                    ),
                    (
                        '{"provenance_type":"org_upload",'
                        '"applicant_name":"Node Only A",'
                        '"applicant_email":"node-only-a@example.test",'
                        '"notes":"PRIVATE_NODE_NOTES_A"}'
                    ),
                    ids["a_node_without_linkedin"],
                    TENANT_A,
                    (
                        '{"resume_text":"PRIVATE_RESUME_NODE_NO_LINKEDIN_A",'
                        '"current_title":"Finance Manager",'
                        '"skills_normalized":["Forecasting"],'
                        '"seniority":"manager","location":{"city":"Delhi"}}'
                    ),
                    (
                        '{"provenance_type":"platform_applicant",'
                        '"applicant_name":"No LinkedIn A",'
                        '"applicant_email":"no-linkedin-a@example.test"}'
                    ),
                    ids["b_node_only"],
                    TENANT_B,
                    (
                        '{"resume_text":"PRIVATE_RESUME_NODE_ONLY_B",'
                        '"linkedin_url":"https://www.linkedin.com/in/private-node-only-b",'
                        '"current_title":"Backend Engineer",'
                        '"skills_normalized":["Python"],'
                        '"seniority":"senior","location":{"city":"Bengaluru"}}'
                    ),
                    (
                        '{"provenance_type":"platform_applicant",'
                        '"applicant_name":"Node Only B",'
                        '"applicant_email":"node-only-b@example.test"}'
                    ),
                ),
            )
            cur.executemany(
                """
                INSERT INTO candidate_provenance
                    (global_candidate_id, source_type, tenant_id, source_detail)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                [
                    (
                        global_id,
                        "platform_applicant",
                        TENANT_A,
                        json.dumps(
                            {
                                "resume_node_id": ids["a_dual_node"],
                                "application_id": "PRIVATE_APPLICATION_A",
                            }
                        ),
                    ),
                    (
                        a_node_only_global_id,
                        "org_upload",
                        TENANT_A,
                        json.dumps(
                            {
                                "resume_node_id": ids["a_node_only"],
                                "org_id": "PRIVATE_ORG_A",
                            }
                        ),
                    ),
                    (
                        a_node_without_linkedin_global_id,
                        "platform_applicant",
                        TENANT_A,
                        json.dumps(
                            {
                                "resume_node_id": ids["a_node_without_linkedin"],
                                "application_id": "PRIVATE_NO_LINKEDIN_A",
                            }
                        ),
                    ),
                    (
                        b_node_only_global_id,
                        "platform_applicant",
                        TENANT_B,
                        json.dumps(
                            {
                                "resume_node_id": ids["b_node_only"],
                                "application_id": "PRIVATE_APPLICATION_B",
                            }
                        ),
                    ),
                    (
                        malformed_provenance_global_id,
                        "org_upload",
                        TENANT_A,
                        json.dumps(
                            {
                                "resume_node_id": "not-a-uuid",
                                "org_id": "PRIVATE_MALFORMED_ORG",
                            }
                        ),
                    ),
                ],
            )

    repo = CandidateRepository(RUNTIME_DSN)
    try:
        yield ids, repo
    finally:
        repo.close()


def test_private_search_is_tenant_scoped_and_includes_non_signal_sources(private_candidates):
    ids, repo = private_candidates
    rows_a, total_a = repo.search_tenant_private_candidates(
        tenant_id=TENANT_A,
        query_terms=["python", "backend", "bengaluru"],
        skills_any=["python", "django"],
        limit=100,
    )
    assert {row.candidate_id for row in rows_a} == {
        ids["a_upload"],
        f"node:{ids['a_dual_node']}",
        f"node:{ids['a_node_only']}",
        f"node:{ids['a_node_without_linkedin']}",
    }
    assert total_a == 4
    dual = next(row for row in rows_a if row.candidate_id == f"node:{ids['a_dual_node']}")
    assert dual.global_candidate_id == ids["a_dual_global"]
    assert dual.display_name == "Applicant A from node"
    assert dual.skills == ["fastapi", "python"]
    node_only = next(row for row in rows_a if row.candidate_id == f"node:{ids['a_node_only']}")
    assert node_only.skills == ["roadmapping"]
    no_linkedin = next(
        row for row in rows_a if row.candidate_id == f"node:{ids['a_node_without_linkedin']}"
    )
    assert no_linkedin.linkedin_url is None
    assert no_linkedin.linkedin_id is None
    upload = next(row for row in rows_a if row.candidate_id == ids["a_upload"])
    assert upload.skills == []
    assert upload.linkedin_id == "private-upload-a"

    rows_b, total_b = repo.search_tenant_private_candidates(
        tenant_id=TENANT_B,
        query_terms=["python"],
        skills_any=["python"],
        limit=100,
    )
    assert {row.candidate_id for row in rows_b} == {
        ids["b_application"],
        f"node:{ids['b_node_only']}",
    }
    assert total_b == 2

    no_context_rows, no_context_total = repo.search_tenant_private_candidates(
        tenant_id=None,
        query_terms=["python"],
        skills_any=["python"],
        limit=100,
    )
    assert no_context_rows == []
    assert no_context_total == 0


def test_private_search_has_honest_limit_and_no_raw_pii(private_candidates):
    _ids, repo = private_candidates
    rows, total = repo.search_tenant_private_candidates(
        tenant_id=TENANT_A,
        query_terms=["python"],
        skills_any=["python"],
        limit=1,
    )
    assert len(rows) == 1
    assert total == 4
    rendered = repr([asdict(row) for row in rows])
    assert "example.test" not in rendered
    assert "+910000" not in rendered
    assert "RAW_PROFILE_PRIVATE" not in rendered
    assert "SOURCE_PAYLOAD_SENTINEL" not in rendered
    assert "NESTED_PRIVATE" not in rendered
    assert "PRIVATE_RESUME_NODE" not in rendered
    assert "node-private-a@example.test" not in rendered
    assert "PRIVATE_NODE_NOTES" not in rendered


def test_private_search_api_projection_omits_raw_private_fields(private_candidates, monkeypatch):
    _ids, repo = private_candidates
    from activekg.api import main

    monkeypatch.setattr(main, "candidate_repo", repo)
    monkeypatch.setattr(main, "JWT_ENABLED", True)
    response = main.search_tenant_private_candidates(
        main.TenantPrivateCandidateSearchRequest(
            query_text="Python backend Bengaluru",
            skills_any=["Python", "Django"],
            limit=100,
        ),
        _rl=None,
        claims=SimpleNamespace(tenant_id=TENANT_A),
    )
    payload = response.model_dump()
    assert payload["surface"] == "tenant_private_v1"
    assert payload["total"] == 4
    assert payload["total_available"] == 4
    assert payload["truncated"] is False
    forbidden = {
        "primary_email",
        "primary_phone",
        "profile",
        "props",
        "payload",
        "resume",
        "notes",
        "job_id",
        "org_id",
    }
    for result in payload["results"]:
        assert forbidden.isdisjoint(result)
