"""Synthetic-only v2 extraction proof. Never opens a provider or production DB."""

import asyncio
import json
import time
from dataclasses import replace

import httpx
import pytest

from activekg.candidate_index import resume_policy as v2
from activekg.candidate_index.contracts import IndexContractError
from activekg.candidate_index.processing import resume_extraction
from activekg.candidate_index.providers import WorkRefused, extract_once
from tests.test_candidate_index import POLICY


def policy():
    return replace(POLICY, **v2.policy_fields())


def raw(skills):
    return {"skills_raw": skills, "skills_normalized": [], "confidence": 0.9}


@pytest.mark.parametrize(
    "entries,source,expected",
    [
        (["AWS (EC2, S3)"], "AWS (EC2, S3)", ["AWS", "EC2", "S3"]),
        (["AWS EC2"], "AWS (EC2, S3)", ["AWS EC2", "AWS", "EC2"]),
        (
            ["Azure App Services"],
            "Azure (App Services, Blob Storage)",
            ["Azure App Services", "Azure", "App Services"],
        ),
        (["AWS App Services"], "AWS (EC2, S3); Azure (App Services)", []),
        (
            ["Google Cloud BigQuery"],
            "Google Cloud: BigQuery, GKE\nAzure: App Services",
            ["Google Cloud BigQuery", "Google Cloud", "BigQuery"],
        ),
        (["AWS App Services"], "AWS: EC2, S3; Azure: App Services", []),
        (["AWS App Services"], "AWS (Azure (App Services), EC2)", []),
        (
            ["Python (programming language)"],
            "Python (programming language)",
            ["Python (programming language)"],
        ),
        (
            ["Tools (C++, C#, .NET, Node.js, SNS/SQS)"],
            "Tools (C++, C#, .NET, Node.js, SNS/SQS)",
            ["Tools", "C++", "C#", ".NET", "Node.js", "SNS/SQS"],
        ),
        (["AWS", "AWS (EC2, S3)", "EC2"], "AWS (EC2, S3)", ["AWS", "EC2", "S3"]),
    ],
)
def test_explicit_group_association(entries, source, expected):
    result = v2.validate(raw(entries), source, {})
    assert result["skills_raw"] == expected
    assert result["skills_normalized"] == [v2.normalized(s) for s in expected]


def test_group_invented_member_is_counted_not_reassigned():
    stats = {}
    result = v2.validate(
        raw(["AWS (EC2, App Services)"]), "AWS (EC2, S3); Azure (App Services)", stats
    )
    assert result["skills_raw"] == ["AWS", "EC2"]
    assert stats["unsupported_group_components"] == 1


def test_expansion_cannot_bypass_100_limit():
    entries = [f"Group{i} (A{i}, B{i})" for i in range(34)]
    stats = {}
    with pytest.raises(WorkRefused):
        v2.validate(raw(entries), "; ".join(entries), stats)
    assert stats["validation_reason"] == "expanded_list_limit"


@pytest.mark.parametrize(
    "skill,text,expected",
    [
        ("C++", "Skills: C++, C#, .NET", True),
        ("C#", "C#, .NET", True),
        (".NET", "C#/.NET", True),
        ("Node.js", "Node.js and SQL", True),
        ("R", "Python / R / Julia", True),
        ("R", "Power BI", False),
        ("C", "C++", False),
        ("C", "C#", False),
        ("Node", "Node.js", False),
        ("SQL", "PostgreSQL", False),
        ("R", "RStudio", False),
        ("Python", "PYTHON. SQL.", True),
        ("Python", "Python not required", True),
        ("AWS App Services", "Azure (App Services), AWS (EC2)", False),
        ("machine learning", "Machine\n  Learning", True),
    ],
)
def test_literal_edges_and_known_context_limitation(skill, text, expected):
    assert v2.supported(skill, text) is expected


def test_supported_unique_skills_only_and_no_normalized_backdoor():
    source = "Azure App Services; Python; C++; Node.js"
    data = raw(["Azure App Services", "Python", " PYTHON ", "C++", "Node.js", "AWS App Services"])
    data.update(
        skills_normalized=["aws", "Invented"],
        primary_skills=["Python", "AWS"],
        years_by_skill={"Python": 3, "AWS": 7},
    )
    diagnostics = {}
    result = resume_extraction(data, source, policy(), POLICY.primary_model_id, diagnostics)
    assert result.namespace["skills_raw"] == ["Azure App Services", "Python", "C++", "Node.js"]
    assert result.namespace["skills_normalized"] == [
        "azure app services",
        "python",
        "c++",
        "node.js",
    ]
    assert result.namespace["primary_skills"] == ["Python"]
    assert result.namespace["years_by_skill"] == {"Python": 3}
    assert diagnostics["unsupported_skills"] == 1
    assert diagnostics["duplicate_skills"] == 1
    assert diagnostics["retained_skills"] == 4
    assert "AWS" not in result.professional_text


def test_100_is_ceiling_not_target_and_no_truncation():
    skills = [f"Tool{i}" for i in range(100)]
    result = resume_extraction(raw(skills), ", ".join(skills), policy(), POLICY.primary_model_id)
    assert result.namespace["skills_raw"] == skills
    assert "Tool99" in "".join(c["text"] for c in result.chunks)
    assert (
        len(
            resume_extraction(
                raw(["Python"]), "Python", policy(), POLICY.primary_model_id
            ).namespace["skills_raw"]
        )
        == 1
    )
    with pytest.raises(WorkRefused):
        resume_extraction(
            raw(skills + ["Tool100"]), ", ".join(skills), policy(), POLICY.primary_model_id
        )


@pytest.mark.parametrize("field,limit", list(v2.LIST_LIMITS.items()))
def test_each_list_limit_matches_prompt_and_schema(field, limit):
    assert f"{field}: {limit}" in v2.prompt("")[0]
    values = [f"Tool{i}" for i in range(limit)]
    data = raw(["Python"])
    data[field] = values
    v2.validate(data, "Python " + " ".join(values), {})
    data[field] = values + ["extra"]
    with pytest.raises(WorkRefused):
        v2.validate(data, "Python " + " ".join(values), {})


def test_nonliteral_function_is_not_removed():
    data = raw(["Python"])
    data["functions"] = ["backend"]
    assert v2.validate(data, "Python APIs", {})["functions"] == ["backend"]


def test_unknown_location_field_is_not_silently_dropped():
    data = raw(["Python"])
    data["location"] = {"city": "Pune", "invented_field": "not allowed"}
    with pytest.raises(WorkRefused):
        resume_extraction(data, "Python", policy(), POLICY.primary_model_id)


def test_full_unicode_namespace_is_bounded_without_trimming():
    skills = ["😀" * 115 + str(i) for i in range(100)]
    diagnostics = {}
    with pytest.raises(WorkRefused):
        v2.validate(raw(skills), " ".join(skills), diagnostics)
    assert diagnostics["validation_reason"] == "namespace_bytes"


def test_output_allowance_changes_policy_identity(monkeypatch):
    before = policy()
    monkeypatch.setattr(v2, "OUTPUT_TOKENS", 2048)
    assert policy().digest != before.digest
    with pytest.raises(WorkRefused, match="policy_mismatch"):
        resume_extraction(raw(["Python"]), "Python", before, POLICY.primary_model_id)


def test_old_policy_keeps_old_bounds():
    with pytest.raises(WorkRefused):
        resume_extraction(
            raw([f"Tool{i}" for i in range(16)]), "Python", POLICY, POLICY.primary_model_id
        )


def test_configuration_requires_explicit_new_pin(monkeypatch):
    from activekg.candidate_index.admission import configuration
    from tests.test_candidate_index_workers import runtime_environment

    old = runtime_environment(monkeypatch, "extract")
    assert configuration().policy == old
    monkeypatch.setenv("CANDIDATE_INDEX_EXTRACTION_ADAPTER", "v2")
    with pytest.raises(IndexContractError, match="policy_not_pinned"):
        configuration()
    new = replace(old, **v2.policy_fields())
    monkeypatch.setenv("CANDIDATE_INDEX_POLICY_SHA256", new.digest)
    assert configuration().policy == new


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_real_http_adapter_allowance_safe_usage_and_privacy_order(finish):
    events, diagnostics = [], {}

    async def authority():
        events.append("privacy")

    async def reply(request):
        assert events == ["privacy"]
        body = json.loads(request.content)
        assert body["max_tokens"] == 4096 and body["temperature"] == 0.1
        assert body["reasoning_effort"] == "medium"
        assert "skills_raw: 100" in body["messages"][0]["content"]
        events.append("send")
        payload = {
            "model": POLICY.primary_model_id,
            "usage": {"completion_tokens": 2030},
            "choices": [
                {
                    "finish_reason": finish,
                    "message": {"role": "assistant", "content": json.dumps(raw(["Python"]))},
                }
            ],
        }
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(json.dumps(payload).encode()),
        )

    async def run():
        return await extract_once(
            "Python",
            POLICY.primary_model_id,
            api_key="fake",
            deadline=time.monotonic() + 1,
            before_send=authority,
            managed_v2=True,
            diagnostics=diagnostics,
            transport=httpx.MockTransport(reply),
        )

    if finish == "length":
        with pytest.raises(WorkRefused):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert diagnostics == {"finish_reason": finish, "completion_tokens": 2030}


def test_slow_cutoff_leaves_low_only_the_remaining_budget():
    calls = []

    async def authority():
        pass

    async def reply(request):
        body = json.loads(request.content)
        calls.append(body["reasoning_effort"])
        if len(calls) == 1:
            await asyncio.sleep(0.04)
            payload = {"model": POLICY.primary_model_id, "choices": [{"finish_reason": "length"}]}
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=httpx.ByteStream(json.dumps(payload).encode()),
            )
        await asyncio.sleep(0.3)
        raise AssertionError("low response must be cancelled at shared deadline")

    async def run():
        deadline = time.monotonic() + 0.15
        for effort, code in [("medium", "output_truncated"), ("low", "provider_timeout")]:
            with pytest.raises(WorkRefused, match=code):
                await extract_once(
                    "Python",
                    POLICY.primary_model_id,
                    api_key="fake",
                    deadline=deadline,
                    before_send=authority,
                    managed_v2=True,
                    reasoning_effort=effort,
                    transport=httpx.MockTransport(reply),
                )
        assert time.monotonic() >= deadline

    asyncio.run(run())
    assert calls == ["medium", "low"]


def test_slow_primary_cannot_give_fallback_a_fresh_budget():
    calls = []

    async def authority():
        pass

    async def slow(request):
        calls.append(json.loads(request.content)["model"])
        await asyncio.sleep(0.15)
        raise AssertionError("must be cancelled")

    async def run():
        deadline = time.monotonic() + 0.03
        with pytest.raises(WorkRefused, match="provider_timeout"):
            await extract_once(
                "Python",
                POLICY.primary_model_id,
                api_key="fake",
                deadline=deadline,
                before_send=authority,
                managed_v2=True,
                transport=httpx.MockTransport(slow),
            )
        with pytest.raises(WorkRefused, match="dispatch_exhausted"):
            await extract_once(
                "Python",
                POLICY.fallback_model_id,
                api_key="fake",
                deadline=deadline,
                before_send=authority,
                managed_v2=True,
                transport=httpx.MockTransport(slow),
            )

    asyncio.run(run())
    assert calls == [POLICY.primary_model_id]


@pytest.mark.parametrize("attempt", [1, 2])
def test_worker_v2_saves_full_list_and_emits_safe_counts(attempt, caplog):
    from dataclasses import asdict

    from activekg.candidate_index.processing import GenerationWorker
    from tests.test_candidate_index_workers import Repository

    current = policy()
    repo = Repository(attempt=attempt)
    skills = [f"Tool{i}" for i in range(70)]
    text = " ".join(skills)
    from activekg.candidate_index.contracts import sha256

    repo.dispatch.update(
        retry_mode="medium",
        policy=asdict(current),
        policy_sha256=current.digest,
        professional_text=text,
        source_manifest={"source_hash": sha256(text)},
    )
    calls = []

    def reply(request):
        body = json.loads(request.content)
        calls.append(body["model"])
        payload = {
            "model": body["model"],
            "usage": {"completion_tokens": 2200},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(raw(skills + ["InventedSkill", "Tool0"])),
                    },
                }
            ],
        }
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(json.dumps(payload).encode()),
        )

    with caplog.at_level("INFO", logger="activekg.candidate_index.processing"):
        assert (
            asyncio.run(
                GenerationWorker(
                    repo, current, api_key="fake", transport=httpx.MockTransport(reply)
                ).process({})
            )
            == "completed"
        )
    assert len(calls) == 1
    assert repo.completes[0]["namespace"]["skills_raw"] == skills
    line = next(
        r.message for r in caplog.records if r.message.startswith("candidate_index_extraction_v2 ")
    )
    metrics = json.loads(line.split(" ", 1)[1])
    assert metrics["unsupported_skills"] == 1 and metrics["duplicate_skills"] == 1
    assert metrics["retained_skills"] == 70 and metrics["completion_tokens"] == 2200
    assert metrics["attempt"] == attempt and len(metrics["generation_hash"]) == 64
    assert "Tool0" not in line and "InventedSkill" not in line and "fake" not in line


def test_chunk_limit_remains_lossless_or_explicit_refusal():
    from activekg.candidate_index.contracts import IndexContractError, chunk_manifest

    with pytest.raises(IndexContractError, match="chunk_overflow"):
        chunk_manifest("x" * (32 * 1200 + 1), policy())


def test_old_prompt_and_schema_untouched_by_v2():
    from activekg.extraction.prompt import build_extraction_prompt
    from activekg.extraction.schema import MAX_SKILLS_RAW

    assert "LIST LIMITS" not in build_extraction_prompt("Python")[0]
    assert MAX_SKILLS_RAW == 15


@pytest.mark.parametrize("long_names", [False, True])
def test_actual_postgres_record_bounds_when_disposable_database_selected(long_names):
    import os

    import psycopg
    from psycopg.conninfo import conninfo_to_dict

    dsn = os.environ.get("RESUME_V2_TEST_DSN")
    if not dsn:
        pytest.skip("explicit disposable byte-bound database not selected")
    info = conninfo_to_dict(dsn)
    assert info["dbname"] == "resume_v2_test"
    assert info["host"].startswith("/tmp/ealana-resume-v2-pg.")
    skills = (
        [f"技術{i}" for i in range(100)]
        if not long_names
        else ["x" * 117 + f"{i:03}" for i in range(100)]
    )
    result = resume_extraction(raw(skills), " ".join(skills), policy(), POLICY.primary_model_id)
    with psycopg.connect(dsn) as conn:
        # Actual PostgreSQL jsonb representation and the shipped extraction
        # table's byte/array checks, in a rollback-only temporary table.
        conn.execute(
            "CREATE TEMP TABLE bounds (namespace jsonb CHECK (octet_length(namespace::text)<=65536), "
            "evidence jsonb CHECK (jsonb_array_length(evidence) BETWEEN 1 AND 32 AND octet_length(evidence::text)<=16384), "
            "chunks jsonb CHECK (jsonb_array_length(chunks) BETWEEN 1 AND 32 AND octet_length(chunks::text)<=262144))"
        )
        conn.execute(
            "INSERT INTO bounds VALUES (%s::jsonb,%s::jsonb,%s::jsonb)",
            tuple(
                json.dumps(v, ensure_ascii=False)
                for v in (result.namespace, result.evidence, result.chunks)
            ),
        )
        assert (
            conn.execute(
                "SELECT jsonb_array_length(namespace->'skills_raw') FROM bounds"
            ).fetchone()[0]
            == 100
        )
        conn.rollback()
