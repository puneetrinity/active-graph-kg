"""Candidate Spark contract tests: no providers, credentials or database required."""

import copy
import hashlib
import json
import os
import subprocess
import time
from uuid import uuid4

import pytest

from activekg.candidate_profiles.spark_contracts import (
    BUNDLED_PURPOSES,
    old,
    parse_command,
    parse_consent_projection,
    parse_exclusion_api,
    parse_interpretation_transfer,
    validate_interpretation_receipt,
)


def interpretation_transfer():
    p = {
        "schemaVersion": 1,
        "status": "proposal",
        "summary": "Change work arrangement",
        "changes": [
            {
                "kind": "preference_section",
                "answer": {
                    "section": "arrangement",
                    "answered": True,
                    "value": {"workArrangements": ["onsite"], "remoteRegions": None},
                },
            }
        ],
        "proposalId": str(uuid4()),
        "expectedDraftVersion": 0,
        "expectedProfileVersion": 0,
        "expectedPreferenceVersion": 0,
        "expectedPreferenceDraftRevision": 0,
        "expiresAt": "2026-12-01T00:00:00.000Z",
    }
    p["proposalHash"] = hashlib.sha256(old.canonical(p).encode()).hexdigest()
    return {
        "schemaVersion": 1,
        "commandId": str(uuid4()),
        "subjectId": str(uuid4()),
        "bindingId": str(uuid4()),
        "consentRevision": 1,
        "use": "change_preferences",
        "sourceProposalId": None,
        "sourceVersion": None,
        "targetRowId": None,
        "proposal": p,
    }


def test_interpretation_is_canonical_proposal_only():
    value = interpretation_transfer()
    parsed = parse_interpretation_transfer(old.canonical(value).encode())
    assert parsed["value"] == value
    assert len(parsed["hash"]) == 64


@pytest.mark.parametrize(
    "field",
    [
        "commandId",
        "subjectId",
        "bindingId",
        "commandHash",
        "proposalId",
        "proposalHash",
        "reason",
        "extra",
    ],
)
def test_interpretation_receipt_rejects_mismatch(field):
    value = interpretation_transfer()
    parsed = parse_interpretation_transfer(old.canonical(value).encode())
    receipt = {
        "schemaVersion": 1,
        "commandId": value["commandId"],
        "subjectId": value["subjectId"],
        "bindingId": value["bindingId"],
        "commandHash": parsed["hash"],
        "proposalId": value["proposal"]["proposalId"],
        "proposalHash": value["proposal"]["proposalHash"],
        "outcome": "stored",
        "reason": None,
    }
    assert validate_interpretation_receipt(value, receipt) == receipt
    receipt[field] = "tampered"
    with pytest.raises(ValueError):
        validate_interpretation_receipt(value, receipt)


def test_interpretation_typescript_parity():
    root = os.environ.get("SPARK_SOURCE_FLOW_ROOT")
    if not root:
        if os.environ.get("SPARK_SOURCE_PARITY_REQUIRED") == "1":
            pytest.fail("Required interpretation parity needs Flow artifact")
        pytest.skip("Explicit Flow path required")
    value = interpretation_transfer()
    raw = old.canonical(value)
    script = """
const {createRequire}=require('node:module');
const req=createRequire(process.env.SPARK_SOURCE_FLOW_ROOT+'/package.json');req('tsx/cjs');
const {parseInterpretationTransfer}=req(process.env.SPARK_SOURCE_FLOW_ROOT+'/server/candidate-product/interpretation.ts');
let data='';process.stdin.on('data',x=>data+=x);process.stdin.on('end',()=>{
try {process.stdout.write(JSON.stringify(parseInterpretationTransfer(data)));}
catch(e){process.stderr.write(String(e));process.exitCode=1;}});
"""
    run = subprocess.run(
        ["node", "-e", script],
        input=raw,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == parse_interpretation_transfer(raw.encode())


def test_interpretation_http_gate_and_strict_envelope(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from activekg.api import auth, candidate_profiles

    value = interpretation_transfer()
    raw = old.canonical(value)
    claims = auth.JWTClaims(
        tenant_id="candidate_" + value["subjectId"],
        actor_id="vantahire-backend",
        actor_type="service",
        scopes=["candidate-profile:owner-write"],
        issuer="vantahire",
        iat=int(time.time()),
        exp=int(time.time()) + 60,
    )
    monkeypatch.setattr(auth, "JWT_ENABLED", True)
    monkeypatch.setattr(auth, "JWT_ALGORITHM", "RS256")
    monkeypatch.setattr(auth, "JWT_AUDIENCE", "activekg")
    for flag in (
        "CANDIDATE_FOUNDATION_ENABLED",
        "CANDIDATE_PROFILE_ENABLED",
        "CANDIDATE_SPARK_ENABLED",
        "CANDIDATE_CONSENT_V2_ENABLED",
        "CANDIDATE_SOURCE_IMPORT_ENABLED",
    ):
        monkeypatch.setenv(flag, "true")
    monkeypatch.delenv("CANDIDATE_INTERPRETATION_ENABLED", raising=False)
    calls = []

    def storage(subject, binding, **kwargs):
        calls.append((subject, binding, kwargs))
        return {"stored": True}

    monkeypatch.setattr(candidate_profiles, "interpretation_storage", storage)
    app = FastAPI()
    app.include_router(candidate_profiles.router)
    app.dependency_overrides[auth.get_jwt_claims] = lambda: claims
    headers = {
        "content-type": "application/json",
        "x-candidate-binding-id": value["bindingId"],
        "x-candidate-command-id": value["commandId"],
        "x-command-sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }
    with TestClient(app) as client:
        assert (
            client.post(
                "/candidate-profiles/interpretations", content=raw, headers=headers
            ).status_code
            == 404
        )
        assert not calls
        monkeypatch.setenv("CANDIDATE_INTERPRETATION_ENABLED", "true")
        response = client.post("/candidate-profiles/interpretations", content=raw, headers=headers)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert len(calls) == 1
        assert (
            client.post(
                "/candidate-profiles/interpretations",
                content=raw,
                headers={**headers, "x-command-sha256": "0" * 64},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/candidate-profiles/interpretations?extra=1", content=raw, headers=headers
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/candidate-profiles/interpretations", content="x" * 131073, headers=headers
            ).status_code
            == 413
        )
        claims.scopes = ["candidate-profile:owner-read"]
        assert (
            client.post(
                "/candidate-profiles/interpretations", content=raw, headers=headers
            ).status_code
            == 403
        )
        assert len(calls) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "hash",
        "prompt",
        "model_response",
        "duplicate",
        "purpose",
        "target",
        "authorisation",
        "nul",
        "over_limit",
    ],
)
def test_interpretation_refuses_unsafe_transfer(mutation):
    value = interpretation_transfer()
    if mutation == "hash":
        value["proposal"]["summary"] = "changed"
    elif mutation in ("prompt", "model_response"):
        value[mutation] = "must never persist this"
    elif mutation == "purpose":
        value["use"] = "source_correction"
    elif mutation == "target":
        value["targetRowId"] = str(uuid4())
    elif mutation == "authorisation":
        value["proposal"]["changes"][0]["answer"] = {
            "section": "authorisation",
            "answered": True,
            "value": [],
        }
    elif mutation == "duplicate":
        value["proposal"]["changes"] *= 2
    elif mutation == "nul":
        value["proposal"]["summary"] = "hidden\x00text"
    elif mutation == "over_limit":
        value["proposal"]["changes"] *= 21
    if mutation != "hash":
        p = copy.deepcopy(value["proposal"])
        p.pop("proposalHash")
        value["proposal"]["proposalHash"] = hashlib.sha256(old.canonical(p).encode()).hexdigest()
    with pytest.raises(ValueError):
        parse_interpretation_transfer(old.canonical(value).encode())


def command(value=None):
    return {
        "schemaVersion": 3,
        "idempotencyKey": str(uuid4()),
        "expectedDraftVersion": 0,
        "expectedProfileVersion": 0,
        "expectedPreferenceVersion": 0,
        "expectedPreferenceDraftRevision": 0,
        "action": "save_preference_section",
        "payload": {
            "section": "employment_types",
            "answered": True,
            "value": ["gig_freelance", "contract"] if value is None else value,
        },
    }


def test_canonical_command_is_stable():
    c = command()
    parsed = parse_command(json.dumps(c).encode())
    assert parse_command(parsed["canonical"].encode()) == parsed
    assert parsed["value"]["payload"]["value"] == ["gig_freelance", "contract"]


@pytest.mark.parametrize(
    "value", [[], ["unknown"], ["contract", "contract"], "contract", None, [True]]
)
def test_no_implied_or_duplicate_employment_choices(value):
    c = command()
    c["payload"]["value"] = value
    with pytest.raises((ValueError, TypeError)):
        parse_command(json.dumps(c).encode())


def test_duplicate_keys_refused():
    raw = json.dumps(command()).replace(
        '"schemaVersion": 3', '"schemaVersion": 3, "schemaVersion": 3'
    )
    with pytest.raises(ValueError):
        parse_command(raw.encode())


def consent():
    return {
        "schemaVersion": 2,
        "subjectId": str(uuid4()),
        "bindingId": str(uuid4()),
        "eventId": str(uuid4()),
        "version": 1,
        "action": "grant",
        "copyVersion": 1,
        "copyHash": "a" * 64,
        "purposes": BUNDLED_PURPOSES,
    }


def test_full_bundle_and_withdrawal():
    c = consent()
    assert parse_consent_projection(json.dumps(c).encode())["value"] == c
    c.update(action="withdraw", copyVersion=None, copyHash=None, purposes=[])
    assert parse_consent_projection(json.dumps(c).encode())["value"] == c


@pytest.mark.parametrize(
    "patch",
    [
        {"purposes": BUNDLED_PURPOSES[:-1]},
        {"purposes": BUNDLED_PURPOSES + ["marketing"]},
        {"version": True},
        {"emailVerified": True},
        {"copyVersion": 0},
    ],
)
def test_invalid_consent_refused(patch):
    with pytest.raises(ValueError):
        parse_consent_projection(json.dumps({**consent(), **patch}).encode())


def test_exclusion_company_requires_exact_registry_identity():
    request = {
        "idempotencyKey": str(uuid4()),
        "expectedPreferenceDraftRevision": 0,
        "restrictions": [{"kind": "exclude_company", "companyId": str(uuid4())}],
    }
    assert (
        parse_exclusion_api(json.dumps(request).encode(), "exclusion_preview_request")["value"]
        == request
    )
    request["restrictions"] = [{"kind": "exclude_company", "companyName": "Ambiguous"}]
    with pytest.raises(ValueError):
        parse_exclusion_api(json.dumps(request).encode(), "exclusion_preview_request")


@pytest.mark.parametrize("kind", ["early_stage", "no_on_call", "hide_me_from_employer"])
def test_unsupported_notes_cannot_become_hard_filters(kind):
    request = {
        "idempotencyKey": str(uuid4()),
        "expectedPreferenceDraftRevision": 0,
        "restrictions": [{"kind": kind}],
    }
    with pytest.raises(ValueError):
        parse_exclusion_api(json.dumps(request).encode(), "exclusion_preview_request")
