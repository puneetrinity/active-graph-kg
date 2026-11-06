"""Immutable resume-transfer contracts, including optional real TS parity."""

import copy
import hashlib
import json
import os
import subprocess

import pytest

from activekg.candidate_profiles import profile_contracts as old
from activekg.candidate_profiles.source_contracts import (
    parse_resume_transfer,
    validate_lookup_transfer,
    validate_resume_transfer,
    validate_source_receipt,
)


def uid(n):
    return f"00000000-0000-4000-8000-{n:012d}"


def fingerprint(value):
    return hashlib.sha256(old.canonical(value).encode()).hexdigest()


def rehash(value):
    value["textHash"] = hashlib.sha256(value["text"].encode()).hexdigest()
    p = value["proposal"]
    p["proposalHash"] = fingerprint({k: v for k, v in p.items() if k != "proposalHash"})
    return value


def fixture():
    prefix, quote = "👩🏽‍💻 नमस्ते\n", "Built Go services for a team of 3."
    evidence = {
        "sourceId": uid(4),
        "sourceVersion": 1,
        "contentHash": "a" * 64,
        "locator": f"text:{len(prefix.encode('utf-16-le')) // 2}:{len(quote)}",
        "quote": quote,
    }
    p = {
        "schemaVersion": 1,
        "proposalId": uid(5),
        "sourceId": uid(4),
        "sourceVersion": 1,
        "proposalHash": "b" * 64,
        "originalDraftVersion": 0,
        "originalProfileVersion": 0,
        "originalPreferenceVersion": 0,
        "originalPreferenceDraftRevision": 0,
        "sourceKind": "resume",
        "facts": [
            {
                "factId": uid(6),
                "kind": "skill",
                "targetRowId": None,
                "value": "Go",
                "evidence": [evidence],
            }
        ],
        "reviewNotes": [],
        "expiresAt": "2026-11-01T00:00:00.000Z",
    }
    return rehash(
        {
            "schemaVersion": 1,
            "commandId": uid(1),
            "subjectId": uid(2),
            "bindingId": uid(3),
            "sourceId": uid(4),
            "sourceVersion": 1,
            "contentHash": "a" * 64,
            "textHash": "b" * 64,
            "policyVersion": "source-v1",
            "consentRevision": 1,
            "text": prefix + quote,
            "proposal": p,
        }
    )


def lookup_fixture(value=None):
    v = copy.deepcopy(value if value is not None else fixture())
    v["policyVersion"] = "rc-profile-v1"
    v["proposal"]["sourceKind"] = "professional_lookup"
    entries = [{"kind": f["kind"], "value": f["value"]} for f in v["proposal"]["facts"]]
    v["text"] = old.canonical(entries)
    v["contentHash"] = hashlib.sha256(v["text"].encode()).hexdigest()
    for f, entry in zip(v["proposal"]["facts"], entries, strict=True):
        quote = old.canonical(entry)
        offset = len(v["text"][: v["text"].index(quote)].encode("utf-16-le")) // 2
        f["evidence"] = [
            {
                "sourceId": v["sourceId"],
                "sourceVersion": v["sourceVersion"],
                "contentHash": v["contentHash"],
                "quote": quote,
                "locator": f"text:{offset}:{len(quote.encode('utf-16-le')) // 2}",
            }
        ]
    return rehash(v)


def test_photo_contract_preserves_optional_field_and_rejects_unsafe_hosts():
    url = "https://media.licdn.com/dms/image/test?e=1234567890"
    from activekg.candidate_profiles.spark_contracts import correction

    assert correction({"factId": uid(7), "kind": "photo", "value": {"url": url}})["value"] == {
        "url": url
    }
    base = {
        "displayName": "Test",
        "headline": "",
        "linkedinUrl": None,
        "currentLocation": None,
        "skills": [],
        "employment": [],
        "education": [],
        "projects": [],
    }
    assert "photoUrl" not in old.profile(base)
    assert old.profile({**base, "photoUrl": url})["photoUrl"] == url
    assert old.profile({**base, "photoUrl": None})["photoUrl"] is None
    for invalid in [
        "https://media.licdn.com.evil.test/a",
        "https://user@media.licdn.com/a",
        "https://media.licdn.com:443/a",
        "https://media.licdn.com/a#x",
        "http://media.licdn.com/a",
        "https://media.licdn.com/" + "a" * 2048,
    ]:
        with pytest.raises(ValueError):
            old.profile({**base, "photoUrl": invalid})


def test_lookup_is_distinct_from_resume_and_strictly_grounded():
    v = lookup_fixture()
    assert validate_lookup_transfer(v)["value"] == v
    with pytest.raises(ValueError):
        validate_resume_transfer(v)
    assert parse_resume_transfer(old.canonical(v).encode(), "professional_lookup")["value"] == v
    for field, replacement in [("policyVersion", "resume-v1"), ("contentHash", "f" * 64)]:
        changed = copy.deepcopy(v)
        changed[field] = replacement
        with pytest.raises(ValueError):
            validate_lookup_transfer(changed)
    changed = copy.deepcopy(v)
    changed["proposal"]["facts"][0]["value"] = "invented"
    with pytest.raises(ValueError):
        validate_lookup_transfer(rehash(changed))


def test_lookup_typescript_parity():
    root = os.environ.get("SPARK_SOURCE_FLOW_ROOT")
    if not root:
        if os.environ.get("SPARK_SOURCE_PARITY_REQUIRED") == "1":
            pytest.fail("Required lookup parity needs Flow artifact")
        pytest.skip("Lookup parity needs explicit Flow path")
    values = [lookup_fixture()]
    unicode = fixture()
    unicode["proposal"]["facts"][0].update(kind="displayName", value="नमस्ते 👩🏽‍💻")
    values.append(lookup_fixture(unicode))
    for field in ("contentHash", "textHash", "policyVersion"):
        changed = lookup_fixture()
        changed[field] = "wrong"
        values.append(changed)
    changed = lookup_fixture()
    changed["proposal"]["facts"][0]["value"] = "invented"
    values.append(rehash(changed))
    script = r"""
const {createRequire}=require('node:module'),fs=require('node:fs');
const req=createRequire(process.env.SPARK_SOURCE_FLOW_ROOT+'/package.json');req('tsx/cjs');
const {validateLookupTransfer}=req(process.env.SPARK_SOURCE_FLOW_ROOT+'/server/candidate-product/spark-contracts.ts');
process.stdout.write(JSON.stringify(JSON.parse(fs.readFileSync(0,'utf8')).map(v=>{
try{return {ok:true,...validateLookupTransfer(v)}}catch{return {ok:false}}})));
"""
    run = subprocess.run(
        ["node", "-e", script],
        input=json.dumps(values),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    for value, actual in zip(values, json.loads(run.stdout), strict=True):
        try:
            expected = {"ok": True, **validate_lookup_transfer(value)}
        except (ValueError, TypeError):
            expected = {"ok": False}
        assert actual == expected


def cases():
    result = [("unicode", fixture(), True)]
    for field in ("text", "policyVersion"):
        v = fixture()
        v[field] += "\x00"
        result.append(("nul_" + field, rehash(v), False))
    v = fixture()
    v["proposal"]["reviewNotes"] = ["hidden\x00note"]
    result.append(("nul_note", rehash(v), False))
    for size, valid in [(2048, True), (2049, False)]:
        v = fixture()
        v["text"] = "Go " + "x" * (size - 3)
        v["proposal"]["facts"][0]["evidence"][0].update(quote=v["text"], locator=f"text:0:{size}")
        result.append((f"quote_{size}", rehash(v), valid))
    v = fixture()
    v["text"] = "Go " + "x" * 2045
    fact = v["proposal"]["facts"][0]
    fact["evidence"][0].update(quote=v["text"], locator="text:0:2048")
    v["proposal"]["facts"] = [{**copy.deepcopy(fact), "factId": uid(100 + i)} for i in range(33)]
    result.append(("aggregate_quotes", rehash(v), False))
    for expiry in ["2026-11-01T00:00Z", "2026-11-01T00:00:00Z", "0000-02-29T00:00:00.1234567Z"]:
        v = fixture()
        v["proposal"]["expiresAt"] = expiry
        result.append((expiry, rehash(v), True))
    for key, val in [
        ("schemaVersion", True),
        ("consentRevision", 0),
        ("sourceVersion", -1),
        ("commandId", "not-uuid"),
        ("extra", True),
        ("textHash", "b" * 64),
    ]:
        v = fixture()
        v[key] = val
        result.append((key, v, False))
    for key, val in [
        ("sourceId", uid(99)),
        ("sourceVersion", 2),
        ("sourceKind", "professional_lookup"),
        ("proposalHash", "c" * 64),
        ("expiresAt", "2026-02-30T00:00:00Z"),
    ]:
        v = fixture()
        v["proposal"][key] = val
        if key != "proposalHash":
            rehash(v)
        result.append(("proposal_" + key, v, False))
    for key, val in [
        ("sourceId", uid(99)),
        ("sourceVersion", 2),
        ("contentHash", "c" * 64),
        ("locator", "text:0:2"),
        ("locator", "text:01:2"),
        ("locator", "text:1:1"),
        ("locator", "text:9007199254740992:1"),
        ("quote", "made up"),
    ]:
        v = fixture()
        v["proposal"]["facts"][0]["evidence"][0][key] = val
        result.append(("evidence_" + key + str(val), rehash(v), False))
    v = fixture()
    v["proposal"]["facts"].append(copy.deepcopy(v["proposal"]["facts"][0]))
    result.append(("duplicate_fact", rehash(v), False))
    v = fixture()
    v["proposal"]["facts"][0]["targetRowId"] = uid(9)
    result.append(("scalar_target", rehash(v), False))
    for size, valid in [(384 * 1024 + 1, True), (512 * 1024, True), (512 * 1024 + 1, False)]:
        v = fixture()
        v["text"] = "x" * size
        v["proposal"]["facts"] = []
        result.append((f"text_{size}", rehash(v), valid))
    for date, current, valid in [
        ("2024-02-29", False, True),
        ("2023-02-29", False, False),
        ("2024-02-29", True, False),
    ]:
        v = fixture()
        f = v["proposal"]["facts"][0]
        f["kind"] = "employment"
        f["value"] = {
            "id": uid(8),
            "employer": "Example",
            "title": "Engineer",
            "description": "",
            "start": None,
            "end": {"value": date, "precision": "day"},
            "current": current,
        }
        result.append((f"row_{date}_{current}", rehash(v), valid))
    return result


@pytest.mark.parametrize("name,value,valid", cases(), ids=[x[0] for x in cases()])
def test_transfer(name, value, valid):
    if not valid:
        with pytest.raises((ValueError, TypeError, UnicodeError)):
            parse_resume_transfer(json.dumps(value).encode())
    else:
        parsed = parse_resume_transfer(json.dumps(value).encode())
        assert parse_resume_transfer(parsed["canonical"].encode()) == parsed
        assert parsed["value"] == value


def test_duplicate_keys_and_raw_byte_limit():
    wire = old.canonical(fixture())
    with pytest.raises(ValueError):
        parse_resume_transfer(
            wire.replace('"commandId":', '"commandId":"duplicate","commandId":', 1).encode()
        )
    with pytest.raises(ValueError):
        parse_resume_transfer(b" " * (2 * 1024 * 1024 + 1))


def test_receipt_identity_and_no_publication():
    request = fixture()
    parsed = validate_resume_transfer(request)
    receipt = {
        k: request[k] for k in ("commandId", "subjectId", "bindingId", "sourceId", "sourceVersion")
    }
    receipt.update(
        schemaVersion=1,
        commandHash=parsed["hash"],
        proposalId=request["proposal"]["proposalId"],
        proposalHash=request["proposal"]["proposalHash"],
        outcome="stored",
        reason=None,
        canonicalProfileChanged=False,
        publicationActive=False,
    )
    assert validate_source_receipt(request, receipt) == receipt
    for key, value in [
        ("commandHash", "a" * 64),
        ("subjectId", uid(99)),
        ("proposalId", uid(99)),
        ("publicationActive", True),
        ("canonicalProfileChanged", True),
        ("reason", "stale_edit"),
    ]:
        with pytest.raises(ValueError):
            validate_source_receipt(request, {**receipt, key: value})


def test_real_typescript_and_frozen_design_parity():
    root = os.environ.get("SPARK_SOURCE_FLOW_ROOT")
    design = os.environ.get("SPARK_SOURCE_DESIGN_CONTRACT")
    if not root or (not design and os.environ.get("SPARK_SOURCE_PARITY_REQUIRED") != "1"):
        if os.environ.get("SPARK_SOURCE_PARITY_REQUIRED") == "1":
            pytest.fail("Required source parity needs pinned Flow and design artifacts")
        pytest.skip("Cross-language source proof needs explicit Flow and frozen design paths")
    script = r"""
const {createRequire}=require('node:module');
const fs=require('node:fs');
const req=createRequire(process.env.SPARK_SOURCE_FLOW_ROOT+'/package.json');req('tsx/cjs');
const product=req(process.env.SPARK_SOURCE_FLOW_ROOT+'/server/candidate-product/spark-contracts.ts');
(async()=>{const design=process.env.SPARK_SOURCE_DESIGN_CONTRACT?await import('file://'+process.env.SPARK_SOURCE_DESIGN_CONTRACT):null;
 const inputs=JSON.parse(fs.readFileSync(0,'utf8'));
 const evaluate=(fn,v)=>{try{return {ok:true,...fn(v)}}catch{return {ok:false}}};
 process.stdout.write(JSON.stringify(inputs.map(v=>({product:evaluate(product.validateResumeTransfer,v),...(design?{design:evaluate(design.validateResumeTransfer,v)}:{})}))));})();
"""
    inputs = cases()
    run = subprocess.run(
        ["node", "-e", script],
        input=json.dumps([x[1] for x in inputs]),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    outputs = json.loads(run.stdout)
    assert len(outputs) == len(inputs)
    for (name, value, valid), output in zip(inputs, outputs, strict=True):
        assert output["product"]["ok"] == valid, name
        if design:
            assert output["design"] == output["product"], name
        if valid:
            assert output["product"] == {"ok": True, **validate_resume_transfer(value)}, name


def test_typescript_receipt_parity():
    root = os.environ.get("SPARK_SOURCE_FLOW_ROOT")
    if not root:
        if os.environ.get("SPARK_SOURCE_PARITY_REQUIRED") == "1":
            pytest.fail("Required receipt parity needs pinned Flow artifact")
        pytest.skip("Cross-language source proof needs explicit Flow path")
    request = fixture()
    receipt = {
        k: request[k] for k in ("commandId", "subjectId", "bindingId", "sourceId", "sourceVersion")
    }
    receipt.update(
        schemaVersion=1,
        commandHash=validate_resume_transfer(request)["hash"],
        proposalId=request["proposal"]["proposalId"],
        proposalHash=request["proposal"]["proposalHash"],
        outcome="stored",
        reason=None,
        canonicalProfileChanged=False,
        publicationActive=False,
    )
    values = [receipt, {**receipt, "outcome": "rejected", "reason": "consent_changed"}]
    values += [
        {**receipt, key: value}
        for key, value in [
            ("commandHash", "b" * 64),
            ("proposalHash", "b" * 64),
            ("sourceVersion", 2),
            ("outcome", "rejected"),
            ("canonicalProfileChanged", True),
            ("publicationActive", True),
        ]
    ]
    script = r"""
const{createRequire}=require('node:module'),fs=require('node:fs');
const req=createRequire(process.env.SPARK_SOURCE_FLOW_ROOT+'/package.json');req('tsx/cjs');
const p=req(process.env.SPARK_SOURCE_FLOW_ROOT+'/server/candidate-product/spark-contracts.ts');
const {request,values}=JSON.parse(fs.readFileSync(0,'utf8'));
process.stdout.write(JSON.stringify(values.map(v=>{try{return{ok:true,value:p.validateSourceReceipt(request,v)}}catch{return{ok:false}}})));
"""
    run = subprocess.run(
        ["node", "-e", script],
        input=json.dumps({"request": request, "values": values}),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    for value, result in zip(values, json.loads(run.stdout), strict=True):
        try:
            expected = {"ok": True, "value": validate_source_receipt(request, value)}
        except ValueError:
            expected = {"ok": False}
        assert result == expected
