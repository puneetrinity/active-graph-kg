"""Pure synthetic wire tests. No database, model or network calls."""

import asyncio
import json
import time
from types import SimpleNamespace
from uuid import uuid4

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI

from activekg.api import candidate_sourcing as api
from activekg.api.candidate_sourcing import (
    SOURCING_SCOPE,
    SourcingAuthError,
    verify_sourcing_authority,
)
from activekg.candidate_index import sourcing
from activekg.candidate_index.sourcing_contracts import (
    REVALIDATE_BODY_BYTES,
    SEARCH_BODY_BYTES,
    SourcingContractError,
    parse_public_applicants,
    parse_sourcing_revalidate,
    parse_sourcing_search,
)


def search_body():
    return {
        "version": 1,
        "requestId": "synthetic-request",
        "flowRunId": str(uuid4()),
        "externalJobId": 189,
        "artifactHash": "a" * 64,
        "query": "Python backend engineer",
        "limit": 100,
    }


def test_public_applicants_contract_has_no_job_or_tenant_authority():
    body = {"version": 1, "requestId": "membership", "publicIds": [str(uuid4())]}
    assert len(parse_public_applicants(json.dumps(body).encode()).publicIds) == 1
    for value in [
        {**body, "tenantId": "org_5"},
        {**body, "externalJobId": 1},
        {**body, "publicIds": body["publicIds"] * 2},
        {**body, "publicIds": []},
        {**body, "publicIds": [str(uuid4()) for _ in range(101)]},
        {**body, "publicIds": ["00000000-0000-0000-0000-000000000000"]},
    ]:
        with pytest.raises(SourcingContractError):
            parse_public_applicants(json.dumps(value).encode())


def test_public_applicants_reader_checks_exact_order_and_boolean(monkeypatch):
    item = str(uuid4())
    command = parse_public_applicants(
        json.dumps({"version": 1, "requestId": "membership", "publicIds": [item]}).encode()
    )
    values = [{"publicId": item, "isApplicant": True}]

    async def query(*_):
        return values

    monkeypatch.setattr(sourcing, "_query", query)
    result = asyncio.run(
        sourcing.public_applicants(command, "org_32", "unused", deadline=time.monotonic() + 1)
    )
    assert result["results"] == values
    for invalid in [
        [{"publicId": str(uuid4()), "isApplicant": True}],
        [{"publicId": item, "isApplicant": "true"}],
        [],
    ]:
        values = invalid
        with pytest.raises(sourcing.SourcingUnavailable):
            asyncio.run(
                sourcing.public_applicants(
                    command, "org_32", "unused", deadline=time.monotonic() + 1
                )
            )


def identity():
    return {
        "candidateId": str(uuid4()),
        "referenceId": str(uuid4()),
        "applicationId": 123,
        "applicationJobId": 188,
        "resumeVersionId": str(uuid4()),
        "sourceId": str(uuid4()),
        "generationId": str(uuid4()),
        "sourceHash": "b" * 64,
    }


def revalidate_body():
    return {
        "version": 1,
        "requestId": "synthetic-request",
        "externalJobId": 189,
        "items": [identity()],
    }


def wire(value):
    return json.dumps(value, ensure_ascii=False).encode()


@pytest.fixture(scope="module")
def signing_keys():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = (
        private.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    return private, public


def service_claims():
    now = int(time.time())
    return {
        "iss": "signal",
        "aud": "activekg",
        "sub": "signal-service",
        "actor_type": "service",
        "tenant_id": "org_32",
        "scopes": [SOURCING_SCOPE],
        "request_id": "synthetic-request",
        "iat": now,
        "exp": now + 300,
        "jti": str(uuid4()),
    }


def test_service_authority_is_signed_and_request_bound(signing_keys):
    private, public = signing_keys
    token = jwt.encode(service_claims(), private, algorithm="RS256")
    result = verify_sourcing_authority(token, "synthetic-request", public_key=public)
    assert result.tenant_id == "org_32"
    assert result.request_id == "synthetic-request"
    with pytest.raises(SourcingAuthError):
        verify_sourcing_authority(token, "other-request", public_key=public)


@pytest.mark.parametrize(
    "field,value",
    [
        ("iss", "vantahire"),
        ("aud", "signal"),
        ("aud", ["activekg"]),
        ("sub", "vantahire-backend"),
        ("actor_type", "user"),
        ("tenant_id", "public"),
        ("tenant_id", "org_0"),
        ("tenant_id", "org_032"),
        ("tenant_id", "org_2147483648"),
        ("tenant_id", None),
        ("scopes", ["kg:read"]),
        ("scopes", [SOURCING_SCOPE, "kg:read"]),
        ("scopes", SOURCING_SCOPE),
        ("request_id", "other-request"),
        ("jti", ""),
        ("jti", "bad\nnonce"),
        ("iat", True),
        ("exp", True),
    ],
)
def test_service_authority_refuses_wrong_claims(signing_keys, field, value):
    private, public = signing_keys
    token = jwt.encode({**service_claims(), field: value}, private, algorithm="RS256")
    with pytest.raises(SourcingAuthError, match="^candidate_sourcing_unauthorized$"):
        verify_sourcing_authority(token, "synthetic-request", public_key=public)


@pytest.mark.parametrize("missing", list(service_claims()))
def test_service_authority_requires_every_claim(signing_keys, missing):
    private, public = signing_keys
    claims = service_claims()
    del claims[missing]
    token = jwt.encode(claims, private, algorithm="RS256")
    with pytest.raises(SourcingAuthError):
        verify_sourcing_authority(token, "synthetic-request", public_key=public)


@pytest.mark.parametrize("issued,expires", [(-301, -1), (0, 301), (60, 300), (-10, -10)])
def test_service_authority_enforces_time_budget(signing_keys, issued, expires):
    private, public = signing_keys
    now = int(time.time())
    claims = {**service_claims(), "iat": now + issued, "exp": now + expires}
    token = jwt.encode(claims, private, algorithm="RS256")
    with pytest.raises(SourcingAuthError):
        verify_sourcing_authority(token, "synthetic-request", public_key=public)


def test_service_authority_rejects_other_keys_algorithms_and_bad_wire(signing_keys):
    _, public = signing_keys
    wrong = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    tokens = [
        jwt.encode(service_claims(), wrong, algorithm="RS256"),
        jwt.encode(service_claims(), "synthetic-test-secret-at-least-32-bytes", algorithm="HS256"),
        "not-a-token",
        "x" * 16385,
        "",
    ]
    for token in tokens:
        with pytest.raises(SourcingAuthError) as failure:
            verify_sourcing_authority(token, "synthetic-request", public_key=public)
        assert str(failure.value) == "candidate_sourcing_unauthorized"
        assert failure.value.__suppress_context__


def test_http_routes_bind_org_and_request_without_any_provider(signing_keys, monkeypatch):
    private, public = signing_keys
    monkeypatch.setattr(api.auth, "JWT_ENABLED", True)
    monkeypatch.setattr(api.auth, "SIGNAL_JWT_PUBLIC_KEY", public)
    monkeypatch.setattr(api, "enabled", lambda _: True)
    monkeypatch.setattr(api, "configuration", lambda: SimpleNamespace(policy="synthetic-policy"))
    monkeypatch.setenv("ACTIVEKG_DSN", "synthetic-no-connection")
    calls = []

    async def fake_search(command, tenant, dsn, policy, *, deadline):
        calls.append((tenant, command.requestId))
        assert policy == "synthetic-policy" and deadline > time.monotonic()
        return {"version": 1, "requestId": command.requestId, "results": []}

    async def fake_revalidate(command, tenant, dsn, *, deadline):
        calls.append((tenant, command.requestId))
        return {"version": 1, "requestId": command.requestId, "results": []}

    monkeypatch.setattr(sourcing, "search", fake_search)
    monkeypatch.setattr(sourcing, "revalidate", fake_revalidate)
    monkeypatch.setattr(sourcing, "public_applicants", fake_revalidate)
    app = FastAPI()
    app.include_router(api.router)
    token = jwt.encode(service_claims(), private, algorithm="RS256")

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            for endpoint, body in [
                ("search", search_body()),
                ("revalidate", revalidate_body()),
                (
                    "public-applicants",
                    {"version": 1, "requestId": "synthetic-request", "publicIds": [str(uuid4())]},
                ),
            ]:
                path = "/organization-candidates/sourcing/" + endpoint
                assert (await client.post(path, json=body)).status_code == 401
                headers = {"authorization": "Bearer " + token}
                assert (
                    await client.post(path, json={**body, "tenantId": "org_99"}, headers=headers)
                ).status_code == 422
                assert (
                    await client.post(
                        path, json={**body, "requestId": "different"}, headers=headers
                    )
                ).status_code == 403
                assert (await client.post(path, json=body, headers=headers)).status_code == 200

    asyncio.run(run())
    assert calls == [("org_32", "synthetic-request")] * 3


def test_response_whitelists_facts_and_counts_withheld_rows():
    body = search_body()
    request = parse_sourcing_search(wire(body))
    row = {
        "identity": identity(),
        "linkedPublicId": None,
        "distance": 0.25,
        "evidence": {
            "version": 1,
            "kind": "candidate_provided",
            "scope": "org_32",
            "sourceVersion": 1,
            "observedAt": "2026-10-16T10:00:00Z",
            "skills": ["C++", "Python"],
            "location": {"city": None, "country": None},
        },
    }
    result = sourcing.search_response([row], request, "org_32")
    assert result["counts"] == {"returned": 1, "withheldInvalidEvidence": 0}
    for changes in [
        {"scope": "org_99"},
        {"skills": ["private@example.test"]},
        {"skills": ["Python"] * 101},
        {"email": "private@example.test"},
    ]:
        bad = {**row, "evidence": {**row["evidence"], **changes}}
        result = sourcing.search_response([bad], request, "org_32")
        assert result["results"] == []
        assert result["counts"]["withheldInvalidEvidence"] == 1
        assert "private@example" not in json.dumps(result)


def test_revalidation_never_replaces_or_omits_requested_tuple():
    request = parse_sourcing_revalidate(wire(revalidate_body()))
    row = {"identity": request.items[0].model_dump(mode="json"), "decision": "allow"}
    assert sourcing.revalidate_response([row], request)["results"] == [row]
    with pytest.raises(sourcing.SourcingUnavailable):
        sourcing.revalidate_response([], request)
    with pytest.raises(sourcing.SourcingUnavailable):
        sourcing.revalidate_response([{**row, "identity": identity()}], request)


def test_shared_deadline_cancels_slow_database_wait(monkeypatch):
    cancelled = []

    async def slow(*_args):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)

    monkeypatch.setattr(sourcing, "_query", slow)
    request = parse_sourcing_revalidate(wire(revalidate_body()))

    async def run():
        with pytest.raises(asyncio.TimeoutError):
            await sourcing.revalidate(request, "org_32", "unused", deadline=time.monotonic() + 0.02)

    asyncio.run(run())
    assert cancelled == [True]


def test_valid_search_and_revalidation_round_trip():
    body = search_body()
    assert parse_sourcing_search(wire(body)).model_dump(mode="json") == body
    body = revalidate_body()
    assert parse_sourcing_revalidate(wire(body)).model_dump(mode="json") == body


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 1.0),
        ("version", 2),
        ("requestId", ""),
        ("requestId", " " * 3),
        ("requestId", "x" * 201),
        ("requestId", "line\nbreak"),
        ("flowRunId", None),
        ("flowRunId", "00000000-0000-0000-0000-000000000000"),
        ("externalJobId", True),
        ("externalJobId", "189"),
        ("externalJobId", 0),
        ("externalJobId", 2_147_483_648),
        ("artifactHash", "A" * 64),
        ("query", " "),
        ("query", "x" * 2001),
        ("query", "bad\x00query"),
        ("limit", True),
        ("limit", 0),
        ("limit", 101),
        ("limit", "1"),
    ],
)
def test_search_refuses_coercions_missing_identity_and_unbounded_values(field, value):
    with pytest.raises(SourcingContractError, match="^candidate_sourcing_body_invalid$"):
        parse_sourcing_search(wire({**search_body(), field: value}))


@pytest.mark.parametrize(
    "key", ["tenantId", "organizationId", "email", "resumeText", "linkedinUrl"]
)
def test_no_body_tenant_or_contact_authority(key):
    for parse, body in (
        (parse_sourcing_search, search_body()),
        (parse_sourcing_revalidate, revalidate_body()),
    ):
        with pytest.raises(SourcingContractError):
            parse(wire({**body, key: "private-value"}))


@pytest.mark.parametrize(
    "field", ["candidateId", "referenceId", "resumeVersionId", "sourceId", "generationId"]
)
@pytest.mark.parametrize("value", [None, "", "not-a-uuid", "00000000-0000-0000-0000-000000000000"])
def test_null_or_fake_private_identity_never_becomes_public(field, value):
    body = revalidate_body()
    body["items"][0][field] = value
    with pytest.raises(SourcingContractError):
        parse_sourcing_revalidate(wire(body))


def test_revalidation_bound_and_duplicate_refusal():
    body = revalidate_body()
    for items in ([], [body["items"][0]] * 2, [identity() for _ in range(101)]):
        with pytest.raises(SourcingContractError):
            parse_sourcing_revalidate(wire({**body, "items": items}))
    hundred = {**body, "items": [identity() for _ in range(100)]}
    assert len(parse_sourcing_revalidate(wire(hundred)).items) == 100


@pytest.mark.parametrize(
    "parse,cap",
    [
        (parse_sourcing_search, SEARCH_BODY_BYTES),
        (parse_sourcing_revalidate, REVALIDATE_BODY_BYTES),
    ],
)
def test_wire_caps_duplicate_keys_and_errors_do_not_echo_input(parse, cap):
    for raw in (
        b"",
        b" " * (cap + 1),
        b"{",
        b"\xff",
        b'{"version":1,"version":1}',
        b'"private@example.invalid"',
    ):
        with pytest.raises(SourcingContractError) as failure:
            parse(raw)
        assert str(failure.value) == "candidate_sourcing_body_invalid"
        assert failure.value.__suppress_context__


def test_search_byte_boundary_is_enforced_without_truncation():
    raw = wire(search_body())
    assert parse_sourcing_search(raw + b" " * (SEARCH_BODY_BYTES - len(raw)))
    with pytest.raises(SourcingContractError):
        parse_sourcing_search(raw + b" " * (SEARCH_BODY_BYTES - len(raw) + 1))
