"""Disposable-only synthetic API transcript for the cross-system contract proof."""

import asyncio
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI

from activekg.api import candidate_sourcing as api
from activekg.candidate_index import resume_history_policy, sourcing
from activekg.candidate_index.processing import resume_extraction
from tests import test_candidate_index_postgres as base

assert os.environ.get("ACTIVEKG_INDEX_TEST_DISPOSABLE") == "1"
output = Path(os.environ["RESUME_XS_FIXTURE"])
assert output.parent == Path("/tmp") and output.name.startswith("resume-xs-")
patch = pytest.MonkeyPatch()
for name, value in {
    "ACTIVEKG_DSN": base.RUNTIME,
    "CANDIDATE_PRIVACY_HMAC_ACTIVE_VERSION": "1",
    "CANDIDATE_PRIVACY_HMAC_KEY_V1": "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE=",
    "CANDIDATE_PRIVACY_INTAKE_ENABLED": "false",
    "CANDIDATE_PRIVACY_FLOW_ISSUER": "vantahire",
    "CANDIDATE_PRIVACY_FLOW_ACTOR_ID": "vantahire-backend",
    "CANDIDATE_PRIVACY_SIGNAL_ISSUER": "signal",
    "CANDIDATE_PRIVACY_SIGNAL_ACTOR_ID": "signal-service",
}.items():
    patch.setenv(name, value)
fixture = base.target.__wrapped__()
owner = next(fixture)  # executes the existing disposable-target/role checks
try:
    owner.execute(
        "UPDATE candidate_index_jobs SET state='cancelled',lease_token=NULL,lease_expires_at=NULL"
    )
    text = "Python. Backend engineer, March 2019 to June 2024."
    patch.setattr(base, "TEXT", text)
    original_body = base._body
    patch.setattr(base, "_body", lambda **kw: original_body(job_id=95202, **kw))
    policy = replace(base.POLICY, **resume_history_policy.policy_fields())
    repo, scope, _, _ = base.source("org_95101", policy=policy)
    dispatch = repo.reserve(repo.claim("extract")[0], policy)
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
    extracted = resume_extraction(raw, text, policy, policy.primary_model_id, {})
    assert repo.complete_extract(
        dispatch,
        namespace=extracted.namespace,
        evidence=extracted.evidence,
        confidence=extracted.confidence,
        model_id=policy.primary_model_id,
        chunks=extracted.chunks,
        professional_text=extracted.professional_text,
    )
    dispatch = repo.reserve(repo.claim("embed")[0], policy)
    assert repo.complete_embed(dispatch, [base.VECTOR for _ in dispatch["extraction"]["chunks"]])
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    patch.setattr(api.auth, "JWT_ENABLED", True)
    patch.setattr(api.auth, "SIGNAL_JWT_PUBLIC_KEY", public)
    patch.setattr(api, "enabled", lambda _: True)
    patch.setattr(api, "configuration", lambda: SimpleNamespace(policy=policy))

    async def embedding(*args, **kwargs):
        return base.VECTOR

    patch.setattr(sourcing, "embed_query_once", embedding)
    app = FastAPI()
    app.include_router(api.router)

    def token(tenant):
        now = int(time.time())
        return jwt.encode(
            {
                "iss": "signal",
                "aud": "activekg",
                "sub": "signal-service",
                "actor_type": "service",
                "tenant_id": tenant,
                "scopes": [api.SOURCING_SCOPE],
                "request_id": "synthetic-xs",
                "iat": now,
                "exp": now + 300,
                "jti": str(uuid4()),
            },
            key,
            algorithm="RS256",
        )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local-test"
        ) as client:
            body = {
                "version": 1,
                "requestId": "synthetic-xs",
                "externalJobId": 95201,
                "flowRunId": str(uuid4()),
                "artifactHash": "a" * 64,
                "query": "Python",
                "limit": 100,
            }

            async def search(tenant, job):
                response = await client.post(
                    "/organization-candidates/sourcing/search",
                    json={**body, "externalJobId": job},
                    headers={"authorization": "Bearer " + token(tenant)},
                )
                assert response.status_code == 200
                return response.json()

            result = await search(scope, 95201)
            assert len(result["results"]) == 1
            assert (await search(scope, 95202))["results"] == []
            assert (await search("org_95102", 95201))["results"] == []
            return result

    response = asyncio.run(run())
    output.write_text(json.dumps({"synthetic": True, "scope": scope, "response": response}))
    print(
        "Synthetic signed Memory API transcript verified: cross-org and same-job excluded; no provider calls"
    )
finally:
    fixture.close()
    patch.undo()
