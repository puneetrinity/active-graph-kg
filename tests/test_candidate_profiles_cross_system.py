"""ASGI boundary checks with synthetic service claims; DB integration is tested separately."""

import asyncio
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from activekg.api import auth, candidate_profiles
from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.candidate_profiles.contracts import parse_command, receipt


@pytest.mark.skipif(
    not os.getenv("CANDIDATE_FOUNDATION_E2E_MEMORY_DATABASE"),
    reason="full disposable schemas required",
)
def test_real_flow_processor_signed_http_and_both_runtime_databases(monkeypatch):
    import psycopg
    import uvicorn
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from psycopg import sql

    directory = os.environ["CANDIDATE_FOUNDATION_TEST_SOCKET"]
    port = int(os.environ["CANDIDATE_FOUNDATION_TEST_PORT"])
    memory_db = os.environ["CANDIDATE_FOUNDATION_E2E_MEMORY_DATABASE"]
    flow_db = os.environ["CANDIDATE_FOUNDATION_FLOW_TEST_DATABASE"]
    assert directory.startswith("/tmp/candidate-s1-")
    memory_suffix = re.fullmatch(r"s1_memory_([a-f0-9]{10})_test", memory_db)
    flow_suffix = re.fullmatch(r"flow_schema_control_test_([a-f0-9]{10})", flow_db)
    assert memory_suffix and flow_suffix
    flow_root = Path(os.environ["CANDIDATE_FOUNDATION_FLOW_ROOT"]).resolve()
    assert (flow_root / "server/candidate-product/processor.ts").is_file()
    user_id = 100_000_000 + int(uuid4().hex[:6], 16)
    with psycopg.connect(host=directory, port=port, user="ews", dbname=flow_db) as conn:
        conn.execute(
            sql.SQL("SET LOCAL ROLE {}").format(
                sql.Identifier("flow_schema_control_test_owner_" + flow_suffix[1])
            )
        )
        conn.execute(
            "INSERT INTO users(id,username,password,role,email_verified,auth_version,first_name,last_name) VALUES(%s,%s,'synthetic','candidate',true,1,'Synthetic','E2E')",
            (user_id, f"s1-{uuid4()}@example.invalid"),
        )
        conn.execute(
            "INSERT INTO users(id,username,password,role,email_verified,auth_version,first_name,last_name) VALUES(%s,%s,'synthetic','candidate',true,1,'Synthetic','Concurrent')",
            (user_id + 1, f"s1-{uuid4()}@example.invalid"),
        )
        conn.execute(
            "INSERT INTO candidate_privacy_sync_state(consumer_name,status,last_success_at) VALUES('flow','healthy',clock_timestamp()) ON CONFLICT(consumer_name) DO UPDATE SET status='healthy',last_success_at=excluded.last_success_at"
        )
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
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
    }.items():
        monkeypatch.setattr(auth, name, value)
    monkeypatch.setenv("CANDIDATE_FOUNDATION_ENABLED", "true")
    monkeypatch.setenv(
        "ACTIVEKG_DSN",
        f"host={directory} port={port} user=s1_runtime_{memory_suffix[1]}_test dbname={memory_db}",
    )
    app = FastAPI()
    app.include_router(candidate_profiles.router)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    script = r"""
import assert from 'node:assert/strict';
import {createHash,randomUUID} from 'node:crypto';
import {pool} from './server/db.ts';
import {foundationRepository} from './server/candidate-product/repository.ts';
import {runFoundationOnce} from './server/candidate-product/processor.ts';
import {deliverFoundation} from './server/candidate-product/memory-client.ts';
import {requireCandidatePrivacyAllowed} from './server/candidate-privacy/decision.ts';
import {captureConsent} from './server/candidate-consent/repository.ts';
const privacy=id=>requireCandidatePrivacyAllowed({type:'candidate_user',id},{globalUse:true,newGlobalOperation:true});
const repo=foundationRepository(pool,privacy);
const actor={userId:Number(process.env.S1_USER),authVersion:1,passwordVersion:createHash('sha256').update('synthetic').digest('hex'),reauthenticatedAt:Date.now()};
try {
 const initial=await repo.ensure(actor,randomUUID());
 const repeatKey=randomUUID();
 const repeated=await Promise.all([repo.ensure(actor,repeatKey),repo.ensure(actor,repeatKey)]);
 assert.deepEqual(repeated[0],repeated[1]);
 let delivered=0;
 const deliver=async(raw,hash,timeout)=>{
   const first=await deliverFoundation(raw,hash,timeout);
   const replay=await deliverFoundation(raw,hash,timeout);
   assert.deepEqual(replay,first); delivered++;
   return first;
 };
 for(let i=0;i<10;i++){
   if(!await runFoundationOnce({db:pool,privacy,enabled:()=>true,deliver}))break;
 }
 const after=await repo.read(actor);
 assert.ok(delivered>0);
 const concurrentActor={...actor,userId:actor.userId+1};
 await Promise.all([
   repo.ensure(concurrentActor,randomUUID()),
   captureConsent('withdraw',{request_id:randomUUID(),expected_version:0},concurrentActor),
 ]);
 await repo.ensure(concurrentActor,randomUUID());
 console.log(JSON.stringify({initial,after,delivered}));
}finally{await pool.end();}
"""
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        env = {
            **os.environ,
            "NODE_ENV": "test",
            "CANDIDATE_FOUNDATION_DISPOSABLE_TEST": "1",
            "VANTAHIRE_JWT_PRIVATE_KEY": private,
            "ACTIVEKG_BASE_URL": f"http://127.0.0.1:{listener.getsockname()[1]}",
            "S1_USER": str(user_id),
            "DATABASE_URL": f"postgresql://flow_schema_control_test_runtime_{flow_suffix[1]}@localhost/{flow_db}?host={directory}&port={port}",
            "DATABASE_SSL": "false",
        }
        result = subprocess.run(
            ["node", "--import", "tsx", "--input-type=module", "-e", script],
            cwd=flow_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        import json

        output = json.loads(result.stdout.strip().splitlines()[-1])
        assert output["after"]["publicationActive"] is False
        assert output["after"]["state"] == "linked"
        with psycopg.connect(host=directory, port=port, user="ews", dbname=flow_db) as conn:
            assert conn.execute(
                "SELECT count(*) FROM candidate_consent_events WHERE user_id=%s", (user_id,)
            ).fetchone() == (0,)
            assert conn.execute(
                "SELECT s.version,s.desired_action,a.subject_id=s.subject_id FROM candidate_consent_subjects s JOIN candidate_product_accounts a USING(user_id) WHERE user_id=%s",
                (user_id + 1,),
            ).fetchall() == [(1, "withdraw", True)]
            assert conn.execute(
                "SELECT action FROM candidate_consent_events WHERE user_id=%s", (user_id + 1,)
            ).fetchall() == [("withdraw",)]
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        assert not thread.is_alive()


RAW = b'{"schemaVersion":1,"commandId":"11111111-1111-4111-8111-111111111111","subjectId":"22222222-2222-4222-8222-222222222222","bindingId":"33333333-3333-4333-8333-333333333333","sequence":1,"action":"ensure_subject"}'


def test_endpoint_replay_and_exact_payload(monkeypatch):
    asyncio.run(_endpoint_replay_and_exact_payload(monkeypatch))


async def _endpoint_replay_and_exact_payload(monkeypatch):
    monkeypatch.setenv("CANDIDATE_FOUNDATION_ENABLED", "true")
    monkeypatch.setattr(auth, "JWT_ENABLED", True)
    monkeypatch.setattr(auth, "JWT_ALGORITHM", "RS256")
    monkeypatch.setattr(auth, "JWT_AUDIENCE", "activekg")
    command, digest = parse_command(RAW)
    expected = receipt(command, digest)
    now = int(time.time())
    claims = JWTClaims(
        f"candidate_{command['subjectId']}",
        "vantahire-backend",
        "service",
        ["candidate-profile:foundation-write"],
        exp=now + 60,
        issuer="vantahire",
        iat=now,
    )
    calls = []

    def store(subject, binding, body, sha):
        calls.append((subject, binding, body, sha))
        return expected, len(calls) > 1

    monkeypatch.setattr(candidate_profiles, "execute", store)
    app = FastAPI()
    app.include_router(candidate_profiles.router)
    app.dependency_overrides[get_jwt_claims] = lambda: claims
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://memory.invalid"
    ) as client:
        headers = {"content-type": "application/json", "x-candidate-command-sha256": digest}
        for status in (201, 200):
            response = await client.post(
                "/candidate-profiles/foundation/commands", content=RAW, headers=headers
            )
            assert response.status_code == status
            assert response.json() == expected
        assert calls[0] == (command["subjectId"], command["bindingId"], command, digest)
        for body in (b" " + RAW, RAW.replace(b'"sequence":1', b'"sequence":1,"sequence":1')):
            assert (
                await client.post(
                    "/candidate-profiles/foundation/commands", content=body, headers=headers
                )
            ).status_code == 400
        assert (
            await client.post(
                "/candidate-profiles/foundation/commands?extra=1", content=RAW, headers=headers
            )
        ).status_code == 400
        assert (
            await client.post(
                "/candidate-profiles/foundation/commands",
                content=RAW,
                headers={**headers, "x-candidate-command-sha256": "a" * 64},
            )
        ).status_code == 400
        assert len(calls) == 2
        claims.tenant_id = "candidate_44444444-4444-4444-8444-444444444444"
        assert (
            await client.post(
                "/candidate-profiles/foundation/commands", content=RAW, headers=headers
            )
        ).status_code == 403
        monkeypatch.setenv("CANDIDATE_FOUNDATION_ENABLED", "false")
        assert (
            await client.post(
                "/candidate-profiles/foundation/commands", content=RAW, headers=headers
            )
        ).status_code == 404
        assert len(calls) == 2


def test_status_binding_scope_and_no_data(monkeypatch):
    asyncio.run(_status_binding_scope_and_no_data(monkeypatch))


async def _status_binding_scope_and_no_data(monkeypatch):
    monkeypatch.setenv("CANDIDATE_FOUNDATION_ENABLED", "true")
    monkeypatch.setattr(auth, "JWT_ENABLED", True)
    monkeypatch.setattr(auth, "JWT_ALGORITHM", "RS256")
    monkeypatch.setattr(auth, "JWT_AUDIENCE", "activekg")
    command, _ = parse_command(RAW)
    now = int(time.time())
    claims = JWTClaims(
        f"candidate_{command['subjectId']}",
        "vantahire-backend",
        "service",
        ["candidate-profile:foundation-read"],
        exp=now + 60,
        issuer="vantahire",
        iat=now,
    )
    calls = []
    expected = {
        "schemaVersion": 1,
        "state": "empty",
        "profileVersion": 0,
        "preferenceVersion": 0,
        "publicationActive": False,
    }

    def read(subject, binding):
        calls.append((subject, binding))
        return expected, True

    monkeypatch.setattr(candidate_profiles, "execute", read)
    app = FastAPI()
    app.include_router(candidate_profiles.router)
    app.dependency_overrides[get_jwt_claims] = lambda: claims
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://memory.invalid"
    ) as client:
        url = f"/candidate-profiles/foundation?binding={command['bindingId']}"
        response = await client.get(url)
        assert response.status_code == 200 and response.json() == expected
        assert calls == [(command["subjectId"], command["bindingId"])]
        for suffix in ("&binding=" + command["bindingId"], "&subjectId=x"):
            assert (await client.get(url + suffix)).status_code == 400
        assert (await client.request("GET", url, content=b"{}")).status_code == 400
        claims.scopes = ["candidate-profile:foundation-write"]
        assert (await client.get(url)).status_code == 403
        assert len(calls) == 1
