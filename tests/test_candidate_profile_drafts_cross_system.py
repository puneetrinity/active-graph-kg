"""Signed S2 Flow-to-Memory HTTP proof on disposable full schemas only."""

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql


def test_profile_http_authority_and_raw_contract(monkeypatch):
    import asyncio

    import httpx
    from fastapi import FastAPI

    from activekg.api import auth, candidate_profiles
    from activekg.api.auth import JWTClaims, get_jwt_claims
    from activekg.candidate_profiles.profile_contracts import canonical, parse_profile_command
    from tests.test_candidate_profile_drafts_postgres import FACTS, command

    monkeypatch.setenv("CANDIDATE_FOUNDATION_ENABLED", "true")
    monkeypatch.setenv("CANDIDATE_PROFILE_ENABLED", "true")
    for key, value in {
        "JWT_ENABLED": True,
        "JWT_ALGORITHM": "RS256",
        "JWT_AUDIENCE": "activekg",
    }.items():
        monkeypatch.setattr(auth, key, value)
    subject, binding, cid = str(uuid4()), str(uuid4()), str(uuid4())
    raw = canonical(command("save_draft", {"profile": FACTS})).encode()
    digest = parse_profile_command(raw)[2]
    now = int(time.time())
    claims = JWTClaims(
        f"candidate_{subject}",
        "vantahire-backend",
        "service",
        ["candidate-profile:owner-write"],
        now + 60,
        "vantahire",
        now,
    )
    calls = []

    def store(*args, **kwargs):
        calls.append((args, kwargs))
        return {"synthetic": True}

    monkeypatch.setattr(candidate_profiles, "execute_profile", store)
    app = FastAPI()
    app.include_router(candidate_profiles.router)
    app.dependency_overrides[get_jwt_claims] = lambda: claims
    headers = {
        "content-type": "application/json",
        "x-candidate-command-id": cid,
        "x-candidate-binding-id": binding,
        "x-candidate-command-sha256": digest,
    }

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local.invalid"
        ) as client:
            endpoint = "/candidate-profiles/profile/commands"
            response = await client.post(endpoint, content=raw, headers=headers)
            assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
            assert calls == [
                ((subject, binding), {"raw": raw, "command_id": cid, "digest": digest})
            ]
            for body in (
                b" " + raw,
                raw.replace(b'"schemaVersion":2', b'"schemaVersion":2,"schemaVersion":2'),
            ):
                assert (
                    await client.post(endpoint, content=body, headers=headers)
                ).status_code == 422
            assert (
                await client.post(endpoint, content=b"x" * (384 * 1024 + 1), headers=headers)
            ).status_code == 413
            assert (
                await client.post(
                    endpoint,
                    content=raw,
                    headers={**headers, "x-candidate-command-sha256": "0" * 64},
                )
            ).status_code == 422
            assert (
                await client.post(
                    endpoint,
                    content=raw,
                    headers=list(headers.items()) + [("x-candidate-binding-id", binding)],
                )
            ).status_code == 400
            assert (
                await client.post(endpoint + "?subject=foreign", content=raw, headers=headers)
            ).status_code == 400
            claims.scopes = ["candidate-profile:foundation-write"]
            assert (await client.post(endpoint, content=raw, headers=headers)).status_code == 403
            assert len(calls) == 1
            monkeypatch.setenv("CANDIDATE_PROFILE_ENABLED", "false")
            assert (await client.post(endpoint, content=raw, headers=headers)).status_code == 404

    asyncio.run(run())


@pytest.mark.skipif(
    not os.getenv("S2_FLOW_DATABASE"), reason="explicit local full-schema pair required"
)
def test_signed_profile_delivery_and_owner_read(monkeypatch):
    import uvicorn
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from fastapi import FastAPI

    from activekg.api import auth, candidate_profiles

    admin_url = os.environ["S2_PROFILE_TEST_ADMIN_URL"]
    parsed = urlparse(admin_url)
    assert (
        parsed.hostname == "127.0.0.1"
        and parsed.path == "/postgres"
        and os.getenv("S2_PROTOTYPE_DISPOSABLE") == "1"
    )
    flow_db = os.environ["S2_FLOW_DATABASE"]
    match = re.fullmatch(r"flow_schema_control_test_([a-f0-9]{10})", flow_db)
    assert match
    flow_runtime = "flow_schema_control_test_runtime_" + match[1]
    flow_owner = "flow_schema_control_test_owner_" + match[1]
    suffix = uuid4().hex[:10]
    owner, runtime, name = (
        f"s2_owner_{suffix}_test",
        f"s2_runtime_{suffix}_test",
        f"s2_memory_{suffix}_test",
    )

    def dsn(db, role="ews"):
        return f"postgresql://{role}@127.0.0.1:{parsed.port}/{db}"

    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS CREATEROLE").format(
                sql.Identifier(owner)
            )
        )
        admin.execute(
            sql.SQL("CREATE DATABASE {} OWNER {} TEMPLATE template0 ENCODING 'UTF8'").format(
                sql.Identifier(name), sql.Identifier(owner)
            )
        )
    with psycopg.connect(dsn(name), autocommit=True) as admin:
        admin.execute("CREATE EXTENSION vector")
        admin.execute("CREATE EXTENSION pgcrypto")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACTIVEKG_")}
    env.update(
        ACTIVEKG_MIGRATE_DSN=dsn(name, owner),
        ACTIVEKG_MIGRATION_APPLY="1",
        ACTIVEKG_SCHEMA_TARGET_ID=str(uuid4()),
        ACTIVEKG_SCHEMA_ENVIRONMENT="development",
        ACTIVEKG_SCHEMA_SOURCE_COMMIT="0" * 40,
        ACTIVEKG_SCHEMA_FRESH_INIT="1",
        ACTIVEKG_RUNTIME_ROLE=runtime,
        ACTIVEKG_RUNTIME_PASSWORD="synthetic-local-only",
    )
    release = subprocess.run(
        [sys.executable, "scripts/init_railway_db.py"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert release.returncode == 0, release.stdout + release.stderr
    with psycopg.connect(dsn(name, owner), autocommit=True) as conn:
        conn.execute(Path(os.environ["S2_PROFILE_SQL"]).read_text())
        from scripts.init_railway_db import (
            _assert_candidate_profile_runtime_privileges,
            _harden_candidate_profile_runtime_privileges,
        )

        with conn.cursor() as cur:
            _harden_candidate_profile_runtime_privileges(cur, runtime)
            _assert_candidate_profile_runtime_privileges(cur, runtime)
    # Provision synthetic owners through the existing S1 setup routine, then
    # independently prepare its already-proven Memory identity anchors.
    user = 100_000_000 + int(uuid4().hex[:5], 16)
    with psycopg.connect(dsn(flow_db, flow_owner)) as conn:
        conn.execute(
            "INSERT INTO candidate_privacy_sync_state(consumer_name,status,last_success_at) VALUES('flow','healthy',clock_timestamp()) ON CONFLICT(consumer_name) DO UPDATE SET status='healthy',last_success_at=excluded.last_success_at"
        )
        for n in range(20):
            conn.execute(
                "INSERT INTO users(id,username,password,role,email_verified,auth_version,first_name,last_name) VALUES(%s,%s,'synthetic','candidate',true,1,'Synthetic','S2')",
                (user + n, f"s2-{uuid4()}@example.invalid"),
            )
            conn.execute(
                "SELECT flow_candidate_foundation_ensure(%s,1,encode(sha256(convert_to('synthetic','UTF8')),'hex'),clock_timestamp(),%s)",
                (user + n, uuid4()),
            )
            conn.execute(
                "UPDATE candidate_product_accounts SET foundation_state='linked',acknowledged_sequence=command_sequence WHERE user_id=%s",
                (user + n,),
            )
        identities = conn.execute(
            "SELECT subject_id,binding_id FROM candidate_product_accounts WHERE user_id BETWEEN %s AND %s",
            (user, user + 19),
        ).fetchall()
    with psycopg.connect(dsn(name, runtime)) as conn:
        for subject, binding in identities:
            conn.execute(
                "SELECT candidate_owned_ensure(%s,%s,%s,1,%s)",
                (subject, binding, uuid4(), "a" * 64),
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
    for k, v in {
        "JWT_ENABLED": True,
        "JWT_ALGORITHM": "RS256",
        "JWT_AUDIENCE": "activekg",
        "JWT_ISSUER": "vantahire",
        "JWT_PUBLIC_KEY": public,
    }.items():
        monkeypatch.setattr(auth, k, v)
    monkeypatch.setenv("CANDIDATE_FOUNDATION_ENABLED", "true")
    monkeypatch.setenv("CANDIDATE_PROFILE_ENABLED", "true")
    monkeypatch.setenv("ACTIVEKG_DSN", dsn(name, runtime))
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
import {randomUUID} from 'node:crypto';
import pg from 'pg';
import {profileRepository} from './server/candidate-product/profile-repository.ts';
import {runProfileBatch} from './server/candidate-product/profile-processor.ts';
import {deliverProfile,readProfile} from './server/candidate-product/profile-client.ts';
import {requireCandidatePrivacyAllowed} from './server/candidate-privacy/decision.ts';
const pool=new pg.Pool({connectionString:process.env.DATABASE_URL});
const repo=profileRepository(pool,id=>requireCandidatePrivacyAllowed({type:'candidate_user',id},{globalUse:false}));
const facts={displayName:'Synthetic S2',headline:'',linkedinUrl:null,currentLocation:{city:'Bengaluru',countryCode:'IN',displayCountry:'India'},skills:['Python'],employment:[],education:[],projects:[]};
const user=Number(process.env.S2_USER);
try{
 const commands=[];
 for(let n=0;n<20;n++)commands.push(await repo.admit({userId:user+n,authVersion:1,passwordVersion:null,reauthenticatedAt:null},JSON.stringify({schemaVersion:2,idempotencyKey:randomUUID(),action:'save_draft',expectedDraftVersion:0,expectedProfileVersion:0,expectedPreferenceVersion:0,payload:{profile:facts}})));
 let sends=0;
 const start=performance.now();
 const processed=await runProfileBatch({enabled:()=>true,claim:repo.claim,authority:repo.authority,finish:repo.finish,deliver:async(lease,signal)=>{
  const first=await deliverProfile(lease,signal);sends++;
  // A lost acknowledgement safely replays the identical command over real HTTP.
  if(lease.userId===user)assert.deepEqual(await deliverProfile(lease,signal),first);
  return first;
 }});
 const elapsed=performance.now()-start;
 assert.equal(processed,20);assert.equal(sends,20);
 for(let n=0;n<20;n++)assert.equal((await repo.readCommand({userId:user+n,authVersion:1},commands[n].commandId)).status,'applied');
 const identity=await repo.owner({userId:user,authVersion:1});
 const view=await readProfile(identity.subjectId,identity.bindingId);
 assert.deepEqual(view.draft,facts);assert.equal(view.draftVersion,1);assert.equal(view.publicationActive,false);assert.equal(view.preferences,null);
 await assert.rejects(()=>readProfile(identity.subjectId,randomUUID()));
 console.log(JSON.stringify({processed,sends,elapsedMs:elapsed,underTwoSeconds:elapsed<2000,draftVersion:view.draftVersion,publicationActive:false}));
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
            "S2_USER": str(user),
            "DATABASE_URL": dsn(flow_db, flow_runtime),
            "DATABASE_SSL": "false",
        }
        result = subprocess.run(
            ["node", "--import", "tsx", "--input-type=module", "-e", script],
            cwd=os.environ["S2_FLOW_ROOT"],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        evidence = json.loads(result.stdout.strip().splitlines()[-1])
        assert evidence["processed"] == 20 and evidence["underTwoSeconds"]
        print("S2 signed HTTP evidence", json.dumps(evidence), "retained", name)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        assert not thread.is_alive()
