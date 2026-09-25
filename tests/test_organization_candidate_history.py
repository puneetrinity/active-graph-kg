from __future__ import annotations

import asyncio
import json
import threading
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from activekg.api import auth
from activekg.api import organization_candidate_history as api
from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.candidate_history.contracts import (
    HistoryConfig,
    HistoryReadRequest,
    Watermark,
    classify_status,
)
from activekg.candidate_history.worker import HistoryProjector

SOURCE = UUID("11111111-1111-4111-8111-111111111111")


@pytest.mark.parametrize(
    "value,state,authority",
    [
        (None, "waiting_source", "awaiting_binding"),
        (
            {"eligible": False, "reason": "privacy_unavailable", "contended": True},
            "privacy_wait",
            "temporarily_unavailable",
        ),
        (
            {"eligible": False, "reason": "privacy_unavailable"},
            "privacy_wait",
            "temporarily_unavailable",
        ),
        (
            {"eligible": False, "reason": "privacy_review"},
            "privacy_wait",
            "temporarily_unavailable",
        ),
        (
            {"eligible": False, "reason": "privacy_restricted"},
            "privacy_restricted",
            "privacy_restricted",
        ),
        ({"eligible": False, "reason": "superseded"}, "waiting_source", "awaiting_binding"),
        ({"eligible": False, "reason": "consent_changed"}, "binding_conflict", "binding_conflict"),
        ({"eligible": True, "reason": None, "source_id": str(SOURCE)}, "applied", "eligible"),
        (
            {
                "eligible": True,
                "reason": None,
                "source_id": str(SOURCE),
                "state": "failed",
                "error_code": "provider_timeout",
            },
            "applied",
            "eligible",
        ),
        (
            {"eligible": True, "reason": None, "source_id": "different"},
            "binding_conflict",
            "binding_conflict",
        ),
        ({"eligible": "true", "reason": None}, "privacy_wait", "temporarily_unavailable"),
        ({"eligible": False, "reason": []}, "privacy_wait", "temporarily_unavailable"),
        ({"eligible": False, "reason": "unknown"}, "privacy_wait", "temporarily_unavailable"),
    ],
    ids=[
        "missing",
        "contended",
        "unavailable",
        "review",
        "restricted",
        "superseded",
        "consent",
        "eligible",
        "embedding-failed",
        "wrong-source",
        "wrong-bool",
        "malformed",
        "unknown",
    ],
)
def test_closed_status_mapping(value, state, authority):
    result = classify_status(value, SOURCE)
    assert (result.state, result.authority) == (state, authority)


@pytest.mark.parametrize(
    "changes",
    [
        {"count": 1},
        {"count": "01"},
        {"count": "1"},
        {"sequence": "1"},
        {"event_id": str(SOURCE)},
        {"count": "9223372036854775808"},
        {"extra": True},
    ],
    ids=[
        "number",
        "leading-zero",
        "missing-watermark",
        "partial-sequence",
        "partial-event",
        "overflow",
        "extra",
    ],
)
def test_watermark_refuses_malformed(changes):
    with pytest.raises(ValidationError):
        Watermark.model_validate_json(
            json.dumps({"count": "0", "event_id": None, "sequence": None, **changes})
        )


def test_decimal_precision_and_request_closed_shape():
    expected = {
        "count": "9007199254740993",
        "event_id": str(SOURCE),
        "sequence": "9223372036854775807",
    }
    request = {
        "schema_version": 1,
        "organization_id": 1,
        "application_id": 2,
        "job_id": 3,
        "reference_id": str(SOURCE),
        "expected": expected,
    }
    assert (
        HistoryReadRequest.model_validate_json(json.dumps(request)).expected.count
        == expected["count"]
    )
    with pytest.raises(ValidationError):
        HistoryReadRequest.model_validate_json(json.dumps({**request, "candidate_id": str(SOURCE)}))


@pytest.mark.parametrize(
    "env",
    [
        {"ORG_CANDIDATE_HISTORY_ENABLED": "yes"},
        {"ORG_CANDIDATE_HISTORY_POLL_MS": "999"},
        {"ORG_CANDIDATE_HISTORY_NEW_LIMIT": "101"},
        {"ORG_CANDIDATE_HISTORY_RETRY_LIMIT": "0"},
        {"ORG_CANDIDATE_HISTORY_POLL_MS": "1e3"},
    ],
    ids=["flag", "poll", "new", "retry", "not-decimal"],
)
def test_config_refuses_out_of_bounds(env):
    with pytest.raises(ValueError):
        HistoryConfig.from_env(env)


def test_projector_single_flight_off_loop_and_drained_shutdown():
    entered, release = threading.Event(), threading.Event()
    calls = []

    class Repo:
        def step(self, new, retry):
            calls.append(threading.current_thread().name)
            entered.set()
            assert release.wait(3)
            return {"outcome": "idle"}

    async def run():
        projector = HistoryProjector(Repo(), HistoryConfig(enabled=True))
        projector.start()
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(0.005)
        assert entered.is_set() and len(calls) == 1 and calls[0].startswith("candidate-history")
        assert not projector.healthy()
        stopping = asyncio.create_task(projector.stop())
        await asyncio.sleep(0.02)
        assert not stopping.done()
        release.set()
        await stopping
        assert len(calls) == 1 and not projector.healthy()

    asyncio.run(run())


@pytest.mark.parametrize("cancel_worker", [True, False], ids=["worker-cancel", "shutdown-cancel"])
def test_cancellation_drains_real_thread_and_releases_executor(cancel_worker):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    class Repo:
        def step(self, *_args):
            entered.set()
            assert release.wait(3)
            finished.set()
            return {"outcome": "idle"}

    async def run():
        projector = HistoryProjector(Repo(), HistoryConfig(enabled=True))
        projector.start()
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.005)
        assert entered.is_set(), "worker thread did not start"
        stopping = asyncio.create_task(projector.stop())
        await asyncio.sleep(0.01)
        (projector._task if cancel_worker else stopping).cancel()
        await asyncio.sleep(0.01)
        assert not finished.is_set() and not stopping.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        assert finished.is_set() and projector._executor is None and projector._task is None

    asyncio.run(run())


def test_future_or_unknown_health_is_not_fresh(monkeypatch):
    import activekg.candidate_history.worker as worker

    class Pending:
        def done(self):
            return False

    projector = HistoryProjector(None, HistoryConfig(enabled=True))
    projector._task = Pending()
    monkeypatch.setattr(worker.time, "monotonic", lambda: 100.0)
    for stamp, expected in [(None, False), (101.0, False), (69.0, False), (99.0, True)]:
        projector._last_success = stamp
        assert projector.healthy() is expected


@pytest.mark.parametrize(
    "change,status",
    [
        ({"issuer": "other"}, 403),
        ({"actor_type": "user"}, 403),
        ({"actor_id": "other"}, 403),
        ({"scopes": ["decision-history:write"]}, 403),
        ({"tenant_id": "org_2"}, 403),
    ],
    ids=["issuer", "actor-type", "actor", "write-only", "tenant"],
)
def test_service_denials_before_repository(monkeypatch, change, status):
    monkeypatch.setattr(auth, "JWT_ENABLED", True)
    monkeypatch.setattr(auth, "JWT_ISSUER", "vantahire")
    monkeypatch.setenv("ORG_CANDIDATE_HISTORY_ENABLED", "true")
    monkeypatch.setenv("ORG_DECISION_INBOX_FLOW_ACTOR_ID", "vantahire-backend")
    app = FastAPI()
    app.include_router(api.router)
    claims = JWTClaims(
        **{
            "issuer": "vantahire",
            "actor_type": "service",
            "actor_id": "vantahire-backend",
            "tenant_id": "org_1",
            "scopes": ["decision-history:read"],
            **change,
        }
    )
    app.dependency_overrides[get_jwt_claims] = lambda: claims

    def forbidden(*args, **kwargs):
        pytest.fail("denied request reached database")

    monkeypatch.setattr(api, "HistoryRepository", forbidden)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/organization-candidate-history/read",
                json={
                    "schema_version": 1,
                    "organization_id": 1,
                    "application_id": 2,
                    "job_id": 3,
                    "reference_id": str(SOURCE),
                    "expected": {"count": "0", "event_id": None, "sequence": None},
                },
            )
            assert response.status_code == status

    asyncio.run(run())
