from unittest.mock import Mock

import pytest
from fastapi import APIRouter, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.testclient import TestClient

from activekg.api import work_timing


def test_executor_wait_and_work_are_separate(monkeypatch):
    times = iter([2.0, 5.0])
    log = Mock()
    monkeypatch.setattr(work_timing, "perf_counter", lambda: next(times))
    monkeypatch.setattr(work_timing, "logger", log)
    assert work_timing.timed_work("sourced_candidate_ingest", 1.0, lambda: "private") == "private"
    assert log.info.call_args.kwargs["extra_fields"] == {
        "timing_id": None,
        "operation": "sourced_candidate_ingest",
        "outcome": "ok",
        "executor_wait_ms": 1000.0,
        "work_ms": 3000.0,
        "total_ms": 4000.0,
    }
    assert "private" not in str(log.mock_calls)


def test_error_is_rethrown_without_logging_message(monkeypatch):
    log = Mock()
    monkeypatch.setattr(work_timing, "logger", log)
    error = RuntimeError("private SQL with credentials")

    def fail():
        raise error

    with pytest.raises(RuntimeError) as caught:
        work_timing.timed_work("privacy_eligibility_batch", 0, fail)
    assert caught.value is error
    assert log.info.call_args.kwargs["extra_fields"]["outcome"] == "error"
    assert "credentials" not in str(log.mock_calls)


def test_request_and_executor_timings_correlate_without_user_data(monkeypatch):
    log = Mock()
    monkeypatch.setattr(work_timing, "logger", log)
    app = FastAPI()
    router = APIRouter(route_class=work_timing.TimingRoute)

    @router.post("/synthetic/{candidate}")
    async def route(request: Request, candidate: str):
        await request.body()
        return await run_in_threadpool(
            lambda: work_timing.timed_work("sourced_candidate_ingest", 0, lambda: {"ok": True})
        )

    app.include_router(router)
    with TestClient(app) as client:
        for _ in range(2):
            assert (
                client.post(
                    "/synthetic/private-person?secret=key", content="private body"
                ).status_code
                == 200
            )
    events = [call.kwargs["extra_fields"] for call in log.info.call_args_list]
    assert len(events) == 4
    assert events[0]["timing_id"] == events[1]["timing_id"]
    assert events[2]["timing_id"] == events[3]["timing_id"]
    assert events[0]["timing_id"] != events[2]["timing_id"]
    assert events[1]["route"] == "/synthetic/{candidate}"
    assert events[1]["status"] == 200
    assert "private" not in str(events)
    assert "secret" not in str(events)
