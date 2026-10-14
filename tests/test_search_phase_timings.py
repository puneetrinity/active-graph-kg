import hashlib

from activekg.api import global_memory
from activekg.common.logger import _log_context


def test_search_phase_log_has_only_counts_times_and_hashed_request(monkeypatch):
    captured = []

    def record(message, extra_fields):
        assert _log_context.get() is None
        captured.append((message, extra_fields))

    monkeypatch.setattr(global_memory.logger, "info", record)
    context = {"request_id": "private-request", "tenant_id": "private-tenant"}
    token = _log_context.set(context)
    try:
        global_memory._log_search_phases("public_v1", 300, 300, 299, 1.23, 2.34, 3.45)
        assert _log_context.get() is context
    finally:
        _log_context.reset(token)
    message, fields = captured[0]
    assert message == "candidate_search_phase_timings"
    assert fields == {
        "surface": "public_v1",
        "request_id_hash": hashlib.sha256(b"private-request").hexdigest(),
        "requested_limit": 300,
        "applied_limit": 300,
        "returned_count": 299,
        "embed_ms": 1.23,
        "query_ms": 2.34,
        "projection_ms": 3.45,
    }
    assert "private-request" not in repr(fields)
    assert "private-tenant" not in repr(fields)


def test_unknown_surface_cannot_put_arbitrary_text_in_log(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected log")

    monkeypatch.setattr(global_memory.logger, "info", forbidden)
    global_memory._log_search_phases("private query text", 1, 1, 0, 0, 0, 0)
