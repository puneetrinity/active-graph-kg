from contextlib import contextmanager

import pytest

from activekg.candidate_profiles.publication import (
    publication_text,
    sweep_publications,
    vector_literal,
)


def test_default_off_does_not_connect(monkeypatch):
    monkeypatch.delenv("CANDIDATE_NETWORK_PUBLICATION_ENABLED", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("network publication must stay off")

    assert sweep_publications("unused", None, connect=forbidden) == 0


def test_network_guard_and_each_required_anchor_mutation(tmp_path):
    from pathlib import Path

    from scripts.candidate_network_guard import REQUIRED, check

    root = Path(__file__).resolve().parents[1]
    assert check(root) == []
    for path in REQUIRED:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((root / path).read_text())
    for path, anchors in REQUIRED.items():
        target = tmp_path / path
        original = target.read_text()
        for anchor in anchors:
            target.write_text(original.replace(anchor, "MUTATED_ANCHOR"))
            assert check(tmp_path), (path, anchor)
            target.write_text(original)


def test_ranking_text_uses_professional_fields_only():
    facts = {
        "displayName": "PRIVATE NAME",
        "headline": "Engineer",
        "skills": ["Python"],
        "email": "PRIVATE EMAIL",
        "photoUrl": "PRIVATE PHOTO",
        "currentLocation": "PRIVATE LOCATION",
        "preferences": "PRIVATE PREFERENCES",
        "employment": [{"title": "Developer", "employer": "Acme", "description": "PRIVATE"}],
        "education": [{"qualification": "BSc", "institution": "PRIVATE SCHOOL"}],
    }
    assert publication_text(facts) == "Engineer\nPython\nDeveloper\nAcme\nBSc"


@pytest.mark.parametrize(
    "vector", [[], [1] * 383, [0] * 384, [float("nan")] * 384, [float("inf")] * 384]
)
def test_invalid_vectors_refused(vector):
    with pytest.raises(ValueError, match="invalid_publication_vector"):
        vector_literal(vector)


def test_vector_has_exact_dimensions():
    result = vector_literal([1] + [0] * 383)
    assert result.startswith("[1.0,0.0,")
    assert len(result.strip("[]").split(",")) == 384


def enable_network(monkeypatch):
    monkeypatch.setenv("CANDIDATE_NETWORK_WORKER_DSN", "worker-test-only")
    monkeypatch.setenv("CANDIDATE_NETWORK_RECOVERY_EPOCH", "b5-unit-epoch-000000001")
    for flag in (
        "CANDIDATE_FOUNDATION_ENABLED",
        "CANDIDATE_PROFILE_ENABLED",
        "CANDIDATE_SPARK_ENABLED",
        "CANDIDATE_CONSENT_V2_ENABLED",
        "CANDIDATE_NETWORK_PUBLICATION_ENABLED",
    ):
        monkeypatch.setenv(flag, "true")


def test_worker_commits_claim_before_encoding_and_checks_finish(monkeypatch):
    enable_network(monkeypatch)
    events = []
    responses = iter(
        [
            True,  # recovery gate open
            {
                "publicationId": "publication",
                "leaseToken": "lease",
                "contentHash": "hash",
                "facts": {"headline": "Engineer"},
            },
            True,  # recovery gate still open at finish
            False,  # Withdrawal won the race: not activated.
            True,
            None,
        ]
    )

    class Cursor:
        def execute(self, sql, params=None):
            events.append((sql, params))

        def fetchone(self):
            return (next(responses),)

    class Connection:
        @contextmanager
        def cursor(self):
            yield Cursor()

    @contextmanager
    def connect(dsn):
        assert dsn == "worker-test-only"
        events.append("begin")
        yield Connection()
        events.append("commit")

    class Encoder:
        def encode(self, texts):
            assert events[-1] == "commit"
            assert texts == ["Engineer"]
            events.append("encode")
            return [[1] + [0] * 383]

    assert sweep_publications("test", Encoder(), connect=connect) == 0
    finish = [event for event in events if isinstance(event, tuple) and "finish" in event[0]]
    assert len(finish) == 1
    assert finish[0][1][:3] == ("publication", "lease", "hash")


@pytest.mark.parametrize(
    "missing",
    [
        "CANDIDATE_FOUNDATION_ENABLED",
        "CANDIDATE_PROFILE_ENABLED",
        "CANDIDATE_SPARK_ENABLED",
        "CANDIDATE_CONSENT_V2_ENABLED",
    ],
)
def test_all_parent_switches_required(monkeypatch, missing):
    enable_network(monkeypatch)
    monkeypatch.delenv(missing)

    def forbidden(*args, **kwargs):
        raise AssertionError("must not connect")

    assert sweep_publications("unused", None, connect=forbidden) == 0


def test_enabled_worker_never_falls_back_to_app_credentials(monkeypatch):
    enable_network(monkeypatch)
    monkeypatch.delenv("CANDIDATE_NETWORK_WORKER_DSN")
    with pytest.raises(ValueError, match="worker_unconfigured"):
        sweep_publications(
            "ordinary-app-dsn", None, connect=lambda *args: pytest.fail("must not connect")
        )


@pytest.mark.parametrize(
    "ids",
    [
        ["not-a-uuid"],
        ["AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"],
        ["aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"] * 2,
    ],
)
def test_validation_refuses_noncanonical_or_duplicate_ids_before_database(monkeypatch, ids):
    from types import SimpleNamespace

    from fastapi import HTTPException, Response

    from activekg.api import global_memory

    enable_network(monkeypatch)
    monkeypatch.setattr(global_memory, "_require_enabled", lambda: None)

    def forbidden(*args):
        raise AssertionError("invalid input must not reach database")

    monkeypatch.setattr(global_memory, "_get_tenant_conn", forbidden)
    response = Response()
    with pytest.raises(HTTPException) as error:
        global_memory.validate_network_publications(
            global_memory.NetworkValidationRequest(publicationIds=ids),
            response,
            SimpleNamespace(tenant_id="organisation-test"),
        )
    assert error.value.status_code == 422
    assert response.headers["Cache-Control"] == "no-store"


def test_validation_requires_tenant_before_database(monkeypatch):
    from types import SimpleNamespace

    from fastapi import HTTPException, Response

    from activekg.api import global_memory

    enable_network(monkeypatch)
    monkeypatch.setattr(global_memory, "_require_enabled", lambda: None)
    with pytest.raises(HTTPException) as error:
        global_memory.validate_network_publications(
            global_memory.NetworkValidationRequest(publicationIds=[]),
            Response(),
            SimpleNamespace(tenant_id=None),
        )
    assert error.value.status_code == 403


@pytest.mark.parametrize("issuer", ["vantahire", None, "signal-impostor"])
def test_result_keys_refused_for_non_signal_issuer_before_database(monkeypatch, issuer):
    from types import SimpleNamespace

    from fastapi import HTTPException, Response

    from activekg.api import auth, global_memory

    enable_network(monkeypatch)
    monkeypatch.setattr(global_memory, "_require_enabled", lambda: None)
    monkeypatch.setattr(auth, "SIGNAL_JWT_ISSUER", "signal")

    def forbidden(*args):
        raise AssertionError("refused caller must not reach database")

    monkeypatch.setattr(global_memory, "_get_tenant_conn", forbidden)
    with pytest.raises(HTTPException) as error:
        global_memory.validate_network_publications(
            global_memory.NetworkValidationRequest(publicationIds=[], includeResultKeys=True),
            Response(),
            SimpleNamespace(tenant_id="organisation-test", issuer=issuer),
        )
    assert error.value.status_code == 403
    assert error.value.detail == "result_keys_forbidden"


def test_result_keys_allowed_for_signal_issuer(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from fastapi import Response

    from activekg.api import auth, global_memory

    enable_network(monkeypatch)
    monkeypatch.setattr(global_memory, "_require_enabled", lambda: None)
    monkeypatch.setattr(auth, "SIGNAL_JWT_ISSUER", "signal")
    calls = []

    class Cursor:
        def execute(self, sql, *args):
            calls.append(sql)

        def fetchone(self):
            return [True] if "recovery_open" in calls[-1] else [[]]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Conn:
        def cursor(self):
            return Cursor()

    @contextmanager
    def conn(tenant):
        yield Conn()

    monkeypatch.setattr(global_memory, "_get_tenant_conn", conn)
    result = global_memory.validate_network_publications(
        global_memory.NetworkValidationRequest(publicationIds=[], includeResultKeys=True),
        Response(),
        SimpleNamespace(tenant_id="organisation-test", issuer="signal"),
    )
    assert result == {"schemaVersion": 1, "publications": [], "resultKeys": []}
    assert any("candidate_network_result_keys" in sql for sql in calls)


def _gate_worker(monkeypatch, gate_answers, claims):
    """Fake worker connection: answers the B5 gate, then claim/finish."""
    enable_network(monkeypatch)
    events = []
    gate = iter(gate_answers)
    claim_rows = iter(claims)

    class Cursor:
        last = ""

        def execute(self, sql, params=None):
            self.last = sql
            events.append((sql, params))

        def fetchone(self):
            if "recovery_open" in self.last:
                return (next(gate),)
            if "claim" in self.last:
                return (next(claim_rows),)
            return (True,)

    class Connection:
        @contextmanager
        def cursor(self):
            yield Cursor()

    @contextmanager
    def connect(dsn):
        yield Connection()

    class Encoder:
        def encode(self, texts):
            return [[1] + [0] * 383]

    return events, connect, Encoder()


def test_closed_recovery_gate_claims_nothing(monkeypatch):
    events, connect, encoder = _gate_worker(monkeypatch, [False], [])
    assert sweep_publications("test", encoder, connect=connect) == 0
    assert not any("claim" in sql for sql, _ in events)


def test_missing_epoch_keeps_gate_closed_without_querying(monkeypatch):
    events, connect, encoder = _gate_worker(monkeypatch, [], [])
    monkeypatch.delenv("CANDIDATE_NETWORK_RECOVERY_EPOCH")
    assert sweep_publications("test", encoder, connect=connect) == 0
    assert events == []


def test_gate_closing_mid_flight_never_finishes(monkeypatch):
    claim = {
        "publicationId": "p",
        "leaseToken": "l",
        "contentHash": "h",
        "facts": {"headline": "E"},
    }
    events, connect, encoder = _gate_worker(monkeypatch, [True, False], [claim])
    assert sweep_publications("test", encoder, connect=connect) == 0
    assert not any("finish" in sql for sql, _ in events)
    assert (
        "SELECT public.candidate_network_recovery_open(%s)",
        ("b5-unit-epoch-000000001",),
    ) in events


@pytest.mark.parametrize("keys", [False, True])
def test_closed_recovery_gate_validates_nothing(monkeypatch, keys):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from fastapi import Response

    from activekg.api import auth, global_memory

    enable_network(monkeypatch)
    monkeypatch.setattr(global_memory, "_require_enabled", lambda: None)
    monkeypatch.setattr(auth, "SIGNAL_JWT_ISSUER", "signal")
    calls = []

    class Cursor:
        def execute(self, sql, *args):
            calls.append(sql)

        def fetchone(self):
            return [False]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Conn:
        def cursor(self):
            return Cursor()

    @contextmanager
    def conn(tenant):
        yield Conn()

    monkeypatch.setattr(global_memory, "_get_tenant_conn", conn)
    result = global_memory.validate_network_publications(
        global_memory.NetworkValidationRequest(
            publicationIds=["aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"], includeResultKeys=keys
        ),
        Response(),
        SimpleNamespace(tenant_id="organisation-test", issuer="signal"),
    )
    assert result["publications"] == [] and result.get("resultKeys", []) == []
    assert not any("candidate_network_validate" in sql or "result_keys" in sql for sql in calls)


@pytest.mark.parametrize("gate_open", [False, True])
def test_search_omits_native_rows_while_recovery_is_closed(monkeypatch, gate_open):
    from types import SimpleNamespace

    from activekg.api import global_memory

    enable_network(monkeypatch)
    monkeypatch.setattr(global_memory, "_require_enabled", lambda: None)
    monkeypatch.setattr(global_memory, "_require_public_enabled", lambda: None)
    monkeypatch.setattr(
        global_memory, "_row_to_dict", lambda cur, row: {"id": row[0], "similarity": 0.5}
    )
    monkeypatch.setattr(global_memory, "_log_search_phases", lambda *a, **k: None)

    class Vector(list):
        def tolist(self):
            return list(self)

    class Encoder:
        def encode(self, texts):
            return [Vector([1.0] + [0.0] * 383)]

    monkeypatch.setattr(global_memory, "_embedder", Encoder())
    calls = []
    native = {"kind": "candidate_network", "identity": "network:x", "similarity": 0.9}

    class Cursor:
        def execute(self, sql, *args):
            calls.append(sql)

        def fetchall(self):
            return [("provider-1",)]

        def fetchone(self):
            return [gate_open] if "recovery_open" in calls[-1] else [[native]]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Conn:
        def cursor(self):
            return Cursor()

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(global_memory, "_get_tenant_conn", lambda tenant: Conn())
    result = global_memory.search_global_candidates(
        global_memory.GlobalCandidateSearchRequest(query_text="engineer", surface="public_v2"),
        SimpleNamespace(tenant_id="organisation-test", issuer="signal"),
    )
    kinds = [row["kind"] for row in result["results"]]
    assert result["surface"] == "public_v2" and result["count"] == len(result["results"])
    if gate_open:
        assert "candidate_network" in kinds
    else:
        assert kinds == ["provider"]
        assert not any("candidate_network_search" in sql for sql in calls)
