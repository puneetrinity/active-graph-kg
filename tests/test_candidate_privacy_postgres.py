from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import UUID, uuid4

import numpy as np
import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from activekg.api.operational import bounded_readiness_check
from activekg.graph.candidate_repository import CandidateRepository
from activekg.graph.models import Candidate, Edge, Node
from activekg.graph.repository import GraphRepository
from activekg.privacy.config import CandidatePrivacyConfig
from activekg.privacy.identity import normalize_privacy_identifier
from activekg.privacy.models import (
    CandidatePrivacyAction,
    CandidatePrivacyAuthorityType,
    CandidatePrivacyDecision,
    CandidatePrivacyReason,
    CandidatePrivacyTransition,
    CanonicalSubject,
)
from activekg.privacy.repository import (
    CandidatePrivacyConflict,
    CandidatePrivacyRepository,
    CandidatePrivacyUnavailable,
)

OWNER_DSN = os.getenv("ACTIVEKG_PRIVACY_TEST_OWNER_DSN")
RUNTIME_DSN = os.getenv("ACTIVEKG_PRIVACY_TEST_RUNTIME_DSN")

pytestmark = pytest.mark.skipif(
    not OWNER_DSN or not RUNTIME_DSN,
    reason="disposable candidate-privacy PostgreSQL DSNs are not configured",
)


def _dsn(dsn: str, database: str) -> str:
    return make_conninfo(dsn, dbname=database)


def _maintenance() -> psycopg.Connection:
    return psycopg.connect(make_conninfo(OWNER_DSN, dbname="postgres"), autocommit=True)


def _drop_database(name: str) -> None:
    with _maintenance() as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))


def _clone_database(name: str) -> tuple[str, str]:
    source = psycopg.conninfo.conninfo_to_dict(OWNER_DSN)["dbname"]
    _drop_database(name)
    with _maintenance() as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
                sql.Identifier(name), sql.Identifier(source)
            )
        )
    return _dsn(OWNER_DSN, name), _dsn(RUNTIME_DSN, name)


def _config(*versions: int, active: int | None = None) -> CandidatePrivacyConfig:
    active_version = active if active is not None else max(versions)
    return CandidatePrivacyConfig(
        active_key_version=active_version,
        keys={version: bytes([version]) * 32 for version in versions},
        intake_enabled=True,
        flow_issuer="flow",
        flow_actor_id="flow-service",
        signal_issuer="signal",
        signal_actor_id="signal-service",
    )


def _create(
    repository: CandidatePrivacyRepository,
    *,
    request_id: UUID,
    action: CandidatePrivacyAction = CandidatePrivacyAction.REQUEST_ERASURE,
    evidence_ref: UUID | None = None,
    identifiers=None,
):
    return repository.create_directive(
        request_id=request_id,
        action=action,
        authority_type=CandidatePrivacyAuthorityType.VERIFIED_CANDIDATE,
        evidence_ref=evidence_ref or UUID("33333333-3333-4333-8333-333333333333"),
        reason=(
            CandidatePrivacyReason.CANDIDATE_ERASURE_REQUEST
            if action is CandidatePrivacyAction.REQUEST_ERASURE
            else CandidatePrivacyReason.CANDIDATE_GLOBAL_OPT_OUT
        ),
        issuer="flow",
        actor_id="flow-service",
        identifiers=identifiers
        or [normalize_privacy_identifier("email", "privacy.person@example.test")],
        canonical=CanonicalSubject(),
        effective_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
    )


def test_privacy_objects_and_runtime_privileges_are_exact() -> None:
    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname='public' AND c.relname IN "
            "('candidate_privacy_directive_events','candidate_privacy_directives',"
            "'candidate_privacy_identity_tokens')"
        )
        assert cur.fetchone()[0] == 3
        cur.execute(
            "SELECT relname, relrowsecurity FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname='public' AND relname LIKE 'candidate_privacy_%' AND relkind='r'"
        )
        assert dict(cur.fetchall()) == {
            "candidate_privacy_directive_events": True,
            "candidate_privacy_directives": True,
            "candidate_privacy_identity_tokens": True,
        }
    with psycopg.connect(RUNTIME_DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM candidate_privacy_directives")
        assert cur.fetchone()[0] == 0
        for statement in (
            "INSERT INTO candidate_privacy_directive_events "
            "(directive_id,directive_version,request_id,event_type,action,scope,"
            "resulting_state,authority_type,evidence_ref,reason_code,issuer,actor_id,"
            "actor_type,key_version,effective_at) VALUES "
            "(gen_random_uuid(),1,gen_random_uuid(),'requested','request_erasure',"
            "'active_profile','requested','privacy_operator',gen_random_uuid(),"
            "'verified_support_request','test','test','service',1,now())",
            "UPDATE candidate_privacy_directives SET updated_at=now()",
            "SELECT nextval('candidate_privacy_directive_events_cursor_seq')",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                cur.execute(statement)
            conn.rollback()
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute("SELECT * FROM candidate_privacy_identity_tokens")
        conn.rollback()
        cur.execute("SELECT candidate_privacy_token_key_versions()")
        assert cur.fetchall() == []


def test_three_event_activation_exact_replay_cas_release_and_append_only() -> None:
    owner_dsn, runtime_dsn = _clone_database("memory_privacy_lifecycle_test")
    repository = CandidatePrivacyRepository(runtime_dsn, config=_config(1))
    request_id = UUID("11111111-1111-4111-8111-111111111111")
    evidence = UUID("33333333-3333-4333-8333-333333333333")
    try:
        returned_request, created = _create(
            repository, request_id=request_id, evidence_ref=evidence
        )
        assert returned_request == request_id
        assert created.version == 3
        assert created.decision is CandidatePrivacyDecision.BLOCK_ALL

        replay_request, replay = _create(repository, request_id=request_id, evidence_ref=evidence)
        assert replay_request == request_id
        assert replay.directive_id == created.directive_id
        assert replay.version == 3

        with pytest.raises(CandidatePrivacyConflict):
            _create(repository, request_id=request_id, evidence_ref=uuid4())

        transition_request = UUID("44444444-4444-4444-8444-444444444444")
        _, released = repository.transition_directive(
            directive_id=created.directive_id,
            expected_version=3,
            request_id=transition_request,
            transition=CandidatePrivacyTransition.RELEASE,
            evidence_ref=UUID("55555555-5555-4555-8555-555555555555"),
            reason=CandidatePrivacyReason.OPERATOR_CORRECTION,
            issuer="flow",
            actor_id="flow-service",
            effective_at=datetime(2026, 8, 23, tzinfo=timezone.utc),
        )
        assert released.version == 4
        assert released.decision is CandidatePrivacyDecision.ALLOW
        replay_request, replay_after_release = _create(
            repository, request_id=request_id, evidence_ref=evidence
        )
        assert replay_request == request_id
        assert replay_after_release.directive_id == created.directive_id
        assert replay_after_release.version == 4
        assert (
            repository.evaluate(
                identifiers=[normalize_privacy_identifier("email", "privacy.person@example.test")]
            )
            is CandidatePrivacyDecision.ALLOW
        )
        with pytest.raises(CandidatePrivacyConflict):
            repository.transition_directive(
                directive_id=created.directive_id,
                expected_version=3,
                request_id=uuid4(),
                transition=CandidatePrivacyTransition.MARK_NEEDS_REVIEW,
                evidence_ref=uuid4(),
                reason=CandidatePrivacyReason.IDENTITY_AMBIGUITY,
                issuer="flow",
                actor_id="flow-service",
            )
        with pytest.raises(CandidatePrivacyConflict):
            repository.transition_directive(
                directive_id=created.directive_id,
                expected_version=4,
                request_id=request_id,
                transition=CandidatePrivacyTransition.MARK_NEEDS_REVIEW,
                evidence_ref=uuid4(),
                reason=CandidatePrivacyReason.IDENTITY_AMBIGUITY,
                issuer="flow",
                actor_id="flow-service",
            )

        high_water, snapshot = repository.snapshot(
            after_directive_id=None,
            high_water_cursor=None,
            limit=100,
        )
        assert len(snapshot) == 1
        with pytest.raises(CandidatePrivacyConflict):
            repository.snapshot(
                after_directive_id=None,
                high_water_cursor=high_water + 1,
                limit=100,
            )

        with psycopg.connect(owner_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT event_type, resulting_state, directive_version "
                "FROM candidate_privacy_directive_events ORDER BY cursor"
            )
            assert cur.fetchall() == [
                ("requested", "requested", 1),
                ("verified", "verified", 2),
                ("activated", "active_quarantine", 3),
                ("released", "released", 4),
            ]
            for statement in (
                "UPDATE candidate_privacy_directive_events SET reason_code='operator_correction'",
                "DELETE FROM candidate_privacy_identity_tokens",
                "TRUNCATE candidate_privacy_directive_events",
                "TRUNCATE candidate_privacy_identity_tokens",
            ):
                with pytest.raises(psycopg.Error):
                    cur.execute(statement)
                conn.rollback()
    finally:
        repository.close()
        _drop_database("memory_privacy_lifecycle_test")


def test_same_request_concurrently_activates_exactly_once() -> None:
    owner_dsn, runtime_dsn = _clone_database("memory_privacy_concurrent_test")
    request_id = UUID("66666666-6666-4666-8666-666666666666")

    def execute() -> UUID:
        repository = CandidatePrivacyRepository(runtime_dsn, config=_config(1))
        try:
            return _create(repository, request_id=request_id)[1].directive_id
        finally:
            repository.close()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            directive_ids = list(executor.map(lambda _value: execute(), range(2)))
        assert directive_ids[0] == directive_ids[1]
        with psycopg.connect(owner_dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM candidate_privacy_directives")
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT count(*) FROM candidate_privacy_directive_events")
            assert cur.fetchone()[0] == 3
    finally:
        _drop_database("memory_privacy_concurrent_test")


def test_ambiguity_is_review_and_key_rotation_is_fail_closed_by_readiness() -> None:
    owner_dsn, runtime_dsn = _clone_database("memory_privacy_rotation_test")
    old_repository = CandidatePrivacyRepository(runtime_dsn, config=_config(1))
    identifier = normalize_privacy_identifier("email", "ambiguous.person@example.test")
    try:
        with psycopg.connect(owner_dsn) as conn, conn.cursor() as cur:
            for tenant_id, candidate_id in (
                ("org-a", UUID("77777777-7777-4777-8777-777777777777")),
                ("org-b", UUID("88888888-8888-4888-8888-888888888888")),
            ):
                cur.execute(
                    "INSERT INTO candidates (candidate_id, tenant_id, scope, display_name) "
                    "VALUES (%s,%s,'shared',%s)",
                    (candidate_id, tenant_id, "Synthetic Privacy Candidate"),
                )
                cur.execute(
                    "INSERT INTO candidate_identifiers "
                    "(candidate_id, tenant_id, identifier_type, value_normalized, value_raw) "
                    "VALUES (%s,%s,'email',%s,%s)",
                    (candidate_id, tenant_id, identifier.normalized, identifier.normalized),
                )
        _, directive = _create(
            old_repository,
            request_id=UUID("99999999-9999-4999-8999-999999999999"),
            identifiers=[identifier],
        )
        assert directive.decision is CandidatePrivacyDecision.REVIEW

        token_only = normalize_privacy_identifier("email", "future.import@example.test")
        _, token_only_directive = _create(
            old_repository,
            request_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            identifiers=[token_only],
        )
        assert token_only_directive.decision is CandidatePrivacyDecision.BLOCK_ALL

        dual_read = CandidatePrivacyRepository(runtime_dsn, config=_config(1, 2, active=2))
        new_only = CandidatePrivacyRepository(runtime_dsn, config=_config(2))
        try:
            replay_request, replay = _create(
                dual_read,
                request_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
                identifiers=[token_only],
            )
            assert replay_request == UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
            assert replay.directive_id == token_only_directive.directive_id
            assert replay.version == 3
            assert dual_read.evaluate(identifiers=[identifier]) is CandidatePrivacyDecision.REVIEW
            assert new_only.evaluate(identifiers=[identifier]) is CandidatePrivacyDecision.REVIEW
            assert (
                dual_read.evaluate(identifiers=[token_only]) is CandidatePrivacyDecision.BLOCK_ALL
            )
            assert new_only.evaluate(identifiers=[token_only]) is CandidatePrivacyDecision.ALLOW
            assert new_only.referenced_key_versions() == {1}

            with patch.dict(
                "os.environ",
                {
                    "ACTIVEKG_SCHEMA_TARGET_ID": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                    "ACTIVEKG_SCHEMA_ENVIRONMENT": "development",
                },
                clear=False,
            ):
                result = bounded_readiness_check(
                    type("Repository", (), {"pool": new_only.pool})(),
                    unsafe_search_configuration=False,
                    jwt_enabled=True,
                    jwt_problems=[],
                    privacy_problems=[],
                    privacy_key_versions={2},
                )
            assert result.ready is False
            assert "candidate_privacy_hmac_version_missing" in result.reasons
        finally:
            dual_read.close()
            new_only.close()
    finally:
        old_repository.close()
        _drop_database("memory_privacy_rotation_test")


def test_multiple_tenant_rows_linked_to_one_global_subject_are_not_ambiguous() -> None:
    owner_dsn, runtime_dsn = _clone_database("memory_privacy_reconciled_test")
    repository = CandidatePrivacyRepository(runtime_dsn, config=_config(1))
    identifier = normalize_privacy_identifier("email", "reconciled.person@example.test")
    global_id = UUID("abababab-abab-4bab-8bab-abababababab")
    try:
        with psycopg.connect(owner_dsn) as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO global_candidates (id) VALUES (%s)", (global_id,))
            for tenant_id, candidate_id in (
                ("org-a", UUID("acacacac-acac-4cac-8cac-acacacacacac")),
                ("org-b", UUID("adadadad-adad-4dad-8dad-adadadadadad")),
            ):
                cur.execute(
                    "INSERT INTO candidates "
                    "(candidate_id, tenant_id, scope, display_name, global_candidate_id) "
                    "VALUES (%s,%s,'shared',%s,%s)",
                    (candidate_id, tenant_id, "Synthetic Reconciled Candidate", global_id),
                )
                cur.execute(
                    "INSERT INTO candidate_identifiers "
                    "(candidate_id, tenant_id, identifier_type, value_normalized, value_raw) "
                    "VALUES (%s,%s,'email',%s,%s)",
                    (candidate_id, tenant_id, identifier.normalized, identifier.normalized),
                )

        _, directive = _create(
            repository,
            request_id=UUID("aeaeaeae-aeae-4eae-8eae-aeaeaeaeaeae"),
            identifiers=[identifier],
        )
        assert directive.decision is CandidatePrivacyDecision.BLOCK_ALL
        with psycopg.connect(owner_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT global_candidate_id, candidate_tenant_id, candidate_id "
                "FROM candidate_privacy_directives WHERE directive_id = %s",
                (directive.directive_id,),
            )
            assert cur.fetchone() == (global_id, None, None)
    finally:
        repository.close()
        _drop_database("memory_privacy_reconciled_test")


def test_github_global_resolution_and_incompatible_canonical_links_are_review() -> None:
    owner_dsn, runtime_dsn = _clone_database("memory_privacy_canonical_test")
    repository = CandidatePrivacyRepository(runtime_dsn, config=_config(1))
    linked_global_id = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    conflicting_global_id = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
    linkedin_global_id = UUID("abababab-abab-4aba-8aba-abababababab")
    candidate_id = UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd")
    legacy_email_global_id = UUID("adadadad-adad-4ada-8ada-adadadadadad")
    primary_email_candidate_id = UUID("dededede-dede-4ede-8ede-dededededede")
    try:
        with psycopg.connect(owner_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO global_candidates (id, github_id) VALUES (%s,%s),(%s,%s)",
                (linked_global_id, "PrivacyPerson", conflicting_global_id, "OtherPerson"),
            )
            cur.execute(
                "INSERT INTO global_candidates (id, linkedin_id, linkedin_url) VALUES (%s,%s,%s)",
                (
                    linkedin_global_id,
                    "PrivacyLinkedIn",
                    "https://www.linkedin.com/in/PrivacyLinkedIn/?trk=legacy",
                ),
            )
            cur.execute(
                "INSERT INTO candidates "
                "(candidate_id, tenant_id, scope, global_candidate_id, display_name) "
                "VALUES (%s,'org-a','shared',%s,'Synthetic Privacy Candidate')",
                (candidate_id, linked_global_id),
            )
            cur.execute(
                "INSERT INTO global_candidates (id, email_hash) VALUES (%s,%s)",
                (
                    legacy_email_global_id,
                    hashlib.sha256(b"first.last@googlemail.com").hexdigest(),
                ),
            )
            cur.execute(
                "INSERT INTO candidates "
                "(candidate_id, tenant_id, scope, primary_email, display_name) "
                "VALUES (%s,'org-a','shared',%s,'Primary Email Only')",
                (primary_email_candidate_id, "primary.only@example.test"),
            )

        github = normalize_privacy_identifier("github_url", "https://github.com/PrivacyPerson")
        _, linked = _create(
            repository,
            request_id=UUID("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"),
            identifiers=[github],
        )
        assert linked.decision is CandidatePrivacyDecision.BLOCK_ALL
        with psycopg.connect(owner_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT global_candidate_id FROM candidate_privacy_directives "
                "WHERE directive_id=%s",
                (linked.directive_id,),
            )
            assert cur.fetchone()[0] == linked_global_id

        linkedin = normalize_privacy_identifier("linkedin_url", "linkedin.com/in/privacylinkedin/")
        _, linkedin_linked = _create(
            repository,
            request_id=UUID("cdcdcdcd-cdcd-4dcd-8dcd-cdcdcdcdcdcd"),
            identifiers=[linkedin],
        )
        with psycopg.connect(owner_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT global_candidate_id FROM candidate_privacy_directives "
                "WHERE directive_id=%s",
                (linkedin_linked.directive_id,),
            )
            assert cur.fetchone()[0] == linkedin_global_id

        _, legacy_email_linked = _create(
            repository,
            request_id=UUID("acacacac-acac-4aca-8aca-acacacacacac"),
            identifiers=[normalize_privacy_identifier("email", "First.Last@GoogleMail.com")],
        )
        _, primary_email_linked = _create(
            repository,
            request_id=UUID("dcdcdcdc-dcdc-4dcd-8dcd-dcdcdcdcdcdc"),
            identifiers=[normalize_privacy_identifier("email", "primary.only@example.test")],
        )
        with psycopg.connect(owner_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT global_candidate_id FROM candidate_privacy_directives "
                "WHERE directive_id=%s",
                (legacy_email_linked.directive_id,),
            )
            assert cur.fetchone()[0] == legacy_email_global_id
            cur.execute(
                "SELECT candidate_tenant_id, candidate_id "
                "FROM candidate_privacy_directives WHERE directive_id=%s",
                (primary_email_linked.directive_id,),
            )
            assert cur.fetchone() == ("org-a", primary_email_candidate_id)

        _, inconsistent = repository.create_directive(
            request_id=UUID("ffffffff-ffff-4fff-8fff-ffffffffffff"),
            action=CandidatePrivacyAction.REQUEST_ERASURE,
            authority_type=CandidatePrivacyAuthorityType.VERIFIED_CANDIDATE,
            evidence_ref=UUID("12121212-1212-4212-8212-121212121212"),
            reason=CandidatePrivacyReason.CANDIDATE_ERASURE_REQUEST,
            issuer="flow",
            actor_id="flow-service",
            identifiers=[],
            canonical=CanonicalSubject(
                global_candidate_id=conflicting_global_id,
                candidate_tenant_id="org-a",
                candidate_id=candidate_id,
            ),
            effective_at=datetime(2026, 8, 22, tzinfo=timezone.utc),
        )
        assert inconsistent.decision is CandidatePrivacyDecision.REVIEW
    finally:
        repository.close()
        _drop_database("memory_privacy_canonical_test")


def test_identifier_evaluation_treats_tenant_only_as_lookup_context() -> None:
    owner_dsn, runtime_dsn = _clone_database("memory_privacy_tenant_context_test")
    repository = CandidatePrivacyRepository(runtime_dsn, config=_config(1))
    identifier = normalize_privacy_identifier("email", "tenant-context@example.test")
    try:
        assert (
            repository.evaluate(
                identifiers=[identifier],
                candidate_tenant_id="tenant-context-only",
            )
            is CandidatePrivacyDecision.ALLOW
        )
        with pytest.raises(CandidatePrivacyUnavailable):
            repository.evaluate(candidate_id=uuid4())
    finally:
        repository.close()
        _drop_database("memory_privacy_tenant_context_test")


def test_repository_and_node_fences_keep_global_opt_out_private_then_hide_erasure() -> None:
    owner_dsn, runtime_dsn = _clone_database("memory_privacy_fences_test")
    privacy = CandidatePrivacyRepository(runtime_dsn, config=_config(1))
    candidates = CandidateRepository(runtime_dsn, privacy_repository=privacy)
    graph = GraphRepository(runtime_dsn)
    candidate_id = UUID("13131313-1313-4313-8313-131313131313")
    node_id = UUID("14141414-1414-4414-8414-141414141414")
    email = "private.workflow@example.test"
    try:
        graph.create_node(
            Node(
                id=str(node_id),
                tenant_id="org-a",
                classes=["Resume"],
                props={
                    "candidate_id": str(candidate_id),
                    "candidate_tenant_id": "org-a",
                    "text": "synthetic resume",
                },
            )
        )
        candidates.create_candidate(
            Candidate(
                candidate_id=str(candidate_id),
                tenant_id="org-a",
                display_name="Synthetic Private Workflow",
                primary_email=email,
                node_id=str(node_id),
            )
        )
        candidates.add_identifier(
            str(candidate_id),
            "email",
            email,
            tenant_id="org-a",
            source="vantahire",
        )

        _, global_opt_out = privacy.create_directive(
            request_id=UUID("15151515-1515-4515-8515-151515151515"),
            action=CandidatePrivacyAction.WITHDRAW_GLOBAL_MATCHING,
            authority_type=CandidatePrivacyAuthorityType.VERIFIED_CANDIDATE,
            evidence_ref=UUID("16161616-1616-4616-8616-161616161616"),
            reason=CandidatePrivacyReason.CANDIDATE_GLOBAL_OPT_OUT,
            issuer="flow",
            actor_id="flow-service",
            identifiers=[normalize_privacy_identifier("email", email)],
            canonical=CanonicalSubject(
                candidate_tenant_id="org-a",
                candidate_id=candidate_id,
            ),
        )
        assert global_opt_out.decision is CandidatePrivacyDecision.BLOCK_GLOBAL
        assert candidates.get_candidate(str(candidate_id), tenant_id="org-a") is not None
        assert len(candidates.list_identifiers(str(candidate_id), tenant_id="org-a")) == 1
        assert graph.get_node(str(node_id), tenant_id="org-a") is not None

        _, erasure = privacy.create_directive(
            request_id=UUID("17171717-1717-4717-8717-171717171717"),
            action=CandidatePrivacyAction.REQUEST_ERASURE,
            authority_type=CandidatePrivacyAuthorityType.VERIFIED_CANDIDATE,
            evidence_ref=UUID("18181818-1818-4818-8818-181818181818"),
            reason=CandidatePrivacyReason.CANDIDATE_ERASURE_REQUEST,
            issuer="flow",
            actor_id="flow-service",
            identifiers=[normalize_privacy_identifier("email", email)],
            canonical=CanonicalSubject(
                candidate_tenant_id="org-a",
                candidate_id=candidate_id,
            ),
        )
        assert erasure.decision is CandidatePrivacyDecision.BLOCK_ALL
        assert candidates.get_candidate(str(candidate_id), tenant_id="org-a") is None
        assert candidates.list_identifiers(str(candidate_id), tenant_id="org-a") == []
        assert graph.get_node(str(node_id), tenant_id="org-a") is None
        assert all(item.id != str(node_id) for item in graph.list_nodes(tenant_id="org-a"))
        with pytest.raises(PermissionError):
            graph.update_node_embedding(
                str(node_id),
                np.zeros(384),
                0.0,
                datetime.now(timezone.utc).isoformat(),
                tenant_id="org-a",
            )

        _, released = privacy.transition_directive(
            directive_id=erasure.directive_id,
            expected_version=3,
            request_id=UUID("19191919-1919-4919-8919-191919191919"),
            transition=CandidatePrivacyTransition.RELEASE,
            evidence_ref=UUID("20202020-2020-4020-8020-202020202020"),
            reason=CandidatePrivacyReason.OPERATOR_CORRECTION,
            issuer="flow",
            actor_id="flow-service",
        )
        assert released.decision is CandidatePrivacyDecision.ALLOW
        # The separate global opt-out remains active, so tenant-private use is
        # restored while global/public use remains blocked.
        assert candidates.get_candidate(str(candidate_id), tenant_id="org-a") is not None
        assert (
            privacy.canonical_decision(candidate_tenant_id="org-a", candidate_id=candidate_id)
            is CandidatePrivacyDecision.BLOCK_GLOBAL
        )
        assert graph.get_node(str(node_id), tenant_id="org-a") is not None
    finally:
        graph.pool.close()
        candidates.close()
        privacy.close()
        _drop_database("memory_privacy_fences_test")


def test_privacy_filtered_ann_underfill_rescans_exactly() -> None:
    """A privacy-filtered IVFFlat miss must not become an empty success."""
    database = "memory_privacy_ann_recall_test"
    owner_dsn, runtime_dsn = _clone_database(database)
    graph: GraphRepository | None = None
    query = np.zeros(384, dtype=np.float32)
    query[0] = 1.0
    query_literal = "[" + ",".join(str(float(value)) for value in query) + "]"
    try:
        with psycopg.connect(runtime_dsn) as runtime_conn, runtime_conn.cursor() as cur:
            cur.execute("SELECT current_user")
            runtime_role = cur.fetchone()[0]

        with psycopg.connect(owner_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("ALTER ROLE {} IN DATABASE {} SET enable_seqscan TO off").format(
                    sql.Identifier(runtime_role), sql.Identifier(database)
                )
            )
            rows: list[tuple[list[str], str, str, str]] = []
            for index in range(240):
                vector = np.zeros(384, dtype=np.float32)
                vector[0] = 1.0
                vector[1] = (index + 1) / 100000.0
                rows.append(
                    (
                        ["SyntheticANN"],
                        '{"text":"ann decoy"}',
                        "[" + ",".join(str(float(value)) for value in vector) + "]",
                        '{"privacy_ann_probe":"decoy"}',
                    )
                )
            for index in range(12):
                vector = np.zeros(384, dtype=np.float32)
                vector[0] = -1.0
                vector[1] = (index + 1) / 100000.0
                rows.append(
                    (
                        ["SyntheticANN"],
                        '{"text":"ann allowed"}',
                        "[" + ",".join(str(float(value)) for value in vector) + "]",
                        '{"privacy_ann_probe":"allowed"}',
                    )
                )
            cur.executemany(
                "INSERT INTO nodes (classes,props,embedding,metadata) "
                "VALUES (%s,%s::jsonb,%s::vector,%s::jsonb)",
                rows,
            )
            cur.execute(
                "CREATE INDEX privacy_ann_recall_ivfflat_idx ON nodes "
                "USING ivfflat (embedding vector_cosine_ops) WITH (lists=20)"
            )
            cur.execute("ANALYZE nodes")

        raw_sql = """
            SELECT id
            FROM nodes
            WHERE embedding IS NOT NULL
              AND (candidate_privacy_node_decision(id) = 'allow' OR (
                   candidate_privacy_node_decision(id) = 'block_global'
                   AND tenant_id IS NOT NULL AND tenant_id IS NOT DISTINCT FROM NULL))
              AND metadata @> '{"privacy_ann_probe":"allowed"}'::jsonb
            ORDER BY embedding <=> %s::vector
            LIMIT 10
        """
        with psycopg.connect(runtime_dsn) as conn, conn.cursor() as cur:
            cur.execute("SET ivfflat.probes = 1")
            cur.execute("EXPLAIN (COSTS OFF) " + raw_sql, (query_literal,))
            plan = "\n".join(row[0] for row in cur.fetchall())
            assert "privacy_ann_recall_ivfflat_idx" in plan
            cur.execute(raw_sql, (query_literal,))
            assert len(cur.fetchall()) < 10

        graph = GraphRepository(runtime_dsn)
        with patch.dict(
            "os.environ",
            {"PGVECTOR_INDEXES": "ivfflat", "IVFFLAT_PROBES": "1"},
            clear=False,
        ):
            vector_results = graph.vector_search(
                query,
                top_k=10,
                compound_filter={"privacy_ann_probe": "allowed"},
            )
            hybrid_results = graph.hybrid_search(
                "ann allowed",
                query,
                top_k=10,
                compound_filter={"privacy_ann_probe": "allowed"},
                use_reranker=False,
            )
        assert len(vector_results) == 10
        assert len(hybrid_results) == 10
        assert all(
            node.metadata.get("privacy_ann_probe") == "allowed"
            for node, _score in vector_results + hybrid_results
        )
    finally:
        if graph is not None:
            graph.pool.close()
        _drop_database(database)


def test_global_opt_out_never_exposes_public_node_and_lineage_stops_at_hidden_parent() -> None:
    owner_dsn, runtime_dsn = _clone_database("memory_privacy_node_scope_test")
    privacy = CandidatePrivacyRepository(runtime_dsn, config=_config(1))
    candidates = CandidateRepository(runtime_dsn, privacy_repository=privacy)
    graph = GraphRepository(runtime_dsn)
    public_node_id = UUID("21212121-2121-4121-8121-212121212121")
    public_candidate_id = UUID("22222222-2222-4222-8222-222222222223")
    root_id = UUID("23232323-2323-4323-8323-232323232323")
    hidden_parent_id = UUID("24242424-2424-4424-8424-242424242424")
    hidden_candidate_id = UUID("25252525-2525-4525-8525-252525252525")
    grandparent_id = UUID("26262626-2626-4626-8626-262626262626")
    try:
        graph.create_node(Node(id=str(public_node_id), classes=["Candidate"], props={}))
        candidates.create_candidate(
            Candidate(
                candidate_id=str(public_candidate_id),
                tenant_id="org-a",
                display_name="Public Projection Link",
                node_id=str(public_node_id),
            )
        )
        _, global_opt_out = privacy.create_directive(
            request_id=UUID("27272727-2727-4727-8727-272727272727"),
            action=CandidatePrivacyAction.WITHDRAW_GLOBAL_MATCHING,
            authority_type=CandidatePrivacyAuthorityType.VERIFIED_CANDIDATE,
            evidence_ref=UUID("28282828-2828-4828-8828-282828282828"),
            reason=CandidatePrivacyReason.CANDIDATE_GLOBAL_OPT_OUT,
            issuer="flow",
            actor_id="flow-service",
            identifiers=[],
            canonical=CanonicalSubject(
                candidate_tenant_id="org-a",
                candidate_id=public_candidate_id,
            ),
        )
        assert global_opt_out.decision is CandidatePrivacyDecision.BLOCK_GLOBAL
        assert graph.get_node(str(public_node_id), tenant_id="org-a") is None

        for node_id, classes in (
            (root_id, ["Document"]),
            (hidden_parent_id, ["Resume"]),
            (grandparent_id, ["Document"]),
        ):
            graph.create_node(Node(id=str(node_id), tenant_id="org-a", classes=classes, props={}))
        candidates.create_candidate(
            Candidate(
                candidate_id=str(hidden_candidate_id),
                tenant_id="org-a",
                display_name="Hidden Lineage Parent",
                node_id=str(hidden_parent_id),
            )
        )
        graph.create_edge(
            Edge(
                src=str(root_id),
                rel="DERIVED_FROM",
                dst=str(hidden_parent_id),
                tenant_id="org-a",
            )
        )
        graph.create_edge(
            Edge(
                src=str(hidden_parent_id),
                rel="DERIVED_FROM",
                dst=str(grandparent_id),
                tenant_id="org-a",
            )
        )
        privacy.create_directive(
            request_id=UUID("29292929-2929-4929-8929-292929292929"),
            action=CandidatePrivacyAction.REQUEST_ERASURE,
            authority_type=CandidatePrivacyAuthorityType.VERIFIED_CANDIDATE,
            evidence_ref=UUID("30303030-3030-4030-8030-303030303030"),
            reason=CandidatePrivacyReason.CANDIDATE_ERASURE_REQUEST,
            issuer="flow",
            actor_id="flow-service",
            identifiers=[],
            canonical=CanonicalSubject(
                candidate_tenant_id="org-a",
                candidate_id=hidden_candidate_id,
            ),
        )
        assert graph.get_lineage(str(root_id), tenant_id="org-a") == []
    finally:
        graph.pool.close()
        candidates.close()
        privacy.close()
        _drop_database("memory_privacy_node_scope_test")
