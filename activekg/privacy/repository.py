"""Database authority and fail-closed fences for candidate privacy."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from psycopg_pool import ConnectionPool

from activekg.privacy.config import CandidatePrivacyConfig
from activekg.privacy.identity import NormalizedIdentifier, identity_token
from activekg.privacy.models import (
    CandidatePrivacyAction,
    CandidatePrivacyAuthorityType,
    CandidatePrivacyDecision,
    CandidatePrivacyReason,
    CandidatePrivacyScope,
    CandidatePrivacyState,
    CandidatePrivacyTransition,
    CanonicalSubject,
    DirectiveRecord,
    decision_for,
)


class CandidatePrivacyError(RuntimeError):
    """Base class whose message is safe and contains no submitted identity."""


class CandidatePrivacyUnavailable(CandidatePrivacyError):
    pass


class CandidatePrivacyConflict(CandidatePrivacyError):
    pass


class CandidatePrivacyRestricted(CandidatePrivacyError):
    pass


class CandidatePrivacyRepository:
    def __init__(
        self,
        dsn: str,
        *,
        pool: ConnectionPool | None = None,
        config: CandidatePrivacyConfig | None = None,
    ) -> None:
        self._owns_pool = pool is None
        self.pool = pool or ConnectionPool(dsn, min_size=1, max_size=5, timeout=30.0, open=True)
        self.config = config

    def close(self) -> None:
        if self._owns_pool:
            self.pool.close()

    @contextmanager
    def _conn(self) -> Iterator[Any]:
        try:
            with self.pool.connection() as conn, conn.transaction():
                yield conn
        except CandidatePrivacyError:
            raise
        except Exception as exc:
            raise CandidatePrivacyUnavailable("candidate privacy authority is unavailable") from exc

    @staticmethod
    def _record(row: Sequence[Any]) -> DirectiveRecord:
        return DirectiveRecord(
            directive_id=UUID(str(row[0])),
            action=CandidatePrivacyAction(row[1]),
            scope=CandidatePrivacyScope(row[2]),
            state=CandidatePrivacyState(row[3]),
            version=int(row[4]),
            effective_at=row[5],
            decision=decision_for(row[3], row[1]),
        )

    @staticmethod
    def _subject_from_matches(
        matches: set[tuple[UUID | None, str | None, UUID | None]],
    ) -> CanonicalSubject:
        if not matches:
            return CanonicalSubject()

        global_ids = {row[0] for row in matches if row[0] is not None}
        candidate_pairs = {(row[1], row[2]) for row in matches if row[2] is not None}
        # Several tenant-local rows can legitimately represent the same person
        # after reconciliation.  If every match carries the same non-null
        # global id, preserve that unambiguous global identity and omit the
        # non-unique tenant pair.  Rows with missing/conflicting global links
        # remain conservative review cases.
        one_global = next(iter(global_ids), None) if len(global_ids) == 1 else None
        candidate_pairs_are_compatible = bool(one_global) and all(
            row[0] == one_global for row in matches
        )
        needs_review = len(global_ids) > 1 or (
            len(candidate_pairs) > 1 and not candidate_pairs_are_compatible
        )
        first = sorted(matches, key=lambda row: tuple("" if v is None else str(v) for v in row))[0]
        return CanonicalSubject(
            global_candidate_id=one_global,
            candidate_tenant_id=first[1] if len(candidate_pairs) <= 1 else None,
            candidate_id=first[2] if len(candidate_pairs) <= 1 else None,
            needs_review=needs_review,
        )

    @classmethod
    def _resolve_subject_on_cursor(
        cls,
        cur: Any,
        identifiers: Sequence[NormalizedIdentifier],
    ) -> CanonicalSubject:
        matches: set[tuple[UUID | None, str | None, UUID | None]] = set()
        for identifier in identifiers:
            for digest in identifier.lookup_digests:
                cur.execute(
                    "SELECT global_candidate_id, candidate_tenant_id, candidate_id "
                    "FROM candidate_privacy_resolve_subject(%s, %s)",
                    (identifier.identifier_type, digest),
                )
                for global_id, tenant_id, candidate_id in cur.fetchall():
                    matches.add(
                        (
                            UUID(str(global_id)) if global_id else None,
                            str(tenant_id) if tenant_id else None,
                            UUID(str(candidate_id)) if candidate_id else None,
                        )
                    )
        return cls._subject_from_matches(matches)

    def resolve_subject(self, identifiers: Sequence[NormalizedIdentifier]) -> CanonicalSubject:
        with self._conn() as conn, conn.cursor() as cur:
            return self._resolve_subject_on_cursor(cur, identifiers)

    @staticmethod
    def _resolve_canonical_on_cursor(
        cur: Any,
        canonical: CanonicalSubject,
    ) -> CanonicalSubject:
        cur.execute(
            "SELECT global_candidate_id, candidate_tenant_id, candidate_id, needs_review "
            "FROM candidate_privacy_resolve_canonical(%s,%s,%s)",
            (
                canonical.global_candidate_id,
                canonical.candidate_tenant_id,
                canonical.candidate_id,
            ),
        )
        row = cur.fetchone()
        if row is None:
            raise CandidatePrivacyUnavailable("candidate privacy canonical resolver failed")
        return CanonicalSubject(
            global_candidate_id=UUID(str(row[0])) if row[0] else None,
            candidate_tenant_id=str(row[1]) if row[1] else None,
            candidate_id=UUID(str(row[2])) if row[2] else None,
            needs_review=bool(row[3]),
        )

    @staticmethod
    def _merge_subjects(a: CanonicalSubject, b: CanonicalSubject) -> CanonicalSubject:
        global_ids = {value for value in (a.global_candidate_id, b.global_candidate_id) if value}
        candidate_pairs = {
            (tenant_id, candidate_id)
            for tenant_id, candidate_id in (
                (a.candidate_tenant_id, a.candidate_id),
                (b.candidate_tenant_id, b.candidate_id),
            )
            if candidate_id is not None
        }
        ambiguous = (
            a.needs_review or b.needs_review or len(global_ids) > 1 or len(candidate_pairs) > 1
        )
        pair = (
            next(iter(candidate_pairs), (None, None)) if len(candidate_pairs) <= 1 else (None, None)
        )
        return CanonicalSubject(
            global_candidate_id=next(iter(global_ids), None) if len(global_ids) <= 1 else None,
            candidate_tenant_id=pair[0],
            candidate_id=pair[1],
            needs_review=ambiguous,
        )

    def _token_payloads(
        self,
        identifiers: Iterable[NormalizedIdentifier],
        *,
        key_version: int | None = None,
    ) -> list[dict[str, Any]]:
        if self.config is None:
            raise CandidatePrivacyUnavailable("candidate privacy HMAC authority is unavailable")
        active, keys = self.config.require_hmac()
        write_version = active if key_version is None else key_version
        key = keys.get(write_version)
        if key is None:
            raise CandidatePrivacyUnavailable("candidate privacy replay key is unavailable")
        return [
            {
                "identifier_type": identifier.identifier_type,
                "key_version": write_version,
                "token": identity_token(key, identifier).hex(),
            }
            for identifier in identifiers
        ]

    def _match_payloads(self, identifiers: Iterable[NormalizedIdentifier]) -> list[dict[str, Any]]:
        if self.config is None:
            raise CandidatePrivacyUnavailable("candidate privacy HMAC authority is unavailable")
        _active, keys = self.config.require_hmac()
        return [
            {
                "identifier_type": identifier.identifier_type,
                "key_version": version,
                "token": identity_token(key, identifier).hex(),
            }
            for identifier in identifiers
            for version, key in sorted(keys.items())
        ]

    def create_directive(
        self,
        *,
        request_id: UUID,
        action: CandidatePrivacyAction,
        authority_type: CandidatePrivacyAuthorityType,
        evidence_ref: UUID,
        reason: CandidatePrivacyReason,
        issuer: str,
        actor_id: str,
        identifiers: Sequence[NormalizedIdentifier],
        canonical: CanonicalSubject,
        effective_at: datetime | None = None,
    ) -> tuple[UUID, DirectiveRecord]:
        if self.config is None:
            raise CandidatePrivacyUnavailable("candidate privacy HMAC authority is unavailable")
        active, _keys = self.config.require_hmac()
        scope = (
            CandidatePrivacyScope.GLOBAL_MATCHING
            if action is CandidatePrivacyAction.WITHDRAW_GLOBAL_MATCHING
            else CandidatePrivacyScope.ACTIVE_PROFILE
        )
        directive_id = uuid4()
        at = effective_at or datetime.now(timezone.utc)
        try:
            with self._conn() as conn, conn.cursor() as cur:
                # Serialize and inspect the request before selecting the write
                # key. An exact replay after a key rotation must be recomputed
                # with its original stored key version; new directives remain
                # single-write on the active version.
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"{issuer}:{request_id}",),
                )
                for identifier in sorted(
                    identifiers, key=lambda item: (item.identifier_type, item.normalized)
                ):
                    for digest in sorted(identifier.lookup_digests):
                        cur.execute(
                            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                            (
                                "candidate-privacy-subject:"
                                f"{identifier.identifier_type}:{digest.hex()}",
                            ),
                        )
                cur.execute(
                    "SELECT key_version FROM candidate_privacy_directive_events "
                    "WHERE issuer = %s AND request_id = %s AND event_type = 'requested'",
                    (issuer, request_id),
                )
                replay = cur.fetchone()
                write_version = int(replay[0]) if replay else active
                tokens = self._token_payloads(identifiers, key_version=write_version)
                resolved = self._resolve_subject_on_cursor(cur, identifiers)
                canonical = self._merge_subjects(
                    resolved,
                    self._resolve_canonical_on_cursor(cur, canonical),
                )
                cur.execute(
                    """
                    SELECT directive_id, action, scope, state, version, effective_at
                    FROM candidate_privacy_create_directive(
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s
                    )
                    """,
                    (
                        directive_id,
                        request_id,
                        action.value,
                        scope.value,
                        authority_type.value,
                        evidence_ref,
                        reason.value,
                        issuer,
                        actor_id,
                        canonical.global_candidate_id,
                        canonical.candidate_tenant_id,
                        canonical.candidate_id,
                        write_version,
                        json.dumps(tokens),
                        canonical.needs_review,
                        at,
                    ),
                )
                row = cur.fetchone()
                if row is None:
                    raise CandidatePrivacyUnavailable(
                        "candidate privacy command returned no result"
                    )
                return request_id, self._record(row)
        except CandidatePrivacyUnavailable as exc:
            message = str(exc.__cause__ or "").lower()
            if "replay conflict" in message or "version conflict" in message:
                raise CandidatePrivacyConflict("candidate privacy request conflicts") from exc
            raise
        except Exception as exc:
            message = str(exc).lower()
            if "replay conflict" in message or "version conflict" in message:
                raise CandidatePrivacyConflict("candidate privacy request conflicts") from exc
            raise CandidatePrivacyUnavailable("candidate privacy command failed") from exc

    def transition_directive(
        self,
        *,
        directive_id: UUID,
        expected_version: int,
        request_id: UUID,
        transition: CandidatePrivacyTransition,
        evidence_ref: UUID,
        reason: CandidatePrivacyReason,
        issuer: str,
        actor_id: str,
        effective_at: datetime | None = None,
    ) -> tuple[UUID, DirectiveRecord]:
        at = effective_at or datetime.now(timezone.utc)
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT directive_id, action, scope, state, version, effective_at
                    FROM candidate_privacy_transition_directive(%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        directive_id,
                        expected_version,
                        request_id,
                        transition.value,
                        evidence_ref,
                        reason.value,
                        issuer,
                        actor_id,
                        at,
                    ),
                )
                row = cur.fetchone()
                if row is None:
                    raise CandidatePrivacyUnavailable(
                        "candidate privacy command returned no result"
                    )
                return request_id, self._record(row)
        except CandidatePrivacyUnavailable as exc:
            message = str(exc.__cause__ or "").lower()
            if "conflict" in message:
                raise CandidatePrivacyConflict("candidate privacy transition conflicts") from exc
            raise
        except Exception as exc:
            message = str(exc).lower()
            if "conflict" in message:
                raise CandidatePrivacyConflict("candidate privacy transition conflicts") from exc
            raise CandidatePrivacyUnavailable("candidate privacy transition failed") from exc

    def _evaluate_on_cursor(
        self,
        cur: Any,
        *,
        identifiers: Sequence[NormalizedIdentifier] = (),
        global_candidate_id: UUID | str | None = None,
        candidate_tenant_id: str | None = None,
        candidate_id: UUID | str | None = None,
    ) -> CandidatePrivacyDecision:
        payloads = self._match_payloads(identifiers) if identifiers else []
        resolved = (
            self._resolve_subject_on_cursor(cur, identifiers) if identifiers else CanonicalSubject()
        )
        supplied = CanonicalSubject(
            global_candidate_id=(
                UUID(str(global_candidate_id)) if global_candidate_id is not None else None
            ),
            # A tenant passed alongside identifier matching is lookup
            # context, not a canonical candidate reference. Preserve it
            # only when the candidate id makes the pair authoritative.
            # A candidate id without its tenant remains fail-closed in
            # the database resolver.
            candidate_tenant_id=(candidate_tenant_id if candidate_id is not None else None),
            candidate_id=UUID(str(candidate_id)) if candidate_id is not None else None,
        )
        canonical = self._merge_subjects(
            resolved,
            self._resolve_canonical_on_cursor(cur, supplied),
        )
        if canonical.needs_review:
            return CandidatePrivacyDecision.REVIEW
        cur.execute(
            """
            SELECT directive_id, action, scope, state, version, effective_at, decision
            FROM candidate_privacy_match(%s::jsonb,%s,%s,%s)
            ORDER BY CASE decision WHEN 'review' THEN 4 WHEN 'block_all' THEN 3
                     WHEN 'block_global' THEN 2 ELSE 1 END DESC,
                     effective_at DESC, version DESC
            LIMIT 1
            """,
            (
                json.dumps(payloads),
                canonical.global_candidate_id,
                canonical.candidate_tenant_id,
                canonical.candidate_id,
            ),
        )
        row = cur.fetchone()
        if row is None:
            return CandidatePrivacyDecision.ALLOW
        record = self._record(row[:6])
        database_decision = CandidatePrivacyDecision(row[6])
        if database_decision is not record.decision:
            raise CandidatePrivacyUnavailable("candidate privacy matcher is inconsistent")
        return database_decision

    def evaluate(
        self,
        *,
        identifiers: Sequence[NormalizedIdentifier] = (),
        global_candidate_id: UUID | str | None = None,
        candidate_tenant_id: str | None = None,
        candidate_id: UUID | str | None = None,
    ) -> CandidatePrivacyDecision:
        with self._conn() as conn, conn.cursor() as cur:
            return self._evaluate_on_cursor(
                cur,
                identifiers=identifiers,
                global_candidate_id=global_candidate_id,
                candidate_tenant_id=candidate_tenant_id,
                candidate_id=candidate_id,
            )

    def evaluate_many(
        self,
        subjects: Sequence[tuple[Sequence[NormalizedIdentifier], CanonicalSubject]],
    ) -> list[CandidatePrivacyDecision]:
        """Evaluate one bounded API batch with three set-based database round trips."""
        if not subjects:
            return []

        identifier_inputs = [
            {
                "subject_index": subject_index,
                "identifier_type": identifier.identifier_type,
                "lookup_digest": digest.hex(),
            }
            for subject_index, (identifiers, _canonical) in enumerate(subjects)
            for identifier in identifiers
            for digest in identifier.lookup_digests
        ]
        supplied = [
            CanonicalSubject(
                global_candidate_id=(
                    UUID(str(canonical.global_candidate_id))
                    if canonical.global_candidate_id is not None
                    else None
                ),
                candidate_tenant_id=(
                    canonical.candidate_tenant_id if canonical.candidate_id is not None else None
                ),
                candidate_id=(
                    UUID(str(canonical.candidate_id))
                    if canonical.candidate_id is not None
                    else None
                ),
            )
            for _identifiers, canonical in subjects
        ]

        with self._conn() as conn, conn.cursor() as cur:
            matches_by_subject: list[set[tuple[UUID | None, str | None, UUID | None]]] = [
                set() for _subject in subjects
            ]
            if identifier_inputs:
                cur.execute(
                    """
                    WITH supplied AS (
                        SELECT
                            (item ->> 'subject_index')::integer AS subject_index,
                            item ->> 'identifier_type' AS identifier_type,
                            decode(item ->> 'lookup_digest', 'hex') AS lookup_digest
                        FROM jsonb_array_elements(%s::jsonb) item
                    )
                    SELECT
                        supplied.subject_index,
                        resolved.global_candidate_id,
                        resolved.candidate_tenant_id,
                        resolved.candidate_id
                    FROM supplied
                    CROSS JOIN LATERAL candidate_privacy_resolve_subject(
                        supplied.identifier_type,
                        supplied.lookup_digest
                    ) resolved
                    ORDER BY supplied.subject_index
                    """,
                    (json.dumps(identifier_inputs),),
                )
                for subject_index, global_id, tenant_id, candidate_id in cur.fetchall():
                    if not 0 <= subject_index < len(subjects):
                        raise CandidatePrivacyUnavailable(
                            "candidate privacy batch resolver is inconsistent"
                        )
                    matches_by_subject[subject_index].add(
                        (
                            UUID(str(global_id)) if global_id else None,
                            str(tenant_id) if tenant_id else None,
                            UUID(str(candidate_id)) if candidate_id else None,
                        )
                    )

            canonical_inputs = [
                {
                    "subject_index": subject_index,
                    "global_candidate_id": (
                        str(canonical.global_candidate_id)
                        if canonical.global_candidate_id is not None
                        else None
                    ),
                    "candidate_tenant_id": canonical.candidate_tenant_id,
                    "candidate_id": (
                        str(canonical.candidate_id) if canonical.candidate_id is not None else None
                    ),
                }
                for subject_index, canonical in enumerate(supplied)
            ]
            cur.execute(
                """
                WITH supplied AS (
                    SELECT
                        (item ->> 'subject_index')::integer AS subject_index,
                        (item ->> 'global_candidate_id')::uuid AS global_candidate_id,
                        item ->> 'candidate_tenant_id' AS candidate_tenant_id,
                        (item ->> 'candidate_id')::uuid AS candidate_id
                    FROM jsonb_array_elements(%s::jsonb) item
                )
                SELECT
                    supplied.subject_index,
                    resolved.global_candidate_id,
                    resolved.candidate_tenant_id,
                    resolved.candidate_id,
                    resolved.needs_review
                FROM supplied
                CROSS JOIN LATERAL candidate_privacy_resolve_canonical(
                    supplied.global_candidate_id,
                    supplied.candidate_tenant_id,
                    supplied.candidate_id
                ) resolved
                ORDER BY supplied.subject_index
                """,
                (json.dumps(canonical_inputs),),
            )
            canonical_rows = cur.fetchall()
            if len(canonical_rows) != len(subjects):
                raise CandidatePrivacyUnavailable(
                    "candidate privacy batch canonical resolver is inconsistent"
                )
            decisions: list[CandidatePrivacyDecision | None] = [None] * len(subjects)
            match_inputs: list[dict[str, Any]] = []
            for expected_index, row in enumerate(canonical_rows):
                subject_index, global_id, tenant_id, candidate_id, needs_review = row
                if subject_index != expected_index:
                    raise CandidatePrivacyUnavailable(
                        "candidate privacy batch canonical resolver is inconsistent"
                    )
                resolved = self._subject_from_matches(matches_by_subject[subject_index])
                canonical = self._merge_subjects(
                    resolved,
                    CanonicalSubject(
                        global_candidate_id=UUID(str(global_id)) if global_id else None,
                        candidate_tenant_id=str(tenant_id) if tenant_id else None,
                        candidate_id=UUID(str(candidate_id)) if candidate_id else None,
                        needs_review=bool(needs_review),
                    ),
                )
                if canonical.needs_review:
                    decisions[subject_index] = CandidatePrivacyDecision.REVIEW
                    continue
                identifiers, _supplied_canonical = subjects[subject_index]
                match_inputs.append(
                    {
                        "subject_index": subject_index,
                        "tokens": self._match_payloads(identifiers) if identifiers else [],
                        "global_candidate_id": (
                            str(canonical.global_candidate_id)
                            if canonical.global_candidate_id is not None
                            else None
                        ),
                        "candidate_tenant_id": canonical.candidate_tenant_id,
                        "candidate_id": (
                            str(canonical.candidate_id)
                            if canonical.candidate_id is not None
                            else None
                        ),
                    }
                )

            if match_inputs:
                cur.execute(
                    """
                    WITH supplied AS (
                        SELECT
                            (item ->> 'subject_index')::integer AS subject_index,
                            COALESCE(item -> 'tokens', '[]'::jsonb) AS tokens,
                            (item ->> 'global_candidate_id')::uuid AS global_candidate_id,
                            item ->> 'candidate_tenant_id' AS candidate_tenant_id,
                            (item ->> 'candidate_id')::uuid AS candidate_id
                        FROM jsonb_array_elements(%s::jsonb) item
                    )
                    SELECT
                        supplied.subject_index,
                        matched.directive_id,
                        matched.action,
                        matched.scope,
                        matched.state,
                        matched.version,
                        matched.effective_at,
                        matched.decision
                    FROM supplied
                    LEFT JOIN LATERAL (
                        SELECT *
                        FROM candidate_privacy_match(
                            supplied.tokens,
                            supplied.global_candidate_id,
                            supplied.candidate_tenant_id,
                            supplied.candidate_id
                        )
                        ORDER BY
                            CASE decision
                                WHEN 'review' THEN 4
                                WHEN 'block_all' THEN 3
                                WHEN 'block_global' THEN 2
                                ELSE 1
                            END DESC,
                            effective_at DESC,
                            version DESC
                        LIMIT 1
                    ) matched ON true
                    ORDER BY supplied.subject_index
                    """,
                    (json.dumps(match_inputs),),
                )
                match_rows = cur.fetchall()
                if len(match_rows) != len(match_inputs):
                    raise CandidatePrivacyUnavailable(
                        "candidate privacy batch matcher is inconsistent"
                    )
                for expected_input, row in zip(match_inputs, match_rows, strict=True):
                    subject_index = row[0]
                    if subject_index != expected_input["subject_index"]:
                        raise CandidatePrivacyUnavailable(
                            "candidate privacy batch matcher is inconsistent"
                        )
                    if row[1] is None:
                        decisions[subject_index] = CandidatePrivacyDecision.ALLOW
                        continue
                    record = self._record(row[1:7])
                    database_decision = CandidatePrivacyDecision(row[7])
                    if database_decision is not record.decision:
                        raise CandidatePrivacyUnavailable(
                            "candidate privacy matcher is inconsistent"
                        )
                    decisions[subject_index] = database_decision

            if any(decision is None for decision in decisions):
                raise CandidatePrivacyUnavailable("candidate privacy batch matcher is incomplete")
            return [decision for decision in decisions if decision is not None]

    def canonical_decision(
        self,
        *,
        global_candidate_id: UUID | str | None = None,
        candidate_tenant_id: str | None = None,
        candidate_id: UUID | str | None = None,
    ) -> CandidatePrivacyDecision:
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT candidate_privacy_decision_for(%s,%s,%s)",
                    (global_candidate_id, candidate_tenant_id, candidate_id),
                )
                row = cur.fetchone()
                if row is None:
                    raise CandidatePrivacyUnavailable("candidate privacy decision is unavailable")
                return CandidatePrivacyDecision(row[0])
        except CandidatePrivacyError:
            raise
        except Exception as exc:
            raise CandidatePrivacyUnavailable("candidate privacy decision is unavailable") from exc

    def node_decision(self, node_id: UUID | str) -> CandidatePrivacyDecision:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT candidate_privacy_node_decision(%s)", (node_id,))
            row = cur.fetchone()
            if row is None:
                raise CandidatePrivacyUnavailable("candidate privacy node decision is unavailable")
            return CandidatePrivacyDecision(row[0])

    def changes(self, *, after_cursor: int, limit: int) -> list[dict[str, Any]]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT cursor, event_id, directive_id, action, scope, resulting_state,
                       directive_version, effective_at
                FROM candidate_privacy_directive_events
                WHERE cursor > %s ORDER BY cursor LIMIT %s
                """,
                (after_cursor, limit),
            )
            return [
                {
                    "cursor": int(row[0]),
                    "event_id": str(row[1]),
                    "directive_id": str(row[2]),
                    "action": row[3],
                    "scope": row[4],
                    "state": row[5],
                    "version": int(row[6]),
                    "effective_at": row[7].isoformat(),
                }
                for row in cur.fetchall()
            ]

    def snapshot(
        self,
        *,
        after_directive_id: UUID | None,
        high_water_cursor: int | None,
        limit: int,
    ) -> tuple[int, list[dict[str, Any]]]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT COALESCE(max(cursor), 0) FROM candidate_privacy_directive_events")
            current_high_water = int(cur.fetchone()[0])
            if high_water_cursor is None:
                high_water_cursor = current_high_water
            elif high_water_cursor > current_high_water:
                raise CandidatePrivacyConflict("candidate privacy snapshot cursor is invalid")
            cur.execute(
                """
                WITH ids AS (
                    SELECT DISTINCT directive_id
                    FROM candidate_privacy_directive_events
                    WHERE cursor <= %s AND (%s::uuid IS NULL OR directive_id > %s::uuid)
                    ORDER BY directive_id LIMIT %s
                )
                SELECT DISTINCT ON (e.directive_id)
                       e.directive_id, e.action, e.scope, e.resulting_state,
                       e.directive_version, e.effective_at
                FROM candidate_privacy_directive_events e
                JOIN ids USING (directive_id)
                WHERE e.cursor <= %s
                ORDER BY e.directive_id, e.directive_version DESC
                """,
                (
                    high_water_cursor,
                    after_directive_id,
                    after_directive_id,
                    limit,
                    high_water_cursor,
                ),
            )
            rows = cur.fetchall()
        items = [
            {
                "directive_id": str(row[0]),
                "action": row[1],
                "scope": row[2],
                "state": row[3],
                "version": int(row[4]),
                "effective_at": row[5].isoformat(),
            }
            for row in rows
        ]
        return high_water_cursor, items

    def referenced_key_versions(self) -> set[int]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT key_version FROM candidate_privacy_token_key_versions()")
            return {int(row[0]) for row in cur.fetchall()}


def require_allowed(decision: CandidatePrivacyDecision, *, global_use: bool) -> None:
    if decision.blocks_all or (global_use and decision.blocks_global):
        raise CandidatePrivacyRestricted("candidate privacy restriction applies")
