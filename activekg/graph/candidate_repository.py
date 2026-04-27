"""Repository for canonical candidate identity.

This sits beside :class:`activekg.graph.repository.GraphRepository` but is
scoped to the candidate identity tables introduced in migration 012:

* ``candidates`` — one row per canonical, source-independent candidate
* ``candidate_identifiers`` — many rows per candidate, normalized merge keys
* ``candidate_source_records`` — many rows per candidate, full provenance

The merge flow is: normalize every identifier an upstream sends, look them up
in ``candidate_identifiers``, and either attach to the existing candidate or
create a new canonical row. Source payloads are recorded verbatim in
``candidate_source_records`` so no upstream data is ever lost.
"""

from __future__ import annotations

import json
from typing import Any

from psycopg_pool import ConnectionPool

from activekg.common.logger import get_enhanced_logger
from activekg.graph.candidate_identifiers import (
    IDENTIFIER_TYPES,
    IdentifierNormalizationError,
    normalize_identifier,
)
from activekg.graph.models import Candidate, CandidateIdentifier, CandidateSourceRecord


class CandidateRepository:
    """Postgres-backed repository for canonical candidate identity."""

    def __init__(self, dsn: str, *, pool: ConnectionPool | None = None):
        self.dsn = dsn
        self.logger = get_enhanced_logger(__name__)
        self._owns_pool = pool is None
        self.pool = pool or ConnectionPool(dsn, min_size=1, max_size=5, timeout=30.0, open=True)

    def close(self) -> None:
        if self._owns_pool:
            self.pool.close()

    # ------------------------------------------------------------------
    # candidates
    # ------------------------------------------------------------------

    def create_candidate(self, candidate: Candidate) -> str:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO candidates (
                        candidate_id, tenant_id, scope, display_name,
                        primary_email, primary_phone, props, metadata, node_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING candidate_id
                    """,
                    (
                        candidate.candidate_id,
                        candidate.tenant_id,
                        candidate.scope,
                        candidate.display_name,
                        candidate.primary_email,
                        candidate.primary_phone,
                        json.dumps(candidate.props),
                        json.dumps(candidate.metadata),
                        candidate.node_id,
                    ),
                )
                return str(cur.fetchone()[0])

    def get_candidate(
        self, candidate_id: str, tenant_id: str | None = None
    ) -> Candidate | None:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                if tenant_id is None:
                    cur.execute(
                        """
                        SELECT candidate_id, tenant_id, scope, display_name, primary_email,
                               primary_phone, props, metadata, node_id, created_at, updated_at
                        FROM candidates WHERE candidate_id = %s
                        """,
                        (candidate_id,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT candidate_id, tenant_id, scope, display_name, primary_email,
                               primary_phone, props, metadata, node_id, created_at, updated_at
                        FROM candidates
                        WHERE candidate_id = %s AND tenant_id IS NOT DISTINCT FROM %s
                        """,
                        (candidate_id, tenant_id),
                    )
                row = cur.fetchone()
                return self._row_to_candidate(row) if row else None

    def update_candidate(
        self,
        candidate_id: str,
        *,
        display_name: str | None = None,
        primary_email: str | None = None,
        primary_phone: str | None = None,
        props: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        node_id: str | None = None,
        tenant_id: str | None = None,
    ) -> bool:
        sets: list[str] = ["updated_at = now()"]
        params: list[Any] = []
        if display_name is not None:
            sets.append("display_name = %s")
            params.append(display_name)
        if primary_email is not None:
            sets.append("primary_email = %s")
            params.append(primary_email)
        if primary_phone is not None:
            sets.append("primary_phone = %s")
            params.append(primary_phone)
        if props is not None:
            sets.append("props = %s")
            params.append(json.dumps(props))
        if metadata is not None:
            sets.append("metadata = %s")
            params.append(json.dumps(metadata))
        if node_id is not None:
            sets.append("node_id = %s")
            params.append(node_id)
        if len(sets) == 1:
            return True

        sql = f"UPDATE candidates SET {', '.join(sets)} WHERE candidate_id = %s"
        params.append(candidate_id)
        if tenant_id is not None:
            sql += " AND tenant_id IS NOT DISTINCT FROM %s"
            params.append(tenant_id)
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.rowcount > 0

    def delete_candidate(self, candidate_id: str, tenant_id: str | None = None) -> bool:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                if tenant_id is None:
                    cur.execute(
                        "DELETE FROM candidates WHERE candidate_id = %s", (candidate_id,)
                    )
                else:
                    cur.execute(
                        "DELETE FROM candidates WHERE candidate_id = %s AND tenant_id IS NOT DISTINCT FROM %s",
                        (candidate_id, tenant_id),
                    )
                return cur.rowcount > 0

    # ------------------------------------------------------------------
    # identifiers
    # ------------------------------------------------------------------

    def add_identifier(
        self,
        candidate_id: str,
        identifier_type: str,
        value: str,
        *,
        tenant_id: str | None = None,
        source: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CandidateIdentifier:
        """Attach a normalized identifier to a canonical candidate.

        Idempotent on ``(tenant_id, identifier_type, value_normalized)``. If the
        identifier already exists on *this* candidate, the row is returned
        unchanged. If it already exists on a *different* candidate the unique
        constraint fires — callers should handle that via
        :meth:`find_candidate_by_identifier` before inserting.
        """
        normalized = normalize_identifier(identifier_type, value)
        ident = CandidateIdentifier(
            candidate_id=candidate_id,
            identifier_type=identifier_type,
            value_normalized=normalized,
            value_raw=value,
            tenant_id=tenant_id,
            source=source,
            confidence=confidence,
            metadata=metadata or {},
        )
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO candidate_identifiers (
                        id, candidate_id, tenant_id, identifier_type,
                        value_normalized, value_raw, source, confidence, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, identifier_type, value_normalized)
                    DO UPDATE SET
                        value_raw = COALESCE(EXCLUDED.value_raw, candidate_identifiers.value_raw),
                        source = COALESCE(EXCLUDED.source, candidate_identifiers.source),
                        confidence = COALESCE(EXCLUDED.confidence, candidate_identifiers.confidence),
                        metadata = candidate_identifiers.metadata || EXCLUDED.metadata,
                        updated_at = now()
                    WHERE candidate_identifiers.candidate_id = EXCLUDED.candidate_id
                    RETURNING id, candidate_id
                    """,
                    (
                        ident.id,
                        ident.candidate_id,
                        ident.tenant_id,
                        ident.identifier_type,
                        ident.value_normalized,
                        ident.value_raw,
                        ident.source,
                        ident.confidence,
                        json.dumps(ident.metadata),
                    ),
                )
                row = cur.fetchone()
                if row is None:
                    # Conflict landed on a different candidate — surface it so
                    # the caller can decide whether to merge candidates.
                    raise IdentifierConflict(
                        identifier_type=identifier_type,
                        value_normalized=normalized,
                        tenant_id=tenant_id,
                    )
                ident.id = str(row[0])
                return ident

    def list_identifiers(
        self, candidate_id: str, tenant_id: str | None = None
    ) -> list[CandidateIdentifier]:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                if tenant_id is None:
                    cur.execute(
                        """
                        SELECT id, candidate_id, tenant_id, identifier_type, value_normalized,
                               value_raw, source, confidence, metadata, created_at, updated_at
                        FROM candidate_identifiers
                        WHERE candidate_id = %s
                        ORDER BY identifier_type, value_normalized
                        """,
                        (candidate_id,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, candidate_id, tenant_id, identifier_type, value_normalized,
                               value_raw, source, confidence, metadata, created_at, updated_at
                        FROM candidate_identifiers
                        WHERE candidate_id = %s AND tenant_id IS NOT DISTINCT FROM %s
                        ORDER BY identifier_type, value_normalized
                        """,
                        (candidate_id, tenant_id),
                    )
                return [self._row_to_identifier(r) for r in cur.fetchall()]

    def find_candidate_by_identifier(
        self,
        identifier_type: str,
        value: str,
        *,
        tenant_id: str | None = None,
    ) -> Candidate | None:
        """Look up a canonical candidate by any of its identifiers."""
        normalized = normalize_identifier(identifier_type, value)
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.candidate_id, c.tenant_id, c.scope, c.display_name, c.primary_email,
                           c.primary_phone, c.props, c.metadata, c.node_id, c.created_at, c.updated_at
                    FROM candidates c
                    JOIN candidate_identifiers i ON i.candidate_id = c.candidate_id
                    WHERE i.identifier_type = %s
                      AND i.value_normalized = %s
                      AND i.tenant_id IS NOT DISTINCT FROM %s
                    LIMIT 1
                    """,
                    (identifier_type, normalized, tenant_id),
                )
                row = cur.fetchone()
                return self._row_to_candidate(row) if row else None

    def remove_identifier(
        self,
        identifier_type: str,
        value: str,
        *,
        tenant_id: str | None = None,
    ) -> bool:
        normalized = normalize_identifier(identifier_type, value)
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM candidate_identifiers
                    WHERE identifier_type = %s
                      AND value_normalized = %s
                      AND tenant_id IS NOT DISTINCT FROM %s
                    """,
                    (identifier_type, normalized, tenant_id),
                )
                return cur.rowcount > 0

    # ------------------------------------------------------------------
    # source records
    # ------------------------------------------------------------------

    def upsert_source_record(self, record: CandidateSourceRecord) -> CandidateSourceRecord:
        """Insert or update a source record. Idempotent on
        ``(tenant_id, source, source_record_type, source_record_id)``."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO candidate_source_records (
                        id, candidate_id, tenant_id, source, source_record_type,
                        source_record_id, source_url, payload, payload_ref, fetched_at,
                        org_id, job_id, effective_recruiter_id, created_by_user_id, resume_source
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, source, source_record_type, source_record_id)
                    DO UPDATE SET
                        candidate_id = EXCLUDED.candidate_id,
                        source_url = COALESCE(EXCLUDED.source_url, candidate_source_records.source_url),
                        payload = EXCLUDED.payload,
                        payload_ref = COALESCE(EXCLUDED.payload_ref, candidate_source_records.payload_ref),
                        fetched_at = COALESCE(EXCLUDED.fetched_at, candidate_source_records.fetched_at),
                        org_id = COALESCE(EXCLUDED.org_id, candidate_source_records.org_id),
                        job_id = COALESCE(EXCLUDED.job_id, candidate_source_records.job_id),
                        effective_recruiter_id = COALESCE(EXCLUDED.effective_recruiter_id, candidate_source_records.effective_recruiter_id),
                        created_by_user_id = COALESCE(EXCLUDED.created_by_user_id, candidate_source_records.created_by_user_id),
                        resume_source = COALESCE(EXCLUDED.resume_source, candidate_source_records.resume_source),
                        updated_at = now()
                    RETURNING id
                    """,
                    (
                        record.id,
                        record.candidate_id,
                        record.tenant_id,
                        record.source,
                        record.source_record_type,
                        record.source_record_id,
                        record.source_url,
                        json.dumps(record.payload),
                        record.payload_ref,
                        record.fetched_at,
                        record.org_id,
                        record.job_id,
                        record.effective_recruiter_id,
                        record.created_by_user_id,
                        record.resume_source,
                    ),
                )
                record.id = str(cur.fetchone()[0])
                return record

    def list_source_records(
        self,
        candidate_id: str,
        *,
        source: str | None = None,
        tenant_id: str | None = None,
    ) -> list[CandidateSourceRecord]:
        sql = [
            """
            SELECT id, candidate_id, tenant_id, source, source_record_type, source_record_id,
                   source_url, payload, payload_ref, fetched_at, created_at, updated_at,
                   org_id, job_id, effective_recruiter_id, created_by_user_id, resume_source
            FROM candidate_source_records
            WHERE candidate_id = %s
            """
        ]
        params: list[Any] = [candidate_id]
        if source is not None:
            sql.append("AND source = %s")
            params.append(source)
        if tenant_id is not None:
            sql.append("AND tenant_id IS NOT DISTINCT FROM %s")
            params.append(tenant_id)
        sql.append("ORDER BY source, source_record_type, source_record_id")
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("\n".join(sql), params)
                return [self._row_to_source_record(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # VantaHire provenance queries
    # ------------------------------------------------------------------

    def find_candidates_by_vantahire_org(
        self, org_id: str, *, tenant_id: str | None = None
    ) -> list[Candidate]:
        """Return canonical candidates that have a VantaHire source record for org_id."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT c.candidate_id, c.tenant_id, c.scope, c.display_name,
                           c.primary_email, c.primary_phone, c.props, c.metadata,
                           c.node_id, c.created_at, c.updated_at
                    FROM candidates c
                    JOIN candidate_source_records sr ON sr.candidate_id = c.candidate_id
                    WHERE sr.source = 'vantahire'
                      AND sr.org_id = %s
                      AND sr.tenant_id IS NOT DISTINCT FROM %s
                    """,
                    (org_id, tenant_id),
                )
                return [self._row_to_candidate(r) for r in cur.fetchall()]

    def find_candidates_by_vantahire_recruiter(
        self, recruiter_id: str, *, tenant_id: str | None = None
    ) -> list[Candidate]:
        """Return canonical candidates with a VantaHire record for effective_recruiter_id."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT c.candidate_id, c.tenant_id, c.scope, c.display_name,
                           c.primary_email, c.primary_phone, c.props, c.metadata,
                           c.node_id, c.created_at, c.updated_at
                    FROM candidates c
                    JOIN candidate_source_records sr ON sr.candidate_id = c.candidate_id
                    WHERE sr.source = 'vantahire'
                      AND sr.effective_recruiter_id = %s
                      AND sr.tenant_id IS NOT DISTINCT FROM %s
                    """,
                    (recruiter_id, tenant_id),
                )
                return [self._row_to_candidate(r) for r in cur.fetchall()]

    def find_candidates_by_vantahire_uploader(
        self, user_id: str, *, tenant_id: str | None = None
    ) -> list[Candidate]:
        """Return canonical candidates with a VantaHire record created by user_id."""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT c.candidate_id, c.tenant_id, c.scope, c.display_name,
                           c.primary_email, c.primary_phone, c.props, c.metadata,
                           c.node_id, c.created_at, c.updated_at
                    FROM candidates c
                    JOIN candidate_source_records sr ON sr.candidate_id = c.candidate_id
                    WHERE sr.source = 'vantahire'
                      AND sr.created_by_user_id = %s
                      AND sr.tenant_id IS NOT DISTINCT FROM %s
                    """,
                    (user_id, tenant_id),
                )
                return [self._row_to_candidate(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # high-level merge helper
    # ------------------------------------------------------------------

    def upsert_candidate_from_source(
        self,
        *,
        source: str,
        source_record_type: str,
        source_record_id: str,
        identifiers: list[tuple[str, str]],
        payload: dict[str, Any] | None = None,
        payload_ref: str | None = None,
        source_url: str | None = None,
        tenant_id: str | None = None,
        display_name: str | None = None,
        org_id: str | None = None,
        job_id: str | None = None,
        effective_recruiter_id: str | None = None,
        created_by_user_id: str | None = None,
        resume_source: str | None = None,
    ) -> Candidate:
        """Resolve or create a canonical candidate for an incoming source record.

        Normalizes every identifier, looks up the first match, and attaches the
        remaining identifiers plus a source record to that candidate. If no
        identifier matches an existing candidate, a new canonical candidate is
        created.
        """
        if not identifiers:
            raise ValueError("at least one identifier is required to merge a candidate")

        normalized_ids: list[tuple[str, str, str]] = []
        for itype, value in identifiers:
            if itype not in IDENTIFIER_TYPES:
                raise IdentifierNormalizationError(f"unknown identifier_type: {itype!r}")
            normalized_ids.append((itype, value, normalize_identifier(itype, value)))

        existing: Candidate | None = None
        for itype, _raw, norm in normalized_ids:
            existing = self._find_candidate_by_normalized(itype, norm, tenant_id=tenant_id)
            if existing is not None:
                break

        candidate = existing or Candidate(
            tenant_id=tenant_id,
            display_name=display_name,
        )
        if existing is None:
            self.create_candidate(candidate)
        elif display_name and not existing.display_name:
            self.update_candidate(
                candidate.candidate_id,
                display_name=display_name,
                tenant_id=tenant_id,
            )
            candidate.display_name = display_name

        for itype, raw, _norm in normalized_ids:
            try:
                self.add_identifier(
                    candidate.candidate_id,
                    itype,
                    raw,
                    tenant_id=tenant_id,
                    source=source,
                )
            except IdentifierConflict:
                # Another candidate already owns this identifier. Skip rather
                # than silently re-pointing it — merging candidates is an
                # explicit operation the caller should drive.
                self.logger.warning(
                    "identifier conflict during merge",
                    extra_fields={
                        "identifier_type": itype,
                        "source": source,
                        "tenant_id": tenant_id,
                    },
                )

        self.upsert_source_record(
            CandidateSourceRecord(
                candidate_id=candidate.candidate_id,
                tenant_id=tenant_id,
                source=source,
                source_record_type=source_record_type,
                source_record_id=source_record_id,
                source_url=source_url,
                payload=payload or {},
                payload_ref=payload_ref,
                org_id=org_id,
                job_id=job_id,
                effective_recruiter_id=effective_recruiter_id,
                created_by_user_id=created_by_user_id,
                resume_source=resume_source,
            )
        )
        return candidate

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _find_candidate_by_normalized(
        self, identifier_type: str, value_normalized: str, *, tenant_id: str | None
    ) -> Candidate | None:
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.candidate_id, c.tenant_id, c.scope, c.display_name, c.primary_email,
                           c.primary_phone, c.props, c.metadata, c.node_id, c.created_at, c.updated_at
                    FROM candidates c
                    JOIN candidate_identifiers i ON i.candidate_id = c.candidate_id
                    WHERE i.identifier_type = %s
                      AND i.value_normalized = %s
                      AND i.tenant_id IS NOT DISTINCT FROM %s
                    LIMIT 1
                    """,
                    (identifier_type, value_normalized, tenant_id),
                )
                row = cur.fetchone()
                return self._row_to_candidate(row) if row else None

    @staticmethod
    def _row_to_candidate(row: tuple[Any, ...]) -> Candidate:
        return Candidate(
            candidate_id=str(row[0]),
            tenant_id=row[1],
            scope=row[2],
            display_name=row[3],
            primary_email=row[4],
            primary_phone=row[5],
            props=row[6] or {},
            metadata=row[7] or {},
            node_id=str(row[8]) if row[8] else None,
            created_at=row[9],
            updated_at=row[10],
        )

    @staticmethod
    def _row_to_identifier(row: tuple[Any, ...]) -> CandidateIdentifier:
        return CandidateIdentifier(
            id=str(row[0]),
            candidate_id=str(row[1]),
            tenant_id=row[2],
            identifier_type=row[3],
            value_normalized=row[4],
            value_raw=row[5],
            source=row[6],
            confidence=row[7],
            metadata=row[8] or {},
            created_at=row[9],
            updated_at=row[10],
        )

    @staticmethod
    def _row_to_source_record(row: tuple[Any, ...]) -> CandidateSourceRecord:
        return CandidateSourceRecord(
            id=str(row[0]),
            candidate_id=str(row[1]),
            tenant_id=row[2],
            source=row[3],
            source_record_type=row[4],
            source_record_id=row[5],
            source_url=row[6],
            payload=row[7] or {},
            payload_ref=row[8],
            fetched_at=row[9],
            created_at=row[10],
            updated_at=row[11],
            org_id=row[12],
            job_id=row[13],
            effective_recruiter_id=row[14],
            created_by_user_id=row[15],
            resume_source=row[16],
        )


class IdentifierConflict(Exception):
    """Raised when an identifier is already attached to a different candidate."""

    def __init__(self, *, identifier_type: str, value_normalized: str, tenant_id: str | None):
        self.identifier_type = identifier_type
        self.value_normalized = value_normalized
        self.tenant_id = tenant_id
        super().__init__(
            f"identifier {identifier_type}={value_normalized!r} already attached to a different candidate"
        )


__all__ = [
    "CandidateRepository",
    "IdentifierConflict",
    "IdentifierNormalizationError",
]
