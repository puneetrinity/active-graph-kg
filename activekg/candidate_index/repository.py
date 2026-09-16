"""Explicit transactions around the eleven generation routines; no import-time IO.

The API supplies a verified scope for source/read operations. Coordination calls
do not accept arbitrary scopes: SQL selects one job and returns only its lease.
Model work must happen after these methods have committed and closed the cursor.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from activekg.candidate_index.contracts import (
    IMMUTABLE_TABLES,
    INDEX_TABLES,
    OWNER_FUNCTIONS,
    RUNTIME_FUNCTIONS,
    IndexPolicy,
    IndexTunables,
    canonical_json,
    sha256,
)

# Derived from PostgreSQL 16, independent of installation role/OIDs. Exact
# structure and effective ACLs are separate gates; source hashing is not readiness.
INDEX_CATALOG_SHA256 = "e8bc38e1da3ae7cdbc11d3cd083083c1761150812daab7a233f96fd158e89cf8"

INDEX_CATALOG_SQL = """
WITH chosen AS (SELECT coalesce(%s,current_user) AS runtime_role),
  relations AS (SELECT c.* FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
    WHERE n.nspname='public' AND (c.relname=ANY(%s) OR
      (left(c.relname,16)='candidate_index_' AND c.relkind IN ('r','p','v','m','f')))),
  routines AS (SELECT p.* FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
    WHERE n.nspname='public' AND left(p.proname,16)='candidate_index_'),
  expected_owner AS (SELECT relowner FROM relations WHERE relname='candidate_index_sources')
SELECT jsonb_build_object(
 'tables', (SELECT jsonb_agg(jsonb_build_object('name',c.relname,'kind',c.relkind,
   'rls',c.relrowsecurity,'force_rls',c.relforcerowsecurity,'owner_matches',c.relowner=o.relowner,
   'columns',(SELECT jsonb_agg(jsonb_build_array(a.attname,format_type(a.atttypid,a.atttypmod),
     a.attnotnull,a.attidentity,a.attgenerated,pg_get_expr(d.adbin,d.adrelid)) ORDER BY a.attnum)
     FROM pg_attribute a LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum
     WHERE a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped),
   'constraints',(SELECT jsonb_agg(jsonb_build_array(x.conname,x.contype,x.convalidated,x.condeferrable,
     x.condeferred,pg_get_constraintdef(x.oid)) ORDER BY x.conname) FROM pg_constraint x WHERE x.conrelid=c.oid),
   'indexes',(SELECT jsonb_agg(jsonb_build_array(k.relname,ix.indisvalid,ix.indisready,pg_get_indexdef(ix.indexrelid)) ORDER BY k.relname)
     FROM pg_index ix JOIN pg_class k ON k.oid=ix.indexrelid WHERE ix.indrelid=c.oid),
   'policies',(SELECT jsonb_agg(jsonb_build_array(p.polname,p.polcmd,p.polpermissive,
     (SELECT jsonb_agg(CASE WHEN r=c.relowner THEN 'index_owner' ELSE r::text END ORDER BY r=c.relowner,r)
       FROM unnest(p.polroles) r),pg_get_expr(p.polqual,p.polrelid),pg_get_expr(p.polwithcheck,p.polrelid)) ORDER BY p.polname)
     FROM pg_policy p WHERE p.polrelid=c.oid),
   'triggers',(SELECT jsonb_agg(jsonb_build_array(t.tgname,t.tgenabled,pg_get_triggerdef(t.oid)) ORDER BY t.tgname)
     FROM pg_trigger t WHERE t.tgrelid=c.oid AND NOT t.tgisinternal)
 ) ORDER BY c.relname) FROM relations c CROSS JOIN expected_owner o),
 'routines',(SELECT jsonb_agg(jsonb_build_array(p.oid::regprocedure::text,p.proowner=o.relowner,
   pg_get_functiondef(p.oid)) ORDER BY p.oid::regprocedure::text) FROM routines p CROSS JOIN expected_owner o)
), jsonb_build_object(
 'tables',(SELECT jsonb_object_agg(c.relname,jsonb_build_object(
   'table',(SELECT jsonb_agg(has_table_privilege(ch.runtime_role,c.oid,privilege) ORDER BY ordinal)
     FROM unnest(ARRAY['SELECT','INSERT','UPDATE','DELETE','TRUNCATE','REFERENCES','TRIGGER']) WITH ORDINALITY p(privilege,ordinal)),
   'columns',(SELECT jsonb_agg(has_any_column_privilege(ch.runtime_role,c.oid,privilege) ORDER BY ordinal)
     FROM unnest(ARRAY['SELECT','INSERT','UPDATE','REFERENCES']) WITH ORDINALITY p(privilege,ordinal)),
   'public',EXISTS(SELECT 1 FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner))) acl WHERE acl.grantee=0),
   'public_columns',EXISTS(SELECT 1 FROM pg_attribute a CROSS JOIN LATERAL aclexplode(a.attacl) acl
     WHERE a.attrelid=c.oid AND acl.grantee=0),
   'grant_option',EXISTS(SELECT 1 FROM aclexplode(c.relacl) acl WHERE acl.grantee<>c.relowner AND acl.is_grantable),
   'column_grant_option',EXISTS(SELECT 1 FROM pg_attribute a CROSS JOIN LATERAL aclexplode(a.attacl) acl
     WHERE a.attrelid=c.oid AND acl.grantee<>c.relowner AND acl.is_grantable)
 )) FROM relations c CROSS JOIN chosen ch),
 'routines',(SELECT jsonb_object_agg(p.oid::regprocedure::text,jsonb_build_array(
   has_function_privilege(ch.runtime_role,p.oid,'EXECUTE'),
   EXISTS(SELECT 1 FROM aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) acl WHERE acl.grantee=0),
   EXISTS(SELECT 1 FROM aclexplode(p.proacl) acl WHERE acl.grantee<>p.proowner AND acl.is_grantable)))
   FROM routines p CROSS JOIN chosen ch)
)
"""


def legacy_applicant_is_managed(cur: psycopg.Cursor, node_id: str, tenant_id: str | None) -> bool:
    """Bound legacy node -> exact 4B evidence -> adopted source, within tenant RLS.

    Never trust the extraction task's mutable metadata/props as an authority flag.
    A managed source stays excluded even if its generation is pending/restricted.
    No source content, identities, model calls, or writes are involved here.
    """
    cur.execute(
        """
        SELECT EXISTS (
          SELECT 1 FROM public.nodes n
          JOIN public.organization_candidate_references r
            ON r.tenant_id=n.tenant_id AND r.application_id::text=n.metadata->>'application_id'
            AND r.job_id::text=n.metadata->>'job_id'
          JOIN public.organization_candidate_resume_evidence e
            ON e.tenant_id=r.tenant_id AND e.reference_id=r.reference_id AND e.candidate_id=r.candidate_id
          JOIN public.candidate_index_sources s
            ON s.scope_key=r.tenant_id AND s.tenant_id=r.tenant_id AND s.reference_id=r.reference_id
            AND s.candidate_id=r.candidate_id AND s.resume_version_id=e.resume_version_id
            AND s.source_kind='organization_application' AND s.source_version=e.version
          WHERE n.id=%s::uuid AND n.tenant_id=%s AND n.metadata->>'source'='vantahire'
            AND r.tenant_id='org_'||(n.metadata->>'org_id')
        ) AS managed
        """,
        (node_id, tenant_id),
    )
    row = cur.fetchone()
    value = row.get("managed") if isinstance(row, dict) else row[0] if row else None
    if type(value) is not bool:
        raise ValueError("candidate_index_legacy_authority_unavailable")
    return value


def catalog_evidence(cur: psycopg.Cursor, role: str | None = None) -> tuple[str, bool]:
    """One catalog query: exact structure plus effective PUBLIC/table/column ACLs."""
    cur.execute(INDEX_CATALOG_SQL, (role, list(INDEX_TABLES)))
    row = cur.fetchone()
    if row is None:
        return "missing", False
    structure, privileges = row
    tables = privileges.get("tables") or {}
    routines = privileges.get("routines") or {}
    # PostgreSQL's regprocedure printer uses spaces only inside type names.
    expected_routines = set(RUNTIME_FUNCTIONS + OWNER_FUNCTIONS)
    valid = set(tables) == set(INDEX_TABLES) and set(routines) == expected_routines
    for name, value in tables.items():
        read = name in IMMUTABLE_TABLES
        valid = valid and value == {
            "table": [read, False, False, False, False, False, False],
            "columns": [read, False, False, False],
            "public": False,
            "public_columns": False,
            "grant_option": False,
            "column_grant_option": False,
        }
    valid = valid and all(
        value == [name in RUNTIME_FUNCTIONS, False, False] for name, value in routines.items()
    )
    return sha256(canonical_json(structure)), bool(valid)


def catalog_ready(cur: psycopg.Cursor, role: str | None = None) -> bool:
    digest, privileges = catalog_evidence(cur, role)
    return privileges and digest == INDEX_CATALOG_SHA256


class IndexRepository:
    def __init__(self, dsn: str, *, tunables: IndexTunables | None = None) -> None:
        self._dsn = dsn
        self.tunables = tunables or IndexTunables()

    @contextmanager
    def transaction(self, scope: str | None = None) -> Iterator[psycopg.Cursor]:
        with psycopg.connect(self._dsn, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='1500ms'")
            cur.execute("SET LOCAL statement_timeout='3s'")
            cur.execute("SET LOCAL idle_in_transaction_session_timeout='4000ms'")
            if scope is not None:
                cur.execute("SELECT set_config('app.current_tenant_id',%s,true)", (scope,))
            yield cur

    @staticmethod
    def capture_on_cursor(
        cur: psycopg.Cursor,
        *,
        kind: str,
        upstream_id: str,
        command_digest: str,
        content_kind: str,
        content: str | None,
        tokens: list[dict[str, Any]],
        key_versions: list[int],
        policy: IndexPolicy,
        priority: str = "interactive",
        tunables: IndexTunables | None = None,
    ) -> dict[str, Any]:
        # Also used inside the existing consent transaction: no nested commit.
        bounds = tunables or IndexTunables()
        cur.execute(
            "SELECT public.candidate_index_capture_source(%s,%s::uuid,%s,%s,%s,%s::jsonb,%s::integer[],%s,%s,%s,%s)",
            (
                kind,
                upstream_id,
                command_digest,
                content_kind,
                content,
                Jsonb(tokens),
                key_versions,
                canonical_json(asdict(policy)),
                priority,
                bounds.pending_per_scope,
                bounds.pending_total,
            ),
        )
        row = cur.fetchone()
        # The existing consent transaction uses dict_row; private intake and
        # worker transactions use tuple_row. Neither gets a nested transaction.
        return next(iter(row.values())) if isinstance(row, dict) else row[0]

    def _options(self, stage: str) -> dict[str, int]:
        if stage not in {"extract", "embed"}:
            raise ValueError("candidate_index_stage_invalid")
        t = self.tunables
        return {
            "lease_ms": t.extraction_lease_ms if stage == "extract" else t.embedding_lease_ms,
            "deadline_ms": t.extraction_dispatch_ms
            if stage == "extract"
            else t.embedding_dispatch_ms,
            "concurrency": t.concurrent_per_stage,
            "scope_concurrency": t.concurrent_per_scope_stage,
            "interactive_weight": t.interactive_weight,
            "maintenance_weight": t.maintenance_weight,
        }

    def claim(self, stage: str) -> list[dict[str, Any]]:
        with self.transaction() as cur:
            cur.execute(
                "SELECT public.candidate_index_claim(%s,%s,%s::jsonb)",
                (stage, self.tunables.claim_limit, Jsonb(self._options(stage))),
            )
            return [row[0] for row in cur.fetchall()]

    def reserve(self, lease: dict[str, Any], policy: IndexPolicy) -> dict[str, Any] | None:
        with self.transaction() as cur:
            cur.execute(
                "SELECT public.candidate_index_claim(%s,1,%s::jsonb,%s::uuid,%s::uuid,%s,%s)",
                (
                    lease["stage"],
                    Jsonb(self._options(lease["stage"])),
                    lease["job_id"],
                    lease["lease_token"],
                    lease["lease_generation"],
                    policy.digest,
                ),
            )
            row = cur.fetchone()
            if row is None:
                return None
            dispatch = row[0]
            # Translate the database's fixed deadline into a conservative local
            # duration. No machine-wall-clock comparison and no renewal on retry.
            cur.execute("SELECT clock_timestamp()")
            database_now = cur.fetchone()[0]
            deadline = datetime.fromisoformat(dispatch["dispatch_deadline"])
            dispatch["remaining_ms"] = max(0, int((deadline - database_now).total_seconds() * 1000))
            return dispatch

    def complete_extract(
        self,
        dispatch: dict[str, Any],
        *,
        namespace: dict[str, Any],
        evidence: list[dict[str, str]],
        confidence: float,
        model_id: str,
        chunks: list[dict[str, Any]],
        professional_text: str,
        parsed_text: str | None = None,
    ) -> bool:
        with self.transaction() as cur:
            cur.execute(
                "SELECT public.candidate_index_complete_extract(%s::uuid,%s::uuid,%s,%s,%s,%s::jsonb,"
                "%s::jsonb,%s,%s,%s::jsonb,%s,%s)",
                (
                    dispatch["job_id"],
                    dispatch["lease_token"],
                    dispatch["lease_generation"],
                    dispatch["input_sha256"],
                    dispatch["policy_sha256"],
                    Jsonb(namespace),
                    Jsonb(evidence),
                    confidence,
                    model_id,
                    Jsonb(chunks),
                    professional_text,
                    parsed_text,
                ),
            )
            return cur.fetchone()[0]

    def complete_embed(self, dispatch: dict[str, Any], vectors: list[list[float]]) -> bool:
        with self.transaction() as cur:
            cur.execute(
                "SELECT public.candidate_index_complete_embed(%s::uuid,%s::uuid,%s,%s::uuid,%s,%s,%s,%s::jsonb)",
                (
                    dispatch["job_id"],
                    dispatch["lease_token"],
                    dispatch["lease_generation"],
                    dispatch["extraction"]["extraction_id"],
                    dispatch["policy_sha256"],
                    dispatch["policy"]["embedding_model_id"],
                    dispatch["policy"]["embedding_artifact_revision"],
                    Jsonb(vectors),
                ),
            )
            return cur.fetchone()[0]

    def fail(self, lease: dict[str, Any], reason: str, retry_ms: int = 0) -> bool:
        with self.transaction() as cur:
            cur.execute(
                "SELECT public.candidate_index_fail(%s::uuid,%s::uuid,%s,%s,%s)",
                (
                    lease["job_id"],
                    lease["lease_token"],
                    lease["lease_generation"],
                    reason,
                    retry_ms,
                ),
            )
            return cur.fetchone()[0]

    def status(self, scope: str, source_id: str) -> dict[str, Any] | None:
        with self.transaction(scope) as cur:
            cur.execute("SELECT public.candidate_index_status(%s::uuid)", (source_id,))
            return cur.fetchone()[0]

    def invalidate(self, scope: str, source_id: str) -> bool:
        with self.transaction(scope) as cur:
            cur.execute("SELECT public.candidate_index_invalidate(%s::uuid)", (source_id,))
            return cur.fetchone()[0]

    def catchup(self, *, maintenance: bool = False) -> int:
        with self.transaction() as cur:
            cur.execute(
                "SELECT public.candidate_index_catchup(%s,%s,%s,%s)",
                (
                    self.tunables.maintenance_batch if maintenance else self.tunables.catchup_batch,
                    maintenance,
                    self.tunables.pending_per_scope,
                    self.tunables.pending_total,
                ),
            )
            return cur.fetchone()[0]
