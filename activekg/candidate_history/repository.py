"""One connection/transaction per bounded history call; never shared with HTTP."""

from __future__ import annotations

import json
from typing import Any

import psycopg

from activekg.candidate_history.contracts import HistoryReadRequest, HistoryResponse


class HistoryUnavailable(RuntimeError):
    def __init__(self, code: str = "temporarily_unavailable"):
        if code not in {
            "temporarily_unavailable",
            "not_found",
            "binding_conflict",
            "privacy_restricted",
        }:
            code = "temporarily_unavailable"
        self.code = code
        super().__init__(code)


class HistoryRepository:
    def __init__(self, dsn: str):
        if not dsn:
            raise HistoryUnavailable()
        self._dsn = dsn

    def _call(self, query: str, args: tuple, *, readonly: bool) -> Any:
        try:
            with psycopg.connect(self._dsn, connect_timeout=3, autocommit=False) as conn:
                # Must be the first statement; SQL functions cannot promote it.
                if readonly:
                    conn.execute("SET TRANSACTION READ ONLY")
                conn.execute("SET LOCAL statement_timeout='3s'")
                conn.execute("SET LOCAL lock_timeout='500ms'")
                conn.execute("SET LOCAL idle_in_transaction_session_timeout='5s'")
                row = conn.execute(query, args).fetchone()
                if row is None or not isinstance(row[0], dict):
                    raise HistoryUnavailable()
                result = row[0]
            return result
        except HistoryUnavailable:
            raise
        except Exception:
            # No DB message, SQL parameters, or DSN may escape this boundary.
            raise HistoryUnavailable() from None

    def step(self, new_limit: int, retry_limit: int) -> dict:
        result = self._call(
            "SELECT public.organization_candidate_history_step(%s,%s)",
            (new_limit, retry_limit),
            readonly=False,
        )
        if result.get("outcome") not in {
            "idle",
            "contended",
            "waiting_reference",
            "waiting_source",
            "privacy_wait",
            "privacy_restricted",
            "binding_conflict",
            "applied",
        }:
            raise HistoryUnavailable()
        return result

    def read(self, command: HistoryReadRequest) -> HistoryResponse:
        e = command.expected
        result = self._call(
            "SELECT public.organization_candidate_history_read(%s,%s,%s,%s,%s,%s,%s)",
            (
                f"org_{command.organization_id}",
                command.application_id,
                command.job_id,
                command.reference_id,
                int(e.sequence) if e.sequence is not None else None,
                e.event_id,
                int(e.count),
            ),
            readonly=True,
        )
        if "error" in result:
            raise HistoryUnavailable(
                result["error"] if isinstance(result["error"], str) else "temporarily_unavailable"
            )
        try:
            response = HistoryResponse.model_validate_json(json.dumps(result))
            b = response.binding
            if (
                b.organization_id,
                b.application_id,
                b.job_id,
                b.reference_id,
                response.freshness.expected,
            ) != (
                command.organization_id,
                command.application_id,
                command.job_id,
                command.reference_id,
                e,
            ):
                raise ValueError()
            return response
        except Exception:
            raise HistoryUnavailable() from None


HISTORY_CATALOG_SHA256 = "5c24c55357fe6e2f6187d142ca23cb7a4b1fdd3add9066ee4d3570bbd41728ce"
HISTORY_CATALOG_SQL = """
WITH chosen AS (SELECT coalesce(%s,current_user) AS runtime_role),
  relations AS (SELECT c.* FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
    WHERE n.nspname='public' AND (c.relname=ANY(%s) OR
      (left(c.relname,31)='organization_candidate_history_' AND c.relkind IN ('r','p','v','m','f')))),
  routines AS (SELECT p.* FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
    WHERE n.nspname='public' AND left(p.proname,31)='organization_candidate_history_'),
  expected_owner AS (SELECT relowner FROM relations WHERE relname='organization_candidate_history_bindings')
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
     (SELECT jsonb_agg(CASE WHEN r=c.relowner THEN 'history_owner' ELSE r::text END ORDER BY r=c.relowner,r)
       FROM unnest(p.polroles) r),pg_get_expr(p.polqual,p.polrelid),pg_get_expr(p.polwithcheck,p.polrelid)) ORDER BY p.polname)
     FROM pg_policy p WHERE p.polrelid=c.oid),
   'triggers',(SELECT jsonb_agg(jsonb_build_array(t.tgname,t.tgenabled,pg_get_triggerdef(t.oid)) ORDER BY t.tgname)
     FROM pg_trigger t WHERE t.tgrelid=c.oid AND NOT t.tgisinternal)
 ) ORDER BY c.relname) FROM relations c CROSS JOIN expected_owner o),
 'input_index',(SELECT pg_get_indexdef(to_regclass('public.och_inbox_app_seq_idx'))),
 'input_policies',(SELECT jsonb_agg(jsonb_build_array(c.relname,p.polname,p.polcmd,p.polpermissive,
   p.polroles=ARRAY[c.relowner],pg_get_expr(p.polqual,p.polrelid),pg_get_expr(p.polwithcheck,p.polrelid)) ORDER BY c.relname,p.polname)
   FROM pg_policy p JOIN pg_class c ON c.oid=p.polrelid
   WHERE c.oid IN ('public.organization_decision_event_inbox'::regclass,'public.organization_decision_stream_state'::regclass)
   ),
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


def catalog_evidence(cur, role=None):
    import hashlib

    from activekg.candidate_history.contracts import OWNER_FUNCTIONS, RUNTIME_FUNCTIONS, TABLES

    cur.execute(HISTORY_CATALOG_SQL, (role, list(TABLES)))
    structure, grants = cur.fetchone()
    digest = hashlib.sha256(
        json.dumps(structure, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    table_acl = grants.get("tables") or {}
    routine_acl = grants.get("routines") or {}
    ok = set(table_acl) == set(TABLES) and set(routine_acl) == set(
        RUNTIME_FUNCTIONS + OWNER_FUNCTIONS
    )
    for value in table_acl.values():
        ok = ok and not any(value.get("table") or []) and not any(value.get("columns") or [])
        ok = ok and not any(
            value.get(k)
            for k in ("public", "public_columns", "grant_option", "column_grant_option")
        )
    for name, value in routine_acl.items():
        ok = ok and value == [name in RUNTIME_FUNCTIONS, False, False]
    return digest, bool(ok)


def catalog_ready(cur, role=None):
    digest, privileges = catalog_evidence(cur, role)
    return privileges and digest == HISTORY_CATALOG_SHA256
