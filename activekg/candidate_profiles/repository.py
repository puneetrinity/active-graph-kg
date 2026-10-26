import os
from typing import Any

import psycopg
from fastapi import HTTPException
from psycopg.rows import dict_row

from activekg.candidate_profiles.contracts import receipt

CATALOG_MD5 = "d3a68eb2ccf60fa8e0c3da68d1996d39"
CATALOG_SQL = """
WITH chosen AS (SELECT coalesce(%s,current_user) AS role),
relations AS (SELECT c.* FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
 WHERE n.nspname='public' AND starts_with(c.relname,'candidate_owned_') AND c.relkind IN ('r','p','v','m','f')),
routines AS (SELECT p.* FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
 WHERE n.nspname='public' AND starts_with(p.proname,'candidate_owned_'))
SELECT md5(jsonb_build_object(
 'tables',(SELECT jsonb_agg(jsonb_build_array(c.relname,c.relkind,c.relrowsecurity,c.relforcerowsecurity,
  (SELECT jsonb_agg(jsonb_build_array(a.attname,format_type(a.atttypid,a.atttypmod),a.attnotnull,a.attidentity,a.attgenerated,
    pg_get_expr(d.adbin,d.adrelid)) ORDER BY a.attnum) FROM pg_attribute a LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum
    WHERE a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped),
  (SELECT jsonb_agg(jsonb_build_array(x.conname,x.convalidated,x.condeferrable,x.condeferred,pg_get_constraintdef(x.oid)) ORDER BY x.conname COLLATE "C") FROM pg_constraint x WHERE x.conrelid=c.oid),
  (SELECT jsonb_agg(jsonb_build_array(i.indisvalid,i.indisready,pg_get_indexdef(i.indexrelid)) ORDER BY pg_get_indexdef(i.indexrelid) COLLATE "C") FROM pg_index i WHERE i.indrelid=c.oid),
  (SELECT jsonb_agg(jsonb_build_array(p.polname,p.polcmd,p.polpermissive,p.polroles=ARRAY[c.relowner],pg_get_expr(p.polqual,p.polrelid),pg_get_expr(p.polwithcheck,p.polrelid)) ORDER BY p.polname COLLATE "C") FROM pg_policy p WHERE p.polrelid=c.oid)
 ) ORDER BY c.relname COLLATE "C") FROM relations c),
 'routines',(SELECT jsonb_agg(jsonb_build_array(p.oid::regprocedure::text,p.proowner=(SELECT relowner FROM relations WHERE relname='candidate_owned_subjects'),pg_get_functiondef(p.oid)) ORDER BY p.oid::regprocedure::text COLLATE "C") FROM routines p),
 'triggers',(SELECT jsonb_agg(jsonb_build_array(t.tgenabled,pg_get_triggerdef(t.oid)) ORDER BY t.tgname COLLATE "C") FROM pg_trigger t WHERE NOT t.tgisinternal AND t.tgrelid IN(SELECT oid FROM relations))
)::text),
 NOT EXISTS(SELECT 1 FROM relations c,chosen ch,unnest(ARRAY['SELECT','INSERT','UPDATE','DELETE','TRUNCATE','REFERENCES','TRIGGER']) v(privilege)
  WHERE has_table_privilege(ch.role,c.oid,v.privilege))
 AND NOT EXISTS(SELECT 1 FROM relations c,chosen ch,unnest(ARRAY['SELECT','INSERT','UPDATE','REFERENCES']) v(privilege)
  WHERE has_any_column_privilege(ch.role,c.oid,v.privilege))
 AND NOT EXISTS(SELECT 1 FROM relations c CROSS JOIN LATERAL aclexplode(coalesce(c.relacl,acldefault('r',c.relowner))) a WHERE a.grantee<>c.relowner)
 AND NOT EXISTS(SELECT 1 FROM relations c JOIN pg_attribute x ON x.attrelid=c.oid CROSS JOIN LATERAL aclexplode(x.attacl) a WHERE a.grantee<>c.relowner)
 AND NOT EXISTS(SELECT 1 FROM routines p,chosen ch WHERE NOT has_function_privilege(ch.role,p.oid,'EXECUTE')
  OR EXISTS(SELECT 1 FROM aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a WHERE a.grantee<>p.proowner AND
    (a.grantee IS DISTINCT FROM (SELECT oid FROM pg_roles WHERE rolname=ch.role) OR a.is_grantable)))
"""


def catalog_evidence(cur, role=None):
    cur.execute(CATALOG_SQL, (role,))
    return tuple(cur.fetchone())


def catalog_ready(cur, role=None):
    digest, permissions = catalog_evidence(cur, role)
    return digest == CATALOG_MD5 and permissions is True


def _connect() -> psycopg.Connection:
    dsn = os.getenv("ACTIVEKG_DSN") or os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("authority_unavailable")
    return psycopg.connect(dsn, row_factory=dict_row, connect_timeout=3)


def _privacy(cur: psycopg.Cursor, subject: str) -> None:
    # This is an empty setup record, not permission. Respect any existing consent withdrawal
    # and privacy restriction; Flow separately checks its current candidate identity.
    cur.execute("SELECT set_config('app.current_tenant_id',%s,true)", (f"candidate_{subject}",))
    cur.execute(
        "SELECT last_action,global_candidate_id FROM candidate_consent_state WHERE subject_id=%s::uuid",
        (subject,),
    )
    state = cur.fetchone()
    if state and state["last_action"] == "withdraw":
        raise HTTPException(451, "privacy_restricted")
    if state and state["global_candidate_id"]:
        cur.execute(
            "SELECT candidate_privacy_global_decision(%s::uuid) AS decision",
            (state["global_candidate_id"],),
        )
        decision = cur.fetchone()
        if not decision or decision["decision"] == "review":
            raise HTTPException(503, "authority_unavailable")
        if decision["decision"] != "allow":
            raise HTTPException(451, "privacy_restricted")


def execute(
    subject: str, binding: str, command: dict[str, Any] | None = None, digest: str | None = None
) -> tuple[dict[str, Any], bool]:
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='1500ms'")
            cur.execute("SET LOCAL statement_timeout='3000ms'")
            cur.execute("SET LOCAL idle_in_transaction_session_timeout='4000ms'")
            _privacy(cur, subject)
            if command is None:
                cur.execute(
                    "SELECT candidate_owned_status(%s::uuid,%s::uuid) AS result", (subject, binding)
                )
            else:
                cur.execute(
                    "SELECT candidate_owned_ensure(%s::uuid,%s::uuid,%s::uuid,%s,%s) AS result",
                    (subject, binding, command["commandId"], command["sequence"], digest),
                )
            row = cur.fetchone()
            if not row or not isinstance(row["result"], dict):
                raise RuntimeError("invalid_result")
            result = row["result"]
            replayed = result.pop("_replayed", None) if command is not None else True
            expected = (
                receipt(command, digest)
                if command is not None
                else {
                    "schemaVersion": 1,
                    "state": "empty",
                    "profileVersion": 0,
                    "preferenceVersion": 0,
                    "publicationActive": False,
                }
            )
            if (
                type(replayed) is not bool
                or result != expected
                or any(type(result[k]) is not type(value) for k, value in expected.items())
            ):
                raise RuntimeError("invalid_result")
            return result, replayed
    except HTTPException:
        raise
    except psycopg.Error as exc:
        marker = exc.diag.message_primary
        if exc.sqlstate == "P0002" and marker == "not_found":
            raise HTTPException(404, "not_found") from None
        if exc.sqlstate == "23505" and marker in {
            "identity_conflict",
            "receipt_conflict",
            "stale_sequence",
        }:
            raise HTTPException(409, marker) from None
        raise HTTPException(503, "authority_unavailable") from None
    except Exception:
        raise HTTPException(503, "authority_unavailable") from None
