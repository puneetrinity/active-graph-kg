"""Privacy-fenced, bounded cache access. No provider client or raw profile storage."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

import psycopg

from activekg.rc_skills.contracts import (
    MAX_RESPONSE_BYTES,
    FinishRequest,
    FinishResponse,
    ReadRequest,
    ReadResponse,
    canonical_json,
)


class RcUnavailable(RuntimeError):
    def __init__(self, code: str = "temporarily_unavailable"):
        self.code = (
            code
            if code
            in {"temporarily_unavailable", "unavailable", "restricted", "conflict", "denied"}
            else "temporarily_unavailable"
        )
        super().__init__(self.code)


class RcRepository:
    def __init__(self, dsn: str):
        if not dsn:
            raise RcUnavailable()
        self._dsn = dsn

    @contextmanager
    def _transaction(self, *, readonly: bool):
        try:
            with psycopg.connect(self._dsn, connect_timeout=3, autocommit=False) as conn:
                if readonly:
                    conn.execute("SET TRANSACTION READ ONLY")
                conn.execute("SET LOCAL statement_timeout='3s'")
                conn.execute("SET LOCAL lock_timeout='500ms'")
                conn.execute("SET LOCAL idle_in_transaction_session_timeout='5s'")
                with conn.cursor() as cur:
                    yield cur
        except RcUnavailable:
            raise
        except Exception:
            # Never expose SQL parameters, candidate details, DSNs or exception causes.
            raise RcUnavailable() from None

    def _privacy(self, cur: Any, candidate_ids: list[str]) -> dict[str, str]:
        # The existing approved-provider mapping is public-only and append-only.
        # Its SELECT grant already exists; no private-profile table is read here.
        from activekg.api.sourced_candidates import _privacy_tokens

        cur.execute(
            "SELECT global_candidate_id::text, canonical_linkedin_url "
            "FROM public.global_candidate_source_identities "
            "WHERE global_candidate_id=ANY(%s::uuid[]) "
            "AND provider_namespace='crustdata' AND record_type='person' "
            "AND verified_issuer='signal' AND verified_actor_id='signal-service' "
            'ORDER BY global_candidate_id, canonical_linkedin_url COLLATE "C"',
            (candidate_ids,),
        )
        urls: dict[str, set[str]] = {}
        for candidate_id, url in cur.fetchall():
            urls.setdefault(candidate_id, set()).add(url)
        tokens = {
            candidate_id: _privacy_tokens(next(iter(values)))
            for candidate_id, values in urls.items()
            if len(values) == 1
        }
        # Same token-before-global lock order as the privacy writer. Lock all
        # batch tokens first, then all identities, to avoid opposite batch order.
        lock_names = sorted(
            {
                f"candidate-privacy-token:{t['identifier_type']}:{t['key_version']}:{t['token']}"
                for values in tokens.values()
                for t in values
            }
        )
        lock_names += [f"candidate-privacy-global:{value}" for value in sorted(tokens)]
        for name in lock_names:
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (name,))
        states: dict[str, str] = {}
        for candidate_id in sorted(tokens):
            cur.execute(
                "SELECT decision FROM public.candidate_privacy_match(%s::jsonb,%s,NULL,NULL)",
                (json.dumps(tokens[candidate_id]), candidate_id),
            )
            decisions = [row[0] for row in cur.fetchall()]
            if any(
                value not in {"allow", "block_all", "block_global", "review"} for value in decisions
            ):
                raise RcUnavailable()
            states[candidate_id] = (
                "restricted" if any(value != "allow" for value in decisions) else "allowed"
            )
        return states

    @staticmethod
    def _result(row: Any) -> dict:
        if row is None or not isinstance(row[0], dict):
            raise RcUnavailable()
        result = row[0]
        if "error" in result:
            raise RcUnavailable(result["error"])
        if len(canonical_json(result).encode()) > MAX_RESPONSE_BYTES:
            raise RcUnavailable()
        return result

    def read(self, service: str, command: ReadRequest) -> ReadResponse:
        with self._transaction(readonly=True) as cur:
            states = self._privacy(cur, command.candidate_ids)
            allowed = [
                value for value in command.candidate_ids if states.get(value.lower()) == "allowed"
            ]
            cur.execute(
                "SELECT public.memory_rc_read(%s,%s::jsonb)",
                (
                    service,
                    canonical_json({"version": 1, "candidateIds": allowed}),
                ),
            )
            response = ReadResponse.model_validate(self._result(cur.fetchone()))
            by_id = {item.global_candidate_id.lower(): item for item in response.candidates}
            if set(by_id) != {value.lower() for value in allowed} or len(by_id) != len(
                response.candidates
            ):
                raise RcUnavailable()
            result = ReadResponse.model_validate(
                {
                    "version": 1,
                    "candidates": [
                        by_id[value.lower()].model_dump(by_alias=True)
                        if value.lower() in by_id
                        else {
                            "globalCandidateId": value,
                            "state": "restricted"
                            if states.get(value.lower()) == "restricted"
                            else "miss",
                            "observation": None,
                        }
                        for value in command.candidate_ids
                    ],
                }
            )
            if len(canonical_json(result.model_dump(by_alias=True)).encode()) > MAX_RESPONSE_BYTES:
                raise RcUnavailable()
            return result

    def finish(self, service: str, command: FinishRequest) -> FinishResponse:
        with self._transaction(readonly=False) as cur:
            state = self._privacy(cur, [command.global_candidate_id]).get(
                command.global_candidate_id.lower()
            )
            if state != "allowed":
                raise RcUnavailable("restricted" if state == "restricted" else "unavailable")
            cur.execute(
                "SELECT public.memory_rc_finish(%s,%s,%s::jsonb)",
                (
                    service,
                    command.origin_attempt,
                    canonical_json(command.model_dump(by_alias=True)),
                ),
            )
            result = FinishResponse.model_validate(self._result(cur.fetchone()))
            if result.content_hash != command.observation.content_hash:
                raise RcUnavailable()
            return result


# Filled only from both real disposable collation proofs, never trusted from DB.
RC_CATALOG_SHA256 = "6de2a5b81f5be4c54af7a969a82cf6ae74f12a86d15315ef61e553905d900a17"
RC_CATALOG_SQL = """
WITH chosen AS (SELECT coalesce(%s,current_user) AS runtime_role),
relations AS (SELECT c.* FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
  WHERE n.nspname='public' AND left(c.relname,9)='rc_skill_' AND c.relkind IN ('r','p','v','m','f')),
routines AS (SELECT p.* FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
  WHERE n.nspname='public' AND left(p.proname,10)='memory_rc_'),
owner AS (SELECT relowner FROM relations WHERE relname='rc_skill_observations')
SELECT jsonb_build_object(
 'tables',(SELECT jsonb_agg(jsonb_build_object(
  'name',c.relname,'kind',c.relkind,'rls',c.relrowsecurity,'force_rls',c.relforcerowsecurity,
  'owner_matches',c.relowner=o.relowner,
  'columns',(SELECT jsonb_agg(jsonb_build_array(a.attname,format_type(a.atttypid,a.atttypmod),
    a.attnotnull,a.attidentity,a.attgenerated,pg_get_expr(d.adbin,d.adrelid)) ORDER BY a.attnum)
    FROM pg_attribute a LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum
    WHERE a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped),
  'constraints',(SELECT jsonb_agg(jsonb_build_array(x.conname,x.contype,x.convalidated,
    x.condeferrable,x.condeferred,pg_get_constraintdef(x.oid)) ORDER BY x.conname COLLATE "C")
    FROM pg_constraint x WHERE x.conrelid=c.oid),
  'indexes',(SELECT jsonb_agg(jsonb_build_array(k.relname,ix.indisvalid,ix.indisready,
    pg_get_indexdef(ix.indexrelid)) ORDER BY k.relname COLLATE "C") FROM pg_index ix
    JOIN pg_class k ON k.oid=ix.indexrelid WHERE ix.indrelid=c.oid),
  'policies',(SELECT jsonb_agg(jsonb_build_array(p.polname,p.polcmd,p.polpermissive,
    p.polroles=ARRAY[c.relowner],pg_get_expr(p.polqual,p.polrelid),pg_get_expr(p.polwithcheck,p.polrelid))
    ORDER BY p.polname COLLATE "C") FROM pg_policy p WHERE p.polrelid=c.oid),
  'triggers',(SELECT jsonb_agg(jsonb_build_array(t.tgname,t.tgenabled,pg_get_triggerdef(t.oid))
    ORDER BY t.tgname COLLATE "C") FROM pg_trigger t WHERE t.tgrelid=c.oid AND NOT t.tgisinternal)
 ) ORDER BY c.relname COLLATE "C") FROM relations c CROSS JOIN owner o),
 'routines',(SELECT jsonb_agg(jsonb_build_array(p.oid::regprocedure::text,p.proowner=o.relowner,
   pg_get_functiondef(p.oid)) ORDER BY p.oid::regprocedure::text COLLATE "C")
   FROM routines p CROSS JOIN owner o)
), jsonb_build_object(
 'tables',(SELECT jsonb_object_agg(c.relname,jsonb_build_object(
  'table',(SELECT jsonb_agg(has_table_privilege(ch.runtime_role,c.oid,privilege) ORDER BY ordinal)
    FROM unnest(ARRAY['SELECT','INSERT','UPDATE','DELETE','TRUNCATE','REFERENCES','TRIGGER']) WITH ORDINALITY p(privilege,ordinal)),
  'columns',(SELECT jsonb_agg(has_any_column_privilege(ch.runtime_role,c.oid,privilege) ORDER BY ordinal)
    FROM unnest(ARRAY['SELECT','INSERT','UPDATE','REFERENCES']) WITH ORDINALITY p(privilege,ordinal)),
  'public',EXISTS(SELECT 1 FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner))) a WHERE a.grantee=0),
  'public_columns',EXISTS(SELECT 1 FROM pg_attribute t CROSS JOIN LATERAL aclexplode(t.attacl) a WHERE t.attrelid=c.oid AND a.grantee=0),
  'grant_option',EXISTS(SELECT 1 FROM aclexplode(c.relacl) a WHERE a.grantee<>c.relowner AND a.is_grantable),
  'column_grant_option',EXISTS(SELECT 1 FROM pg_attribute t CROSS JOIN LATERAL aclexplode(t.attacl) a WHERE t.attrelid=c.oid AND a.grantee<>c.relowner AND a.is_grantable)
 )) FROM relations c CROSS JOIN chosen ch),
 'routines',(SELECT jsonb_object_agg(p.oid::regprocedure::text,jsonb_build_array(
   has_function_privilege(ch.runtime_role,p.oid,'EXECUTE'),
   EXISTS(SELECT 1 FROM aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a WHERE a.grantee=0),
   EXISTS(SELECT 1 FROM aclexplode(p.proacl) a WHERE a.grantee<>p.proowner AND a.is_grantable)))
   FROM routines p CROSS JOIN chosen ch)
)
"""


def catalog_evidence(cur: Any, role: str | None = None) -> tuple[str, bool]:
    from activekg.rc_skills.contracts import RUNTIME_FUNCTIONS, TABLES, content_hash

    cur.execute(RC_CATALOG_SQL, (role,))
    structure, grants = cur.fetchone()
    tables, routines = grants.get("tables") or {}, grants.get("routines") or {}
    ok = set(tables) == set(TABLES) and set(routines) == set(RUNTIME_FUNCTIONS)
    for values in tables.values():
        ok = ok and not any(values.get("table") or []) and not any(values.get("columns") or [])
        ok = ok and not any(
            values.get(key)
            for key in ("public", "public_columns", "grant_option", "column_grant_option")
        )
    return content_hash(structure), bool(
        ok and all(v == [True, False, False] for v in routines.values())
    )


def catalog_ready(cur: Any, role: str | None = None) -> bool:
    digest, privileges = catalog_evidence(cur, role)
    return privileges and digest == RC_CATALOG_SHA256
