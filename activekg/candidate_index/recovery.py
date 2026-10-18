"""Owner-only explicit recovery. No provider client, queue reset or automatic scan."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from activekg.candidate_index.contracts import (
    ORG_TENANT,
    IndexPolicy,
    IndexTunables,
    canonical_json,
)


def checked_pairs(scope: str, pairs: list[dict[str, str]]) -> list[dict[str, str]]:
    if not ORG_TENANT.fullmatch(scope) or not isinstance(pairs, list) or not 1 <= len(pairs) <= 100:
        raise ValueError("recovery_selection_invalid")
    output = []
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {"source_id", "generation_id"}:
            raise ValueError("recovery_selection_invalid")
        output.append({key: str(UUID(value)) for key, value in pair.items()})
    if len({p["source_id"] for p in output}) != len(output):
        raise ValueError("recovery_selection_invalid")
    return sorted(output, key=lambda p: p["source_id"])


class ResumeRecovery:
    def __init__(
        self, dsn: str, target_id: str, policy: IndexPolicy, tunables: IndexTunables | None = None
    ):
        self.dsn = dsn
        self.target_id = str(UUID(target_id))
        self.policy = policy
        self.tunables = tunables or IndexTunables()

    def _identity(self, cur: psycopg.Cursor) -> None:
        cur.execute("SELECT target_id::text FROM activekg_schema_control.target_identity")
        row = cur.fetchone()
        if row != (self.target_id,):
            raise ValueError("recovery_target_mismatch")

    def plan(self, scope: str, pairs: list[dict[str, str]]) -> dict[str, Any]:
        pairs = checked_pairs(scope, pairs)
        with psycopg.connect(self.dsn, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute("SET LOCAL statement_timeout='3s'")
            self._identity(cur)
            cur.execute(
                "SELECT candidate_index_plan_recovery(%s,%s,%s)",
                (scope, Jsonb(pairs), canonical_json(asdict(self.policy))),
            )
            plan = cur.fetchone()[0]
            cur.execute(
                "SELECT encode(sha256(convert_to(%s::jsonb::text,'UTF8')),'hex')", (Jsonb(plan),)
            )
            return {"target_id": self.target_id, "plan_hash": cur.fetchone()[0], "plan": plan}

    def apply(self, envelope: dict[str, Any], approved_hash: str, max_calls: int) -> dict[str, Any]:
        if (
            set(envelope) != {"target_id", "plan_hash", "plan"}
            or envelope["target_id"] != self.target_id
            or envelope["plan_hash"] != approved_hash
            or type(max_calls) is not int
            or not 0 <= max_calls <= 200
            or envelope["plan"]["policy_sha256"] != self.policy.digest
            or envelope["plan"]["policy_json"] != canonical_json(asdict(self.policy))
        ):
            raise ValueError("recovery_approval_mismatch")
        with psycopg.connect(self.dsn, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout='3s'")
            self._identity(cur)
            cur.execute(
                "SELECT set_config('app.recovery_scope_limit',%s,true), "
                "set_config('app.recovery_total_limit',%s,true)",
                (str(self.tunables.pending_per_scope), str(self.tunables.pending_total)),
            )
            cur.execute(
                "SELECT candidate_index_apply_recovery(%s,%s,%s,%s)",
                (envelope["plan"]["scope"], Jsonb(envelope["plan"]), approved_hash, max_calls),
            )
            return cur.fetchone()[0]
