"""Bounded native-publication embedding. Never embeds a private profile wholesale."""

import math
import os
from collections.abc import Callable
from typing import Any

import psycopg

from activekg.candidate_profiles.config import network_publication_enabled, network_recovery_epoch

PUBLICATION_DIMENSIONS = 384
PUBLICATION_BATCH_LIMIT = 5
NETWORK_TABLES = (
    "candidate_network_subjects",
    "candidate_network_publications",
    "candidate_network_recovery_audit",
    "candidate_network_recovery_attestations",
    "candidate_network_recovery_holds",
)
NETWORK_WORKER_ROLE = "activekg_network_worker"
NETWORK_WORKER_FUNCTIONS = (
    "candidate_network_claim()",
    "candidate_network_finish(uuid,uuid,text,vector)",
    "candidate_network_recovery_open(text)",
)
NETWORK_RUNTIME_FUNCTIONS = (
    "candidate_network_search(vector,integer,jsonb)",
    "candidate_network_validate(uuid[])",
    "candidate_network_owner_status(uuid,uuid)",
    "candidate_network_owner_status_detail(uuid,uuid)",
    "candidate_network_result_keys(uuid[])",
    "candidate_network_recovery_open(text)",
)
NETWORK_TRIGGERS = (
    "candidate_network_anchor_created",
    "candidate_network_confirmed",
    "candidate_network_consent_changed",
    "candidate_network_profile_changed",
    "candidate_network_operator_erased",
)
# Filled from the full-schema migration rehearsal, never a hand-written guess.
NETWORK_CATALOG_MD5 = "883c5ea23ce7de2d15c3b1b71163d31f"


def network_catalog_evidence(cur, role=None):
    from activekg.candidate_profiles.repository import CATALOG_SQL

    runtime = ",".join(
        "'" + signature.split("(")[0] + "'" for signature in NETWORK_RUNTIME_FUNCTIONS
    )
    query = (
        CATALOG_SQL.replace(
            "starts_with(c.relname,'candidate_owned_')",
            "starts_with(c.relname,'candidate_network_')",
        )
        .replace(
            "starts_with(p.proname,'candidate_owned_')",
            "starts_with(p.proname,'candidate_network_')",
        )
        .replace(
            " 'tables',",
            " 'privacy_resolver',pg_get_functiondef('public.candidate_privacy_resolve_subject(text,bytea)'::regprocedure), 'tables',",
        )
        .replace("relname='candidate_owned_subjects'", "relname='candidate_network_subjects'")
        .replace(
            "t.tgrelid IN(SELECT oid FROM relations)",
            "(t.tgrelid IN(SELECT oid FROM relations) OR t.tgname IN ('"
            + "','".join(NETWORK_TRIGGERS)
            + "'))",
        )
        .replace(
            "NOT has_function_privilege(ch.role,p.oid,'EXECUTE')",
            "has_function_privilege(ch.role,p.oid,'EXECUTE') IS DISTINCT FROM (p.proname IN ("
            + runtime
            + "))",
        )
        .replace(
            "(a.grantee IS DISTINCT FROM (SELECT oid FROM pg_roles WHERE rolname=ch.role) OR a.is_grantable)",
            "(a.is_grantable OR NOT ((a.grantee IS NOT DISTINCT FROM "
            "(SELECT oid FROM pg_roles WHERE rolname=ch.role) AND p.proname IN ("
            + runtime
            + ")) OR "
            "(a.grantee IS NOT DISTINCT FROM (SELECT oid FROM pg_roles WHERE rolname='activekg_network_worker') "
            "AND p.proname IN ('candidate_network_claim','candidate_network_finish','candidate_network_recovery_open'))))",
        )
    )
    cur.execute(query, (role,))
    return tuple(cur.fetchone())


def network_catalog_ready(cur, role=None):
    digest, permissions = network_catalog_evidence(cur, role)
    return digest == NETWORK_CATALOG_MD5 and permissions is True


def network_recovery_open(cur) -> bool:
    """B5 gate: true only for an attested current epoch. Fails closed."""
    epoch = network_recovery_epoch()
    if epoch is None:
        return False
    cur.execute("SELECT public.candidate_network_recovery_open(%s)", (epoch,))
    row = cur.fetchone()
    return bool(row and row[0] is True)


def publication_text(facts: dict[str, Any]) -> str:
    """Identity, location, photos, contacts and preferences are not ranking input."""
    parts: list[str] = []
    headline = facts.get("headline")
    if isinstance(headline, str):
        parts.append(headline)
    for skill in facts.get("skills", []):
        if isinstance(skill, str):
            parts.append(skill)
    for row in facts.get("employment", []):
        if isinstance(row, dict):
            for field in ("title", "employer"):
                if isinstance(row.get(field), str):
                    parts.append(row[field])
    for row in facts.get("education", []):
        if isinstance(row, dict) and isinstance(row.get("qualification"), str):
            parts.append(row["qualification"])
    return "\n".join(parts)


def vector_literal(values: Any) -> str:
    vector = [float(value) for value in values]
    if (
        len(vector) != PUBLICATION_DIMENSIONS
        or not all(math.isfinite(value) for value in vector)
        or not any(vector)
    ):
        raise ValueError("invalid_publication_vector")
    return "[" + ",".join(str(value) for value in vector) + "]"


def sweep_publications(
    dsn: str,
    embedder: Any,
    *,
    connect: Callable[..., Any] = psycopg.connect,
) -> int:
    if not network_publication_enabled():
        return 0
    # The ordinary worker/app DSN cannot activate publications. Never fall back
    # to it or to owner credentials when this dedicated capability is missing.
    worker_dsn = os.getenv("CANDIDATE_NETWORK_WORKER_DSN")
    if not worker_dsn:
        raise ValueError("candidate_network_worker_unconfigured")
    activated = 0
    for _ in range(PUBLICATION_BATCH_LIMIT):
        # Commit the lease before expensive work. A crash is recovered by the
        # bounded database lease, never by retaining an open transaction.
        with connect(worker_dsn) as conn, conn.cursor() as cur:
            # B5: nothing is claimed while recovery is closed, so a closed gate
            # never revokes or advances staged work; it simply waits.
            if not network_recovery_open(cur):
                break
            cur.execute("SELECT public.candidate_network_claim()")
            row = cur.fetchone()
            claim = row[0] if row else None
        if claim is None:
            break
        text = publication_text(claim["facts"])
        if not text.strip():
            # No fabricated ranking vector. Lease recovery eventually holds it.
            continue
        vector = vector_literal(embedder.encode([text])[0])
        with connect(worker_dsn) as conn, conn.cursor() as cur:
            # In-flight work from before a closure cannot activate; its lease
            # expires and the publication is claimed again once reopened.
            if not network_recovery_open(cur):
                break
            cur.execute(
                "SELECT public.candidate_network_finish(%s::uuid,%s::uuid,%s,%s::vector)",
                (claim["publicationId"], claim["leaseToken"], claim["contentHash"], vector),
            )
            result = cur.fetchone()
            activated += int(bool(result and result[0]))
    return activated
