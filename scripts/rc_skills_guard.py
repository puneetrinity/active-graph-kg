#!/usr/bin/env python3
"""Discover all RC cache writers and check service/privacy/no-provider boundaries."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class GuardError(RuntimeError):
    pass


def validate(root: Path = ROOT) -> None:
    migration = (root / "db/migrations/030_rc_skill_observations.sql").read_text()
    repository = (root / "activekg/rc_skills/repository.py").read_text()
    router = (root / "activekg/rc_skills/router.py").read_text()
    contracts = (root / "activekg/rc_skills/contracts.py").read_text()
    if re.findall(r"CREATE TABLE public\.(\w+)", migration) != ["rc_skill_observations"]:
        raise GuardError("rc_table_inventory")
    if re.findall(r"CREATE FUNCTION public\.(\w+)", migration) != [
        "memory_rc_read",
        "memory_rc_finish",
    ]:
        raise GuardError("rc_routine_inventory")
    for token in (
        "FORCE ROW LEVEL SECURITY",
        "ON DELETE RESTRICT",
        "SECURITY DEFINER",
        "SET search_path=pg_catalog,public",
        "candidate_privacy_global_decision",
        "candidate_privacy_append_only",
        "origin_service,origin_attempt",
        "interval '26280 hours'",
        'ORDER BY k COLLATE "C"',
        "RC_INVALID_HASH",
    ):
        if token not in migration:
            raise GuardError("rc_sql_authority")
    if migration.count("SECURITY DEFINER") != 2 or migration.count("FROM PUBLIC;") != 3:
        raise GuardError("rc_wrapper_privileges")
    if re.search(r"\b(?:NO FORCE|DISABLE) ROW LEVEL SECURITY\b", migration):
        raise GuardError("rc_rls_disabled")
    writes = re.findall(
        r"\b(?:INSERT\s+INTO|UPDATE(?!\s+(?:OR|ON|SET)\b)|DELETE\s+FROM)\s+(?:public\.)?(\w+)",
        migration,
        re.I,
    )
    if writes != ["rc_skill_observations"]:
        raise GuardError("rc_only_append_observations")
    for path in (root / "activekg").rglob("*.py"):
        source = path.read_text()
        if re.search(
            r"\b(?:INSERT\s+INTO|UPDATE\s+|DELETE\s+FROM)\s*(?:public\.)?rc_skill_observations",
            source,
            re.I,
        ):
            raise GuardError("rc_raw_runtime_writer")
    for path in (root / "activekg/rc_skills").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [v.name for v in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                if any(
                    name.split(".")[0] in {"requests", "httpx", "aiohttp", "urllib", "socket"}
                    for name in names
                ):
                    raise GuardError("rc_memory_provider_network_forbidden")
    for token in (
        "candidate-privacy-token:",
        "candidate-privacy-global:",
        "candidate_privacy_match(",
        "readonly=True",
        "readonly=False",
        "SET TRANSACTION READ ONLY",
        "connect_timeout=3",
        "FinishResponse.model_validate",
        "ReadResponse.model_validate",
    ):
        if token not in repository:
            raise GuardError("rc_cache_privacy_boundary")
    for token in (
        "claims.issuer != auth.SIGNAL_JWT_ISSUER",
        'claims.actor_type != "service"',
        'claims.actor_id != "signal-service"',
        "claims.scopes != [scope]",
        "claims.tenant_id",
        "if not enabled:",
        "MAX_BODY_BYTES",
        "Depends(require_reader)",
        "Depends(require_writer)",
        "rc-skills:read",
        "rc-skills:write",
    ):
        if token not in router:
            raise GuardError("rc_service_authority")
    if 'extra="forbid"' not in contracts or "HORIZON_DAYS = 1095" not in contracts:
        raise GuardError("rc_closed_wire")
    if "REVERSE_CONTACT_API_KEY" in repository + router + contracts:
        raise GuardError("rc_memory_provider_key_forbidden")


if __name__ == "__main__":
    validate()
    print("rc-skills-guard: PASS (static authority; not database or provider proof)")
