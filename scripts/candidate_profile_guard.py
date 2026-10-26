"""Static S1 wiring checks; runtime and database tests remain mandatory."""

from pathlib import Path


def check(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    required = {
        "activekg/api/candidate_profiles.py": [
            "if not enabled():",
            "require_authority",
            "parse_command",
            "run_in_threadpool",
            "candidate-profile:foundation-write",
            "candidate-profile:foundation-read",
        ],
        "activekg/candidate_profiles/repository.py": [
            "_privacy(cur, subject)",
            "candidate_owned_ensure",
            "candidate_owned_status",
            "catalog_ready",
        ],
        "activekg/candidate_profiles/authority.py": [
            "RS256",
            "claims.scopes != [scope]",
            "claims.iat",
            "60",
        ],
        "activekg/api/main.py": ["app.include_router(candidate_profiles_router)"],
        "activekg/api/operational.py": ["catalog_ready"],
        "db/migrations/040_candidate_owned_foundation.sql": [
            "FORCE ROW LEVEL SECURITY",
            "REVOKE",
            "stale_sequence",
            "identity_conflict",
        ],
    }
    errors = []
    for file, anchors in required.items():
        text = (root / file).read_text()
        errors.extend(f"{file}: missing {anchor}" for anchor in anchors if anchor not in text)
    return errors


if __name__ == "__main__":
    errors = check()
    if errors:
        raise SystemExit("\n".join(errors))
    print("candidate-profile guard: OK")
