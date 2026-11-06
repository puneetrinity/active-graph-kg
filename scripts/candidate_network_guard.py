"""P7 source-wiring guard; PostgreSQL catalogue and lifecycle tests are mandatory too."""

from pathlib import Path

REQUIRED = {
    "activekg/candidate_profiles/config.py": (
        "def network_publication_enabled()",
        "CANDIDATE_NETWORK_PUBLICATION_ENABLED",
    ),
    "activekg/embedding/global_candidates.py": ("sweep_publications(self.dsn, self.embedder)",),
    "activekg/candidate_profiles/publication.py": (
        "if not network_publication_enabled():",
        "PUBLICATION_BATCH_LIMIT = 5",
        "candidate_network_claim()",
        "candidate_network_finish(%s::uuid,%s::uuid,%s,%s::vector)",
        "NETWORK_CATALOG_MD5",
        "network_catalog_ready",
    ),
    "activekg/api/global_memory.py": (
        "/global-candidates/network/validate",
        "/global-candidates/network/status",
        'Depends(require_scope("kg:read"))',
        "subject = require_owner(claims)",
        "public.candidate_network_validate(%s::uuid[])",
        "public.candidate_network_result_keys(%s::uuid[])",
        "public.candidate_network_owner_status_detail(%s,%s)",
    ),
    "db/migrations/044_candidate_network_result_dedup.sql": (
        "candidate_network_eligible(p.subject_id,p.profile_version,p.preference_version,p.consent_revision)",
        "version=item.profile_version",
        "REVOKE ALL ON FUNCTION public.candidate_network_result_keys(uuid[]) FROM PUBLIC",
    ),
    "scripts/init_railway_db.py": (
        "_harden_candidate_network_runtime_privileges",
        "NETWORK_RUNTIME_FUNCTIONS",
    ),
    "db/migrations/042_candidate_network_publication.sql": (
        "FORCE ROW LEVEL SECURITY",
        "candidate_network_eligible",
        "cc.approved",
        "cs.highest_version=cs.effective_version",
        "candidate_profile_confirmations",
        "candidate_privacy_global_decision",
        "candidate_network_operator_erased",
        "p.lease_token IS DISTINCT FROM token",
        "p.content_hash IS DISTINCT FROM fingerprint",
        "p.attempts>=3",
        "LIMIT n",
        "REVOKE ALL ON FUNCTION",
        "REVOKE ALL ON TABLE",
    ),
}


def check(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    errors = []
    for path, anchors in REQUIRED.items():
        source = (root / path).read_text()
        errors.extend(f"{path}: missing {anchor}" for anchor in anchors if anchor not in source)
    return errors


if __name__ == "__main__":
    errors = check()
    if errors:
        raise SystemExit("\n".join(errors))
    print("candidate-network guard: OK")
