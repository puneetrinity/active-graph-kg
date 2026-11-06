"""Owner-private profile authority; never implies network publication permission."""

from fastapi import HTTPException

from activekg.candidate_profiles.authority import require_authority


def require_owner(claims, *, write=False):
    return require_authority(
        claims, "candidate-profile:owner-write" if write else "candidate-profile:owner-read"
    )


def check_owner_privacy(cur, subject):
    # A network withdrawal stops global use, not the owner's private editing.
    # Erasure and ambiguous privacy authority still fail closed. Flow independently
    # checks its current verified account before admitting and sending a command.
    cur.execute("SELECT set_config('app.current_tenant_id',%s,true)", (f"candidate_{subject}",))
    cur.execute(
        "SELECT global_candidate_id FROM candidate_consent_state WHERE subject_id=%s::uuid",
        (subject,),
    )
    row = cur.fetchone()
    if not row or not row["global_candidate_id"]:
        return
    cur.execute(
        "SELECT candidate_privacy_global_decision(%s::uuid) AS decision",
        (row["global_candidate_id"],),
    )
    result = cur.fetchone()
    decision = result["decision"] if result else None
    if decision == "block_all":
        raise HTTPException(451, "candidate_privacy_restricted")
    if decision not in ("allow", "block_global"):
        raise HTTPException(503, "authority_unavailable")
