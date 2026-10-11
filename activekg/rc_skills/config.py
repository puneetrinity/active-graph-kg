"""Dark-by-default gate; never reads provider credentials."""

import os


def rc_skills_enabled() -> bool:
    value = os.getenv("RC_SKILLS_ENABLED", "false").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError("RC_SKILLS_ENABLED must be true or false")
    return value == "true"
