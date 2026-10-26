import os


def enabled() -> bool:
    return os.getenv("CANDIDATE_FOUNDATION_ENABLED") == "true"
