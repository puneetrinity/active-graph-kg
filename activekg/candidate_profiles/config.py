import os
import re


def enabled() -> bool:
    return os.getenv("CANDIDATE_FOUNDATION_ENABLED") == "true"


def profile_enabled() -> bool:
    return enabled() and os.getenv("CANDIDATE_PROFILE_ENABLED") == "true"


def consent_v2_enabled() -> bool:
    return enabled() and os.getenv("CANDIDATE_CONSENT_V2_ENABLED") == "true"


def spark_enabled() -> bool:
    return profile_enabled() and os.getenv("CANDIDATE_SPARK_ENABLED") == "true"


def source_import_enabled() -> bool:
    return (
        spark_enabled()
        and consent_v2_enabled()
        and os.getenv("CANDIDATE_SOURCE_IMPORT_ENABLED") == "true"
    )


def interpretation_enabled() -> bool:
    return source_import_enabled() and os.getenv("CANDIDATE_INTERPRETATION_ENABLED") == "true"


def network_publication_enabled() -> bool:
    return (
        spark_enabled()
        and consent_v2_enabled()
        and os.getenv("CANDIDATE_NETWORK_PUBLICATION_ENABLED") == "true"
    )


_RECOVERY_EPOCH = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


def network_recovery_epoch() -> str | None:
    """Restore gate epoch (B5). Deployment configuration, outside the database.

    Native Spark discovery opens only when Memory holds an owner-run attestation for
    this exact value; a missing or malformed epoch keeps it closed.
    """
    value = os.getenv("CANDIDATE_NETWORK_RECOVERY_EPOCH")
    return value if value and _RECOVERY_EPOCH.fullmatch(value) else None
