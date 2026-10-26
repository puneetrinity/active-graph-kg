import time

from fastapi import HTTPException

from activekg.api import auth
from activekg.api.auth import JWTClaims
from activekg.candidate_profiles.contracts import UUID


def require_authority(claims: JWTClaims | None, scope: str) -> str:
    if not auth.JWT_ENABLED or claims is None:
        raise HTTPException(401, "service_auth_required")
    if (
        auth.JWT_ALGORITHM != "RS256"
        or auth.JWT_AUDIENCE != "activekg"
        or claims.issuer != "vantahire"
        or claims.actor_id != "vantahire-backend"
        or claims.actor_type != "service"
        or claims.scopes != [scope]
        or not isinstance(claims.tenant_id, str)
        or not claims.tenant_id.startswith("candidate_")
        or not UUID.fullmatch(claims.tenant_id[len("candidate_") :])
    ):
        raise HTTPException(403, "service_auth_denied")
    if (
        type(claims.iat) is not int
        or type(claims.exp) is not int
        or not 0 < claims.exp - claims.iat <= 60
        or claims.iat > time.time() + auth.JWT_LEEWAY_SECONDS
        or claims.exp < time.time() - auth.JWT_LEEWAY_SECONDS
    ):
        raise HTTPException(403, "service_auth_denied")
    return claims.tenant_id[len("candidate_") :]
