"""JWT authentication middleware for Active Graph KG.

Extracts tenant_id and actor_id from JWT claims for RLS context and audit trail.
"""

from __future__ import annotations

import os

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWTError

from activekg.common.logger import get_enhanced_logger
from activekg.observability.metrics import access_violations_total

logger = get_enhanced_logger(__name__)

# JWT Configuration (from env)
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")  # RS256 public key or HS256 shared secret
JWT_PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY")  # RS256 public key (Vanta's key)
SIGNAL_JWT_PUBLIC_KEY = os.getenv("SIGNAL_JWT_PUBLIC_KEY")  # RS256 public key (Signal's key)
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "RS256")  # RS256 for production, HS256 for dev
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "activekg")
JWT_ISSUER = os.getenv("JWT_ISSUER")  # e.g., "vantahire" (primary issuer)
JWT_ENABLED = os.getenv("JWT_ENABLED", "false").lower() == "true"
JWT_LEEWAY_SECONDS = int(os.getenv("JWT_LEEWAY_SECONDS", "30"))  # Clock skew tolerance

# Multi-issuer support: map issuer → public key
_ISSUER_KEY_MAP: dict[str, str] = {}
if JWT_PUBLIC_KEY:
    # Vanta's key (primary issuer, or JWT_ISSUER value)
    _primary_issuer = JWT_ISSUER or "vantahire"
    _ISSUER_KEY_MAP[_primary_issuer] = JWT_PUBLIC_KEY
if SIGNAL_JWT_PUBLIC_KEY:
    _ISSUER_KEY_MAP["signal"] = SIGNAL_JWT_PUBLIC_KEY

security = HTTPBearer(auto_error=False)


class JWTClaims:
    """Parsed JWT claims for Active Graph KG."""

    def __init__(
        self,
        tenant_id: str,
        actor_id: str,
        actor_type: str = "user",
        scopes: list[str] | None = None,
        exp: int | None = None,
    ):
        self.tenant_id = tenant_id
        self.actor_id = actor_id
        self.actor_type = actor_type
        self.scopes = scopes or []
        self.exp = exp


def verify_jwt(token: str) -> JWTClaims:
    """Verify and decode JWT token.

    Args:
        token: JWT token string

    Returns:
        JWTClaims object with tenant_id, actor_id, scopes

    Raises:
        HTTPException: If token is invalid or expired
    """
    # Multi-issuer: peek at unverified 'iss' claim to select the right public key
    key: str | None = None
    detected_issuer: str | None = None
    if _ISSUER_KEY_MAP and JWT_ALGORITHM == "RS256":
        try:
            unverified = jwt.decode(token, options={"verify_signature": False})
            detected_issuer = unverified.get("iss")
            if detected_issuer and detected_issuer in _ISSUER_KEY_MAP:
                key = _ISSUER_KEY_MAP[detected_issuer]
        except Exception:
            pass  # Fall through to default key

    if not key:
        key = JWT_PUBLIC_KEY or JWT_SECRET_KEY
    if not key:
        raise HTTPException(
            status_code=500, detail="JWT_SECRET_KEY or JWT_PUBLIC_KEY not configured"
        )

    # Determine which issuer to verify against.
    # PyJWT 2.8 requires `issuer` as a string, not a list.
    # Use the detected issuer (from unverified peek) if it's in our key map;
    # otherwise fall back to JWT_ISSUER.
    accepted_issuer: str | None = None
    if detected_issuer and detected_issuer in _ISSUER_KEY_MAP:
        accepted_issuer = detected_issuer
    elif _ISSUER_KEY_MAP and len(_ISSUER_KEY_MAP) == 1:
        accepted_issuer = next(iter(_ISSUER_KEY_MAP))
    elif JWT_ISSUER:
        accepted_issuer = JWT_ISSUER

    try:
        # Decode and verify token with leeway for clock skew
        payload = jwt.decode(
            token,
            key,
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=accepted_issuer,
            leeway=JWT_LEEWAY_SECONDS,  # Tolerate clock skew
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
                "verify_aud": True,
                "verify_iss": bool(accepted_issuer),
            },
        )

        # Extract claims
        tenant_id = payload.get("tenant_id")
        actor_id = payload.get("sub")  # Standard JWT claim for subject/user ID
        actor_type = payload.get("actor_type", "user")

        # Normalize scopes to list[str] from any of:
        #   - "scopes": ["search:read", "kg:write"]   (list of strings)
        #   - "scopes": "search:read kg:write"          (space-delimited string)
        #   - "scope":  "search:read kg:write"          (OAuth2 fallback field)
        # Invalid/non-string values are silently ignored.
        raw_scopes = payload.get("scopes")
        if raw_scopes is None:
            raw_scopes = payload.get("scope")

        if isinstance(raw_scopes, list):
            scopes = [s for s in raw_scopes if isinstance(s, str)]
        elif isinstance(raw_scopes, str):
            scopes = raw_scopes.split() if raw_scopes else []
        else:
            scopes = []

        exp = payload.get("exp")

        if not tenant_id:
            raise HTTPException(status_code=401, detail="Missing tenant_id claim in JWT")

        if not actor_id:
            raise HTTPException(status_code=401, detail="Missing sub (actor_id) claim in JWT")

        logger.info(
            "JWT verified",
            extra_fields={"tenant_id": tenant_id, "actor_id": actor_id, "scopes": scopes},
        )

        return JWTClaims(
            tenant_id=tenant_id, actor_id=actor_id, actor_type=actor_type, scopes=scopes, exp=exp
        )

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="JWT token has expired") from None
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=401, detail="JWT token has invalid audience") from None
    except jwt.InvalidIssuerError:
        raise HTTPException(status_code=401, detail="JWT token has invalid issuer") from None
    except PyJWTError as e:
        logger.warning("JWT verification failed", extra_fields={"error": str(e)})
        raise HTTPException(status_code=401, detail=f"Invalid JWT token: {str(e)}") from e


async def get_jwt_claims(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> JWTClaims | None:
    """FastAPI dependency to extract and verify JWT claims.

    Usage:
        @app.get("/nodes/{node_id}")
        def get_node(node_id: str, claims: JWTClaims = Depends(get_jwt_claims)):
            repo.get_node(node_id, tenant_id=claims.tenant_id)

    Returns:
        JWTClaims if JWT_ENABLED=true and valid token provided
        None if JWT_ENABLED=false (dev mode)

    Raises:
        HTTPException 401 if JWT_ENABLED=true and token is missing/invalid
    """
    # Dev mode: JWT disabled
    if not JWT_ENABLED:
        logger.debug("JWT authentication disabled (dev mode)")
        return None

    # Production mode: require JWT
    if not credentials:
        try:
            access_violations_total.labels(type="missing_token").inc()
        except Exception:
            pass
        raise HTTPException(
            status_code=401, detail="Missing Authorization header. JWT required for this endpoint."
        )

    return verify_jwt(credentials.credentials)


def require_scope(required_scope: str):
    """Factory that returns a FastAPI dependency enforcing a specific scope.

    Usage:
        @app.post("/search", dependencies=[Depends(require_scope("search:read"))])
        def search(...):
            ...
    """

    async def _check(claims: JWTClaims | None = Depends(get_jwt_claims)):
        if JWT_ENABLED and claims and required_scope not in claims.scopes:
            try:
                access_violations_total.labels(type="scope_denied").inc()
            except Exception:
                pass
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required scope: {required_scope}",
            )

    return _check
