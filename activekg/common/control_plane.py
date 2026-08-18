"""Import-light authentication for operational control-plane endpoints.

The verifier deliberately has no FastAPI dependency so the API and standalone
workers share exactly one fail-closed bearer-token contract.
"""

from __future__ import annotations

import hmac
import os

CONTROL_PLANE_TOKEN_ENV = "ACTIVEKG_CONTROL_PLANE_TOKEN"
_MIN_TOKEN_LENGTH = 32


class ControlPlaneUnavailable(RuntimeError):
    """The service has no valid control-plane credential configured."""


class ControlPlaneUnauthorized(RuntimeError):
    """The request did not present the configured control-plane credential."""


def verify_control_plane_authorization(authorization: str | None) -> None:
    """Validate one exact ``Authorization: Bearer`` credential.

    Configuration is read at request time so a missing or malformed secret
    fails closed. Neither the supplied credential nor the configured value is
    ever returned or logged.
    """

    configured = os.getenv(CONTROL_PLANE_TOKEN_ENV, "")
    if len(configured) < _MIN_TOKEN_LENGTH:
        raise ControlPlaneUnavailable("control-plane authentication is unavailable")

    scheme, separator, supplied = (authorization or "").partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not supplied:
        raise ControlPlaneUnauthorized("control-plane authentication required")

    if not hmac.compare_digest(supplied, configured):
        raise ControlPlaneUnauthorized("control-plane authentication required")
