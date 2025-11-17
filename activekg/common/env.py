from __future__ import annotations

from typing import Optional, Sequence
import os


def env_str(name_or_names: str | Sequence[str], default: str = "") -> str:
    """Return the first non-empty environment variable from a name or list of names.

    Ensures a string is always returned (never None) for easier type-checking.
    """
    if isinstance(name_or_names, str):
        val = os.getenv(name_or_names)
        return val if val is not None and val != "" else default
    for name in name_or_names:
        val = os.getenv(name)
        if val:
            return val
    return default


def env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int = 0) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except Exception:
        return default


def env_float(name: str, default: float = 0.0) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except Exception:
        return default

