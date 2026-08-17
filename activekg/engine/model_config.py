"""Service-local model selection and production safety guards."""

from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_GROQ_FAST_MODEL = "openai/gpt-oss-20b"
DEFAULT_GROQ_LARGE_MODEL = "openai/gpt-oss-120b"

RETIRED_GROQ_MODELS = frozenset(
    {
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    }
)


def is_railway_production(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get("RAILWAY_ENVIRONMENT_NAME", "").strip().lower() == "production"


def resolve_groq_model(
    env_name: str,
    default: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    env = os.environ if environ is None else environ
    configured = env.get(env_name, "").strip()

    if not configured:
        if is_railway_production(env):
            raise RuntimeError(f"{env_name} must be explicitly configured in production")
        configured = default

    if configured in RETIRED_GROQ_MODELS:
        raise RuntimeError(f"{env_name} points to retired Groq model: {configured}")

    return configured


def resolve_model_for_backend(
    env_name: str,
    default: str,
    *,
    backend: str,
    enabled: bool,
    environ: Mapping[str, str] | None = None,
) -> str:
    env = os.environ if environ is None else environ

    if enabled and backend.strip().lower() == "groq":
        return resolve_groq_model(env_name, default, env)

    return env.get(env_name, "").strip() or default
