import pytest

from activekg.engine.model_config import (
    DEFAULT_GROQ_FAST_MODEL,
    DEFAULT_GROQ_LARGE_MODEL,
    RETIRED_GROQ_MODELS,
    resolve_groq_model,
    resolve_model_for_backend,
)


def test_development_uses_supported_defaults():
    env = {"RAILWAY_ENVIRONMENT_NAME": "development"}

    assert resolve_groq_model("LLM_MODEL", DEFAULT_GROQ_FAST_MODEL, env) == DEFAULT_GROQ_FAST_MODEL
    assert (
        resolve_groq_model("EXTRACTION_FALLBACK_MODEL", DEFAULT_GROQ_LARGE_MODEL, env)
        == DEFAULT_GROQ_LARGE_MODEL
    )


@pytest.mark.parametrize("env_name", ["LLM_MODEL", "EXTRACTION_PRIMARY_MODEL"])
def test_production_requires_explicit_active_groq_models(env_name):
    env = {"RAILWAY_ENVIRONMENT_NAME": "production"}

    with pytest.raises(RuntimeError, match=f"{env_name} must be explicitly configured"):
        resolve_groq_model(env_name, DEFAULT_GROQ_FAST_MODEL, env)


@pytest.mark.parametrize("model", sorted(RETIRED_GROQ_MODELS))
def test_retired_groq_models_are_rejected(model):
    env = {
        "RAILWAY_ENVIRONMENT_NAME": "production",
        "LLM_MODEL": model,
    }

    with pytest.raises(RuntimeError, match="points to retired Groq model"):
        resolve_groq_model("LLM_MODEL", DEFAULT_GROQ_FAST_MODEL, env)


def test_supported_override_is_trimmed_and_returned():
    env = {
        "RAILWAY_ENVIRONMENT_NAME": "production",
        "LLM_MODEL": "  openai/gpt-oss-20b  ",
    }

    assert resolve_groq_model("LLM_MODEL", DEFAULT_GROQ_FAST_MODEL, env) == "openai/gpt-oss-20b"


def test_non_groq_backend_is_not_subject_to_groq_guard():
    env = {
        "RAILWAY_ENVIRONMENT_NAME": "production",
        "ASK_FALLBACK_MODEL": "gpt-4o-mini",
    }

    assert (
        resolve_model_for_backend(
            "ASK_FALLBACK_MODEL",
            "gpt-4o-mini",
            backend="openai",
            enabled=True,
            environ=env,
        )
        == "gpt-4o-mini"
    )


def test_disabled_api_model_does_not_gate_extraction_model_validation():
    production = {"RAILWAY_ENVIRONMENT_NAME": "production"}

    assert (
        resolve_model_for_backend(
            "LLM_MODEL",
            DEFAULT_GROQ_FAST_MODEL,
            backend="groq",
            enabled=False,
            environ=production,
        )
        == DEFAULT_GROQ_FAST_MODEL
    )
    with pytest.raises(RuntimeError, match="EXTRACTION_PRIMARY_MODEL"):
        resolve_groq_model("EXTRACTION_PRIMARY_MODEL", DEFAULT_GROQ_FAST_MODEL, production)
