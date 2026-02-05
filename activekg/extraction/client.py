"""Groq extraction client with two-tier model fallback."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from pydantic import ValidationError

from activekg.extraction.prompt import build_extraction_prompt
from activekg.extraction.schema import ExtractionResult

logger = logging.getLogger(__name__)

# Model tiers
PRIMARY_MODEL = os.getenv("EXTRACTION_PRIMARY_MODEL", "llama-3.1-8b-instant")
FALLBACK_MODEL = os.getenv("EXTRACTION_FALLBACK_MODEL", "llama-3.3-70b-versatile")
CONFIDENCE_THRESHOLD = float(os.getenv("EXTRACTION_CONFIDENCE_THRESHOLD", "0.65"))
MAX_TOKENS = int(os.getenv("EXTRACTION_MAX_TOKENS", "1024"))


class ExtractionClient:
    """Client for resume extraction using Groq with two-tier fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        primary_model: str = PRIMARY_MODEL,
        fallback_model: str = FALLBACK_MODEL,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        max_tokens: int = MAX_TOKENS,
    ):
        """Initialize extraction client.

        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env)
            primary_model: Primary model for extraction (fast, cheap)
            fallback_model: Fallback model for retries (slower, better)
            confidence_threshold: Threshold below which to retry with fallback
            max_tokens: Max tokens for LLM response (defaults to EXTRACTION_MAX_TOKENS env)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.confidence_threshold = confidence_threshold
        self.max_tokens = max_tokens

        # Lazy import to avoid circular deps
        self._llm_primary: Any = None
        self._llm_fallback: Any = None

    def _get_llm(self, model: str) -> Any:
        """Get or create LLM provider for model."""
        from activekg.engine.llm_provider import LLMProvider

        return LLMProvider(
            backend="groq",
            model=model,
            api_key=self.api_key,
            temperature=0.1,  # Low temp for structured output
            max_tokens=self.max_tokens,
        )

    @property
    def llm_primary(self) -> Any:
        if self._llm_primary is None:
            self._llm_primary = self._get_llm(self.primary_model)
        return self._llm_primary

    @property
    def llm_fallback(self) -> Any:
        if self._llm_fallback is None:
            self._llm_fallback = self._get_llm(self.fallback_model)
        return self._llm_fallback

    def extract(self, resume_text: str) -> tuple[ExtractionResult, str]:
        """Extract structured fields from resume text.

        Uses two-tier approach:
        1. Try primary model (fast, cheap)
        2. If JSON invalid, missing required fields, or low confidence → retry with fallback

        Args:
            resume_text: Raw resume text

        Returns:
            Tuple of (ExtractionResult, model_used)

        Raises:
            ExtractionError: If both tiers fail
        """
        system_msg, user_prompt = build_extraction_prompt(resume_text)

        # Try primary model first
        result, needs_fallback, error_reason = self._try_extract(
            self.llm_primary, system_msg, user_prompt, self.primary_model
        )

        if result and not needs_fallback:
            logger.info(
                "Extraction succeeded with primary model",
                extra={
                    "model": self.primary_model,
                    "confidence": result.confidence,
                    "skills_count": len(result.primary_skills),
                },
            )
            return result, self.primary_model

        # Fallback to larger model
        logger.info(
            "Falling back to larger model",
            extra={
                "reason": error_reason,
                "primary_model": self.primary_model,
                "fallback_model": self.fallback_model,
            },
        )

        result, needs_fallback, error_reason = self._try_extract(
            self.llm_fallback, system_msg, user_prompt, self.fallback_model
        )

        if result:
            logger.info(
                "Extraction succeeded with fallback model",
                extra={
                    "model": self.fallback_model,
                    "confidence": result.confidence,
                    "skills_count": len(result.primary_skills),
                },
            )
            return result, self.fallback_model

        # Both failed
        raise ExtractionError(f"Extraction failed on both models: {error_reason}")

    def _try_extract(
        self, llm: Any, system_msg: str, user_prompt: str, model_name: str
    ) -> tuple[ExtractionResult | None, bool, str]:
        """Try extraction with a single model.

        Args:
            llm: LLMProvider instance
            system_msg: System prompt
            user_prompt: User prompt with resume text
            model_name: Model name for logging

        Returns:
            Tuple of (result, needs_fallback, error_reason)
        """
        start_time = time.time()

        try:
            # Generate with no stop sequences to get full JSON
            response = llm.generate(
                user_prompt,
                system_message=system_msg,
                stop=[],  # Don't stop early
                max_tokens=self.max_tokens,
            )

            latency_ms = (time.time() - start_time) * 1000
            logger.debug(
                "LLM extraction response",
                extra={"model": model_name, "latency_ms": latency_ms, "response_len": len(response)},
            )

            # Parse JSON
            result = self._parse_response(response)

            # Check if we need fallback
            if not result.has_required_fields():
                return result, True, "missing_required_fields"

            if result.confidence < self.confidence_threshold:
                return result, True, f"low_confidence_{result.confidence}"

            return result, False, ""

        except json.JSONDecodeError as e:
            return None, True, f"json_invalid: {e}"
        except ValidationError as e:
            return None, True, f"validation_failed: {e}"
        except Exception as e:
            logger.warning(
                "Extraction attempt failed",
                extra={"model": model_name, "error": str(e)},
            )
            return None, True, f"exception: {e}"

    def _parse_response(self, response: str) -> ExtractionResult:
        """Parse LLM response into ExtractionResult.

        Handles common issues like markdown code blocks, extra text.
        """
        text = response.strip()

        # Strip markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Find JSON object boundaries
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise json.JSONDecodeError("No JSON object found", text, 0)

        json_str = text[start:end]
        data = json.loads(json_str)

        return ExtractionResult(**data)


class ExtractionError(Exception):
    """Raised when extraction fails after all retries."""

    pass
