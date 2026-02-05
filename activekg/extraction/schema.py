"""Pydantic schemas for resume extraction output."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Configurable caps
MAX_PRIMARY_SKILLS = max(0, int(os.getenv("EXTRACTION_MAX_PRIMARY_SKILLS", "12")))
MAX_RECENT_TITLES = max(0, int(os.getenv("EXTRACTION_MAX_RECENT_TITLES", "3")))
MAX_CERTIFICATIONS = max(0, int(os.getenv("EXTRACTION_MAX_CERTIFICATIONS", "10")))
MAX_INDUSTRIES = max(0, int(os.getenv("EXTRACTION_MAX_INDUSTRIES", "5")))


class ExtractionResult(BaseModel):
    """Structured extraction output from resume parsing."""

    # Must-have fields (Phase 2)
    primary_skills: list[str] = Field(
        default_factory=list,
        description="Top 8-12 technical and professional skills",
        min_length=0,
        max_length=MAX_PRIMARY_SKILLS or 1,
    )
    recent_job_titles: list[str] = Field(
        default_factory=list,
        description="1-3 most recent job titles",
        min_length=0,
        max_length=MAX_RECENT_TITLES or 1,
    )
    years_experience_total: int | str | None = Field(
        default=None,
        description="Total years of experience (number or bucket like '5-7')",
    )

    # Optional fields
    certifications: list[str] | None = Field(
        default=None,
        description="Professional certifications (AWS, PMP, etc.)",
    )
    industries: list[str] | None = Field(
        default=None,
        description="Industries worked in (e.g., 'Finance', 'Healthcare', 'Tech')",
    )

    # Extraction metadata
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model confidence score (0-1)",
    )

    @field_validator("primary_skills", mode="before")
    @classmethod
    def clean_skills(cls, v: list[str] | None) -> list[str]:
        if not v:
            return []
        # Dedupe and limit to configured cap
        seen: set[str] = set()
        result: list[str] = []
        for skill in v:
            normalized = skill.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(skill.strip())
        return result[:MAX_PRIMARY_SKILLS]

    @field_validator("recent_job_titles", mode="before")
    @classmethod
    def clean_titles(cls, v: list[str] | None) -> list[str]:
        if not v:
            return []
        # Dedupe and limit to configured cap
        seen: set[str] = set()
        result: list[str] = []
        for title in v:
            normalized = title.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(title.strip())
        return result[:MAX_RECENT_TITLES]

    @field_validator("certifications", mode="before")
    @classmethod
    def clean_certifications(cls, v: list[str] | None) -> list[str] | None:
        if not v:
            return None
        # Dedupe and limit to configured cap
        seen: set[str] = set()
        result: list[str] = []
        for cert in v:
            normalized = cert.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(cert.strip())
        return result[:MAX_CERTIFICATIONS] if result else None

    @field_validator("industries", mode="before")
    @classmethod
    def clean_industries(cls, v: list[str] | None) -> list[str] | None:
        if not v:
            return None
        # Dedupe and limit to configured cap
        seen: set[str] = set()
        result: list[str] = []
        for industry in v:
            normalized = industry.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(industry.strip())
        return result[:MAX_INDUSTRIES] if result else None

    def has_required_fields(self) -> bool:
        """Check if extraction has minimum required fields."""
        return len(self.primary_skills) >= 1 or len(self.recent_job_titles) >= 1

    def to_props(self) -> dict:
        """Convert to node props dict for storage."""
        props: dict = {
            "primary_skills": self.primary_skills,
            "recent_job_titles": self.recent_job_titles,
        }
        if self.years_experience_total is not None:
            props["years_experience_total"] = self.years_experience_total
        if self.certifications:
            props["certifications"] = self.certifications
        if self.industries:
            props["industries"] = self.industries
        return props


class ExtractionStatus(BaseModel):
    """Extraction status metadata stored in node props."""

    status: Literal["queued", "processing", "ready", "failed", "skipped"] = "queued"
    error: str | None = None
    confidence: float | None = None
    extracted_at: str | None = None
    extraction_version: str | None = None
    model_used: str | None = None

    def to_props(self) -> dict:
        """Convert to node props dict for storage."""
        return {
            "extraction_status": self.status,
            "extraction_error": self.error,
            "extraction_confidence": self.confidence,
            "extracted_at": self.extracted_at,
            "extraction_version": self.extraction_version,
            "extraction_model": self.model_used,
        }
