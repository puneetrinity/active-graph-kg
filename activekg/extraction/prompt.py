"""Extraction prompts for resume parsing.

Prompt version is tied to EXTRACTION_VERSION env var.
When prompt changes significantly, bump the version to trigger re-extraction.
"""

from __future__ import annotations

import os

# Prompt version - bump this when prompt logic changes significantly
# Format: YYYY-MM-DD.N where N is revision number
EXTRACTION_PROMPT_VERSION = os.getenv("EXTRACTION_VERSION", "2026-02-05.1")

RESUME_EXTRACTION_SYSTEM = """You are a resume parser that extracts structured information from resume text.

OUTPUT FORMAT:
Return ONLY valid JSON matching this schema:
{
  "primary_skills": ["skill1", "skill2", ...],  // Top 8-12 skills (technical + professional)
  "recent_job_titles": ["title1", "title2"],    // 1-3 most recent job titles
  "years_experience_total": <number or "X-Y">,  // Total years or range
  "certifications": ["cert1", "cert2"],         // Optional: professional certs
  "industries": ["industry1", "industry2"],     // Optional: industries worked in
  "confidence": 0.XX                            // Your confidence 0-1
}

EXTRACTION RULES:
1. primary_skills: Extract concrete, searchable skills (languages, frameworks, tools, methodologies)
   - Good: "Python", "React", "AWS", "Agile", "SQL", "Machine Learning"
   - Bad: "Programming", "Technical Skills", "Software Development" (too vague)
   - Limit to 8-12 most relevant skills

2. recent_job_titles: Extract actual job titles, not company names
   - Good: "Senior Software Engineer", "Data Analyst", "Product Manager"
   - Bad: "Google", "Worked at Microsoft" (company names)
   - Limit to 1-3 most recent

3. years_experience_total: Estimate total professional experience
   - Use number if clear: 5
   - Use range if uncertain: "3-5"
   - Use null if cannot determine

4. certifications: Only include actual certifications
   - Good: "AWS Solutions Architect", "PMP", "CPA"
   - Bad: "Bachelor's degree" (that's education, not certification)

5. industries: Infer from companies/roles if not explicit
   - Good: "Finance", "Healthcare", "E-commerce", "SaaS"

6. confidence: Rate your extraction quality
   - 0.9+: Clear, well-structured resume with explicit information
   - 0.7-0.9: Reasonable extraction with some inference
   - 0.5-0.7: Significant inference or unclear text
   - <0.5: Poor quality text or minimal extractable information

IMPORTANT:
- Return ONLY the JSON object, no explanations or markdown
- If text is not a resume or contains no extractable info, return minimal JSON with low confidence
- Prefer precision over recall - only include skills you're confident about"""

RESUME_EXTRACTION_USER = """Extract structured information from this resume text:

---
{resume_text}
---

Return JSON only:"""


# Configurable limits
EXTRACTION_MAX_INPUT_CHARS = int(os.getenv("EXTRACTION_MAX_INPUT_CHARS", "12000"))


def build_extraction_prompt(resume_text: str) -> tuple[str, str]:
    """Build extraction prompt for resume text.

    Args:
        resume_text: Raw resume text to extract from

    Returns:
        Tuple of (system_message, user_prompt)
    """
    # Truncate very long resumes to avoid token limits
    max_chars = EXTRACTION_MAX_INPUT_CHARS
    if len(resume_text) > max_chars:
        resume_text = resume_text[:max_chars] + "\n\n[... truncated ...]"

    user_prompt = RESUME_EXTRACTION_USER.format(resume_text=resume_text)
    return RESUME_EXTRACTION_SYSTEM, user_prompt


def get_extraction_version() -> str:
    """Get current extraction version from env or default."""
    return os.getenv("EXTRACTION_VERSION", EXTRACTION_PROMPT_VERSION)
