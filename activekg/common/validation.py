"""Comprehensive input validation for all API endpoints."""

import re
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from enum import StrEnum as StrEnum
else:  # pragma: no cover - runtime fallback for Python < 3.11
    try:
        from enum import StrEnum as StrEnum  # Python 3.11+
    except Exception:

        class StrEnum(str, Enum):  # noqa: UP042
            """Fallback StrEnum for Python < 3.11."""

            pass


class SeniorityLevel(StrEnum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"


class SearchFilters(BaseModel):
    """Validated search filters."""

    min_experience: int | None = Field(None, ge=0, le=50, description="Minimum years of experience")
    max_experience: int | None = Field(None, ge=0, le=50, description="Maximum years of experience")
    seniority_levels: list[SeniorityLevel] | None = Field(
        None, description="Required seniority levels"
    )
    required_skills: list[str] | None = Field(None, max_length=20, description="Required skills")
    excluded_skills: list[str] | None = Field(None, max_length=10, description="Skills to exclude")

    @field_validator("required_skills", "excluded_skills")
    @classmethod
    def validate_skills(cls, v):
        if v is None:
            return v
        sanitized = []
        for skill in v:
            if not isinstance(skill, str):
                continue
            clean_skill = re.sub(r"[^a-zA-Z0-9\s\-\+\#\.]", "", skill.strip())
            if clean_skill and len(clean_skill) <= 50:
                sanitized.append(clean_skill)
        return sanitized[:20] if sanitized else None

    @field_validator("max_experience")
    @classmethod
    def validate_experience_range(cls, v, info):
        if (
            v is not None
            and "min_experience" in info.data
            and info.data["min_experience"] is not None
        ):
            if v < info.data["min_experience"]:
                raise ValueError("max_experience must be greater than or equal to min_experience")
        return v


class SearchRequest(BaseModel):
    """Validated search request."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    num_results: int = Field(10, ge=1, le=100, description="Number of results to return")
    filters: SearchFilters | None = Field(None, description="Search filters")
    include_debug: bool = Field(False, description="Include debug information in response")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        clean_query = re.sub(r"\s+", " ", v.strip())
        dangerous_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"eval\s*\(",
            r"document\.",
            r"window\.",
        ]
        for pattern in dangerous_patterns:
            clean_query = re.sub(pattern, "", clean_query, flags=re.IGNORECASE)
        if not clean_query:
            raise ValueError("Query cannot be empty after sanitization")
        return clean_query


class KGSearchRequest(BaseModel):
    """Knowledge Graph semantic search request."""

    query: str = Field(..., min_length=1, max_length=2000, description="Semantic search query text")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    metadata_filters: dict[str, Any] | None = Field(
        None, description="Simple equality filters (key-value pairs)"
    )
    compound_filter: dict[str, Any] | None = Field(
        None, description="JSONB containment filter for nested/typed queries"
    )
    tenant_id: str | None = Field(None, max_length=100, description="Tenant ID for multi-tenancy")
    use_weighted_score: bool = Field(
        False, description="Apply recency/drift weighting (default: False)"
    )
    use_hybrid: bool = Field(
        False, description="Use hybrid BM25+vector search with score fusion (default: False)"
    )
    use_reranker: bool = Field(
        True, description="Apply cross-encoder reranking to hybrid results (default: True)"
    )
    decay_lambda: float = Field(
        0.01, ge=0.0, le=1.0, description="Age decay rate (default: 0.01 = ~1% per day)"
    )
    drift_beta: float = Field(
        0.1, ge=0.0, le=1.0, description="Drift penalty weight (default: 0.1)"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        clean_query = re.sub(r"\s+", " ", v.strip())
        if not clean_query:
            raise ValueError("Query cannot be empty")
        return clean_query


class IndexBuildRequest(BaseModel):
    """Validated index build request."""

    data_source: str = Field(..., min_length=1, max_length=500, description="Path to the data file")
    force_rebuild: bool = Field(False, description="Force rebuild even if indexes exist")
    backup_existing: bool = Field(True, description="Backup existing indexes before rebuild")

    @field_validator("data_source")
    @classmethod
    def validate_data_source(cls, v):
        clean_path = v.strip()
        if ".." in clean_path or clean_path.startswith("/"):
            raise ValueError("Invalid data source path")


class AskRequest(BaseModel):
    """LLM-powered Q&A request with grounded citations."""

    question: str = Field(
        ..., min_length=1, max_length=1000, description="Question to answer using KG context"
    )
    max_results: int | None = Field(5, ge=1, le=20, description="Max context nodes to retrieve")
    tenant_id: str | None = Field(None, max_length=100, description="Tenant ID for multi-tenancy")
    use_weighted_score: bool = Field(
        True, description="Use recency/drift weighting for context (default: True)"
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        clean_question = re.sub(r"\s+", " ", v.strip())
        if not clean_question:
            raise ValueError("Question cannot be empty")
        return clean_question


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    components: dict[str, dict[str, Any]]
    llm_backend: str | None = None
    llm_model: str | None = None


class MetricsResponse(BaseModel):
    counters: dict[str, float]
    gauges: dict[str, float]
    histograms: dict[str, dict[str, float]]
    timestamp: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict[str, Any] | None = None
    timestamp: str
    request_id: str | None = None


def validate_pagination(offset: int = 0, limit: int = 10) -> tuple[int, int]:
    offset = max(0, min(offset, 10000))
    limit = max(1, min(limit, 100))
    return offset, limit


def sanitize_text_input(text: str, max_length: int = 1000) -> str:
    if not isinstance(text, str):
        return ""
    clean_text = text.strip()[:max_length]
    clean_text = re.sub(r"<[^>]+>", "", clean_text)
    clean_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", clean_text)
    return clean_text


def validate_document_structure(doc: dict[str, Any]) -> bool:
    required_fields = ["id", "name"]
    for field in required_fields:
        if field not in doc:
            return False
    if not isinstance(doc["id"], str) or len(doc["id"]) > 100:
        return False
    if not isinstance(doc["name"], str) or len(doc["name"]) > 200:
        return False
    if "experience_years" in doc:
        if not isinstance(doc["experience_years"], (int, float)) or doc["experience_years"] < 0:
            return False
    if "skills" in doc:
        if not isinstance(doc["skills"], list):
            return False
        for skill in doc["skills"]:
            if not isinstance(skill, str) or len(skill) > 100:
                return False
    return True


class NodeCreate(BaseModel):
    """Validated node creation request.

    Security note:
        When JWT_ENABLED=true, tenant_id from JWT claims overrides this field.
        The tenant_id field is only used in dev mode (JWT_ENABLED=false).
    """

    classes: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Node class labels (e.g., ['Person', 'Employee'])",
    )
    props: dict[str, Any] = Field(..., description="Node properties (arbitrary JSON)")
    payload_ref: str | None = Field(
        None, max_length=500, description="External payload reference (URL, S3 key, etc.)"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    refresh_policy: dict[str, Any] = Field(
        default_factory=dict, description="Auto-refresh policy configuration"
    )
    triggers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Trigger configs to activate on embedding updates (e.g., [{name, threshold}])",
    )
    tenant_id: str | None = Field(
        None,
        max_length=100,
        description="Tenant ID (dev mode only, overridden by JWT in production)",
    )
    extract_before_embed: bool | None = Field(
        None,
        description="If true, extract structured fields before embedding. "
        "If false, embed immediately and extract async. "
        "Defaults to EXTRACTION_MODE env var (async if unset).",
    )

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, v):
        if not v:
            raise ValueError("At least one class label is required")
        # Limit each class name length
        for class_name in v:
            if not isinstance(class_name, str) or len(class_name) > 100:
                raise ValueError("Class names must be strings under 100 characters")
        return v[:10]  # Max 10 classes

    @field_validator("payload_ref")
    @classmethod
    def validate_payload_ref(cls, v):
        if v is None:
            return v
        # Basic validation for common payload ref formats
        if len(v) > 500:
            raise ValueError("payload_ref must be under 500 characters")
        return v.strip()

    @field_validator("triggers")
    @classmethod
    def normalize_triggers(cls, v):
        """Accept list[str] or list[dict] and normalize to [{name, threshold}]."""
        if not v:
            return []
        out: list[dict[str, Any]] = []
        for item in v:
            if isinstance(item, str):
                name = item.strip()
                if name:
                    out.append({"name": name, "threshold": 0.85})
            elif isinstance(item, dict):
                name = item.get("name")
                if not name or not isinstance(name, str):
                    continue
                thr = item.get("threshold", 0.85)
                try:
                    thrf = float(thr)
                except Exception:
                    thrf = 0.85
                out.append({"name": name, "threshold": thrf})
        # Dedupe by name
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for t in out:
            if t["name"] in seen:
                continue
            seen.add(t["name"])
            deduped.append(t)
        return deduped


class NodeBatchCreate(BaseModel):
    """Validated batch node creation request."""

    nodes: list[NodeCreate] = Field(..., min_length=1, max_length=500)
    tenant_id: str | None = Field(
        None,
        max_length=100,
        description="Tenant ID (dev mode only, overridden by JWT in production)",
    )
    continue_on_error: bool = Field(True, description="If false, abort on first error")
    extract_before_embed: bool | None = Field(
        None,
        description="If true, extract structured fields before embedding for all nodes. "
        "Overrides per-node setting. Defaults to EXTRACTION_MODE env var.",
    )


class EdgeCreate(BaseModel):
    """Validated edge creation request.

    Security note:
        When JWT_ENABLED=true, tenant_id from JWT claims overrides this field.
        The tenant_id field is only used in dev mode (JWT_ENABLED=false).
    """

    src: str = Field(..., min_length=1, max_length=100, description="Source node ID")
    dst: str = Field(..., min_length=1, max_length=100, description="Target node ID")
    rel: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Relationship type (e.g., 'WORKS_WITH', 'REPORTS_TO')",
    )
    props: dict[str, Any] = Field(
        default_factory=dict, description="Edge properties (arbitrary JSON)"
    )
    tenant_id: str | None = Field(
        None,
        max_length=100,
        description="Tenant ID (dev mode only, overridden by JWT in production)",
    )

    @field_validator("src", "dst")
    @classmethod
    def validate_node_ids(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Node IDs must be non-empty strings")
        if len(v) > 100:
            raise ValueError("Node IDs must be under 100 characters")
        return v.strip()

    @field_validator("rel")
    @classmethod
    def validate_rel(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Relationship type must be a non-empty string")
        if len(v) > 100:
            raise ValueError("Relationship type must be under 100 characters")
        # Uppercase convention validation (optional, can be removed if not needed)
        return v.strip()
