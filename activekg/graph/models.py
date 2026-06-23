from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class Node:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str | None = None
    classes: list[str] = field(default_factory=list)
    props: dict[str, Any] = field(default_factory=dict)
    payload_ref: str | None = None
    embedding: np.ndarray | None = None
    embedding_status: str | None = None
    embedding_error: str | None = None
    embedding_attempts: int | None = None
    embedding_updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    refresh_policy: dict[str, Any] = field(default_factory=dict)
    triggers: list[dict[str, Any]] = field(default_factory=list)
    version: int = 1
    # Active refresh tracking (explicit columns for query performance)
    last_refreshed: datetime | None = None
    drift_score: float | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Edge:
    src: str
    rel: str
    dst: str
    props: dict[str, Any] = field(default_factory=dict)
    tenant_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Candidate:
    """Canonical, source-independent candidate owned by ActiveKG."""

    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str | None = None
    scope: str = "shared"
    display_name: str | None = None
    primary_email: str | None = None
    primary_phone: str | None = None
    props: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    node_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CandidateIdentifier:
    """A normalized identifier (email, phone, URL, upstream id) attached to a
    canonical candidate. ``value_normalized`` is the merge key; ``value_raw``
    preserves what the upstream actually sent."""

    candidate_id: str
    identifier_type: str
    value_normalized: str
    value_raw: str | None = None
    tenant_id: str | None = None
    source: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CandidateSourceRecord:
    """A single upstream record (VantaHire application, Signal profile, etc.)
    attached to a canonical candidate. Preserves full provenance."""

    candidate_id: str
    source: str
    source_record_type: str
    source_record_id: str
    tenant_id: str | None = None
    source_url: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    payload_ref: str | None = None
    fetched_at: datetime | None = None
    # Structured VantaHire provenance — populated for source='vantahire' records
    # so downstream Talent Search can filter by org/job/recruiter without JSONB scans.
    org_id: str | None = None
    job_id: str | None = None
    effective_recruiter_id: str | None = None
    created_by_user_id: str | None = None
    resume_source: str | None = None
    # Structured Signal tags — populated for source='signal' records so tag-based
    # candidate search uses a GIN index rather than scanning JSONB payloads.
    job_tags: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SignalTagSearchRow:
    """A single result from a Signal tag-based candidate search."""

    candidate_id: str
    display_name: str | None
    primary_email: str | None
    scope: str
    tenant_id: str | None
    signal_source_record_id: str
    stored_tags: list[str]
    overlap_count: int
    overlap_ratio: float
