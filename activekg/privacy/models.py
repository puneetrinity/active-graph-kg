"""Import-light types for the reversible candidate-privacy control plane."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class CandidatePrivacyAction(StrEnum):
    WITHDRAW_GLOBAL_MATCHING = "withdraw_global_matching"
    REQUEST_ERASURE = "request_erasure"


class CandidatePrivacyScope(StrEnum):
    GLOBAL_MATCHING = "global_matching"
    ACTIVE_PROFILE = "active_profile"


class CandidatePrivacyState(StrEnum):
    REQUESTED = "requested"
    VERIFIED = "verified"
    ACTIVE_QUARANTINE = "active_quarantine"
    NEEDS_REVIEW = "needs_review"
    RELEASED = "released"
    SUPERSEDED = "superseded"
    HARD_PURGE_ELIGIBLE = "hard_purge_eligible"


class CandidatePrivacyDecision(StrEnum):
    ALLOW = "allow"
    BLOCK_GLOBAL = "block_global"
    BLOCK_ALL = "block_all"
    REVIEW = "review"

    @property
    def blocks_all(self) -> bool:
        return self in {self.BLOCK_ALL, self.REVIEW}

    @property
    def blocks_global(self) -> bool:
        return self is not self.ALLOW


class CandidatePrivacyAuthorityType(StrEnum):
    VERIFIED_CANDIDATE = "verified_candidate"
    PRIVACY_OPERATOR = "privacy_operator"


class CandidatePrivacyReason(StrEnum):
    CANDIDATE_GLOBAL_OPT_OUT = "candidate_global_opt_out"
    CANDIDATE_ERASURE_REQUEST = "candidate_erasure_request"
    VERIFIED_SUPPORT_REQUEST = "verified_support_request"
    IDENTITY_AMBIGUITY = "identity_ambiguity"
    OPERATOR_CORRECTION = "operator_correction"


class CandidatePrivacyTransition(StrEnum):
    RELEASE = "release"
    MARK_NEEDS_REVIEW = "mark_needs_review"


@dataclass(frozen=True)
class CanonicalSubject:
    global_candidate_id: UUID | None = None
    candidate_tenant_id: str | None = None
    candidate_id: UUID | None = None
    needs_review: bool = False


@dataclass(frozen=True)
class DirectiveRecord:
    directive_id: UUID
    action: CandidatePrivacyAction
    scope: CandidatePrivacyScope
    state: CandidatePrivacyState
    version: int
    effective_at: datetime
    decision: CandidatePrivacyDecision


def decision_for(
    state: CandidatePrivacyState | str,
    action: CandidatePrivacyAction | str,
) -> CandidatePrivacyDecision:
    normalized_state = CandidatePrivacyState(state)
    normalized_action = CandidatePrivacyAction(action)
    if normalized_state in {
        CandidatePrivacyState.NEEDS_REVIEW,
        CandidatePrivacyState.HARD_PURGE_ELIGIBLE,
    }:
        return CandidatePrivacyDecision.REVIEW
    if normalized_state is CandidatePrivacyState.ACTIVE_QUARANTINE:
        if normalized_action is CandidatePrivacyAction.REQUEST_ERASURE:
            return CandidatePrivacyDecision.BLOCK_ALL
        return CandidatePrivacyDecision.BLOCK_GLOBAL
    return CandidatePrivacyDecision.ALLOW
