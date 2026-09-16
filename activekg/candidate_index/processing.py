"""Generation-bound orchestration: short DB calls, cancellable work, fenced commit.

No timer or client starts on import. Legacy Redis processing does not call this
module; the two worker entrypoints adopt its explicit lifecycle separately.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from activekg.candidate_index.contracts import (
    ACTIVE_SOURCE_KINDS,
    IndexContractError,
    IndexPolicy,
    IndexTunables,
    canonical_json,
    chunk_manifest,
    sha256,
    validate_vectors,
)
from activekg.candidate_index.providers import (
    WorkRefused,
    embed_once,
    extract_once,
    parse_original,
    remaining,
)
from activekg.candidate_index.repository import IndexRepository

PROFESSIONAL_FIELDS = (
    "current_title",
    "primary_titles",
    "skills_raw",
    "skills_normalized",
    "domains",
    "functions",
    "certifications",
    "industries",
    "primary_skills",
    "recent_job_titles",
)


@dataclass(frozen=True)
class Extraction:
    namespace: dict[str, Any]
    evidence: list[dict[str, str]]
    confidence: float
    professional_text: str
    chunks: list[dict[str, Any]]
    model_id: str

    def arguments(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _bounded_value(value: Any, depth: int = 0) -> None:
    if depth > 4:
        raise WorkRefused("incomplete")
    if value is None or type(value) is bool:
        return
    if isinstance(value, str):
        if len(value) > 512 or re.search(r"[\x00-\x1f\x7f]", value):
            raise WorkRefused("incomplete")
        return
    if type(value) in (int, float):
        if abs(value) > 1000 or not math.isfinite(value):
            raise WorkRefused("incomplete")
        return
    if isinstance(value, list):
        if len(value) > 100:
            raise WorkRefused("incomplete")
        for item in value:
            _bounded_value(item, depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 40:
            raise WorkRefused("incomplete")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise WorkRefused("incomplete")
            _bounded_value(item, depth + 1)
        return
    raise WorkRefused("incomplete")


def _assemble(
    namespace: dict[str, Any],
    fields: tuple[str, ...],
    text: str,
    confidence: float,
    policy: IndexPolicy,
    model: str,
) -> Extraction:
    evidence = []
    lines = []
    for field in fields:
        value = namespace.get(field)
        values = [value] if isinstance(value, str) else value or []
        if values:
            lines.append(field + ": " + "; ".join(values))
        for item in values:
            if len(evidence) < 32 and 2 <= len(item) <= 512 and item.lower() in text.lower():
                evidence.append({"field": field, "text": item})
    if not evidence:
        raise WorkRefused("incomplete")
    # Only professional fields. No raw document, display name, email, phone,
    # LinkedIn URL or applicant locator is copied into search chunks.
    professional = "\n".join(lines)
    try:
        chunks = chunk_manifest(professional, policy)
    except IndexContractError as exc:
        raise WorkRefused(
            "chunk_overflow" if str(exc) == "candidate_index_chunk_overflow" else "incomplete"
        ) from None
    return Extraction(namespace, evidence, confidence, professional, chunks, model)


def resume_extraction(
    raw: dict[str, Any], text: str, policy: IndexPolicy, model: str
) -> Extraction:
    from pydantic import ValidationError

    from activekg.extraction import schema

    _bounded_value(raw)
    if not isinstance(raw, dict) or set(raw) - set(schema.ExtractionResult.model_fields):
        raise WorkRefused("incomplete")
    if len(canonical_json(raw).encode()) > 65536:
        raise WorkRefused("incomplete")
    for field in ("current_title", "seniority"):
        value = raw.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise WorkRefused("incomplete")
    if (
        raw.get("seniority") is not None
        and raw["seniority"].strip().lower() not in schema._SENIORITY_ALLOWED
    ):
        raise WorkRefused("incomplete")
    for field in ("total_years_experience", "years_experience_total"):
        value = raw.get(field)
        if value is None:
            continue
        if type(value) in (int, float) and 0 <= value <= 100:
            continue
        if isinstance(value, str) and re.fullmatch(r"[0-9]{1,2}(?:-[0-9]{1,2})?", value):
            ends = [int(n) for n in value.split("-")]
            if len(ends) == 1 or ends[0] <= ends[1]:
                continue
        raise WorkRefused("incomplete")
    list_limits = {
        "primary_titles": schema.MAX_PRIMARY_TITLES,
        "skills_raw": schema.MAX_SKILLS_RAW,
        "skills_normalized": schema.MAX_SKILLS_NORMALIZED,
        "domains": schema.MAX_DOMAINS,
        "functions": schema.MAX_FUNCTIONS,
        "certifications": schema.MAX_CERTIFICATIONS,
        "industries": schema.MAX_INDUSTRIES,
        "primary_skills": schema.MAX_PRIMARY_SKILLS,
        "recent_job_titles": schema.MAX_RECENT_TITLES,
    }
    for field, bound in list_limits.items():
        values = raw.get(field)
        if values is None and field in {"certifications", "industries"}:
            continue
        if field in raw and (
            not isinstance(values, list)
            or len(values) > bound
            or any(not isinstance(v, str) or not v.strip() for v in values)
        ):
            raise WorkRefused("incomplete")
    location = raw.get("location")
    if location is not None and (
        not isinstance(location, dict) or set(location) - set(schema.LocationInfo.model_fields)
    ):
        raise WorkRefused("incomplete")
    years = raw.get("years_by_skill")
    if years is not None and (
        not isinstance(years, dict)
        or len(years) > schema.MAX_YEARS_BY_SKILL
        or any(type(v) not in (int, float) or not 0 <= v <= 100 for v in years.values())
    ):
        raise WorkRefused("incomplete")
    if type(raw.get("confidence")) not in (int, float):
        raise WorkRefused("incomplete")
    if raw["confidence"] < policy.minimum_confidence:
        raise WorkRefused("low_confidence")
    try:
        parsed = schema.ExtractionResult.model_validate(raw, strict=True)
    except (ValidationError, ValueError, TypeError, AttributeError):
        raise WorkRefused("incomplete") from None
    return _assemble(
        parsed.model_dump(mode="json"), PROFESSIONAL_FIELDS, text, parsed.confidence, policy, model
    )


def profile_extraction(profile: dict[str, Any], policy: IndexPolicy) -> Extraction:
    from pydantic import ValidationError

    from activekg.api.candidate_consent import Profile

    try:
        approved = Profile.model_validate(profile)
    except (ValidationError, ValueError, TypeError):
        raise WorkRefused("incomplete") from None
    # Immutable approved snapshot is input only. Missing professional fields are
    # not invented and a name/location/URL-only snapshot cannot be indexed.
    namespace = {"headline": approved.headline, "skills": list(approved.skills)}
    return _assemble(
        namespace,
        ("headline", "skills"),
        canonical_json(profile),
        1.0,
        policy,
        "deterministic-profile:v1",
    )


class GenerationWorker:
    def __init__(
        self,
        repository: IndexRepository,
        policy: IndexPolicy,
        *,
        api_key: str | None = None,
        transport: Any = None,
    ) -> None:
        self.repository = repository
        self.policy = policy
        self.tunables: IndexTunables = repository.tunables
        self.api_key = os.environ.get("GROQ_API_KEY", "") if api_key is None else api_key
        self.transport = transport

    async def _authority(self, dispatch: dict[str, Any], deadline: float) -> None:
        remaining(deadline)
        try:
            current = await asyncio.to_thread(
                self.repository.status, dispatch["scope_key"], dispatch["source_id"]
            )
        except Exception:
            raise WorkRefused("privacy_unavailable") from None
        if not current or current.get("eligible") is not True:
            reason = (current or {}).get("reason")
            code = {
                "privacy_review": "privacy_review",
                "privacy_unavailable": "privacy_unavailable",
                "consent_changed": "consent_changed",
            }.get(reason, "privacy_restricted")
            raise WorkRefused(code)
        if current.get("generation_id") != dispatch["generation_id"]:
            raise WorkRefused("consent_changed")
        remaining(deadline)

    async def process(self, lease: dict[str, Any]) -> str:
        started = time.monotonic()
        try:
            dispatch = await asyncio.to_thread(self.repository.reserve, lease, self.policy)
        except Exception:
            return "privacy_unavailable"  # No external work, no error string/identity.
        if dispatch is None:
            return "not_reserved"
        try:
            if (
                dispatch["source_kind"] not in ACTIVE_SOURCE_KINDS
                or IndexPolicy(**dispatch["policy"]) != self.policy
                or dispatch["policy_sha256"] != self.policy.digest
            ):
                raise WorkRefused("policy_mismatch")
            remaining_ms = dispatch.get("remaining_ms")
            if type(remaining_ms) is not int or remaining_ms <= 0:
                raise WorkRefused("dispatch_exhausted")
            deadline = started + remaining_ms / 1000
            await self._authority(dispatch, deadline)
            if dispatch["stage"] == "extract":
                deterministic = dispatch["content_kind"] == "approved_profile"
                if deterministic:
                    if (
                        dispatch["source_kind"] != "candidate_consent"
                        or dispatch["source_manifest"].get("resume_version_id") is not None
                    ):
                        raise WorkRefused("policy_mismatch")
                    extraction = profile_extraction(dispatch["approved_profile"], self.policy)
                    parsed_text = None
                else:
                    attempt = dispatch["attempt"]
                    if (
                        type(attempt) is not int
                        or not 1 <= attempt <= self.tunables.extraction_attempts
                    ):
                        raise WorkRefused("dispatch_exhausted")
                    parsed_text = None
                    text = dispatch.get("professional_text")
                    if text is None:
                        if dispatch["content_kind"] != "original_bytes":
                            raise WorkRefused("source_missing")
                        parsed_text = await parse_original(
                            dispatch["original_bytes"], deadline=deadline
                        )
                        text = parsed_text
                    if (
                        not isinstance(text, str)
                        or not text.strip()
                        or len(text.encode()) > 2 * 1024 * 1024
                    ):
                        raise WorkRefused("incomplete")
                    # Hashes are checked at source capture; do not compare the
                    # consent profile hash against its separate selected resume.
                    if (
                        dispatch["source_kind"] == "organization_application"
                        and dispatch["content_kind"] == "pinned_text"
                    ):
                        if sha256(text) != dispatch["source_manifest"]["source_hash"]:
                            raise WorkRefused("incomplete")
                    model = (
                        self.policy.primary_model_id
                        if attempt == 1
                        else self.policy.fallback_model_id
                    )
                    raw = await extract_once(
                        text,
                        model,
                        api_key=self.api_key,
                        deadline=deadline,
                        before_send=lambda: self._authority(dispatch, deadline),
                        transport=self.transport,
                    )
                    extraction = resume_extraction(raw, text, self.policy, model)
                remaining(deadline)
                completed = await asyncio.to_thread(
                    self.repository.complete_extract,
                    dispatch,
                    **extraction.arguments(),
                    parsed_text=parsed_text,
                )
            elif dispatch["stage"] == "embed":
                if (
                    type(dispatch["attempt"]) is not int
                    or not 1 <= dispatch["attempt"] <= self.tunables.embedding_attempts
                ):
                    raise WorkRefused("dispatch_exhausted")
                chunks = dispatch["extraction"]["chunks"]
                texts = [c["text"] for c in chunks]
                # Validate the full ordered manifest again before the model.
                if chunk_manifest("".join(texts), self.policy) != chunks:
                    raise WorkRefused("incomplete")
                vectors = await embed_once(
                    texts,
                    self.policy.embedding_model_id,
                    self.policy.embedding_artifact_revision,
                    deadline=deadline,
                )
                remaining(deadline)
                completed = await asyncio.to_thread(
                    self.repository.complete_embed, dispatch, validate_vectors(vectors, len(chunks))
                )
            else:
                raise WorkRefused("policy_mismatch")
            return "completed" if completed else "fenced"
        except WorkRefused as exc:
            reason, retry_ms = exc.code, exc.retry_ms
        except (IndexContractError, ValueError, TypeError, KeyError, UnicodeError):
            reason, retry_ms = "incomplete", 0
        except Exception:
            reason, retry_ms = "provider_unavailable", 1000
        try:
            await asyncio.to_thread(self.repository.fail, dispatch, reason, retry_ms)
        except Exception:
            return "settlement_unavailable"  # Reserved ambiguity is retained, never reset.
        return reason

    async def tick(self, stage: str) -> list[str]:
        if stage not in {"extract", "embed"}:
            raise ValueError("candidate_index_stage_invalid")
        # Resume only already accepted sources, not a historical reindex.
        if stage == "extract":
            await asyncio.to_thread(self.repository.catchup)
            await asyncio.to_thread(self.repository.catchup, maintenance=True)
        leases = []
        for _ in range(min(self.tunables.claim_limit, self.tunables.concurrent_per_stage)):
            claimed = await asyncio.to_thread(self.repository.claim, stage)
            if not claimed:
                break
            leases.extend(claimed)
        return await asyncio.gather(*(self.process(lease) for lease in leases))

    async def run(
        self, stage: str, stop: asyncio.Event, observed: Callable[[str], None] | None = None
    ) -> None:
        while not stop.is_set():
            try:
                outcomes = await self.tick(stage)
                status = "error" if "settlement_unavailable" in outcomes else "ready"
            except Exception:
                # No loop-level retry reset and no source/exception logging.
                status = "error"
            if observed is not None:
                observed(status)
            try:
                await asyncio.wait_for(stop.wait(), self.tunables.poll_ms / 1000)
            except TimeoutError:
                pass


def worker_configuration(stage: str):
    """Refuse bad placement/config before either entrypoint opens a connection.

    Disabled lanes do not demand a model artifact or claim historical work.
    These are configuration checks, not provider calls or model warmups.
    """
    from activekg.candidate_index.admission import configuration, require_placement
    from activekg.candidate_index.providers import cached_embedding_path

    if stage not in {"extract", "embed"}:
        raise IndexContractError("candidate_index_stage_invalid")
    if not require_placement(stage):
        return None
    config = configuration()
    if stage == "extract" and not os.environ.get("GROQ_API_KEY", "").strip():
        raise IndexContractError("candidate_index_extraction_key_missing")
    if stage == "embed":
        try:
            cached_embedding_path(
                config.policy.embedding_model_id, config.policy.embedding_artifact_revision
            )
        except WorkRefused:
            raise IndexContractError("candidate_index_embedding_artifact_missing") from None
    return config


class GenerationRuntime:
    """One explicit, cancellable generation loop alongside a legacy Redis loop.

    Constructing this object starts nothing. Signal handlers request cancellation
    immediately; close joins the loop, child reaping, and bounded SQL settlement.
    No daemon thread is treated as successfully stopped merely because it was
    asked to stop. The in-memory status never performs IO or exposes source data.
    """

    def __init__(self, worker: GenerationWorker, stage: str) -> None:
        if stage not in {"extract", "embed"}:
            raise IndexContractError("candidate_index_stage_invalid")
        self.worker = worker
        self.stage = stage
        self._lock = threading.Lock()
        self._started = threading.Event()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._status = "starting"
        self._observed_at: float | None = None

    def _observe(self, status: str) -> None:
        with self._lock:
            self._status = status
            self._observed_at = time.monotonic()

    def status(self) -> str:
        with self._lock:
            status, observed_at = self._status, self._observed_at
        if self._stopping.is_set():
            return "stopped"
        if self._thread is not None and self._started.is_set() and not self._thread.is_alive():
            return "error"
        if observed_at is None:
            return "starting"
        budget_ms = (
            self.worker.tunables.extraction_dispatch_ms
            if self.stage == "extract"
            else self.worker.tunables.embedding_dispatch_ms
        )
        if time.monotonic() - observed_at > (budget_ms + 15000) / 1000:
            return "stale"
        return status

    def start(self) -> None:
        if self._thread is not None or self._stopping.is_set():
            raise IndexContractError("candidate_index_runtime_already_started")

        async def main() -> None:
            self._loop = asyncio.get_running_loop()
            self._task = asyncio.current_task()
            self._started.set()
            if not self._stopping.is_set():
                await self.worker.run(self.stage, asyncio.Event(), self._observe)

        def run() -> None:
            try:
                asyncio.run(main())
            except asyncio.CancelledError:
                pass  # Reserved ambiguity survives cancellation/restart.
            except Exception:
                self._observe("error")
            finally:
                self._loop = None
                self._task = None
                self._started.set()

        self._thread = threading.Thread(
            target=run, name=f"candidate-index-{self.stage}", daemon=True
        )
        self._thread.start()
        if not self._started.wait(3):
            self.request_stop()
            raise IndexContractError("candidate_index_runtime_start_timeout")

    def request_stop(self) -> None:
        if self._stopping.is_set():
            return  # Do not interrupt child reaping with a second cancellation.
        self._stopping.set()
        loop, task = self._loop, self._task
        if loop is not None and task is not None:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # A concurrently closed loop has already finished.

    def close(self) -> None:
        self.request_stop()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise IndexContractError("candidate_index_runtime_stop_timeout")
