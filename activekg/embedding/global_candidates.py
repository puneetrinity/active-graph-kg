"""Embedding producer for global_candidates (#29 slice 4, Stage-4 blob text).

The Redis-queue worker only embeds graph ``nodes`` — ``global_candidates``
rows had NO producer, so every row sat at the default
``embedding_status='queued'`` forever and vector retrieval over the platform
pool was structurally impossible. This sweep polls for queued rows in
batches, builds the profile text with the SAME builder the search endpoint
conceptually pairs with, embeds via the worker's provider, and marks rows
ready/failed. Runs opportunistically inside the worker loop; no Redis
plumbing needed.

Stage-4: the embed text is blob-enriched. Thin normalized fields alone
(headline-only for sourced rows — the signal->global mirror never writes
role_family/skills) ordered an 815-row segment so badly that half the
fit-top-100 fell below vector rank 500 (Stage-3 offline gate). The sweep now
joins the freshest tenant-side crustdata blob via the #29 link column and
embeds titles/companies/education too. EMBED_VERSION stamps each row; bumping
it makes the sweep re-embed the whole pool (blanket re-embed = version drain,
no manual backfill).
"""

from __future__ import annotations

import time
from typing import Any

import psycopg

from activekg.common.logger import get_enhanced_logger

logger = get_enhanced_logger(__name__)

# Bump on ANY change to build_candidate_embedding_text (or the underlying
# model): the sweep re-embeds every ready/skipped row with a lower version.
EMBED_VERSION = 3

# The encoder truncates long inputs anyway (MiniLM ~256 wordpieces); this cap
# just keeps pathological blobs from wasting tokenizer time. Order the parts
# most-informative-first so truncation costs the least.
_MAX_TEXT_CHARS = 1200


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


def _employment_parts(profile: dict[str, Any]) -> list[str]:
    exp = _as_dict(_as_dict(profile.get("experience")).get("employment_details"))
    parts: list[str] = []
    current = [e for e in _as_list(exp.get("current")) if isinstance(e, dict)]
    cur = [
        f"{e['title']} at {e['name']}" if e.get("name") else str(e["title"])
        for e in current[:3]
        if e.get("title")
    ]
    if cur:
        parts.append("; ".join(cur))
    past = [e for e in _as_list(exp.get("past")) if isinstance(e, dict)]
    prev = [
        f"{e['title']} at {e['name']}" if e.get("name") else str(e["title"])
        for e in past[:5]
        if e.get("title")
    ]
    if prev:
        parts.append("previously: " + "; ".join(prev))
    return parts


def _blob_parts(profile: Any, *, have_headline: bool) -> list[str]:
    """Text parts from the tenant-side crustdata blob (nested /person/search
    schema with flat-schema fallbacks). Fail-open: any malformed blob yields
    an empty list and the row embeds with normalized fields only."""
    if not isinstance(profile, dict) or not profile:
        return []
    parts: list[str] = []
    basic = _as_dict(profile.get("basic_profile"))
    if not have_headline:
        headline = basic.get("headline") or profile.get("headline")
        if headline:
            parts.append(str(headline))
    parts.extend(_employment_parts(profile))
    if not parts:
        # Flat-schema fallback: at least the current title.
        ct = basic.get("current_title") or profile.get("current_title")
        if ct:
            parts.append(str(ct))
    schools = _as_list(_as_dict(profile.get("education")).get("schools"))
    edu = [
        f"{s['degree']}, {s['school']}" if s.get("school") else str(s["degree"])
        for s in schools[:2]
        if isinstance(s, dict) and s.get("degree")
    ]
    if edu:
        parts.append("education: " + "; ".join(edu))
    return parts


def build_candidate_embedding_text(row: dict[str, Any]) -> str:
    """ONE text builder shared by this producer and the search endpoint's query
    side: candidates and JD queries must live in the same vector space.
    ``row`` may carry ``crustdata_profile`` (freshest tenant-side blob)."""
    parts: list[str] = []
    for key in ("headline", "role_family", "seniority_band"):
        v = row.get(key)
        if v:
            parts.append(str(v))
    parts.extend(_blob_parts(row.get("crustdata_profile"), have_headline=bool(row.get("headline"))))
    skills = row.get("skills_normalized")
    if skills:
        parts.append("skills: " + ", ".join(skills[:30]))
    loc = ", ".join(
        str(v) for v in [row.get("location_city"), row.get("location_country_code")] if v
    )
    if loc:
        parts.append(loc)
    return ". ".join(parts)[:_MAX_TEXT_CHARS]


_SELECT_COLS = (
    "gc.id, gc.headline, gc.role_family, gc.seniority_band, gc.skills_normalized, "
    "gc.location_city, gc.location_country_code, tc.profile AS crustdata_profile"
)


class GlobalCandidateEmbedder:
    """Batch-sweeps global_candidates rows that still need an embedding."""

    def __init__(
        self,
        dsn: str,
        embedder: Any,
        *,
        batch_size: int = 64,
        sweep_interval_seconds: float = 15.0,
        enabled: bool = True,
    ) -> None:
        self.dsn = dsn
        self.embedder = embedder
        self.batch_size = batch_size
        self.sweep_interval = sweep_interval_seconds
        self.enabled = enabled
        self._last_sweep = 0.0

    def maybe_sweep(self) -> int:
        """Called from the worker loop; rate-limited by sweep_interval."""
        if not self.enabled:
            return 0
        now = time.monotonic()
        if now - self._last_sweep < self.sweep_interval:
            return 0
        self._last_sweep = now
        try:
            return self._sweep_once()
        except Exception as e:  # never take down the node-embedding loop
            logger.error("global_candidates embedding sweep failed", extra={"error": str(e)})
            return 0

    def _sweep_once(self) -> int:
        conn = psycopg.connect(self.dsn, autocommit=False)
        try:
            with conn.cursor() as cur:
                # FOR UPDATE SKIP LOCKED: safe under multiple worker replicas.
                # Freshest tenant blob via the #29 link column (worker runs on
                # the owner DSN, so RLS doesn't hide rows; a missing blob just
                # yields NULL and the row embeds with normalized fields only).
                # Version drain: ready/skipped rows built with an older
                # EMBED_VERSION re-embed after queued rows are served —
                # skipped_empty is included because the blob may now provide
                # text where the normalized fields had none.
                cur.execute(
                    f"""
                    SELECT {_SELECT_COLS}
                    FROM global_candidates gc
                    LEFT JOIN LATERAL (
                        SELECT c.profile
                        FROM candidates c
                        WHERE c.global_candidate_id = gc.id
                          AND c.profile IS NOT NULL
                          AND c.profile <> '{{}}'::jsonb
                        ORDER BY c.updated_at DESC
                        LIMIT 1
                    ) tc ON true
                    WHERE gc.embedding_status = 'queued'
                       OR (
                            gc.embedding_status IN ('ready', 'skipped_empty')
                            AND gc.embed_version < %s
                          )
                    ORDER BY (gc.embedding_status = 'queued') DESC, gc.updated_at ASC
                    LIMIT %s
                    FOR UPDATE OF gc SKIP LOCKED
                    """,
                    (EMBED_VERSION, self.batch_size),
                )
                rows = cur.fetchall()
                if not rows:
                    conn.commit()
                    return 0
                cols = [d.name for d in cur.description]
                dicts = [dict(zip(cols, r, strict=False)) for r in rows]

                texts: list[str] = []
                embeddable: list[dict[str, Any]] = []
                empty_ids: list[str] = []
                for d in dicts:
                    text = build_candidate_embedding_text(d)
                    if text.strip():
                        texts.append(text)
                        embeddable.append(d)
                    else:
                        empty_ids.append(str(d["id"]))

                if empty_ids:
                    # Nothing embeddable — skip, don't loop forever.
                    cur.execute(
                        "UPDATE global_candidates"
                        " SET embedding_status = 'skipped_empty', embed_version = %s"
                        " WHERE id = ANY(%s::uuid[])",
                        (EMBED_VERSION, empty_ids),
                    )

                done = 0
                if embeddable:
                    vectors = self.embedder.encode(texts)
                    for d, vec in zip(embeddable, vectors, strict=False):
                        vec_literal = "[" + ",".join(f"{x:.6f}" for x in vec.tolist()) + "]"
                        cur.execute(
                            "UPDATE global_candidates"
                            " SET embedding = %s::vector, embedding_status = 'ready',"
                            "     embed_version = %s, updated_at = now()"
                            " WHERE id = %s",
                            (vec_literal, EMBED_VERSION, d["id"]),
                        )
                        done += 1

            conn.commit()
            if done or empty_ids:
                logger.info(
                    "global_candidates embedding sweep",
                    extra={"embedded": done, "skipped_empty": len(empty_ids)},
                )
            return done
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
