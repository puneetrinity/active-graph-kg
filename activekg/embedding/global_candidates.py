"""Embedding producer for global_candidates (#29 slice 4).

The Redis-queue worker only embeds graph ``nodes`` — ``global_candidates``
rows had NO producer, so every row sat at the default
``embedding_status='queued'`` forever and vector retrieval over the platform
pool was structurally impossible. This sweep polls for queued rows in
batches, builds the profile text with the SAME builder the search endpoint
conceptually pairs with, embeds via the worker's provider, and marks rows
ready/failed. Runs opportunistically inside the worker loop; no Redis
plumbing needed.
"""

from __future__ import annotations

import time
from typing import Any

import psycopg

from activekg.common.logger import get_enhanced_logger

logger = get_enhanced_logger(__name__)


def build_candidate_embedding_text(row: dict[str, Any]) -> str:
    """ONE text builder shared by this producer and the search endpoint's query
    side: candidates and JD queries must live in the same vector space."""
    parts: list[str] = []
    for key in ("name", "headline", "role_family", "seniority_band"):
        v = row.get(key)
        if v:
            parts.append(str(v))
    skills = row.get("skills_normalized")
    if skills:
        parts.append("skills: " + ", ".join(skills[:30]))
    loc = ", ".join(
        str(v) for v in [row.get("location_city"), row.get("location_country_code")] if v
    )
    if loc:
        parts.append(loc)
    return ". ".join(parts)

_SELECT_COLS = (
    "id, name, headline, role_family, seniority_band, skills_normalized, "
    "location_city, location_country_code"
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
                cur.execute(
                    f"""
                    SELECT {_SELECT_COLS}
                    FROM global_candidates
                    WHERE embedding_status = 'queued'
                    ORDER BY updated_at ASC
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                    """,
                    (self.batch_size,),
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
                    # Nothing to embed (no name/headline/skills) — skip, don't loop forever.
                    cur.execute(
                        "UPDATE global_candidates SET embedding_status = 'skipped_empty'"
                        " WHERE id = ANY(%s::uuid[])",
                        (empty_ids,),
                    )

                done = 0
                if embeddable:
                    vectors = self.embedder.encode(texts)
                    for d, vec in zip(embeddable, vectors, strict=False):
                        vec_literal = "[" + ",".join(f"{x:.6f}" for x in vec.tolist()) + "]"
                        cur.execute(
                            "UPDATE global_candidates"
                            " SET embedding = %s::vector, embedding_status = 'ready',"
                            "     updated_at = now()"
                            " WHERE id = %s",
                            (vec_literal, d["id"]),
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
