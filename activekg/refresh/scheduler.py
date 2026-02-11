from __future__ import annotations

import time
from datetime import datetime, timezone
from time import time as _now

from apscheduler.schedulers.background import BackgroundScheduler
from prometheus_client import Counter, Histogram

from activekg.common.logger import get_enhanced_logger
from activekg.observability.metrics import (
    track_node_refresh_latency,
    track_nodes_refreshed,
    track_refresh_cycle_nodes,
    track_schedule_run,
)

# Prometheus metrics for Drive poller
connector_poller_runs_total = Counter(
    "connector_poller_runs_total", "Total connector poller runs", ["provider", "tenant"]
)

connector_poller_errors_total = Counter(
    "connector_poller_errors_total",
    "Total connector poller errors",
    ["provider", "tenant", "error_type"],
)

connector_poller_latency_seconds = Histogram(
    "connector_poller_latency_seconds",
    "Connector poller run duration in seconds",
    ["provider", "tenant"],
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)


class RefreshScheduler:
    """APScheduler-based refresh engine.

    - Scans nodes with due refresh_policy
    - Re-embeds content and computes drift_score
    - Appends events and updates DB
    - Optionally runs trigger engine after refreshes
    """

    def __init__(self, repository, embedding_provider, trigger_engine=None, gcs_poller_enabled=True):
        self.repo = repository
        self.embedder = embedding_provider
        self.trigger_engine = trigger_engine
        self.gcs_poller_enabled = gcs_poller_enabled
        self.scheduler = BackgroundScheduler()
        self.logger = get_enhanced_logger(__name__)
        # Track last run timestamps per job for inter-run observations
        self._last_times: dict[str, float] = {}
        self.last_runs: dict[str, dict] = {}

    def start(self):
        # Run refresh cycle every minute
        self.scheduler.add_job(self.run_cycle, "interval", minutes=1, id="refresh_cycle")

        # Run trigger engine every 2 minutes if available
        if self.trigger_engine:
            self.scheduler.add_job(self.run_triggers, "interval", minutes=2, id="trigger_cycle")

        # Run daily purge of soft-deleted nodes at 02:00 UTC
        self.scheduler.add_job(self.run_purge, "cron", hour=2, minute=0, id="purge_deleted_cycle")

        # Run Drive poller (connector changes feed) every 60s
        # Uses per-tenant Redis lock to honor tenant-specific poll_interval_seconds
        self.scheduler.add_job(self.run_drive_poller, "interval", seconds=60, id="drive_poller")

        # Run GCS poller every 60s (mirrors Drive poller)
        if self.gcs_poller_enabled:
            self.scheduler.add_job(self.run_gcs_poller, "interval", seconds=60, id="gcs_poller")

        self.scheduler.start()
        self.logger.info(
            "RefreshScheduler started",
            extra_fields={
                "has_triggers": self.trigger_engine is not None,
                "purge_enabled": True,
                "gcs_poller_enabled": self.gcs_poller_enabled,
            },
        )

    def shutdown(self):
        self.scheduler.shutdown(wait=False)
        self.logger.info("RefreshScheduler stopped")

    def run_cycle(self):
        """Scan for due nodes and refresh."""
        self.logger.info("Refresh cycle begin")
        # Observe inter-run interval for refresh_cycle
        now_ts = _now()
        last = self._last_times.get("refresh_cycle")
        try:
            if last is not None:
                track_schedule_run(
                    "refresh_cycle", kind="interval", inter_run_s=max(0.0, now_ts - last)
                )
            else:
                track_schedule_run("refresh_cycle", kind="interval", inter_run_s=None)
        except Exception:
            pass
        self._last_times["refresh_cycle"] = now_ts
        due = self.repo.find_nodes_due_for_refresh()
        refreshed_ids = []

        for node in due:
            try:
                t0 = _now()
                text = self.repo.load_payload_text(node)

                # Guard against empty/None payloads
                if not text or not text.strip():
                    self.logger.warning(
                        f"Skipping refresh for node {node.id}: empty or whitespace-only payload"
                    )
                    continue

                old = node.embedding
                new = self.embedder.encode([text])[0]
                drift = (
                    1.0 - float((old @ new) / ((old**2).sum() ** 0.5 * (new**2).sum() ** 0.5))
                    if (old is not None)
                    else 0.0
                )

                # Update node embedding, drift, and timestamp (now in dedicated columns)
                timestamp = datetime.now(timezone.utc).isoformat()
                self.repo.update_node_embedding(
                    node.id, new, drift, timestamp, tenant_id=node.tenant_id
                )

                # Write to embedding history
                self.repo.write_embedding_history(
                    node.id, drift, embedding_ref=node.payload_ref, tenant_id=node.tenant_id
                )

                # Emit event if drift exceeds threshold (default 0.1)
                drift_threshold = node.refresh_policy.get("drift_threshold", 0.1)
                if drift > drift_threshold:
                    self.repo.append_event(
                        node.id,
                        "refreshed",
                        {
                            "drift_score": drift,
                            "last_refreshed": timestamp,
                            "threshold_exceeded": True,
                        },
                        tenant_id=node.tenant_id,
                        actor_id="scheduler",
                        actor_type="scheduler",
                    )
                    self.logger.info(
                        "Drift threshold exceeded",
                        extra_fields={
                            "node_id": str(node.id),
                            "drift": drift,
                            "threshold": drift_threshold,
                        },
                    )

                # Track successfully refreshed nodes
                refreshed_ids.append(node.id)
                try:
                    track_node_refresh_latency(max(0.0, _now() - t0), result="ok")
                except Exception:
                    pass
                try:
                    track_nodes_refreshed(result="ok", count=1)
                except Exception:
                    pass

            except Exception as e:
                self.logger.error(
                    "Refresh failed", extra_fields={"node_id": str(node.id), "error": str(e)}
                )
                try:
                    track_node_refresh_latency(max(0.0, _now() - t0), result="error")
                except Exception:
                    pass
                try:
                    track_nodes_refreshed(result="error", count=1)
                except Exception:
                    pass
        self.logger.info(
            "Refresh cycle end", extra_fields={"count": len(due), "refreshed": len(refreshed_ids)}
        )

        # Observe per-cycle refreshed nodes
        try:
            track_refresh_cycle_nodes(len(refreshed_ids))
        except Exception:
            pass
        # Run triggers ONLY for refreshed nodes (efficient)
        if refreshed_ids and self.trigger_engine:
            self.trigger_engine.run_for(refreshed_ids)

    def run_triggers(self):
        """Execute trigger engine to check for pattern matches."""
        if not self.trigger_engine:
            return

        try:
            self.logger.info("Trigger cycle begin")
            now_ts = _now()
            last = self._last_times.get("trigger_cycle")
            try:
                if last is not None:
                    track_schedule_run(
                        "trigger_cycle", kind="interval", inter_run_s=max(0.0, now_ts - last)
                    )
                else:
                    track_schedule_run("trigger_cycle", kind="interval", inter_run_s=None)
            except Exception:
                pass
            self._last_times["trigger_cycle"] = now_ts
            count = self.trigger_engine.run()
            self.logger.info("Trigger cycle end", extra_fields={"fired": count})
            self.last_runs["trigger_cycle"] = {"fired": count, "ts": now_ts}
        except Exception as e:
            self.logger.error("Trigger cycle failed", extra_fields={"error": str(e)})

    def run_purge(self):
        """Execute daily purge of soft-deleted nodes across all tenants."""
        try:
            self.logger.info("Purge cycle begin")
            now_ts = _now()
            last = self._last_times.get("purge_deleted_cycle")
            try:
                if last is not None:
                    track_schedule_run(
                        "purge_deleted_cycle", kind="cron", inter_run_s=max(0.0, now_ts - last)
                    )
                else:
                    track_schedule_run("purge_deleted_cycle", kind="cron", inter_run_s=None)
            except Exception:
                pass
            self._last_times["purge_deleted_cycle"] = now_ts

            # Purge across all tenants (tenant_id=None means all tenants)
            result = self.repo.purge_deleted_nodes(
                tenant_id=None,  # All tenants
                batch_size=500,
                dry_run=False,
            )

            self.logger.info(
                "Purge cycle end",
                extra_fields={
                    "total_purged": result.get("total_purged", 0),
                    "tenants_processed": result.get("tenants_processed", 0),
                },
            )
            self.last_runs["purge_deleted_cycle"] = {"result": result, "ts": now_ts}
        except Exception as e:
            self.logger.error("Purge cycle failed", extra_fields={"error": str(e)})

    def run_drive_poller(self):
        """Poll Google Drive changes for all enabled tenants and enqueue events.

        Strategy:
        - List enabled Drive configs from config store
        - For each tenant, acquire a Redis lock honoring poll_interval_seconds
        - Instantiate DriveConnector and call list_changes(cursor)
        - Enqueue ChangeItems to Redis queue connector:drive:{tenant}:queue
        - Update cursor in connector_cursors table
        - Register tenant in active registry for worker discovery
        """
        try:
            now_ts = _now()
            last = self._last_times.get("drive_poller")
            try:
                if last is not None:
                    track_schedule_run(
                        "drive_poller", kind="interval", inter_run_s=max(0.0, now_ts - last)
                    )
                else:
                    track_schedule_run("drive_poller", kind="interval", inter_run_s=None)
            except Exception:
                pass
            self._last_times["drive_poller"] = now_ts
            # Lazy imports to avoid hard deps if not used
            import json
            from datetime import datetime

            from activekg.common.metrics import get_redis_client
            from activekg.connectors.config_store import get_config_store
            from activekg.connectors.cursor_store import get_cursor, set_cursor
            from activekg.connectors.providers.drive import DriveConnector

            store = get_config_store()
            redis_client = get_redis_client()

            # List drive configs (metadata), then fetch full config per tenant
            meta = store.list_all(provider="drive")
            for entry in meta:
                tenant_id = entry.get("tenant_id")
                if not tenant_id or entry.get("enabled") is False:
                    continue

                cfg = store.get(tenant_id, "drive")
                if not cfg:
                    continue

                # Increment poller runs metric at start of processing
                connector_poller_runs_total.labels(provider="drive", tenant=tenant_id).inc()

                # Start timer for latency tracking
                poll_start_time = time.time()

                # Per-tenant poll interval
                interval = int(cfg.get("poll_interval_seconds", 300))
                lock_key = f"connector:drive:{tenant_id}:poll_lock"
                # NX lock with TTL=interval to avoid per-tenant over-polling
                try:
                    if not redis_client.set(lock_key, "1", ex=interval, nx=True):
                        continue  # respect interval
                except Exception as e:
                    # If Redis is unavailable, proceed (best effort)
                    self.logger.warning(
                        "Drive poller: Redis lock failed, proceeding",
                        extra_fields={"tenant_id": tenant_id, "error": str(e)},
                    )

                try:
                    connector = DriveConnector(tenant_id=tenant_id, config=cfg)
                except Exception as e:
                    connector_poller_errors_total.labels(
                        provider="drive", tenant=tenant_id, error_type="connector_init"
                    ).inc()
                    self.logger.error(
                        "Drive poller: connector init failed",
                        extra_fields={"tenant_id": tenant_id, "error": str(e)},
                    )
                    continue

                # Get last cursor and poll changes
                try:
                    cursor = get_cursor(tenant_id, "drive")
                    changes, next_cursor = connector.list_changes(cursor)
                except Exception as e:
                    connector_poller_errors_total.labels(
                        provider="drive", tenant=tenant_id, error_type="list_changes"
                    ).inc()
                    self.logger.error(
                        "Drive poller: list_changes failed",
                        extra_fields={"tenant_id": tenant_id, "error": str(e)},
                    )
                    continue

                # Enqueue changes
                queued = 0
                for ch in changes:
                    try:
                        item = {
                            "uri": ch.uri,
                            "operation": ch.operation,
                            "etag": ch.etag,
                            "modified_at": ch.modified_at.isoformat()
                            if ch.modified_at
                            else datetime.utcnow().isoformat(),
                            "tenant_id": tenant_id,
                        }
                        redis_client.lpush(f"connector:drive:{tenant_id}:queue", json.dumps(item))
                        queued += 1
                    except Exception as e:
                        self.logger.error(
                            "Drive poller: enqueue failed",
                            extra_fields={"tenant_id": tenant_id, "error": str(e)},
                        )

                # Register tenant in active registry for worker discovery
                try:
                    registry_key = "connector:active_tenants"
                    registry_entry = json.dumps({"provider": "drive", "tenant_id": tenant_id})
                    redis_client.sadd(registry_key, registry_entry)
                except Exception:
                    pass

                # Persist next cursor
                if next_cursor:
                    try:
                        set_cursor(tenant_id, "drive", next_cursor)
                    except Exception as e:
                        connector_poller_errors_total.labels(
                            provider="drive", tenant=tenant_id, error_type="cursor_save"
                        ).inc()
                        self.logger.error(
                            "Drive poller: save cursor failed",
                            extra_fields={"tenant_id": tenant_id, "error": str(e)},
                        )

                self.logger.info(
                    "Drive poll cycle",
                    extra_fields={
                        "tenant_id": tenant_id,
                        "queued": queued,
                        "had_next": bool(next_cursor),
                    },
                )

                # Record latency
                poll_duration = time.time() - poll_start_time
                connector_poller_latency_seconds.labels(provider="drive", tenant=tenant_id).observe(
                    poll_duration
                )

        except Exception as e:
            self.logger.error("Drive poller failed", extra_fields={"error": str(e)})

    def run_gcs_poller(self):
        """Poll GCS changes for all enabled tenants and enqueue events.

        Strategy (mirrors run_drive_poller):
        - List enabled GCS configs from config store
        - For each tenant, acquire a Redis lock honoring poll_interval_seconds
        - Instantiate GCSConnector and call list_changes(cursor)
        - Enqueue ChangeItems to Redis queue connector:gcs:{tenant}:queue
        - Update cursor in connector_cursors table
        - Register tenant in active registry for worker discovery
        """
        try:
            now_ts = _now()
            last = self._last_times.get("gcs_poller")
            try:
                if last is not None:
                    track_schedule_run(
                        "gcs_poller", kind="interval", inter_run_s=max(0.0, now_ts - last)
                    )
                else:
                    track_schedule_run("gcs_poller", kind="interval", inter_run_s=None)
            except Exception:
                pass
            self._last_times["gcs_poller"] = now_ts
            # Lazy imports to avoid hard deps if not used
            import json
            from datetime import datetime

            from activekg.common.metrics import get_redis_client
            from activekg.connectors.config_store import get_config_store
            from activekg.connectors.cursor_store import get_cursor, set_cursor
            from activekg.connectors.providers.gcs import GCSConnector

            store = get_config_store()
            redis_client = get_redis_client()

            # List GCS configs (metadata), then fetch full config per tenant
            meta = store.list_all(provider="gcs")
            for entry in meta:
                tenant_id = entry.get("tenant_id")
                if not tenant_id or entry.get("enabled") is False:
                    continue

                cfg = store.get(tenant_id, "gcs")
                if not cfg:
                    continue

                # Increment poller runs metric at start of processing
                connector_poller_runs_total.labels(provider="gcs", tenant=tenant_id).inc()

                # Start timer for latency tracking
                poll_start_time = time.time()

                # Per-tenant poll interval
                interval = int(cfg.get("poll_interval_seconds", 300))
                lock_key = f"connector:gcs:{tenant_id}:poll_lock"
                # NX lock with TTL=interval to avoid per-tenant over-polling
                try:
                    if not redis_client.set(lock_key, "1", ex=interval, nx=True):
                        continue  # respect interval
                except Exception as e:
                    # If Redis is unavailable, proceed (best effort)
                    self.logger.warning(
                        "GCS poller: Redis lock failed, proceeding",
                        extra_fields={"tenant_id": tenant_id, "error": str(e)},
                    )

                try:
                    connector = GCSConnector(tenant_id=tenant_id, config=cfg)
                except Exception as e:
                    connector_poller_errors_total.labels(
                        provider="gcs", tenant=tenant_id, error_type="connector_init"
                    ).inc()
                    self.logger.error(
                        "GCS poller: connector init failed",
                        extra_fields={"tenant_id": tenant_id, "error": str(e)},
                    )
                    continue

                # Get last cursor and poll changes
                try:
                    cursor = get_cursor(tenant_id, "gcs")
                    changes, next_cursor = connector.list_changes(cursor)
                except Exception as e:
                    connector_poller_errors_total.labels(
                        provider="gcs", tenant=tenant_id, error_type="list_changes"
                    ).inc()
                    self.logger.error(
                        "GCS poller: list_changes failed",
                        extra_fields={"tenant_id": tenant_id, "error": str(e)},
                    )
                    continue

                # Enqueue changes
                queued = 0
                for ch in changes:
                    try:
                        item = {
                            "uri": ch.uri,
                            "operation": ch.operation,
                            "etag": ch.etag,
                            "modified_at": ch.modified_at.isoformat()
                            if ch.modified_at
                            else datetime.utcnow().isoformat(),
                            "tenant_id": tenant_id,
                        }
                        redis_client.lpush(f"connector:gcs:{tenant_id}:queue", json.dumps(item))
                        queued += 1
                    except Exception as e:
                        self.logger.error(
                            "GCS poller: enqueue failed",
                            extra_fields={"tenant_id": tenant_id, "error": str(e)},
                        )

                # Register tenant in active registry for worker discovery
                try:
                    registry_key = "connector:active_tenants"
                    registry_entry = json.dumps({"provider": "gcs", "tenant_id": tenant_id})
                    redis_client.sadd(registry_key, registry_entry)
                except Exception:
                    pass

                # Persist next cursor
                if next_cursor:
                    try:
                        set_cursor(tenant_id, "gcs", next_cursor)
                    except Exception as e:
                        connector_poller_errors_total.labels(
                            provider="gcs", tenant=tenant_id, error_type="cursor_save"
                        ).inc()
                        self.logger.error(
                            "GCS poller: save cursor failed",
                            extra_fields={"tenant_id": tenant_id, "error": str(e)},
                        )

                self.logger.info(
                    "GCS poll cycle",
                    extra_fields={
                        "tenant_id": tenant_id,
                        "queued": queued,
                        "had_next": bool(next_cursor),
                    },
                )

                # Record latency
                poll_duration = time.time() - poll_start_time
                connector_poller_latency_seconds.labels(provider="gcs", tenant=tenant_id).observe(
                    poll_duration
                )

        except Exception as e:
            self.logger.error("GCS poller failed", extra_fields={"error": str(e)})
