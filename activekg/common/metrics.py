import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import redis

try:
    import redis as _redis
except Exception:  # pragma: no cover
    _redis = None


@dataclass
class MetricPoint:
    timestamp: float
    value: float
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Advanced metrics collection with structured logging support."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._metrics: dict[str, deque[MetricPoint]] = defaultdict(lambda: deque(maxlen=max_history))
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.RLock()

    def increment_counter(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ):
        """Increment a counter metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value
            self._metrics[key].append(MetricPoint(time.time(), self._counters[key], labels or {}))

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None):
        """Set a gauge metric value."""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
            self._metrics[key].append(MetricPoint(time.time(), value, labels or {}))

    def record_histogram(self, name: str, value: float, labels: dict[str, str] | None = None):
        """Record a value in a histogram."""
        with self._lock:
            key = self._make_key(name, labels)
            self._histograms[key].append(value)
            self._metrics[key].append(MetricPoint(time.time(), value, labels or {}))

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get current counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0.0)

    def get_gauge(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get current gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key, 0.0)

    def get_histogram_stats(
        self, name: str, labels: dict[str, str] | None = None
    ) -> dict[str, float]:
        """Get histogram statistics."""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])
        if not values:
            return {}

        arr = np.array(values)
        return {
            "count": len(values),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
        }

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics in a structured format."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: self.get_histogram_stats(k) for k in self._histograms.keys()},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create a unique key for the metric."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"


# Global metrics instance
metrics = MetricsCollector()


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, metric_name: str, labels: dict[str, str] | None = None):
        self.metric_name = metric_name
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            metrics.record_histogram(
                self.metric_name, duration * 1000, self.labels
            )  # Store in milliseconds


# -----------------------------
# Redis helper (singleton)
# -----------------------------
_redis_client = None


def get_redis_client() -> "redis.Redis[bytes]":
    """Return a Redis client from REDIS_URL.

    Falls back to redis://localhost:6379/0 if REDIS_URL is unset.
    Lazily initializes a singleton client for the process.
    """
    global _redis_client
    if _redis is None:
        raise RuntimeError("redis library not installed")

    if _redis_client is None:
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = _redis.from_url(url)
        try:
            _redis_client.ping()
        except Exception:
            # Defer connection errors to caller; client constructed
            pass
    return _redis_client
