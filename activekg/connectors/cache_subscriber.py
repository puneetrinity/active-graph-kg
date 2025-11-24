"""Redis pub/sub subscriber for connector config cache invalidation.

Subscribes to cache invalidation messages and evicts entries from local cache
to maintain consistency across multiple workers.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from typing import Any

from prometheus_client import Counter

logger = logging.getLogger(__name__)

# Redis pub/sub (optional - graceful degradation if not available)
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - cache subscriber disabled")

# Prometheus metrics
connector_pubsub_messages_total = Counter(
    "connector_pubsub_messages_total", "Total Redis pub/sub messages received", ["operation"]
)
connector_pubsub_reconnect_total = Counter(
    "connector_pubsub_reconnect_total", "Total Redis pub/sub reconnection attempts"
)
connector_pubsub_shutdown_total = Counter(
    "connector_pubsub_shutdown_total", "Total Redis pub/sub clean shutdowns"
)
connector_pubsub_invalid_msg_total = Counter(
    "connector_pubsub_invalid_msg_total", "Total invalid pub/sub messages dropped", ["reason"]
)


class CacheSubscriber:
    """Subscribes to Redis pub/sub for cache invalidation."""

    def __init__(self, redis_url: str, config_store):
        """Initialize cache subscriber.

        Args:
            redis_url: Redis connection URL
            config_store: ConnectorConfigStore instance to invalidate cache
        """
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis not available, cannot start cache subscriber")

        self.redis_url = redis_url
        self.config_store = config_store
        self.channel = "connector:config:changed"

        self._redis_client: redis.Redis | None = None
        self._pubsub: redis.client.PubSub | None = None
        self._subscriber_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._reconnect_delay = 1.0  # Start with 1 second
        self._max_reconnect_delay = 60.0  # Max 60 seconds

        # Health tracking
        self._connected = False
        self._last_message_ts: datetime | None = None
        self._reconnect_count = 0

    def start(self) -> None:
        """Start subscriber thread."""
        if self._running:
            logger.warning("Cache subscriber already running")
            return

        self._running = True
        self._subscriber_thread = threading.Thread(
            target=self._subscribe_loop, daemon=True, name="cache-subscriber"
        )
        self._subscriber_thread.start()
        logger.info("Cache subscriber started")

    def stop(self) -> None:
        """Stop subscriber thread gracefully."""
        if not self._running:
            return

        logger.info("Stopping cache subscriber...")

        # Signal thread to stop
        self._running = False
        self._stop_event.set()

        # Wait for thread to exit
        if self._subscriber_thread:
            self._subscriber_thread.join(timeout=5.0)

        # Now close connections (after thread has exited)
        if self._pubsub:
            try:
                self._pubsub.unsubscribe()
                self._pubsub.close()
            except Exception as e:
                logger.debug(f"Pubsub close (expected during shutdown): {e}")

        if self._redis_client:
            try:
                self._redis_client.close()
            except Exception as e:
                logger.debug(f"Redis close (expected during shutdown): {e}")

        connector_pubsub_shutdown_total.inc()
        logger.info("Cache subscriber stopped cleanly")

    def _connect(self) -> bool:
        """Connect to Redis and subscribe to channel.

        Returns:
            True if connection successful
        """
        try:
            self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self._redis_client.ping()
            self._pubsub = self._redis_client.pubsub()
            self._pubsub.subscribe(self.channel)
            logger.info(f"Subscribed to Redis channel: {self.channel}")
            self._reconnect_delay = 1.0  # Reset delay on successful connection
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis for pub/sub: {e}")
            self._connected = False
            self._reconnect_count += 1
            connector_pubsub_reconnect_total.inc()
            return False

    def _subscribe_loop(self) -> None:
        """Main subscriber loop with reconnection logic."""
        while self._running and not self._stop_event.is_set():
            if not self._connect():
                logger.warning(f"Retrying in {self._reconnect_delay}s...")
                # Use wait() instead of sleep to respond immediately to stop_event
                if self._stop_event.wait(timeout=self._reconnect_delay):
                    break
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
                continue

            try:
                # Listen for messages
                if self._pubsub is None:
                    logger.error("PubSub connection is None, reconnecting...")
                    continue

                for message in self._pubsub.listen():
                    if not self._running or self._stop_event.is_set():
                        break

                    if message["type"] == "message":
                        self._handle_message(message["data"])

            except Exception as e:
                # Check if it's a shutdown error (expected)
                if self._stop_event.is_set():
                    logger.debug(f"Subscriber loop exit during shutdown: {e}")
                    break

                logger.error(f"Error in subscriber loop: {e}")
                connector_pubsub_reconnect_total.inc()
                # Use wait() instead of sleep to respond immediately to stop_event
                if self._stop_event.wait(timeout=self._reconnect_delay):
                    break
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

    def _handle_message(self, data: str) -> None:
        """Handle cache invalidation message with validation.

        Args:
            data: JSON message with tenant_id, provider, operation
        """
        # Valid operations per user spec
        VALID_OPERATIONS = {"upsert", "enable", "disable", "delete", "rotate"}

        try:
            # Parse JSON
            try:
                msg = json.loads(data)
            except json.JSONDecodeError as e:
                connector_pubsub_invalid_msg_total.labels(reason="invalid_json").inc()
                logger.warning(f"Malformed JSON in cache invalidation message: {e}")
                return

            # Validate required fields
            tenant_id = msg.get("tenant_id")
            provider = msg.get("provider")
            operation = msg.get("operation")

            if not tenant_id:
                connector_pubsub_invalid_msg_total.labels(reason="missing_tenant_id").inc()
                logger.warning(f"Cache invalidation message missing tenant_id: {msg}")
                return

            if not provider:
                connector_pubsub_invalid_msg_total.labels(reason="missing_provider").inc()
                logger.warning(f"Cache invalidation message missing provider: {msg}")
                return

            if not operation:
                connector_pubsub_invalid_msg_total.labels(reason="missing_operation").inc()
                logger.warning(f"Cache invalidation message missing operation: {msg}")
                return

            # Validate operation value
            if operation not in VALID_OPERATIONS:
                connector_pubsub_invalid_msg_total.labels(reason="invalid_operation").inc()
                logger.warning(
                    f"Invalid operation '{operation}' (expected one of {VALID_OPERATIONS})"
                )
                return

            # Valid message - evict from local cache
            cache_key = (tenant_id, provider)
            if cache_key in self.config_store._cache:
                del self.config_store._cache[cache_key]
                logger.debug(f"Cache invalidated: {tenant_id}/{provider} (op={operation})")

            # Track last message timestamp for health monitoring
            self._last_message_ts = datetime.utcnow()
            connector_pubsub_messages_total.labels(operation=operation).inc()

        except Exception as e:
            connector_pubsub_invalid_msg_total.labels(reason="unexpected_error").inc()
            logger.error(f"Unexpected error handling cache invalidation message: {e}")

    def get_health(self) -> dict[str, Any]:
        """Get subscriber health status.

        Returns:
            Health status dict with connected, last_message_ts, reconnects
        """
        health: dict[str, Any] = {"connected": self._connected, "reconnects": self._reconnect_count}

        if self._last_message_ts:
            health["last_message_ts"] = self._last_message_ts.isoformat() + "Z"
        else:
            health["last_message_ts"] = None

        return health


# Global subscriber instance
_subscriber: CacheSubscriber | None = None


def start_subscriber(redis_url: str, config_store) -> None:
    """Start global cache subscriber.

    Args:
        redis_url: Redis connection URL
        config_store: ConnectorConfigStore instance
    """
    global _subscriber

    if _subscriber is not None:
        logger.warning("Cache subscriber already started")
        return

    if not REDIS_AVAILABLE:
        logger.warning("Redis not available, cache subscriber disabled")
        return

    try:
        _subscriber = CacheSubscriber(redis_url, config_store)
        _subscriber.start()
    except Exception as e:
        logger.error(f"Failed to start cache subscriber: {e}")


def stop_subscriber() -> None:
    """Stop global cache subscriber."""
    global _subscriber

    if _subscriber is not None:
        _subscriber.stop()
        _subscriber = None


def get_subscriber_health() -> dict[str, Any] | None:
    """Get global cache subscriber health status.

    Returns:
        Health status dict or None if subscriber not running
    """
    global _subscriber

    if _subscriber is not None:
        return _subscriber.get_health()
    return None
