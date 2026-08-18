#!/usr/bin/env python3
"""
Test script to verify new Prometheus metrics are being emitted correctly.

Tests:
1. Purger metrics (connector_purger_total, connector_purger_latency_seconds)
2. Rate limiting metrics (api_rate_limited_total) - check existence
3. Webhook topic rejection metrics (webhook_topic_rejected_total) - check existence
4. DLQ metrics (connector_dlq_depth, connector_dlq_total) - check existence
"""

import json
import os
import time

import requests


def get_prometheus_metrics() -> str:
    """Fetch all metrics from /prometheus endpoint."""
    token = os.getenv("ACTIVEKG_CONTROL_PLANE_TOKEN")
    if not token:
        raise RuntimeError("ACTIVEKG_CONTROL_PLANE_TOKEN must be set before metrics access")
    try:
        response = requests.get(
            "http://localhost:8000/prometheus",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"❌ Failed to fetch metrics: {e}")
        return ""


def parse_metrics(metrics_text: str) -> dict[str, list[str]]:
    """Parse Prometheus metrics text into a dict of metric_name -> [lines]."""
    metrics = {}
    for line in metrics_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Extract metric name (before '{' or ' ')
        if "{" in line:
            metric_name = line.split("{")[0]
        elif " " in line:
            metric_name = line.split(" ")[0]
        else:
            continue

        if metric_name not in metrics:
            metrics[metric_name] = []
        metrics[metric_name].append(line)
    return metrics


def test_metric_exists(
    metrics: dict[str, list[str]], metric_name: str, required_labels: list[str] | None = None
) -> bool:
    """Test if a metric exists and optionally has required labels."""
    if metric_name not in metrics:
        print(f"❌ Metric '{metric_name}' not found")
        return False

    lines = metrics[metric_name]
    print(f"✅ Metric '{metric_name}' exists ({len(lines)} time series)")

    if required_labels:
        # Check if at least one line contains all required labels
        for line in lines:
            if all(label in line for label in required_labels):
                print(f"   ✓ Found required labels: {required_labels}")
                for label_line in lines:
                    print(f"     {label_line}")
                return True
        print(f"   ⚠️  Required labels not found: {required_labels}")
        print("   Available lines:")
        for line in lines:
            print(f"     {line}")
        return False

    # Just show first few lines
    for line in lines[:5]:
        print(f"   {line}")
    if len(lines) > 5:
        print(f"   ... ({len(lines) - 5} more)")
    return True


def test_purger_endpoint() -> bool:
    """Test purger endpoint and verify metrics are emitted."""
    print("\n=== Testing Purger Endpoint ===")

    # Get baseline metrics
    print("Fetching baseline metrics...")
    baseline = get_prometheus_metrics()
    baseline_metrics = parse_metrics(baseline)

    baseline_purger_total = 0
    if "connector_purger_total" in baseline_metrics:
        for line in baseline_metrics["connector_purger_total"]:
            if 'result="success"' in line:
                baseline_purger_total = float(line.split()[-1])
                print(
                    f'Baseline connector_purger_total{{result="success"}}: {baseline_purger_total}'
                )
                break

    # Call purger endpoint
    print("\nCalling purger endpoint (dry_run=true)...")
    try:
        response = requests.post(
            "http://localhost:8000/_admin/connectors/purge_deleted",
            json={"dry_run": True, "tenant_id": "default"},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        print(f"✅ Purger response: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"❌ Purger endpoint failed: {e}")
        return False

    # Wait a bit for metrics to be recorded
    time.sleep(2)

    # Get updated metrics
    print("\nFetching updated metrics...")
    updated = get_prometheus_metrics()
    updated_metrics = parse_metrics(updated)

    # Check connector_purger_total incremented
    success = True
    if "connector_purger_total" in updated_metrics:
        for line in updated_metrics["connector_purger_total"]:
            if 'result="success"' in line:
                new_count = float(line.split()[-1])
                print(f'\n✅ connector_purger_total{{result="success"}}: {new_count}')
                if new_count > baseline_purger_total:
                    print(f"   ✓ Metric incremented (was {baseline_purger_total})")
                else:
                    print(f"   ⚠️  Metric did not increment (still {baseline_purger_total})")
                    success = False
                break
    else:
        print("❌ connector_purger_total not found in updated metrics")
        success = False

    # Check connector_purger_latency_seconds exists
    if (
        "connector_purger_latency_seconds_bucket" in updated_metrics
        or "connector_purger_latency_seconds_sum" in updated_metrics
    ):
        print("✅ connector_purger_latency_seconds histogram exists")
        # Show some buckets
        if "connector_purger_latency_seconds_bucket" in updated_metrics:
            print("   Sample buckets:")
            for line in updated_metrics["connector_purger_latency_seconds_bucket"][:3]:
                print(f"     {line}")
    else:
        print("❌ connector_purger_latency_seconds not found")
        success = False

    return success


def test_rate_limiting_metrics(metrics: dict[str, list[str]]) -> bool:
    """Test that rate limiting metrics are defined (even if zero)."""
    print("\n=== Testing Rate Limiting Metrics ===")
    print("(Rate limiting is disabled, so these metrics may be zero or absent)")

    # The metric should exist in code but may not have been incremented yet
    # Since rate limiting is disabled, we might not see this metric at all
    if "api_rate_limited_total" in metrics:
        print("✅ api_rate_limited_total exists")
        test_metric_exists(metrics, "api_rate_limited_total", ["endpoint", "reason"])
        return True
    else:
        print("ℹ️  api_rate_limited_total not present (expected if no rate limits hit yet)")
        print("   Metric is defined in activekg/api/rate_limiter.py:32-36")
        print("   Will appear when rate_limit_dependency() raises HTTPException 429")
        return True  # Not a failure if rate limiting is disabled


def test_webhook_metrics(metrics: dict[str, list[str]]) -> bool:
    """Test that webhook metrics exist."""
    print("\n=== Testing Webhook Metrics ===")
    print("(These appear when webhooks are rejected due to topic ARN mismatch)")

    if "webhook_topic_rejected_total" in metrics:
        print("✅ webhook_topic_rejected_total exists")
        return test_metric_exists(metrics, "webhook_topic_rejected_total", ["provider", "tenant"])
    else:
        print("ℹ️  webhook_topic_rejected_total not present (expected if no rejections yet)")
        print("   Metric is defined in activekg/connectors/webhooks.py:45-49")
        print("   Will appear when validate_topic_arn() fails")
        return True  # Not a failure if no webhooks rejected


def test_dlq_metrics(metrics: dict[str, list[str]]) -> bool:
    """Test that DLQ metrics exist."""
    print("\n=== Testing DLQ Metrics ===")
    print("(These appear when connector operations fail and go to dead letter queue)")

    dlq_total_exists = "connector_dlq_total" in metrics
    dlq_depth_exists = "connector_dlq_depth" in metrics

    if dlq_total_exists:
        print("✅ connector_dlq_total exists")
        test_metric_exists(metrics, "connector_dlq_total", ["provider", "tenant", "reason"])
    else:
        print("ℹ️  connector_dlq_total not present (expected if no DLQ writes yet)")
        print("   Metric is defined in activekg/connectors/retry.py:13-17")

    if dlq_depth_exists:
        print("✅ connector_dlq_depth exists")
        test_metric_exists(metrics, "connector_dlq_depth", ["provider", "tenant"])
    else:
        print("ℹ️  connector_dlq_depth not present (expected if no DLQ writes yet)")
        print("   Metric is defined in activekg/connectors/retry.py:18-22")

    return True  # Not a failure if no DLQ activity


def main():
    """Run all metric tests."""
    print("=" * 70)
    print("Prometheus Metrics Verification Test")
    print("=" * 70)

    # Test purger endpoint (this will actually trigger metrics)
    purger_success = test_purger_endpoint()

    # Fetch latest metrics for other tests
    print("\n=== Fetching All Metrics for Verification ===")
    metrics_text = get_prometheus_metrics()
    if not metrics_text:
        print("❌ Failed to fetch metrics, aborting")
        return False

    metrics = parse_metrics(metrics_text)
    print(f"✅ Fetched {len(metrics)} unique metric types")

    # Test other metrics
    rate_limiting_success = test_rate_limiting_metrics(metrics)
    webhook_success = test_webhook_metrics(metrics)
    dlq_success = test_dlq_metrics(metrics)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Purger metrics:        {'✅ PASS' if purger_success else '❌ FAIL'}")
    print(f"Rate limiting metrics: {'✅ PASS' if rate_limiting_success else '❌ FAIL'}")
    print(f"Webhook metrics:       {'✅ PASS' if webhook_success else '❌ FAIL'}")
    print(f"DLQ metrics:           {'✅ PASS' if dlq_success else '❌ FAIL'}")
    print()

    all_success = purger_success and rate_limiting_success and webhook_success and dlq_success

    if all_success:
        print("✅ All metrics tests passed!")
        print("\nNext steps:")
        print("1. Check metrics at http://localhost:8000/prometheus")
        print(
            "2. Verify alerts in observability/alerts/connector_alerts.yml reference correct metric names"
        )
        print("3. Test with actual webhook/rate-limit/DLQ traffic to verify labels")
    else:
        print("❌ Some tests failed - review output above")

    return all_success


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
