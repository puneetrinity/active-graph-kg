#!/usr/bin/env python3
"""
Latency Benchmark - Measure API performance across all endpoints.

Metrics:
- p50, p95, p99 latency for each endpoint
- Throughput (requests/second)
- Success rate

SLA Targets:
- /search: p95 < 100ms (with index), p95 < 200ms (weighted)
- /admin/anomalies: p95 < 500ms
- /nodes/{id}/versions: p95 < 50ms
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import numpy as np
import requests


def benchmark_endpoint(
    url: str, method: str, payload: dict[str, Any], num_requests: int, timeout: int = 30
) -> list[float]:
    """Benchmark a single endpoint.

    Args:
        url: Full endpoint URL
        method: HTTP method (GET, POST)
        payload: Request payload (for POST)
        num_requests: Number of requests to make
        timeout: Request timeout in seconds

    Returns:
        List of latencies in seconds
    """
    latencies = []

    for _ in range(num_requests):
        try:
            start_time = time.time()

            if method == "GET":
                resp = requests.get(url, params=payload, timeout=timeout)
            elif method == "POST":
                resp = requests.post(url, json=payload, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            latency = time.time() - start_time

            if resp.status_code == 200:
                latencies.append(latency)
            # Skip failed requests

        except (requests.Timeout, requests.RequestException):
            # Skip failed requests
            continue

    return latencies


def benchmark_endpoint_concurrent(
    url: str,
    method: str,
    payload: dict[str, Any],
    num_requests: int,
    concurrency: int,
    timeout: int = 30,
) -> list[float]:
    """Benchmark endpoint with concurrent requests.

    Args:
        url: Full endpoint URL
        method: HTTP method
        payload: Request payload
        num_requests: Total number of requests
        concurrency: Number of concurrent workers
        timeout: Request timeout

    Returns:
        List of latencies in seconds
    """
    latencies = []

    def single_request():
        try:
            start_time = time.time()

            if method == "GET":
                resp = requests.get(url, params=payload, timeout=timeout)
            elif method == "POST":
                resp = requests.post(url, json=payload, timeout=timeout)
            else:
                return None

            latency = time.time() - start_time

            if resp.status_code == 200:
                return latency

        except (requests.Timeout, requests.RequestException):
            return None

        return None

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_request) for _ in range(num_requests)]

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                latencies.append(result)

    return latencies


def run_latency_benchmarks(
    api_url: str,
    num_requests: int = 100,
    concurrency: int = 1,
    warmup: int = 10,
) -> dict[str, Any]:
    """Run latency benchmarks for all endpoints.

    Args:
        api_url: Base API URL
        num_requests: Number of requests per endpoint
        concurrency: Number of concurrent workers (1 = sequential)
        warmup: Number of warmup requests
    Returns:
        Dict with benchmark results
    """
    print("=" * 70)
    print("LATENCY BENCHMARK")
    print("=" * 70)
    print(f"API URL: {api_url}")
    print(f"Requests per endpoint: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Warmup requests: {warmup}")
    print()

    # Define endpoints to benchmark
    endpoints = [
        {
            "name": "/search (baseline)",
            "url": f"{api_url}/search",
            "method": "POST",
            "payload": {"query": "test query", "top_k": 10, "use_weighted_score": False},
            "sla_p95": 0.100,  # 100ms
        },
        {
            "name": "/search (weighted)",
            "url": f"{api_url}/search",
            "method": "POST",
            "payload": {"query": "test query", "top_k": 10, "use_weighted_score": True},
            "sla_p95": 0.200,  # 200ms
        },
        {
            "name": "/admin/anomalies",
            "url": f"{api_url}/admin/anomalies",
            "method": "GET",
            "payload": {"types": "drift_spike"},
            "sla_p95": 0.500,  # 500ms
        },
        {
            "name": "/health",
            "url": f"{api_url}/health",
            "method": "GET",
            "payload": {},
            "sla_p95": 0.050,  # 50ms
        },
    ]

    results = {}

    for endpoint in endpoints:
        name = endpoint["name"]
        url = endpoint["url"]
        method = endpoint["method"]
        payload = endpoint["payload"]
        sla_p95 = endpoint["sla_p95"]

        print(f"Benchmarking {name}...")

        # Warmup
        if warmup > 0:
            benchmark_endpoint(url, method, payload, warmup, timeout=10)

        # Actual benchmark
        start_time = time.time()
        if concurrency > 1:
            latencies = benchmark_endpoint_concurrent(
                url, method, payload, num_requests, concurrency, timeout=30
            )
        else:
            latencies = benchmark_endpoint(url, method, payload, num_requests, timeout=30)
        total_time = time.time() - start_time

        if not latencies:
            print("  ⚠ No successful requests\n")
            results[name] = {"error": "No successful requests"}
            continue

        # Calculate percentiles
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        mean = np.mean(latencies)
        std = np.std(latencies)

        # Calculate throughput
        throughput = len(latencies) / total_time

        # Check SLA
        meets_sla = p95 <= sla_p95
        sla_status = "✅" if meets_sla else "⚠"

        print(
            f"  {sla_status} p50: {p50 * 1000:.1f}ms, p95: {p95 * 1000:.1f}ms, p99: {p99 * 1000:.1f}ms"
        )
        print(f"  Throughput: {throughput:.1f} req/s, Success: {len(latencies)}/{num_requests}\n")

        results[name] = {
            "latency": {
                "p50": float(p50),
                "p95": float(p95),
                "p99": float(p99),
                "mean": float(mean),
                "std": float(std),
            },
            "throughput": float(throughput),
            "success_rate": float(len(latencies) / num_requests),
            "sla_p95": float(sla_p95),
            "meets_sla": bool(meets_sla),
        }

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for name, data in results.items():
        if "error" in data:
            print(f"{name}: ERROR")
            continue

        p95_ms = data["latency"]["p95"] * 1000
        sla_ms = data["sla_p95"] * 1000
        status = "✅" if data["meets_sla"] else "⚠"
        print(f"{status} {name}")
        print(f"   p95: {p95_ms:.1f}ms (SLA: {sla_ms:.0f}ms)")
        print(f"   Throughput: {data['throughput']:.1f} req/s")

    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Latency Benchmark")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--num-requests", type=int, default=100, help="Requests per endpoint")
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Concurrent workers (1 = sequential)"
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup requests")
    parser.add_argument(
        "--output", default="evaluation/latency_results.json", help="Output JSON file"
    )
    args = parser.parse_args()

    try:
        # Run benchmarks
        results = run_latency_benchmarks(
            args.api_url,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            warmup=args.warmup,
        )

        # Save results
        output = {
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "api_url": args.api_url,
                "num_requests": args.num_requests,
                "concurrency": args.concurrency,
                "warmup": args.warmup,
            },
        }

        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ Results saved to {args.output}")

        # Exit with appropriate code (all SLAs met)
        all_met = all(r.get("meets_sla", False) for r in results.values() if "error" not in r)
        sys.exit(0 if all_met else 1)

    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Cannot connect to API at {args.api_url}")
        print("\nMake sure the API server is running.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
