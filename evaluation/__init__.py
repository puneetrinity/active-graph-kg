"""Evaluation harness for Active Graph KG.

Measures performance across key dimensions:
- Drift cohort analysis (quality before/after refresh)
- Weighted search evaluation (recency vs accuracy trade-off)
- Latency benchmarks (p50/p95/p99)
- Freshness SLA monitoring (% nodes refreshed on time)
"""

__version__ = "1.0.0"
