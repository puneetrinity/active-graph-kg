#!/usr/bin/env python3
"""
Consolidate evaluation results into unified JSON and Markdown summary.

Reads all evaluation/*_results.json files and creates:
- evaluation/results.json - Combined JSON results
- evaluation/SUMMARY.md - Human-readable markdown report
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_result(file_path: Path) -> dict[str, Any] | None:
    """Load a single result file."""
    if not file_path.exists():
        return None
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ Failed to load {file_path.name}: {e}")
        return None


def generate_markdown_summary(results: dict[str, Any], timestamp: str) -> str:
    """Generate markdown summary from consolidated results."""
    md = [
        "# Active Graph KG - Evaluation Summary",
        "",
        f"**Generated**: {timestamp}",
        "",
        "---",
        "",
    ]

    # Latency Benchmark
    if "latency" in results and results["latency"]:
        lat = results["latency"]
        md.extend(
            [
                "## 1. Latency Benchmark",
                "",
                "| Endpoint | p50 | p95 | p99 | SLA | Status |",
                "|----------|-----|-----|-----|-----|--------|",
            ]
        )

        for endpoint, data in lat.get("results", {}).items():
            if data.get("success", False):
                p50 = data.get("p50_ms", 0)
                p95 = data.get("p95_ms", 0)
                p99 = data.get("p99_ms", 0)
                sla = data.get("sla_ms", "N/A")
                status = "✅" if data.get("sla_met", False) else "⚠️"
                md.append(
                    f"| {endpoint} | {p50:.1f}ms | {p95:.1f}ms | {p99:.1f}ms | <{sla}ms | {status} |"
                )

        md.append("")

    # Freshness SLA
    if "freshness" in results and results["freshness"]:
        fresh = results["freshness"]
        md.extend(
            [
                "## 2. Freshness SLA Monitor",
                "",
                f"- **On-time**: {fresh.get('on_time_count', 0)} ({fresh.get('on_time_percent', 0):.1f}%)",
                f"- **At-risk**: {fresh.get('at_risk_count', 0)} ({fresh.get('at_risk_percent', 0):.1f}%)",
                f"- **Overdue**: {fresh.get('overdue_count', 0)} ({fresh.get('overdue_percent', 0):.1f}%)",
                "",
            ]
        )

    # Weighted Search
    if "weighted_search" in results and results["weighted_search"]:
        ws = results["weighted_search"]
        baseline = ws.get("baseline", {})
        weighted = ws.get("weighted", {})
        md.extend(
            [
                "## 3. Weighted Search Evaluation",
                "",
                "| Metric | Baseline | Weighted | Delta |",
                "|--------|----------|----------|-------|",
                f"| Recall@10 | {baseline.get('recall', {}).get('recall@10', 0):.3f} | {weighted.get('recall', {}).get('recall@10', 0):.3f} | {ws.get('delta_recall_10', 0):+.3f} |",
                f"| MRR | {baseline.get('mrr', 0):.3f} | {weighted.get('mrr', 0):.3f} | {ws.get('delta_mrr', 0):+.3f} |",
                f"| Avg Age (days) | {baseline.get('avg_age_days', 0):.1f} | {weighted.get('avg_age_days', 0):.1f} | {ws.get('delta_age', 0):+.1f} |",
                "",
            ]
        )

    # Drift Cohort
    if "drift_cohort" in results and results["drift_cohort"]:
        dc = results["drift_cohort"]
        if dc.get("nodes_refreshed", 0) > 0:
            baseline = dc.get("baseline", {})
            post_refresh = dc.get("post_refresh", {})
            md.extend(
                [
                    "## 4. Drift Cohort Analysis",
                    "",
                    f"- **High-drift nodes found**: {dc.get('high_drift_count', 0)}",
                    f"- **Nodes refreshed**: {dc.get('nodes_refreshed', 0)}",
                    "",
                    "| Metric | Before | After | Delta |",
                    "|--------|--------|-------|-------|",
                    f"| Recall@10 | {baseline.get('recall@10', 0):.3f} | {post_refresh.get('recall@10', 0):.3f} | {dc.get('delta_recall10', 0):+.3f} |",
                    f"| MRR | {baseline.get('mrr', 0):.3f} | {post_refresh.get('mrr', 0):.3f} | {dc.get('delta_mrr', 0):+.3f} |",
                    "",
                ]
            )
        else:
            md.extend(
                [
                    "## 4. Drift Cohort Analysis",
                    "",
                    "⚠️ No high-drift nodes found (empty database or low drift threshold)",
                    "",
                ]
            )

    md.extend(
        [
            "---",
            "",
            "## Summary",
            "",
            f"**Total Evaluations**: {sum(1 for k, v in results.items() if v and k != 'metadata')}",
            "",
            "**Next Steps**:",
            "1. Review individual result files for detailed metrics",
            "2. Fill out evaluation/REPORT_TEMPLATE.md with findings",
            "3. Share results with stakeholders",
            "",
        ]
    )

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Consolidate evaluation results")
    parser.add_argument("--output-dir", default="evaluation", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    timestamp = datetime.now().isoformat()

    print("=" * 70)
    print("Consolidating Evaluation Results")
    print("=" * 70)

    # Load all result files
    results = {
        "metadata": {"generated_at": timestamp, "version": "1.0.0"},
        "latency": load_result(output_dir / "latency_results.json"),
        "freshness": load_result(output_dir / "freshness_results.json"),
        "weighted_search": load_result(output_dir / "weighted_search_results.json"),
        "drift_cohort": load_result(output_dir / "drift_cohort_results.json"),
    }

    # Count loaded results
    loaded = sum(1 for k, v in results.items() if v and k != "metadata")
    print(f"✓ Loaded {loaded}/4 result files")

    # Write consolidated JSON
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Wrote consolidated results to {results_file}")

    # Write markdown summary
    summary = generate_markdown_summary(results, timestamp)
    summary_file = output_dir / "SUMMARY.md"
    with open(summary_file, "w") as f:
        f.write(summary)
    print(f"✓ Wrote summary to {summary_file}")

    print("\n" + "=" * 70)
    print("✅ Consolidation complete")
    print("=" * 70)
    print(f"  Results:  {results_file}")
    print(f"  Summary:  {summary_file}")
    print()


if __name__ == "__main__":
    main()
