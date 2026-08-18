#!/bin/bash
#
# Run all Active Graph KG evaluations
#
# Usage:
#   ./run_all.sh [--api-url URL]
#
# Options:
#   --api-url URL    API base URL (default: http://localhost:8000)
#

set -e  # Exit on error

# Default configuration
API_URL="http://localhost:8000"
OUTPUT_DIR="evaluation"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo "Active Graph KG Evaluation Suite"
echo "======================================================================"
echo "API URL: $API_URL"
echo "Output directory: $OUTPUT_DIR"
echo "Timestamp: $(date -Iseconds)"
echo ""

# Check API is running
echo "Checking API health..."
if ! curl -s -f "$API_URL/health" > /dev/null 2>&1; then
    echo "❌ Error: Cannot connect to API at $API_URL"
    echo "   Make sure the API server is running."
    exit 1
fi
echo "✓ API is reachable"
echo ""

# 1. Latency Benchmark
echo "======================================================================"
echo "1/5: Latency Benchmark"
echo "======================================================================"
python3 "$OUTPUT_DIR/latency_benchmark.py" \
    --api-url "$API_URL" \
    --num-requests 100 \
    --warmup 10 \
    --output "$OUTPUT_DIR/latency_results.json" || true
echo ""

# 2. Freshness SLA Monitor
echo "======================================================================"
echo "2/5: Freshness SLA Monitor"
echo "======================================================================"
python3 "$OUTPUT_DIR/freshness_monitor.py" \
    --api-url "$API_URL" \
    --output "$OUTPUT_DIR/freshness_results.json" || true
echo ""

# 3. Weighted Search Evaluation (requires datasets)
echo "======================================================================"
echo "3/5: Weighted Search Evaluation"
echo "======================================================================"
if [ -f "$OUTPUT_DIR/datasets/test_queries.json" ] && [ -f "$OUTPUT_DIR/datasets/ground_truth.json" ]; then
    python3 "$OUTPUT_DIR/weighted_search_eval.py" \
        --api-url "$API_URL" \
        --queries "$OUTPUT_DIR/datasets/test_queries.json" \
        --ground-truth "$OUTPUT_DIR/datasets/ground_truth.json" \
        --top-k 20 \
        --output "$OUTPUT_DIR/weighted_search_results.json" || true
else
    echo "⚠ Skipping: Test datasets not found"
    echo "   Create: $OUTPUT_DIR/datasets/test_queries.json"
    echo "   Create: $OUTPUT_DIR/datasets/ground_truth.json"
fi
echo ""

# 4. Drift Cohort Analysis (requires datasets)
echo "======================================================================"
echo "4/5: Drift Cohort Analysis"
echo "======================================================================"
if [ -f "$OUTPUT_DIR/datasets/test_queries.json" ] && [ -f "$OUTPUT_DIR/datasets/ground_truth.json" ]; then
    python3 "$OUTPUT_DIR/drift_cohort_analysis.py" \
        --api-url "$API_URL" \
        --queries "$OUTPUT_DIR/datasets/test_queries.json" \
        --ground-truth "$OUTPUT_DIR/datasets/ground_truth.json" \
        --drift-threshold 0.2 \
        --output "$OUTPUT_DIR/drift_cohort_results.json" || true
else
    echo "⚠ Skipping: Test datasets not found"
fi
echo ""

# Summary
echo "======================================================================"
echo "Evaluation Complete"
echo "======================================================================"
echo "Results saved to:"
echo "  - $OUTPUT_DIR/latency_results.json"
echo "  - $OUTPUT_DIR/freshness_results.json"
echo "  - $OUTPUT_DIR/weighted_search_results.json"
echo "  - $OUTPUT_DIR/drift_cohort_results.json"
echo ""
echo "Next steps:"
echo "  1. Review results in each JSON file"
echo "  2. Fill out evaluation/REPORT_TEMPLATE.md"
echo "  3. Share report with stakeholders"
echo ""
echo "Timestamp: $(date -Iseconds)"
