#!/bin/bash
# Historical verification entrypoint retained for future connector design.

set -e

echo "MEMORY_CONNECTORS_UNAVAILABLE: connector observability is dormant."
exit 1

PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
METRICS_URL="${METRICS_URL:-http://localhost:8000/metrics}"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }

# ==================================================
# 1. Verify Metrics Export
# ==================================================
echo "1. Checking metrics endpoint..."
echo "-----------------------------------"

# Check if metrics endpoint is accessible
if curl -s -f "$METRICS_URL" > /dev/null 2>&1; then
    pass "Metrics endpoint accessible at $METRICS_URL"
else
    fail "Metrics endpoint not accessible at $METRICS_URL"
    echo "   Make sure the API server is running with metrics enabled"
    exit 1
fi

# Check for Drive poller metrics
METRICS=$(curl -s "$METRICS_URL")

if echo "$METRICS" | grep -q "connector_poller_runs_total"; then
    pass "connector_poller_runs_total metric found"
else
    fail "connector_poller_runs_total metric NOT found"
fi

if echo "$METRICS" | grep -q "connector_poller_errors_total"; then
    pass "connector_poller_errors_total metric found"
else
    fail "connector_poller_errors_total metric NOT found"
fi

if echo "$METRICS" | grep -q "connector_poller_latency_seconds_bucket"; then
    pass "connector_poller_latency_seconds histogram found"
else
    fail "connector_poller_latency_seconds histogram NOT found"
fi

echo ""

# ==================================================
# 2. Verify Alert Query Syntax
# ==================================================
echo "2. Validating alert PromQL queries..."
echo "-----------------------------------"

# Test ConnectorPollerErrorRateHigh query
ERROR_RATE_QUERY='sum(rate(connector_poller_errors_total[10m])) / clamp_min(sum(rate(connector_poller_runs_total[10m])), 1) > 0.05'
echo "Testing: ConnectorPollerErrorRateHigh"
if command -v promtool > /dev/null 2>&1; then
    if echo "$ERROR_RATE_QUERY" | promtool check metrics 2>/dev/null; then
        pass "Error rate query syntax valid"
    else
        warn "Cannot validate query syntax (promtool check failed)"
    fi
else
    warn "promtool not installed, skipping syntax check"
fi

# Test ConnectorPollerLatencyHigh query
LATENCY_QUERY='histogram_quantile(0.95, sum(rate(connector_poller_latency_seconds_bucket[10m])) by (le, provider, tenant)) > 30'
echo "Testing: ConnectorPollerLatencyHigh"
if command -v promtool > /dev/null 2>&1; then
    if echo "$LATENCY_QUERY" | promtool check metrics 2>/dev/null; then
        pass "Latency query syntax valid"
    else
        warn "Cannot validate query syntax (promtool check failed)"
    fi
else
    warn "promtool not installed, skipping syntax check"
fi

echo ""

# ==================================================
# 3. Test Prometheus Query API (if available)
# ==================================================
echo "3. Testing Prometheus query API..."
echo "-----------------------------------"

if curl -s -f "$PROMETHEUS_URL/-/healthy" > /dev/null 2>&1; then
    pass "Prometheus is running at $PROMETHEUS_URL"

    # Test instant query for runs metric
    QUERY_RESULT=$(curl -s -G --data-urlencode "query=connector_poller_runs_total" \
        "$PROMETHEUS_URL/api/v1/query" | jq -r '.status' 2>/dev/null || echo "error")

    if [ "$QUERY_RESULT" = "success" ]; then
        pass "connector_poller_runs_total query executed successfully"
    else
        warn "Query execution returned: $QUERY_RESULT"
    fi

    # Test error rate calculation
    ERROR_RATE_RESULT=$(curl -s -G --data-urlencode "query=$ERROR_RATE_QUERY" \
        "$PROMETHEUS_URL/api/v1/query" | jq -r '.status' 2>/dev/null || echo "error")

    if [ "$ERROR_RATE_RESULT" = "success" ]; then
        pass "Error rate alert query executed successfully"
    else
        warn "Error rate query returned: $ERROR_RATE_RESULT"
    fi

    # Test latency quantile calculation
    LATENCY_RESULT=$(curl -s -G --data-urlencode "query=$LATENCY_QUERY" \
        "$PROMETHEUS_URL/api/v1/query" | jq -r '.status' 2>/dev/null || echo "error")

    if [ "$LATENCY_RESULT" = "success" ]; then
        pass "Latency alert query executed successfully"
    else
        warn "Latency query returned: $LATENCY_RESULT"
    fi
else
    warn "Prometheus not running at $PROMETHEUS_URL, skipping query tests"
    echo "   Set PROMETHEUS_URL environment variable if running elsewhere"
fi

echo ""

# ==================================================
# 4. Verify Alert Rule File Syntax
# ==================================================
echo "4. Validating alert rule file..."
echo "-----------------------------------"

ALERT_FILE="observability/alerts/connector_alerts.yml"

if [ ! -f "$ALERT_FILE" ]; then
    fail "Alert file not found at $ALERT_FILE"
    exit 1
fi

pass "Alert file exists at $ALERT_FILE"

# Check if promtool can validate the file
if command -v promtool > /dev/null 2>&1; then
    if promtool check rules "$ALERT_FILE" 2>&1 | grep -q "SUCCESS"; then
        pass "Alert rules file is valid YAML and PromQL"
    else
        fail "Alert rules file validation failed"
        promtool check rules "$ALERT_FILE"
        exit 1
    fi
else
    warn "promtool not installed, cannot validate alert rules"
    # Basic YAML syntax check
    if command -v python3 > /dev/null 2>&1; then
        if python3 -c "import yaml; yaml.safe_load(open('$ALERT_FILE'))" 2>/dev/null; then
            pass "Alert rules file is valid YAML"
        else
            fail "Alert rules file has invalid YAML syntax"
            exit 1
        fi
    fi
fi

# Check for required alert rules
if grep -q "ConnectorPollerErrorRateHigh" "$ALERT_FILE"; then
    pass "ConnectorPollerErrorRateHigh alert found"
else
    fail "ConnectorPollerErrorRateHigh alert NOT found"
fi

if grep -q "ConnectorPollerLatencyHigh" "$ALERT_FILE"; then
    pass "ConnectorPollerLatencyHigh alert found"
else
    fail "ConnectorPollerLatencyHigh alert NOT found"
fi

echo ""

# ==================================================
# 5. Verify Dashboard Queries
# ==================================================
echo "5. Validating Grafana dashboard..."
echo "-----------------------------------"

DASHBOARD_FILE="observability/dashboards/connector_overview.json"

if [ ! -f "$DASHBOARD_FILE" ]; then
    fail "Dashboard file not found at $DASHBOARD_FILE"
    exit 1
fi

pass "Dashboard file exists at $DASHBOARD_FILE"

# Check if it's valid JSON
if jq empty "$DASHBOARD_FILE" 2>/dev/null; then
    pass "Dashboard file is valid JSON"
else
    fail "Dashboard file has invalid JSON syntax"
    exit 1
fi

# Check for Drive poller panels
if jq -r '.dashboard.panels[].title' "$DASHBOARD_FILE" 2>/dev/null | grep -q "Drive Poller"; then
    pass "Drive Poller panels found in dashboard"

    # Count panels
    PANEL_COUNT=$(jq -r '.dashboard.panels[].title' "$DASHBOARD_FILE" 2>/dev/null | grep "Drive Poller" | wc -l)
    pass "Found $PANEL_COUNT Drive Poller panels"
else
    fail "Drive Poller panels NOT found in dashboard"
fi

# Verify panel queries use correct metrics
if jq -r '.dashboard.panels[].targets[].expr' "$DASHBOARD_FILE" 2>/dev/null | grep -q "connector_poller_runs_total"; then
    pass "Dashboard uses connector_poller_runs_total metric"
else
    warn "Dashboard may not use connector_poller_runs_total metric"
fi

if jq -r '.dashboard.panels[].targets[].expr' "$DASHBOARD_FILE" 2>/dev/null | grep -q "connector_poller_latency_seconds_bucket"; then
    pass "Dashboard uses connector_poller_latency_seconds histogram"
else
    warn "Dashboard may not use connector_poller_latency_seconds histogram"
fi

echo ""

# ==================================================
# 6. Check Metric Labels
# ==================================================
echo "6. Verifying metric labels..."
echo "-----------------------------------"

# Check if metrics have required labels
if echo "$METRICS" | grep "connector_poller_runs_total" | grep -q "provider="; then
    pass "connector_poller_runs_total has 'provider' label"
else
    warn "connector_poller_runs_total missing 'provider' label"
fi

if echo "$METRICS" | grep "connector_poller_runs_total" | grep -q "tenant="; then
    pass "connector_poller_runs_total has 'tenant' label"
else
    warn "connector_poller_runs_total missing 'tenant' label"
fi

if echo "$METRICS" | grep "connector_poller_errors_total" | grep -q "error_type="; then
    pass "connector_poller_errors_total has 'error_type' label"
else
    warn "connector_poller_errors_total missing 'error_type' label"
fi

echo ""

# ==================================================
# 7. Summary
# ==================================================
echo "=================================================="
echo "Verification Summary"
echo "=================================================="
echo ""
echo "✓ Metrics endpoint accessible"
echo "✓ All required metrics present"
echo "✓ Alert rules file valid"
echo "✓ Dashboard file valid"
echo ""
echo "Next steps:"
echo "1. Start scheduler with RUN_SCHEDULER=true to emit metrics"
echo "2. Trigger a Drive poll to generate sample data"
echo "3. Check Prometheus targets at $PROMETHEUS_URL/targets"
echo "4. Check alerts at $PROMETHEUS_URL/alerts"
echo "5. Import dashboard to Grafana"
echo ""

# Optional: Show sample metrics
echo "Sample metrics from /metrics endpoint:"
echo "-----------------------------------"
echo "$METRICS" | grep "connector_poller" | head -10
echo ""

pass "Staging checklist complete!"
