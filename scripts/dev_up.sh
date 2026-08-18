#!/bin/bash

# dev_up.sh - Clean development workflow for Active Graph KG
# Usage:
#   ./scripts/dev_up.sh                    # Start server only
#   ./scripts/dev_up.sh --seed             # Start server and seed data
#   ./scripts/dev_up.sh --seed --refresh   # Start, seed, and refresh embeddings
#   ./scripts/dev_up.sh --smoke            # Start and run smoke tests
#   ./scripts/dev_up.sh --cosine           # Start in cosine mode (default is RRF)

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse command line arguments
DO_SEED=false
DO_REFRESH=false
DO_SMOKE=false
USE_COSINE=false

for arg in "$@"; do
    case $arg in
        --seed) DO_SEED=true ;;
        --refresh) DO_REFRESH=true ;;
        --smoke) DO_SMOKE=true ;;
        --cosine) USE_COSINE=true ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --seed       Seed test data after server starts"
            echo "  --refresh    Refresh embeddings by class (requires --seed)"
            echo "  --smoke      Run smoke tests after server starts"
            echo "  --cosine     Use cosine scoring mode (default is RRF)"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Change to project directory
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)

echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}Active Graph KG - Development Setup${NC}"
echo -e "${GREEN}===========================================${NC}"

# Step 1: Clean up old processes
echo -e "\n${YELLOW}[1/6] Cleaning up old processes...${NC}"
pkill -9 -f "uvicorn activekg" 2>/dev/null || true
sleep 2
echo "✓ Old processes killed"

# Step 2: Clear Python cache
echo -e "\n${YELLOW}[2/6] Clearing Python cache...${NC}"
find activekg -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find activekg -type f -name "*.pyc" -delete 2>/dev/null || true
echo "✓ Python cache cleared"

# Step 3: Activate venv and set environment
echo -e "\n${YELLOW}[3/6] Setting up environment...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${RED}ERROR: venv not found. Please create it first:${NC}"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

source venv/bin/activate
echo "✓ Virtual environment activated"

# Common environment variables
export PYTHONDONTWRITEBYTECODE=1
export ACTIVEKG_DSN="${ACTIVEKG_DSN:-postgresql:///activekg?host=/var/run/postgresql&port=5433}"
export GROQ_API_KEY="${GROQ_API_KEY:-your-groq-api-key-here}"
export JWT_ENABLED=false
export RATE_LIMIT_ENABLED=false
export LLM_ENABLED=true
export LLM_BACKEND=groq
export LLM_MODEL=llama-3.1-8b-instant
export RUN_SCHEDULER=false
export AUTO_EMBED_ON_CREATE=true
export ACTIVEKG_DEV_TENANT=default
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"

# Scoring mode configuration
if [ "$USE_COSINE" = true ]; then
    echo "  Scoring Mode: Cosine (RRF disabled)"
    export HYBRID_RRF_ENABLED=false
    export RAW_LOW_SIM_THRESHOLD=0.15
    export ASK_SIM_THRESHOLD=0.20
    export RERANK_SKIP_TOPSIM=0.80
else
    echo "  Scoring Mode: RRF (default)"
    export HYBRID_RRF_ENABLED=true
    export RRF_LOW_SIM_THRESHOLD=0.01
    export ASK_SIM_THRESHOLD=0.01
    export HYBRID_RRF_K=60
    export HYBRID_RERANKER_BASE=20
    export HYBRID_RERANKER_BOOST=45
    export HYBRID_ADAPTIVE_THRESHOLD=0.55
    export MAX_RERANK_BUDGET_MS=250
fi

echo "  DSN: $ACTIVEKG_DSN"
echo "  LLM: $LLM_BACKEND ($LLM_MODEL)"
echo "  Tenant: $ACTIVEKG_DEV_TENANT"

# Step 4: Start Redis if needed
echo -e "\n${YELLOW}[4/6] Starting Redis...${NC}"
redis-server --daemonize yes 2>/dev/null || echo "  Redis already running"
sleep 1
echo "✓ Redis ready"

# Step 5: Start server
echo -e "\n${YELLOW}[5/6] Starting API server...${NC}"
LOG_FILE="$PROJECT_DIR/dev_server.log"
PID_FILE="$PROJECT_DIR/dev_server.pid"

uvicorn activekg.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"
echo "  Server PID: $SERVER_PID"
echo "  Log file: $LOG_FILE"

# Wait for server to be healthy
echo "  Waiting for server to start..."
MAX_RETRIES=15
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is healthy and ready${NC}"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo -e "${RED}ERROR: Server failed to start${NC}"
        echo "Last 20 lines of log:"
        tail -20 "$LOG_FILE"
        exit 1
    fi
    sleep 2
done

# Display server info
echo ""
curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "Server is running"

# Step 6: Optional tasks
echo -e "\n${YELLOW}[6/6] Running optional tasks...${NC}"

# Seed data if requested
if [ "$DO_SEED" = true ]; then
    echo -e "\n${YELLOW}Seeding test data...${NC}"
    if [ -f "evaluation/datasets/seed_nodes.json" ]; then
        python3 -c "
import json
import requests

with open('evaluation/datasets/seed_nodes.json') as f:
    nodes = json.load(f)

for node in nodes:
    resp = requests.post('http://localhost:8000/nodes', json=node)
    if resp.status_code in [200, 201]:
        print(f\"✓ Seeded {node.get('classes', [])[0] if node.get('classes') else 'node'}\")
    else:
        print(f\"✗ Failed to seed node: {resp.status_code}\")
"
        echo -e "${GREEN}✓ Data seeded${NC}"
    else
        echo -e "${YELLOW}⚠ seed_nodes.json not found, skipping${NC}"
    fi
fi

# Refresh embeddings if requested
if [ "$DO_REFRESH" = true ]; then
    if [ "$DO_SEED" = false ]; then
        echo -e "${YELLOW}⚠ --refresh requires --seed, skipping${NC}"
    else
        echo -e "\n${YELLOW}Refreshing embeddings by class...${NC}"
        if [ -f "scripts/admin_refresh_by_class.py" ]; then
            python3 scripts/admin_refresh_by_class.py --class Job --tenant-id default --batch-size 100
            echo -e "${GREEN}✓ Embeddings refreshed${NC}"
        else
            echo -e "${YELLOW}⚠ admin_refresh_by_class.py not found, skipping${NC}"
        fi
    fi
fi

# Run smoke tests if requested
if [ "$DO_SMOKE" = true ]; then
    echo -e "\n${YELLOW}Running smoke tests...${NC}"

    # Test 1: direct hybrid search
    echo -e "\n  Test 1: Hybrid search"
    HYBRID_RESULT=$(curl -s -X POST http://localhost:8000/search \
        -H "Content-Type: application/json" \
        -d '{"query":"machine learning engineer frameworks","use_hybrid":true,"top_k":5}' \
        | python3 -c "import sys, json; d=json.load(sys.stdin); print(json.dumps({'count': d.get('count'), 'applied_limit': d.get('applied_limit')}, indent=2))")
    echo "$HYBRID_RESULT"

    # Test 2: direct vector search
    echo -e "\n  Test 2: Vector-only search"
    VECTOR_RESULT=$(curl -s -X POST http://localhost:8000/search \
        -H "Content-Type: application/json" \
        -d '{"query":"machine learning engineer frameworks","use_hybrid":false,"top_k":5}' \
        | python3 -c "import sys, json; d=json.load(sys.stdin); print(json.dumps({'count': d.get('count'), 'applied_limit': d.get('applied_limit')}, indent=2))")
    echo "$VECTOR_RESULT"

    # Test 3: Embedding info
    echo -e "\n  Test 3: Embedding configuration"
    EMBED_RESULT=$(curl -s http://localhost:8000/debug/embed_info \
        | python3 -c "import sys, json; d=json.load(sys.stdin); print(json.dumps({'backend': d.get('backend'), 'model': d.get('model'), 'dimension': d.get('vector_dimension',{}).get('db_dim')}, indent=2))")
    echo "$EMBED_RESULT"

    echo -e "\n${GREEN}✓ Smoke tests completed${NC}"
fi

# Final summary
echo -e "\n${GREEN}===========================================${NC}"
echo -e "${GREEN}Development server is ready!${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""
echo "Server URL:    http://localhost:8000"
echo "Health Check:  http://localhost:8000/health"
echo "API contract:  offline only (cd frontend && npm run generate-api)"
echo "Log File:      $LOG_FILE"
echo "PID File:      $PID_FILE"
echo ""
echo "To stop the server:"
echo "  kill \$(cat $PID_FILE)"
echo "  # or"
echo "  pkill -f 'uvicorn activekg'"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_FILE"
echo ""
