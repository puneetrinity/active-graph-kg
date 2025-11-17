# actvgraph-kg - Quick Start

**One-Line Definition:** A self-refreshing knowledge graph where nodes auto-refresh, track drift, and emit semantic triggers.

**Status:** ✅ Production Ready (Dual ANN, RLS, JWT auth, rate limiting, metrics)
**Last Updated:** 2025-11-17

---

## 🚀 Quick Sanity Steps (5 Minutes)

### **1. Initialize Database**
```bash
# Start PostgreSQL
docker compose up -d

# Run schema (creates tables, indexes, patterns table)
psql -h localhost -U activekg -d activekg -f db/init.sql

# Enable hybrid text search (BM25) for hybrid retrieval
psql -h localhost -U activekg -d activekg -f db/migrations/add_text_search.sql

# Verify tables created
psql -h localhost -U activekg -d activekg -c "\dt"
# Expected: nodes, edges, node_versions, events, embedding_history, patterns
```

### **2. Enable Vector Index**
```bash
# Choose IVFFLAT for <1M vectors (recommended for most cases)
psql -h localhost -U activekg -d activekg -f enable_vector_index.sql

# Verify index created
psql -h localhost -U activekg -d activekg -c "\di idx_nodes_embedding*"
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Start API**
```bash
# Database (or use DATABASE_URL for Railway/Heroku)
export ACTIVEKG_DSN="postgresql://activekg:activekg@localhost:5432/activekg"

# Embedding config
export EMBEDDING_BACKEND="sentence-transformers"
export EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Ask endpoint tuning
export ASK_SIM_THRESHOLD=0.30
export ASK_MAX_TOKENS=256
export ASK_MAX_SNIPPETS=3
export ASK_SNIPPET_LEN=300
export HYBRID_RERANKER_CANDIDATES=20

# Run scheduler on exactly one instance
export RUN_SCHEDULER=true  # false on replicas

# Optional: JWT auth (disable for quick dev)
export JWT_ENABLED=false
# When enabled, use:
# export JWT_ENABLED=true
# export JWT_SECRET_KEY='your-32-char-secret-key-here'
# export JWT_ALGORITHM=HS256
# export JWT_AUDIENCE=activekg
# export JWT_ISSUER=https://auth.yourcompany.com

uvicorn activekg.api.main:app --reload
```

### **5. Run Smoke Test**
```bash
# In separate terminal
python scripts/smoke_test.py
```

**Expected Output:**
```
✓ API is running
=== Test 1: Refresh Cycle ===
✓ Created node: <uuid>
...
=== Test 4: Semantic Search ===
✓ Search returned N results
✅ SMOKE TEST PASSED
```

---

## 📋 Smoke Test Scenarios

The smoke test validates these critical flows:

### **Test 1: Refresh Cycle with Drift Gating**
1. Creates node with `refresh_policy: {"interval": "1m", "drift_threshold": 0.15}`
2. Waits 65 seconds for scheduler cycle
3. Verifies `refreshed` event only emitted if drift > 0.15
4. Checks `embedding_history` table has entry

**What this proves:**
- Scheduler runs and selects due nodes
- Drift calculated correctly (cosine distance)
- Events gated by threshold
- History persisted

### **Test 2: Pattern Registration & Trigger Firing**
1. Registers fraud detection pattern via `POST /triggers`
2. Verifies pattern persisted in DB (not in-memory)
3. Creates node with matching content + trigger
4. Waits 125 seconds for trigger cycle
5. Checks for `trigger_fired` events

**What this proves:**
- Patterns stored in database
- Trigger engine runs periodically
- Similarity thresholds work
- Events emitted correctly

### **Test 3: Lineage Chain Traversal**
1. Creates 3 nodes: A (child), B (intermediate), C (parent)
2. Creates edges: A→B→C with `rel='DERIVED_FROM'`
3. Calls `GET /lineage/A?max_depth=5`
4. Verifies 2 ancestors returned (B at depth 1, C at depth 2)

**What this proves:**
- Recursive CTE works
- DERIVED_FROM convention followed
- Edge metadata preserved
- Depth limiting works

### **Test 4: Semantic Search**
1. Searches for "machine learning neural networks research"
2. Verifies results returned with similarity scores
3. Checks scores are in range [0, 1]

**What this proves:**
- pgvector cosine distance works
- Embeddings generated correctly
- Top-K ranking works
- API integration complete

---

## 🎯 Manual Testing Examples

### **Create Self-Refreshing Node**
```bash
# With interval policy
curl -X POST http://localhost:8000/nodes \
  -H "Content-Type: application/json" \
  -d '{
    "classes": ["Document"],
    "props": {"text": "AI safety research findings", "category": "research"},
    "refresh_policy": {"interval": "5m", "drift_threshold": 0.15}
  }'

# Response: {"id": "<node-id>"}

# With cron policy (every 5 minutes, UTC)
curl -X POST http://localhost:8000/nodes \
  -H "Content-Type: application/json" \
  -d '{
    "classes": ["Document"],
    "props": {"text": "Daily market report", "category": "finance"},
    "refresh_policy": {"cron": "*/5 * * * *", "drift_threshold": 0.1}
  }'

# Cron examples:
#   "*/5 * * * *"  - Every 5 minutes
#   "0 * * * *"    - Every hour at :00
#   "0 2 * * *"    - Every day at 2:00 AM UTC
#   "0 9 * * 1"    - Every Monday at 9:00 AM UTC
#
# Note: Cron takes precedence over interval if both present
```

### **Semantic Search**
```bash
# Basic search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning optimization techniques",
    "top_k": 5,
    "metadata_filters": {"category": "research"}
  }'

# Response: {"query": "...", "results": [...], "count": N}
```

### **Weighted Search (Recency/Drift)**
```bash
# Search with recency/drift weighting (fresher nodes rank higher)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning optimization techniques",
    "top_k": 5,
    "use_weighted_score": true,
    "decay_lambda": 0.01,
    "drift_beta": 0.1
  }'

# Parameters:
#   use_weighted_score: Enable weighted ranking (default: false)
#   decay_lambda: Age decay rate, 0.01 = ~1% penalty per day (default: 0.01)
#   drift_beta: Drift penalty weight, 0.1 = 10% penalty per drift unit (default: 0.1)
#
# Formula: weighted_score = similarity * exp(-decay_lambda * age_days) * (1 - drift_beta * drift_score)
```

### **Register Trigger Pattern**
```bash
curl -X POST http://localhost:8000/triggers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "fraud_alert",
    "example_text": "suspicious wire transfer to offshore account",
    "description": "Detects potential fraud patterns"
  }'

# Response: {"status": "registered", "name": "fraud_alert"}
```

### **List Patterns (Verify DB Persistence)**
```bash
curl http://localhost:8000/triggers

# Response: {"patterns": [{"name": "fraud_alert", "description": "...", ...}], "count": 1}
```

### **Create Lineage Edge**
```bash
curl -X POST http://localhost:8000/edges \
  -H "Content-Type: application/json" \
  -d '{
    "src": "<child-node-id>",
    "rel": "DERIVED_FROM",
    "dst": "<parent-node-id>",
    "props": {"transform": "summarize", "confidence": 0.95}
  }'

# Response: {"status": "created", "src": "...", "rel": "DERIVED_FROM", "dst": "..."}
```

### **Traverse Lineage**
```bash
curl "http://localhost:8000/lineage/<node-id>?max_depth=5"

# Response: {"node_id": "...", "ancestors": [...], "depth": N}
```

### **List Events**
```bash
# All events
curl "http://localhost:8000/events?limit=20"

# Refreshed events only
curl "http://localhost:8000/events?event_type=refreshed&limit=20"

# Events for specific node
curl "http://localhost:8000/events?node_id=<node-id>&limit=20"
```

---

## 🔍 Verification Queries (SQL)

### **Check Explicit Columns**
```sql
SELECT id, classes, last_refreshed, drift_score
FROM nodes
WHERE last_refreshed IS NOT NULL
LIMIT 5;
```

### **Check Patterns Table (DB-Backed)**
```sql
SELECT name, description, created_at, updated_at
FROM patterns
ORDER BY created_at DESC;
```

### **Check Embedding History**
```sql
SELECT node_id, drift_score, created_at
FROM embedding_history
ORDER BY created_at DESC
LIMIT 10;
```

### **Check Lineage Edges**
```sql
SELECT src, rel, dst, props
FROM edges
WHERE rel = 'DERIVED_FROM';
```

### **Check Events by Type**
```sql
SELECT type, COUNT(*) as count
FROM events
GROUP BY type
ORDER BY count DESC;
```

### **Check Vector Index Exists**
```sql
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'nodes' AND indexname LIKE '%embedding%';
```

---

## 🚨 Troubleshooting

### **Issue: "relation 'patterns' does not exist"**
**Solution:** Re-run schema initialization
```bash
psql -h localhost -U activekg -d activekg -f db/init.sql
```

### **Issue: "column 'last_refreshed' does not exist"**
**Solution:** Drop and recreate nodes table (⚠️ deletes data)
```bash
psql -h localhost -U activekg -d activekg -c "DROP TABLE IF EXISTS nodes CASCADE;"
psql -h localhost -U activekg -d activekg -f db/init.sql
```

### **Issue: Smoke test hangs on "Waiting 65 seconds"**
**Cause:** Scheduler not running in background

**Solution:** Start scheduler in separate terminal:
```python
# scheduler_runner.py
from activekg.refresh.scheduler import RefreshScheduler
from activekg.triggers.trigger_engine import TriggerEngine
from activekg.triggers.pattern_store import PatternStore
from activekg.graph.repository import GraphRepository
from activekg.engine.embedding_provider import EmbeddingProvider
import os

DSN = os.getenv("ACTIVEKG_DSN", "postgresql://activekg:activekg@localhost:5432/activekg")

repo = GraphRepository(DSN)
embedder = EmbeddingProvider()
pattern_store = PatternStore(DSN)
trigger_engine = TriggerEngine(pattern_store, repo)

scheduler = RefreshScheduler(repo, embedder, trigger_engine)
scheduler.start()

print("Scheduler running... Press Ctrl+C to stop")
try:
    import time
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    scheduler.shutdown()
    print("\nScheduler stopped")
```

```bash
python scheduler_runner.py
```

### **Issue: Search returns empty results**
**Cause:** No nodes with embeddings in database

**Solution:** Create test nodes via API (see examples above)

### **Issue: pgvector adapter error**
**Cause:** Vector extension not registered

**Already Fixed:** `register_vector()` called in repository.py:25 and pattern_store.py:23

---

## 📁 Key Files Reference

| File | Purpose | Lines |
|------|---------|-------|
| `db/init.sql` | Schema with explicit columns, patterns table, indexes | 93 |
| `activekg/graph/repository.py` | CRUD, vector_search, lineage, payload loaders | ~417 |
| `activekg/refresh/scheduler.py` | Refresh cycle with drift gating, trigger integration | 94 |
| `activekg/triggers/pattern_store.py` | DB-backed pattern persistence | 78 |
| `activekg/api/main.py` | All 11 API endpoints | 249 |
| `scripts/smoke_test.py` | E2E validation | 233 |
| `enable_vector_index.sql` | Vector index creation | 17 |

---

## ⏭️ Next Steps After Quickstart

1. **Enable Production Security** (See `docs/operations/security.md`):
   - Row-Level Security (RLS) for multi-tenancy
   - Actor IDs in events
   - File/HTTP/S3 size limits

2. **Evaluate Hybrid Retrieval**:
   - Ensure `db/migrations/add_text_search.sql` applied
   - Use `/search` with `use_hybrid=true` or rely on hybrid inside `/ask`
   - Tune `HYBRID_RERANKER_CANDIDATES`

3. **Deploy to Production**:
   - Railway: `railway up`
   - RunPod: See `llm/runpod-serverless/` for example
   - Docker: `docker build -t activekg .`

4. **Monitor KPIs** (See `IMPLEMENTATION_STATUS.md`):
   - Refresh coverage
   - Drift distribution (p50, p95)
   - Trigger precision
   - Search latency

---

## 📚 Documentation Index

- `README.md` - Project overview
- **`QUICKSTART.md` (this file)** - 5-minute setup
- `IMPLEMENTATION_STATUS.md` - Complete feature inventory
- `PHASE1_VERIFICATION.md` - Detailed verification guide
- `PHASE_1_5_TODO.md` - Minor gaps + roadmap
- `docs/operations/security.md` - Hardening checklist

---

**Status:** Phase 1 Complete ✅
**Time to Working Demo:** 5 minutes
**Time to Production:** +1-2 days (security hardening)
### **Q&A (Citations + Lineage)**
```bash
# Non-streaming
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What vector databases are discussed?", "max_results": 3}'

# Streaming SSE
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What vector databases are discussed?", "max_results": 3}'
```
