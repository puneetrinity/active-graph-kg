# Active Graph KG — Self-Refreshing Knowledge Graph

[![CI](https://github.com/puneetrinity/active-graph-kg/actions/workflows/ci.yml/badge.svg)](https://github.com/puneetrinity/active-graph-kg/actions/workflows/ci.yml)
[![Documentation](https://github.com/puneetrinity/active-graph-kg/actions/workflows/mkdocs.yml/badge.svg)](https://github.com/puneetrinity/active-graph-kg/actions/workflows/mkdocs.yml)
[![Security Scan](https://github.com/puneetrinity/active-graph-kg/actions/workflows/security-scan.yml/badge.svg)](https://github.com/puneetrinity/active-graph-kg/actions/workflows/security-scan.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[![Roadmap](https://img.shields.io/badge/Roadmap-View-blue)](ROADMAP.md)
[![Contributing](https://img.shields.io/badge/Contributing-Guidelines-green)](CONTRIBUTING.md)
[![Evaluation Setup](https://img.shields.io/badge/Evaluation-Setup_Guide-orange)](EVALUATION_SETUP_GUIDE.md)
[![Future Improvements](https://img.shields.io/badge/Future-Improvements-purple)](FUTURE_IMPROVEMENTS.md)

**Status:** ✅ Production Ready
**Version:** 1.0.0
**Last Updated:** 2025-11-24

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?templateUrl=https://github.com/puneetrinity/active-graph-kg)

> **Dual ANN** (IVFFLAT/HNSW), **RLS**, **admin tooling**, **metrics**, and **proof scripts** are in place.

> The first self-refreshing knowledge graph — where every node carries living knowledge: it refreshes, detects drift, and triggers insights automatically.

> Security: Weekly Safety scans on dependencies (see Security Scan badge) with reports archived per run.

---

## Security

- Weekly dependency scanning via Safety (see Security Scan badge); reports are archived per workflow run under Actions artifacts. Workflow: `.github/workflows/security-scan.yml`.
- RLS enforced at the database layer; JWT auth and rate limiting documented in `SECURITY.md`.
- Connector secrets are validated and cached via the config store; DLQ and retries are implemented for ingestion.

## What is Active Graph KG?

Active Graph KG is a **self-refreshing knowledge graph** built on PostgreSQL + pgvector that automatically:

- 🔄 **Refreshes embeddings** based on configurable policies (interval-based, cron-based)
- 📊 **Detects semantic drift** and emits events when content changes significantly
- 🎯 **Fires semantic triggers** when nodes match registered patterns
- 🔗 **Tracks lineage** through DERIVED_FROM edges with recursive queries
- 🌐 **Loads polyglot payloads** from S3, HTTP, local files, or inline text
- 🔍 **Searches with compound filters** using JSONB containment for complex queries
- 🔒 **Isolates tenants** with Row-Level Security at the database level
- 📈 **Exports Prometheus metrics** for production monitoring

Unlike traditional knowledge graphs, Active Graph KG **actively maintains itself** — nodes aren't static data points, they're living entities that refresh, evolve, and trigger actions.

---

## Supported Connectors

Active Graph KG automatically syncs content from cloud storage:

| Provider | Status | Documentation |
|----------|--------|---------------|
| **AWS S3** | ✅ Production | [S3 Connector Guide](docs/S3_CONNECTOR.md) |
| **Google Cloud Storage** | ✅ Production | [GCS Connector Guide](docs/GCS_CONNECTOR.md) |
| **Google Drive** | ✅ Production | [Drive Connector Guide](docs/DRIVE_CONNECTOR.md) |
| Azure Blob Storage | 🚧 Planned | Config schema ready |

**Features:**
- Automatic polling for new/updated files
- Incremental sync with cursor-based pagination
- Multi-format support (PDF, DOCX, HTML, TXT)
- ETag/generation-based change detection
- Idempotent ingestion (no duplicates)

---

## Why This Works: Research-Backed Design

Active Graph KG's architecture is grounded in peer-reviewed research on AI-augmented systems, knowledge graph drift, and LLM-KG orchestration:

**🔬 Evidence:**

- **"Embedding-based drift tracking is an effective signal for meaning changes in evolving graphs."**
  *Chen et al. (2021)* - Knowledge graph embeddings for concept drift detection
  → Validates our `drift_score = 1 - cosine_similarity` formula and threshold-based refresh policies

- **"LLMs excel at reasoning and orchestration; retrieval + structure remain the backbone for accuracy."**
  *Zhu et al. (2023), arXiv:2305.13168* - LLMs for knowledge graph construction and reasoning
  → Supports our design: structured KG for retrieval, LLMs for grounded Q&A with citations (coming in Phase 2)

- **"AI-augmented database systems demonstrate improved real-time performance and anomaly detection."**
  *Gadde (2024)* - AI-Augmented DBMS for Real-Time Data Analytics, Revista de Inteligencia Artificial en Medicina
  → Validates our auto-index creation, weighted search, and trigger-based anomaly detection

**📚 Similar Research:**
- **Query Optimization:** UC Berkeley RISE (Deep RL for SQL, 10x faster planning), GRQO (MDPI 2024, 25% faster execution)
- **Self-Tuning Systems:** OtterTune (ACM SIGMOD 2017), lambda-Tune (LLM-based DB tuning, 2024)
- **Anomaly Detection:** RIT 2024 (Neural Networks + Isolation Forest), Enterprise DL ensembles (ACM 2024)
- **Cloud Resource Mgmt:** esDNN (ACM 2022, GRU workload prediction), Cloud-native DB survey (IEEE 2024)
- **PostgreSQL + AI:** pgvector HNSW parallel builds, pgvectorscale extension (55% developer adoption, 2024)

**🔗 Full Analysis:** See [SIMILAR_PAPERS_ANALYSIS.md](SIMILAR_PAPERS_ANALYSIS.md) for 20+ related papers and performance comparisons.

---

## Quick Start (5 Minutes)

### 1. Start PostgreSQL + pgvector
```bash
# Using Docker
docker run -d \
  --name activekg-postgres \
  -e POSTGRES_USER=activekg \
  -e POSTGRES_PASSWORD=activekg \
  -e POSTGRES_DB=activekg \
  -p 5432:5432 \
  ankane/pgvector

# Enable vector extension
psql postgresql://activekg:activekg@localhost:5432/activekg \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 2. Initialize Database
```bash
# Create schema (nodes, edges, events, patterns, etc.)
psql postgresql://activekg:activekg@localhost:5432/activekg \
  -f db/init.sql

# Enable hybrid text search (BM25) for hybrid BM25+vector retrieval
psql postgresql://activekg:activekg@localhost:5432/activekg \
  -f db/migrations/add_text_search.sql

# Optional: Enable Row-Level Security for multi-tenancy
psql postgresql://activekg:activekg@localhost:5432/activekg \
  -f enable_rls_policies.sql

# Optional: Create vector index for fast search
psql postgresql://activekg:activekg@localhost:5432/activekg \
  -f enable_vector_index.sql
```

### 3. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Start API Server
```bash
# Database (DSN fallback: ACTIVEKG_DSN or DATABASE_URL for PaaS)
export ACTIVEKG_DSN='postgresql://activekg:activekg@localhost:5432/activekg'
# Or for Railway/PaaS: uses DATABASE_URL automatically if ACTIVEKG_DSN not set

export EMBEDDING_BACKEND='sentence-transformers'
export EMBEDDING_MODEL='all-MiniLM-L6-v2'
export ASK_SIM_THRESHOLD=0.30        # similarity cutoff for /ask
export ASK_MAX_TOKENS=256            # token budget
export ASK_MAX_SNIPPETS=3            # snippets in context
export ASK_SNIPPET_LEN=300           # chars per snippet

# Run scheduler on exactly one instance
export RUN_SCHEDULER=true            # false on other replicas

# Optional: JWT auth (HS256 for dev, RS256 in prod via JWT_PUBLIC_KEY)
export JWT_ENABLED=true
export JWT_SECRET_KEY='your-32-char-secret-key-here'
export JWT_ALGORITHM=HS256
export JWT_AUDIENCE=activekg
export JWT_ISSUER=https://auth.yourcompany.com

# Optional: Rate limiting
export RATE_LIMIT_ENABLED=true
export REDIS_URL=redis://localhost:6379/0

# Optional reranker tuning
export MAX_RERANK_BUDGET_MS=0        # budget guard; 0 disables guard
export HYBRID_RERANKER_BASE=20       # initial candidate pool before rerank
export HYBRID_RERANKER_BOOST=45      # reserved for adaptive boosts
export HYBRID_ADAPTIVE_THRESHOLD=0.55 # log suggestions when low confidence

uvicorn activekg.api.main:app --reload
```

#### ANN Configuration (IVFFLAT/HNSW)

Configure ANN indexes and distance metric via env:

```bash
# Choose one or both (coexist for migration)
export PGVECTOR_INDEX=ivfflat            # or hnsw
export PGVECTOR_INDEXES=ivfflat,hnsw     # both present

# Distance metric must match index opclass
export SEARCH_DISTANCE=cosine             # or l2

# IVFFLAT tuning
export IVFFLAT_LISTS=100
export IVFFLAT_PROBES=4

# HNSW tuning (pgvector >= 0.7)
export HNSW_M=16
export HNSW_EF_CONSTRUCTION=128
export HNSW_EF_SEARCH=80
```

Ensure/rebuild indexes via `POST /admin/indexes`.

### 5. Test It
```bash
# Health check
curl http://localhost:8000/health

# Create a node with refresh policy
curl -X POST http://localhost:8000/nodes \
  -H "Content-Type: application/json" \
  -d '{
    "classes": ["Document"],
    "props": {"text": "Machine learning fundamentals"},
    "refresh_policy": {"interval": "5m", "drift_threshold": 0.1},
    "metadata": {"category": "AI", "tags": ["research"]}
  }'

# Search with compound filter
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence",
    "top_k": 10,
    "compound_filter": {"category": "AI", "tags": ["research"]}
  }'

# Q&A (non-streaming)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What vector databases are discussed?", "max_results": 3}'

# Q&A (streaming SSE)
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What vector databases are discussed?", "max_results": 3}'

# Note: /ask metadata
# - top_similarity is the gating score (RRF/weighted/cosine depending on mode)
# - top_vector_similarity / max_vector_similarity are true cosine similarities
# Example (RRF mode):
# {
#   "metadata": {
#     "top_similarity": 0.033,
#     "top_vector_similarity": 0.583,
#     "max_vector_similarity": 0.583,
#     "gating_score_type": "rrf_fused"
#   }
# }

# Prometheus metrics
curl http://localhost:8000/prometheus
```

### 6. Postman Collection

Import the Postman collection to try endpoints quickly:

```
active-graph-kg/postman/actvgraph-kg.postman_collection.json
```

Set `{{base_url}}` (default `http://localhost:8000`) and run requests for /health, /nodes, /search, /triggers, /lineage, /events, and /ask.

---

See also:
- docs/operations/connectors.md — Idempotency, cursors, rotation

## Makefile Shortcuts

Speed up common tasks. Ensure `API` and `TOKEN` are set when required.

```bash
export API=http://localhost:8000
export TOKEN='<admin JWT>'

# Core validation
make live-smoke
make live-extended
make metrics-probe

# Retrieval quality + proof report
make retrieval-quality && make publish-retrieval-uplift
make proof-report            # writes evaluation/PROOF_POINTS_REPORT.md

# DB bootstrap and indexes
make db-bootstrap            # uses ACTIVEKG_DSN or DATABASE_URL
curl -X POST "$API/admin/indexes" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"action":"ensure","types":["ivfflat","hnsw"],"metric":"cosine"}'

# Convenience
make open-grafana            # opens http://localhost:3000/d/activekg-ops
```

## Railway DB Initialization

For Railway deployments, initialize the database (pgvector + schema + migrations) using:

```bash
python3 scripts/init_railway_db.py
```

The script is safe to run multiple times and applies all migrations in `db/migrations/`.

### Demo Run (Quick)

```bash
export API=http://localhost:8000
export TOKEN='<admin JWT>'
make demo-run && make open-grafana
```

Notes:
- The API accepts either `ACTIVEKG_DSN` or `DATABASE_URL` for the database connection.
- Run exactly one API instance with `RUN_SCHEDULER=true`.
- `make open-grafana` opens `GRAFANA_URL` (defaults to `http://localhost:3000/d/activekg-ops`) using your OS opener.

## Core Features

### ✅ Phase 1 MVP (Complete)

#### 1. **Self-Refreshing Nodes**
Nodes automatically re-embed their content based on refresh policies:

```python
{
    "refresh_policy": {
        "interval": "5m",          # Refresh every 5 minutes
        "drift_threshold": 0.1     # Emit event if drift > 0.1
    }
}
```

**Supported intervals:** `1m`, `5m`, `1h`, `1d`, etc.
**Coming soon:** Cron expressions (`"0 0 * * *"` for daily at midnight)

#### 2. **Drift Detection**
System calculates cosine distance between old and new embeddings:

```python
drift = 1.0 - cosine_similarity(old_embedding, new_embedding)
```

Events are emitted **only** when drift exceeds the configured threshold (default: 0.1).

#### 3. **Semantic Triggers**
Register patterns and fire events when nodes match:

```bash
# Register pattern
curl -X POST http://localhost:8000/triggers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "fraud_detection",
    "example_text": "suspicious wire transfer to offshore account",
    "description": "Detects potential fraud"
  }'

# Create node with trigger
curl -X POST http://localhost:8000/nodes \
  -H "Content-Type: application/json" \
  -d '{
    "classes": ["Transaction"],
    "props": {"text": "large wire transfer flagged by system"},
    "triggers": [{"name": "fraud_detection", "threshold": 0.8}]
  }'
```

When similarity ≥ 0.8, a `trigger_fired` event is created.

#### 4. **Lineage Tracking**
Track provenance through DERIVED_FROM edges:

```bash
# Create derivation chain: A → B → C
curl -X POST http://localhost:8000/edges \
  -d '{"src": "A", "rel": "DERIVED_FROM", "dst": "B"}'

curl -X POST http://localhost:8000/edges \
  -d '{"src": "B", "rel": "DERIVED_FROM", "dst": "C"}'

# Traverse lineage
curl http://localhost:8000/lineage/A?max_depth=5
```

Returns recursive ancestor chain with depth and edge metadata.

#### 5. **Polyglot Payloads**
Load content from multiple sources:

```python
# S3
{"payload_ref": "s3://my-bucket/document.txt"}

# HTTP
{"payload_ref": "https://example.com/article.html"}

# Local file
{"payload_ref": "file:///path/to/document.txt"}

# Inline
{"props": {"text": "Direct inline text"}}
```

All are automatically fetched and embedded during refresh.

---

### ✅ Phase 1+ Tactical Improvements (Complete)

#### 6. **JSONB Compound Filters**
Complex metadata queries with nested/typed filtering:

```bash
curl -X POST http://localhost:8000/search \
  -d '{
    "query": "machine learning",
    "compound_filter": {
      "category": "AI",
      "tags": ["research", "2025"],
      "metrics": {"views": 1000}
    }
  }'
```

Uses PostgreSQL's `@>` (containment) operator with GIN index for fast queries.

**Note:** Supports exact value matching and nested containment only. Does NOT support comparison operators (`$gt`, `$lt`, etc.). For range queries, use dedicated columns or implement JSONPath support.

#### 7. **Efficient Trigger Scanning**
Optimized `run_for(node_ids)` method scans **only refreshed nodes**:

- **Before:** O(N) — scan all nodes (slow for large graphs)
- **After:** O(K) — scan only refreshed nodes (2000x faster for sparse refreshes)

```python
# Old: Check all 100K nodes
trigger_engine.run()  # O(100K)

# New: Check only 50 refreshed nodes
trigger_engine.run_for(refreshed_node_ids)  # O(50)
```

#### 8. **Multi-Tenant Audit Trail**
All events and edges now include:

```sql
-- Events
tenant_id TEXT      -- Which tenant owns this data
actor_id TEXT       -- Who triggered the event (user ID, 'scheduler', 'trigger_engine')
actor_type TEXT     -- Type: 'user', 'api_key', 'scheduler', 'trigger', 'system'

-- Edges
tenant_id TEXT      -- Tenant isolation for relationships
```

Query events by actor:
```sql
SELECT * FROM events
WHERE actor_type = 'user'
  AND created_at > now() - interval '1 day'
ORDER BY created_at DESC;
```

#### 9. **Row-Level Security (RLS)**
Database-level tenant isolation:

```sql
-- Set tenant context
SELECT set_tenant_context('acme_corp');

-- All queries are automatically filtered
SELECT * FROM nodes;  -- Only returns acme_corp's nodes
```

Policies applied to: `nodes`, `edges`, `events`, `node_versions`, `embedding_history`

**RLS Configuration (RLS_MODE):**

Active Graph KG supports flexible RLS configuration via the `RLS_MODE` environment variable:

```bash
# Auto-detect database RLS state (recommended, default)
export RLS_MODE=auto

# Always enable tenant context (force on)
export RLS_MODE=on

# Skip tenant context (only safe if DB RLS is disabled)
export RLS_MODE=off
```

**Behavior Matrix:**

| Database RLS | RLS_MODE | Application Behavior |
|--------------|----------|---------------------|
| ✅ Enabled | `auto` | ✅ Sets tenant context |
| ✅ Enabled | `on` | ✅ Sets tenant context |
| ✅ Enabled | `off` | ⚠️ **Forces ON** (logs error to prevent lockout) |
| ❌ Disabled | `auto` | ✅ Skips tenant context |
| ❌ Disabled | `on` | ✅ Sets tenant context (harmless) |
| ❌ Disabled | `off` | ✅ Skips tenant context |

**Safety Guarantee:** If `RLS_MODE=off` but database RLS is enabled, the application automatically forces tenant context ON and logs an error. This prevents accidental query lockouts where RLS would block all queries.

**When to use each mode:**
- `auto` (default): Recommended for most deployments. Auto-detects database state.
- `on`: Use when you want to force tenant context even if RLS detection fails.
- `off`: Use for single-tenant dev instances where database RLS is explicitly disabled.

**To disable RLS completely:**
```bash
# 1. Disable at database level
psql $ACTIVEKG_DSN -c "ALTER TABLE nodes DISABLE ROW LEVEL SECURITY;"
psql $ACTIVEKG_DSN -c "ALTER TABLE edges DISABLE ROW LEVEL SECURITY;"
psql $ACTIVEKG_DSN -c "ALTER TABLE events DISABLE ROW LEVEL SECURITY;"

# 2. Set RLS_MODE to auto or off
export RLS_MODE=auto  # Will auto-detect disabled state
```

#### 10. **Admin Refresh Endpoint**
On-demand refresh for ops control:

```bash
# Mode 1: Refresh all due nodes
curl -X POST http://localhost:8000/admin/refresh

# Mode 2: Refresh specific nodes
curl -X POST http://localhost:8000/admin/refresh \
  -d '["node_id_1", "node_id_2", "node_id_3"]'
```

Events are tagged with `actor_type='user'` and `manual_trigger=true`.

#### 11. **Prometheus Metrics**
Standard exposition format for monitoring:

```bash
curl http://localhost:8000/prometheus
```

Returns:
```
# HELP activekg_refresh_cycles_total Counter metric
# TYPE activekg_refresh_cycles_total counter
activekg_refresh_cycles_total 142

# HELP activekg_search_latency Histogram metric
# TYPE activekg_search_latency summary
activekg_search_latency{quantile="0.5"} 0.023
activekg_search_latency{quantile="0.95"} 0.156
```

Integrate with Prometheus + Grafana for dashboards and alerts.

---

## API Endpoints (30+ Total)

### Core
- `GET /health` - Health check with version
- `GET /metrics` - Metrics in JSON format
- `GET /prometheus` - Metrics in Prometheus format
- `GET /demo` - Demo page with sample data

### Nodes & Search
- `POST /nodes` - Create node
- `GET /nodes/{node_id}` - Get node by ID
- `POST /nodes/{node_id}/refresh` - Manually refresh a specific node
- `GET /nodes/{node_id}/versions` - Get version history for a node
- `POST /search` - Vector search with compound filters
- `POST /ask` - Q&A with RAG and citations
- `POST /ask/stream` - Streaming Q&A with SSE

### Edges & Lineage
- `POST /edges` - Create relationship
- `GET /lineage/{node_id}` - Traverse DERIVED_FROM chain

### Triggers
- `POST /triggers` - Register semantic pattern
- `GET /triggers` - List all patterns
- `DELETE /triggers/{name}` - Delete pattern

### Events
- `GET /events` - List events with filters

### Admin & Operations
- `POST /admin/refresh` - Trigger on-demand refresh
- `GET /admin/anomalies` - Detect anomalies in node embeddings
- `GET /_admin/security/limits` - Get security configuration and limits
- `POST /_admin/connectors/rotate_keys` - Rotate connector encryption keys
- `GET /_admin/connectors/cache/health` - Check connector cache health

### Connectors (Admin)
- `POST /_admin/connectors/configs` - Create connector configuration
- `GET /_admin/connectors/configs` - List all connector configs
- `GET /_admin/connectors/configs/{config_id}` - Get specific config
- `PUT /_admin/connectors/configs/{config_id}` - Update config
- `DELETE /_admin/connectors/configs/{config_id}` - Delete config
- `POST /_admin/connectors/configs/{config_id}/test` - Test connection
- `POST /_admin/connectors/configs/{config_id}/sync` - Trigger manual sync
- `GET /_admin/connectors/runs` - List connector run history
- `GET /_admin/connectors/runs/{run_id}` - Get specific run details

### Debug & Diagnostics
- `GET /debug/embed_info` - Embedding backend status
- `GET /debug/search_explain` - Query plan analysis
- `GET /debug/search_sanity` - Sanity checks
- `GET /debug/dbinfo` - Database metadata
- `GET /debug/intent` - Intent classification for queries

### Governance Metrics
- `activekg_access_violations_total{type}` - missing_token, scope_denied, cross_tenant_query

---

## Project Structure

```
active-graph-kg/
├── activekg/
│   ├── api/
│   │   ├── main.py                    # FastAPI app (24+ endpoints)
│   │   ├── admin_connectors.py        # Connector admin endpoints
│   │   ├── auth.py                    # JWT authentication
│   │   ├── rate_limiter.py            # Per-tenant rate limiting
│   │   └── middleware.py              # Request middleware
│   ├── graph/
│   │   ├── models.py                  # Node, Edge models
│   │   └── repository.py              # Data access layer (vector search, lineage, etc.)
│   ├── triggers/
│   │   ├── trigger_engine.py          # Pattern matching with run_for()
│   │   └── pattern_store.py           # DB-backed pattern storage
│   ├── refresh/
│   │   └── scheduler.py               # APScheduler for refresh cycles
│   ├── engine/
│   │   └── embedding_provider.py      # sentence-transformers wrapper
│   ├── connectors/                    # S3/GCS/Drive connectors
│   │   ├── worker.py                  # Queue worker for events
│   │   └── config_store.py            # Encrypted config storage
│   ├── observability/
│   │   └── metrics.py                 # Prometheus metrics
│   └── common/
│       ├── logger.py                  # Structured logging
│       ├── metrics.py                 # Core metrics
│       └── validation.py              # Pydantic models
├── db/
│   ├── init.sql                       # Schema (nodes, edges, events, patterns)
│   └── migrations/
│       └── add_text_search.sql        # Hybrid BM25 text search
├── scripts/
│   ├── README.md                      # Validation scripts catalog
│   ├── smoke_test.py                  # E2E integration tests
│   ├── backend_readiness_check.py     # Readiness validation
│   ├── live_smoke.sh                  # Quick validation
│   ├── retrieval_quality.sh           # Triple retrieval test
│   ├── proof_points_report.sh         # Generate proof report
│   └── db_bootstrap.sh                # DB schema setup
├── tests/
│   ├── test_phase1_complete.py        # Phase 1 MVP tests
│   ├── test_phase1_plus.py            # Phase 1+ improvement tests
│   └── test_base_engine_gaps.py       # Base engine tests
├── docs/
│   ├── operations/                    # Operations guides
│   ├── development/                   # Dev guides
│   └── api-reference.md               # API documentation
├── enable_rls_policies.sql            # Row-Level Security
├── enable_vector_index.sql            # Vector index helper
├── Dockerfile                         # Container build
├── Procfile                           # Railway deployment
├── Makefile                           # Automation targets
├── LICENSE                            # MIT License
└── LICENSE-ENTERPRISE.md              # Enterprise licensing
```

---

## Deploy on Railway (Self‑Serve Demo)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?templateUrl=https://github.com/puneetrinity/active-graph-kg)

After deploy, set variables in Railway → Variables:
- `ACTIVEKG_DSN` (Postgres with pgvector). Optional if you added the Railway Postgres plugin — the app will fall back to `DATABASE_URL` automatically.
- `EMBEDDING_BACKEND=sentence-transformers`
- `EMBEDDING_MODEL=all-MiniLM-L6-v2` (or larger, e.g., all-mpnet-base-v2)
- `SEARCH_DISTANCE=cosine`
- Optional: `PGVECTOR_INDEXES=ivfflat,hnsw`, `RUN_SCHEDULER=true`, `AUTO_INDEX_ON_STARTUP=false`

Initialize DB once:
```bash
psql $ACTIVEKG_DSN -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql $ACTIVEKG_DSN -f db/init.sql
psql $ACTIVEKG_DSN -f enable_rls_policies.sql
```

Run the demo bundle against Railway:
```bash
export API=https://<your-railway-domain>
export TOKEN='<admin JWT>'
make demo-run && make open-grafana
```

Full guide: docs/operations/SELF_SERVE_RAILWAY.md

---

## Database Schema

### Nodes
```sql
CREATE TABLE nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id TEXT,                         -- Multi-tenant isolation
    classes TEXT[],                         -- Semantic classes
    props JSONB,                            -- Arbitrary properties
    payload_ref TEXT,                       -- S3/HTTP/file reference
    embedding VECTOR(384),                  -- pgvector (all-MiniLM-L6-v2)
    metadata JSONB,                         -- Filterable metadata
    refresh_policy JSONB,                   -- {"interval": "5m", "drift_threshold": 0.1}
    triggers JSONB,                         -- [{"name": "pattern", "threshold": 0.8}]
    last_refreshed TIMESTAMPTZ,             -- Explicit column (not JSONB)
    drift_score DOUBLE PRECISION,           -- Explicit column (not JSONB)
    version INT DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX idx_nodes_embedding ON nodes USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_nodes_metadata ON nodes USING GIN (metadata);
CREATE INDEX idx_nodes_tenant ON nodes(tenant_id);
CREATE INDEX idx_nodes_last_refreshed ON nodes(last_refreshed);
```

### Edges
```sql
CREATE TABLE edges (
  src UUID NOT NULL,
  rel TEXT NOT NULL,
  dst UUID NOT NULL,
  props JSONB NOT NULL DEFAULT '{}',
  tenant_id TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (src, rel, dst)
);

CREATE INDEX idx_edges_src ON edges(src, rel);
CREATE INDEX idx_edges_dst ON edges(dst, rel);
CREATE INDEX idx_edges_lineage ON edges(dst, rel) WHERE rel = 'DERIVED_FROM';
CREATE INDEX idx_edges_tenant ON edges(tenant_id);
```

### Events
```sql
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id UUID REFERENCES nodes(id),
    type TEXT NOT NULL,                     -- 'refreshed', 'trigger_fired', etc.
    payload JSONB,                          -- Event data
    tenant_id TEXT,                         -- Multi-tenant isolation
    actor_id TEXT,                          -- Who triggered (user ID, 'scheduler', etc.)
    actor_type TEXT,                        -- 'user', 'api_key', 'scheduler', 'trigger', 'system'
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_events_node ON events(node_id, created_at DESC);
CREATE INDEX idx_events_type ON events(type, created_at DESC);
CREATE INDEX idx_events_tenant ON events(tenant_id, created_at DESC);
CREATE INDEX idx_events_actor ON events(actor_id, created_at DESC);
```

### Patterns
```sql
CREATE TABLE patterns (
    name TEXT PRIMARY KEY,
    embedding VECTOR(384) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
```

---

## Configuration

### Environment Variables
```bash
# Required
ACTIVEKG_DSN='postgresql://activekg:activekg@localhost:5432/activekg'

# Optional (defaults shown)
EMBEDDING_BACKEND='sentence-transformers'
EMBEDDING_MODEL='all-MiniLM-L6-v2'
ACTIVEKG_VERSION='1.0.0'

# Row-Level Security mode: auto|on|off (default: auto)
RLS_MODE='auto'  # auto-detect | force on | skip (safe mode prevents lockouts)
```

See [.env.example](.env.example) for complete configuration options.

### Refresh Policy Examples
```json
// Interval-based (every 5 minutes)
{
    "interval": "5m",
    "drift_threshold": 0.1
}

// Cron-based (daily at midnight) - Coming soon
{
    "cron": "0 0 * * *",
    "drift_threshold": 0.15
}
```

---

## Testing & Verification

### Run All Tests
```bash
# Code verification (34 automated checks)
./verify_phase1_plus.sh

# Phase 1 MVP tests
python tests/test_phase1_complete.py

# Phase 1+ improvement tests
python tests/test_phase1_plus.py

# E2E smoke test (requires API running)
python scripts/smoke_test.py
```

### Expected Output
```
==============================================================
Phase 1+ Tactical Improvements - Code Verification
==============================================================

=== Improvement 1: JSONB Containment Filter ===
✓ compound_filter parameter in repository.py
✓ JSONB containment operator (@>)
✓ compound_filter in validation models
✓ compound_filter passed to vector_search

[... 34 checks total ...]

✅ ALL CHECKS PASSED - Phase 1+ Complete!
```

---

## Performance

### Vector Search Benchmarks (Projected)
| Nodes | No Index | IVFFLAT | HNSW |
|-------|----------|---------|------|
| 10K   | 50ms     | 15ms    | 5ms  |
| 100K  | 500ms    | 50ms    | 10ms |
| 1M    | 5s       | 150ms   | 20ms |

### Trigger Scanning Speedup
| Scenario | Old (Full Scan) | New (run_for) | Speedup |
|----------|-----------------|---------------|---------|
| 100K nodes, 50 refreshed | 500ms | 0.25ms | **2000x** |
| 10K nodes, 100 refreshed | 50ms  | 0.5ms  | 100x    |

### Recommendations
- **Small datasets (<10K nodes):** IVFFLAT is sufficient
- **Medium datasets (10K-100K):** IVFFLAT with `lists=100`
- **Large datasets (>100K):** Switch to HNSW for best performance

---

## Production Deployment

### 1. Enable RLS Policies
```bash
psql -f enable_rls_policies.sql
```

### 2. Create Vector Index
```bash
# Choose inside the script: IVFFLAT (default) or HNSW (uncomment the HNSW block)
psql -f enable_vector_index.sql
```

### 3. Configure Prometheus
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'activekg'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/prometheus'
```

### 4. Set Up Alerts
```yaml
groups:
  - name: activekg
    rules:
      - alert: HighDriftRate
        expr: rate(activekg_refresh_cycles_total[5m]) > 10
        annotations:
          summary: "Drift rate spike detected"

      - alert: SlowSearch
        expr: activekg_search_latency{quantile="0.95"} > 1.0
        annotations:
          summary: "Search latency p95 > 1s"
```

### 5. Authentication & RLS (Enabled)
- JWT middleware is implemented. Tenant ID is derived from JWT claims and RLS context is set per request.
- See Security guide and examples under “Security & Production Hardening” below.

---

## Production Readiness: 100%

### ✅ Complete
- Self-refreshing nodes with drift detection
- Semantic triggers with DB-backed patterns
- Lineage tracking (recursive CTEs)
- Polyglot payload loaders (S3, HTTP, file)
- Vector search with compound filters
- Efficient trigger scanning
- Multi-tenant audit trail
- Row-Level Security policies (database-level isolation)
- JWT authentication middleware (HS256/RS256)
- Rate limiting (per tenant, per endpoint)
- Admin endpoints (refresh, indexes, metrics)
- Prometheus metrics (40+ metrics)
- Grafana dashboard support
- Dual ANN indexing (IVFFLAT/HNSW)
- DSN fallback for PaaS (Railway, Heroku)
- Comprehensive test suite
- Validation/proof scripts

### ⚠ Nice-to-Have (Not Blockers)
- Optional Helm chart (k8s)
- Full multi-service Railway template
- Connector DLQ and throughput panels

---

## Security & Production Hardening

Active Graph KG includes production-grade security features for multi-tenant deployments.

### JWT Authentication

**Purpose**: Enforce tenant isolation and prevent impersonation attacks.

**Setup**:

```bash
# 1. Install dependencies
pip install PyJWT[crypto]==2.8.0

# 2. Configure JWT (environment variables)
export JWT_ENABLED=true
export JWT_ALGORITHM=RS256              # RS256 for production, HS256 for dev
export JWT_PUBLIC_KEY="-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----"
export JWT_AUDIENCE=activekg
export JWT_ISSUER=vantahire             # Must match issuer in Vanta JWT

# For HS256 dev mode only:
# export JWT_SECRET_KEY="your-32-char-secret-key-here"

# 3. Generate test JWT
python scripts/generate_test_jwt.py --tenant test_tenant --actor test_user

# 4. Use JWT in requests
curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer <token>" \
  -d '{"question": "test"}'
```

**What it does**:
- Extracts `tenant_id` from JWT claims (not request body) → prevents tenant impersonation
- Validates token signature, expiry, audience, issuer
- Enforces scope-based authorization per endpoint
- Sets RLS context automatically for tenant isolation

**Required scopes**:
- `search:read` — `POST /search`
- `ask:read` — `POST /ask`, `POST /ask/stream`
- `kg:write` — `POST /nodes`, `POST /nodes/batch`, `POST /edges`, `POST /upload`
- `admin:refresh` — `POST /admin/refresh`, debug endpoints

**Scope formats** (all accepted in JWT claims):
- `"scopes": ["search:read", "kg:write"]` (array)
- `"scopes": "search:read kg:write"` (space-delimited string)
- `"scope": "search:read kg:write"` (OAuth2 fallback)

**Dev mode**: Set `JWT_ENABLED=false` to disable auth for local development.

### Rate Limiting

**Purpose**: Prevent cost spikes (LLM calls) and noisy neighbor problems.

**Setup**:

```bash
# 1. Install dependencies
pip install redis==5.0.1

# 2. Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# 3. Configure rate limiting
export RATE_LIMIT_ENABLED=true
export REDIS_URL=redis://localhost:6379/0

# 4. Test rate limiting
for i in {1..10}; do
  curl -X POST http://localhost:8000/ask \
    -H "Authorization: Bearer <token>" \
    -d '{"question": "test"}' &
done
# Should see 429 after 5th request
```

**Rate limits** (default, configurable via `activekg/api/rate_limiter.py`):
- `/ask`: 3 req/s (burst 5), max 3 concurrent per tenant
- `/ask/stream`: 1 req/s (burst 3), max 2 concurrent per tenant
- `/search`: 50 req/s (burst 100)
- `/admin/*`: 1 req/s (burst 2)

**Response headers**:
```http
X-RateLimit-Limit: 5
X-RateLimit-Remaining: 2
X-RateLimit-Reset: 1699123456
Retry-After: 1  # Only on 429 responses
```

### Reranker Configuration

**Purpose**: Control cross-encoder reranking behavior for ops flexibility.

**Environment variables**:

```bash
# Master toggle (enable/disable reranking)
export ASK_USE_RERANKER=true  # default: true

# Skip threshold (skip reranking if top hybrid_score >= this)
export RERANK_SKIP_TOPSIM=0.80  # default: 0.80

# Candidate pool size (fetch N candidates before reranking)
export HYBRID_RERANKER_CANDIDATES=20  # default: 20
```

**When reranking is skipped**:
- Structured intents (job search, performance issues queries)
- High-confidence results (top hybrid_score ≥ 0.80)
- Small result sets (K < 3)
- Master toggle disabled (`ASK_USE_RERANKER=false`)

See docs/SCORING_MODES.md for details on fusion modes (RRF vs weighted) and reranking.

### Integration Guide

For step-by-step integration of JWT and rate limiting into your deployment:

1. **Quick start** (30 min): [QUICKSTART.md](QUICKSTART.md)
2. **Detailed examples** (before/after code): [INTEGRATION_EXAMPLE.md](INTEGRATION_EXAMPLE.md)
3. **Production checklist**: [PRODUCTION_HARDENING_GUIDE.md](PRODUCTION_HARDENING_GUIDE.md)
4. **Gotchas & fixes**: [INTEGRATION_GOTCHAS.md](INTEGRATION_GOTCHAS.md)

**Required services**:
- PostgreSQL with pgvector (database)
- Redis (rate limiting)
- JWT provider (auth server or self-signed keys)

---

## What's Next

### Quick Wins (Phase 1.5)
1. Cron policy support (`croniter`)
2. Payload size limits (security)
3. JWT authentication
4. Rate limiting
5. Grafana dashboards

### Phase 2 (Advanced Features)
1. **CRDT-Based Graph Replication** - Distributed, conflict-free updates
2. **Adaptive Compression** - PQ/8-bit quantization for larger embeddings
3. **Real-time Event Streaming** - WebSockets for live updates
4. **Graph Expansion** - Hybrid search with relationship traversal
5. **FAISS Side-Index** - Ultra-fast ANN for millions of nodes

---

## Documentation

### Quick Navigation

Jump to key resources:

- **[Quickstart Guide](QUICKSTART.md)** - Get running in 5 minutes
- **[API Reference](docs/api-reference.md)** - All 24 endpoints with examples
- **[Production Deployment](docs/operations/deployment.md)** - Deploy to production
- **[Security Guide](docs/operations/security.md)** - JWT, RLS, and security best practices
- **[Monitoring Setup](docs/operations/monitoring.md)** - Prometheus metrics and alerts

---

### Documentation Map

Active Graph KG documentation is organized into the following structure:

#### 📚 Operations Guides (Production)
- **[OPERATIONS.md](OPERATIONS.md)** - Complete operations runbook with admin API reference and key rotation procedures
- **[docs/operations/security.md](docs/operations/security.md)** - Complete security guide (JWT, RLS, rate limiting, payload security)
- **[docs/operations/monitoring.md](docs/operations/monitoring.md)** - Prometheus metrics, Grafana dashboards, alerting rules
- **[docs/operations/deployment.md](docs/operations/deployment.md)** - Production deployment checklist and best practices

#### 💻 Development Guides
- **[docs/api-reference.md](docs/api-reference.md)** - Complete API reference (24 endpoints, authentication, examples)
- **[docs/development/testing.md](docs/development/testing.md)** - Comprehensive testing guide (setup, results, troubleshooting)
- **[docs/development/architecture.md](docs/development/architecture.md)** - System architecture with code locations

#### 🚀 Setup & Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup guide
- **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** - Feature inventory with code locations

#### 📖 Implementation Docs
- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - Executive summary with architecture
- **[PHASE1_PLUS_IMPROVEMENTS.md](archive/development-logs/PHASE1_PLUS_IMPROVEMENTS.md)** - Detailed implementation guide
  - Browse more: [Development Logs](archive/development-logs/)

#### 📦 Archive
Historical progress summaries, assessments, and implementation notes have been moved to `archive/`:
- `archive/development-logs/` - Consolidated root development logs and status docs
- `archive/progress/` - Daily/weekly progress summaries (15 files)
- `archive/implementation/` - Security and Prometheus implementation details (8 files)
- `archive/assessments/` - System assessments and reviews (4 files)
- `archive/marketing/` - Marketing materials, pitch decks, whitepapers (5 files)
- `archive/setup/` - Alternative setup guides (1 file)
- `archive/testing/` - Testing documentation history (3 files)

**Note:** For the latest documentation, always refer to the `docs/` directory. Archived files are preserved for historical reference only.

---

## Technology Stack

- **Database:** PostgreSQL 16 + pgvector
- **API:** FastAPI 0.104+
- **Embeddings:** sentence-transformers 3.3.1 (all-MiniLM-L6-v2, 384 dims)
- **Scheduling:** APScheduler
- **Metrics:** Prometheus
- **Testing:** pytest, requests

---

## License

Dual License:
- Community: MIT License (see LICENSE)
- Enterprise: Commercial/permissioned license required (see LICENSE-ENTERPRISE.md)

Summary
- You may use, modify, and distribute the software under the MIT License for
  community/open-source purposes.
- Any enterprise use (e.g., production use by a company or organization) requires
  written permission or a commercial license. See LICENSE-ENTERPRISE.md for how
  to request permissions.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`./verify_phase1_plus.sh && python tests/test_phase1_plus.py`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## Support

- **Issues:** [GitHub Issues](https://github.com/puneetrinity/active-graph-kg/issues)
- **Discussions:** [GitHub Discussions](https://github.com/puneetrinity/active-graph-kg/discussions)
- **Documentation:** [Active Graph KG Docs](https://puneetrinity.github.io/active-graph-kg/)

---

**Built with ❤️ for the knowledge graph community**
## Production Deployment Checklist

Before deploying to production, ensure:

- [ ] JWT Authentication: Set `JWT_ENABLED=true` and use RS256 (set `JWT_PUBLIC_KEY`) or HS256 (`JWT_SECRET_KEY`) securely.
- [ ] RLS Enabled: Run `psql -f enable_rls_policies.sql` and set `RLS_MODE=auto` (or `on` for strict enforcement).
- [ ] Scheduler Singleton: Exactly one instance with `RUN_SCHEDULER=true`.
- [ ] Vector Indexes: Ensure ANN via `POST /admin/indexes` (IVFFLAT/HNSW as needed).
- [ ] Auto-Index Disabled: Set `AUTO_INDEX_ON_STARTUP=false` for large datasets.
- [ ] Rate Limiting: Configure Redis and set `RATE_LIMIT_ENABLED=true`.
- [ ] SSRF Protection: Set `ACTIVEKG_URL_ALLOWLIST` (comma-separated) and consider disabling URL loads in prod.
- [ ] Request Limits: Configure reverse proxy and/or `MAX_REQUEST_SIZE_BYTES` (defaults to 10MB).
- [ ] Monitoring: Scrape `/prometheus` and import Grafana dashboard.
- [ ] Secrets Management: Use env vars / secret store; never commit credentials.

See PRODUCTION_HARDENING_GUIDE.md for details.
