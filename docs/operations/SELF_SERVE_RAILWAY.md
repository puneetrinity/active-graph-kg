# Self‑Serve Demo on Railway (One‑Click Friendly)

This guide packages Active Graph KG for Railway deployment. Choose your deployment model:

**Basic Deployment (API Only):**
- API server + managed Postgres with pgvector
- Manual node creation via `/nodes` and `/upload` endpoints
- Synchronous embedding generation

**Full Deployment (API + Workers):**
- API server + Connector Worker + Embedding Worker
- Automatic GCS/S3/Drive ingestion with polling
- Async embedding generation via Redis queues
- Production-ready for high-volume document processing

**Database Options:**
- Option A (Recommended): Neon/Aiven Postgres (pgvector supported)
- Option B (Advanced): Self-hosted Railway Postgres service using `pgvector/pgvector:pg16` image

All options support "near one‑click" via the Deploy button with minimal env setup.

---

## Prerequisites
- Railway account (paid 32 GB plan recommended for larger embedding models)
- Postgres with pgvector (`CREATE EXTENSION vector;`). Neon or Aiven support this.
- Redis (required for connector workers and async embeddings, optional for rate limiting only)

---

## One‑Click Style Deploy (API)

Add this badge to your repo README (already included in the main README section if you choose to):

```md
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?templateUrl=https://github.com/puneetrinity/active-graph-kg)
```

When Railway imports the repo, it will use Nixpacks or the provided Dockerfile/Procfile to build the API.

### Configure Environment Variables
Set these variables in Railway → Variables for the API service (note: the app will also accept `DATABASE_URL` as a fallback DSN when using the Railway Postgres plugin):

Required
- `ACTIVEKG_DSN` — e.g., `postgresql://USER:PASSWORD@HOST:5432/DBNAME`
- `EMBEDDING_BACKEND=sentence-transformers`
- `EMBEDDING_MODEL=all-MiniLM-L6-v2` (or a larger model like `all-mpnet-base-v2`)
- `SEARCH_DISTANCE=cosine` (or `l2` to match your index opclass)

Recommended
- `PGVECTOR_INDEXES=ivfflat,hnsw` (coexist for migration)
- `AUTO_INDEX_ON_STARTUP=false` (prod-like; manage via admin endpoint)
- `RUN_SCHEDULER=true` (exactly one instance)
- `WORKERS=2` (tune up/down based on CPU)
- `TRANSFORMERS_CACHE=/workspace/cache` + attach a persistent volume to avoid re-downloading models

Security
- Dev: `JWT_SECRET_KEY=<dev-secret>` and `JWT_ALGORITHM=HS256`
- Prod: `JWT_PUBLIC_KEY=<RS256 public>` (preferred) and disable HS256
- If using /ask, set your LLM provider key (e.g., `GROQ_API_KEY`)

Rate Limiting (optional)
- `RATE_LIMIT_ENABLED=true`
- `REDIS_URL=redis://<host>:6379/0`

### Initialize the Database
Option 1 (recommended): run the bootstrap helper (uses ACTIVEKG_DSN or DATABASE_URL automatically):
```bash
make db-bootstrap
```

Option 2: run these once from your laptop or any psql client:
```bash
export ACTIVEKG_DSN='postgresql://USER:PASSWORD@HOST:5432/DBNAME'
psql $ACTIVEKG_DSN -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql $ACTIVEKG_DSN -f db/init.sql
psql $ACTIVEKG_DSN -f enable_rls_policies.sql
# Optional: text search
psql $ACTIVEKG_DSN -f db/migrations/add_text_search.sql
```

Build ANN indexes (non-blocking, concurrent):
```bash
curl -X POST "$API/admin/indexes" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"action":"ensure","types":["ivfflat","hnsw"],"metric":"cosine"}'
```

### Validate the Demo
```bash
export API=https://<your-railway-domain>
export TOKEN='<admin JWT>'
make demo-run
make open-grafana  # if you have Grafana connected
```

---

## Option B: Postgres as a Railway Service (Advanced)

Create a new Railway service with Docker image:
- Image: `pgvector/pgvector:pg16`
- Expose port 5432
- Set env: `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- Use a persistent volume

Configure the API service’s `ACTIVEKG_DSN` to point to the DB service hostname inside Railway.

Initialize and index as above (`db/init.sql`, `enable_rls_policies.sql`, `/admin/indexes`).

---

## Multi-Service Architecture (Connectors & Workers)

For production deployments with **automatic connector ingestion** (GCS/S3/Drive) and **async embedding generation**, deploy multiple Railway services:

### Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│   API Server    │────▶│  Connector Worker │────▶│  Embedding Worker  │
│  (railway.json) │     │(railway.connector│     │ (railway.worker    │
│                 │     │  -worker.json)   │     │     .json)         │
│ - REST API      │     │                  │     │                    │
│ - Scheduler     │     │ - Polls GCS/S3   │     │ - Generates        │
│ - Queue jobs    │     │ - Extracts text  │     │   embeddings       │
│                 │     │ - Creates chunks │     │ - Updates vectors  │
└─────────────────┘     └──────────────────┘     └────────────────────┘
         │                       │                         │
         └───────────────────────┴─────────────────────────┘
                                 │
                         ┌───────▼────────┐
                         │  Redis (Queue) │
                         │  PostgreSQL    │
                         └────────────────┘
```

### Service 1: API Server

**Config:** `railway.json`
**Start Command:** `uvicorn activekg.api.main:app --host 0.0.0.0 --port $PORT`

**Environment Variables:**
```bash
# Core
ACTIVEKG_DSN=postgresql://...  # Or use DATABASE_URL
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Scheduler (IMPORTANT: Set true on EXACTLY ONE instance)
RUN_SCHEDULER=true
RUN_GCS_POLLER=true  # Enable GCS auto-polling

# Async embeddings (recommended for production)
EMBEDDING_ASYNC=true
REDIS_URL=redis://...  # Required for workers

# Workers
WORKERS=2  # API server worker processes
```

**Responsibilities:**
- REST API endpoints (`/search`, `/ask`, `/upload`, `/nodes`, etc.)
- Admin endpoints (`/_admin/connectors/*/ingest`, `/queue-status`)
- Background scheduler (polls GCS/S3/Drive, enqueues jobs)
- Health checks and metrics

### Service 2: Connector Worker

**Config:** `railway.connector-worker.json`
**Start Command:** `python -m activekg.connectors.worker`

**Environment Variables:**
```bash
# Same database and Redis as API
ACTIVEKG_DSN=postgresql://...
REDIS_URL=redis://...

# Worker-specific
CONNECTOR_WORKER_BATCH_SIZE=10
CONNECTOR_WORKER_POLL_INTERVAL=5  # seconds
```

**Responsibilities:**
- Polls Redis queues: `connector:{provider}:{tenant}:queue`
- Fetches files from GCS/S3/Drive
- Extracts text (PDF, DOCX, HTML, TXT)
- Creates parent + chunk nodes in database
- Enqueues embedding jobs for chunks

**When to deploy:**
- ✅ If using GCS/S3/Drive connectors with automatic polling
- ✅ If using manual `/ingest` endpoints for bulk operations
- ❌ Not needed if only using `/upload` or manual node creation

### Service 3: Embedding Worker

**Config:** `railway.worker.json`
**Start Command:** `python -m activekg.embedding.worker`

**Environment Variables:**
```bash
# Same database and Redis as API
ACTIVEKG_DSN=postgresql://...
REDIS_URL=redis://...
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Worker-specific
EMBEDDING_WORKER_BATCH_SIZE=32
EMBEDDING_WORKER_POLL_INTERVAL=2  # seconds
```

**Responsibilities:**
- Polls Redis queue: `embedding:{tenant}:queue`
- Generates embeddings for nodes/chunks
- Updates `nodes.embedding` and `nodes.embedding_queued` in database
- Handles retries and error logging

**When to deploy:**
- ✅ If `EMBEDDING_ASYNC=true` (recommended for production)
- ❌ Not needed if embeddings are generated synchronously

---

## Redis Setup (Required for Workers)

Add Redis plugin in Railway:

1. **In Railway Dashboard:**
   - Go to your project
   - Click "New" → "Database" → "Add Redis"
   - Railway automatically sets `REDIS_URL` env var

2. **Verify Redis URL format:**
   ```bash
   REDIS_URL=redis://default:PASSWORD@HOST:PORT
   ```

3. **Share Redis across services:**
   - Ensure all 3 services have access to same `REDIS_URL`
   - Use Railway's "Reference Variables" feature

**Redis is used for:**
- Connector ingestion queues (`connector:{provider}:{tenant}:queue`)
- Embedding job queues (`embedding:{tenant}:queue`)
- Scheduler locks (prevent duplicate polling)
- Rate limiting (if `RATE_LIMIT_ENABLED=true`)

---

## GCS Connector Credentials on Railway

### Option 1: File-Based Credentials (Recommended)

**Step 1:** Upload credentials to Railway secrets
```bash
# In Railway dashboard
Settings → Secrets → Upload File
# Upload your gcs-credentials.json
```

**Step 2:** Set environment variable
```bash
GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcs-credentials.json
# Or
service_account_json_path=/secrets/gcs-credentials.json  # In connector config
```

**Advantages:**
- ✅ More secure (not visible in env var UI)
- ✅ No truncation issues with large JSON
- ✅ Easier to rotate

### Option 2: Inline JSON (Fallback)

**Use only for small service account keys:**
```bash
GOOGLE_CREDENTIALS_JSON='{"type":"service_account","project_id":"...",...}'
```

**Or in connector config:**
```json
{
  "config": {
    "credentials_json": "{\"type\":\"service_account\",...}",
    "bucket": "my-bucket"
  }
}
```

**Limitations:**
- ⚠️ Visible in Railway UI (less secure)
- ⚠️ May be truncated if very large
- ⚠️ Escaping issues with nested JSON

### Authentication Priority Order

The connector tries credentials in this order:
1. `credentials_json` (inline JSON in config)
2. `service_account_json_path` (file path in config)
3. `GOOGLE_CREDENTIALS_JSON` (env var - inline JSON)
4. `GOOGLE_APPLICATION_CREDENTIALS` (env var - file path)
5. Default credentials (gcloud, workload identity)

---

## Deployment Checklist

### Basic Deployment (API Only)
- [ ] API service deployed with `railway.json`
- [ ] PostgreSQL with pgvector provisioned
- [ ] Database initialized (`db/init.sql`, `enable_rls_policies.sql`)
- [ ] JWT configured and tokens generated
- [ ] ANN indexes created via `/admin/indexes`
- [ ] Health check passes (`/health`)

### Full Deployment (With Connectors)
- [ ] **API service** deployed with `RUN_SCHEDULER=true`
- [ ] **Connector worker** deployed with `railway.connector-worker.json`
- [ ] **Embedding worker** deployed with `railway.worker.json`
- [ ] **Redis** plugin added and `REDIS_URL` shared across services
- [ ] **GCS credentials** uploaded and configured
- [ ] GCS connector registered via `POST /_admin/connectors/gcs/register`
- [ ] Test ingestion: `POST /_admin/connectors/gcs/ingest` (dry_run=true)
- [ ] Monitor queues: `GET /_admin/connectors/gcs/queue-status`
- [ ] Verify chunks created and embeddings generated

### Monitoring
- [ ] Check API logs for scheduler runs
- [ ] Check connector worker logs for file processing
- [ ] Check embedding worker logs for vector generation
- [ ] Monitor Redis queue depths
- [ ] Verify Prometheus metrics (if enabled)

---

## Notes & Limits

### General
- Railway Postgres plugin may not support `vector` extension; use Neon/Aiven if needed.
- Keep `AUTO_INDEX_ON_STARTUP=false` if your DB role is limited; use the admin endpoint for index ops.
- Larger embedding models (mpnet/e5) fit within 32 GB RAM; expect slower CPU embedding vs GPU.

### Scheduler
- **CRITICAL:** Run `RUN_SCHEDULER=true` on exactly ONE API instance
- If you scale API horizontally, set `RUN_SCHEDULER=false` on replica instances
- Scheduler uses Redis locks to prevent duplicate work per tenant

### Workers
- Connector worker and embedding worker can scale horizontally (multiple instances OK)
- Workers use Redis queues for coordination (no duplicate processing)
- If a worker crashes, jobs remain in queue for retry

### Cost Optimization
- Start with 1 instance each (API + Connector Worker + Embedding Worker)
- Scale workers horizontally based on queue depth
- Use smaller embedding models for cost savings (`all-MiniLM-L6-v2` vs `all-mpnet-base-v2`)

---

## Troubleshooting

### "No workers processing files"
- Check connector worker logs for errors
- Verify `REDIS_URL` is set and accessible
- Check Redis queue: `redis-cli LLEN connector:gcs:default:queue`

### "Embeddings not generated"
- Check embedding worker logs
- Verify `EMBEDDING_ASYNC=true` in API
- Check Redis queue: `redis-cli LLEN embedding:default:queue`

### "Duplicate polling"
- Verify only ONE API instance has `RUN_SCHEDULER=true`
- Check for stale Redis locks: `redis-cli KEYS connector:*:poll_lock`

### "GCS authentication failed"
- Verify credentials file path or inline JSON
- Check Railway secrets are mounted correctly
- Test with `gsutil ls gs://your-bucket` using same credentials
