# Self‑Serve Demo on Railway (One‑Click Friendly)

> **Connector deployment is unavailable.** Current deployments may run the API plus embedding/extraction workers,
> but no S3/GCS/Drive connector service, poller or credential path.

This guide packages Active Graph KG for Railway deployment. Choose your deployment model:

**Basic Deployment (API Only):**
- API server + managed Postgres with pgvector
- Manual node creation via `/nodes` and `/upload` endpoints
- Synchronous embedding generation

**Full Deployment (API + Supported Workers):**
- API server + Embedding Worker + Extraction Worker
- Async embedding generation via Redis queues
- No connector worker, provider polling or connector credential setup

**Database Options:**
- Option A (Recommended): Neon/Aiven Postgres (pgvector supported)
- Option B (Advanced): Self-hosted Railway Postgres service using `pgvector/pgvector:pg16` image

All options support "near one‑click" via the Deploy button with minimal env setup.

---

## Prerequisites
- Railway account (paid 32 GB plan recommended for larger embedding models)
- Postgres with pgvector (`CREATE EXTENSION vector;`). Neon or Aiven support this.
- Redis (required for async embedding/extraction queues; optional for rate limiting)

---

## One‑Click Style Deploy (API)

Add this badge to your repo README (already included in the main README section if you choose to):

```md
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?templateUrl=https://github.com/puneetrinity/active-graph-kg)
```

When Railway imports the repo, it will use Nixpacks or the provided Dockerfile/Procfile to build the API.

### Configure Environment Variables
Set these variables in Railway → Variables for the API service. Railway checks
public, constant-cost `/health`. Operators check migration/RLS/runtime-role/JWT
dependency readiness separately through token-protected `/readyz`; dependency
failure must not create a Railway restart loop. The start script removes
`DATABASE_URL` and all migration/adoption variables before Uvicorn.

Required on API and workers
- `ACTIVEKG_DSN` — runtime DSN as the restricted role, e.g.
  `postgresql://activekg_app:SECRET@HOST:5432/DBNAME`
- `ACTIVEKG_SCHEMA_TARGET_ID` — opaque UUID for this adopted database
- `ACTIVEKG_SCHEMA_ENVIRONMENT=production`
- `JWT_ENABLED=true` plus key configuration (see `.env.example`)
- `ACTIVEKG_CONTROL_PLANE_TOKEN` — API-only high-entropy secret used by
  `/readyz`, `/metrics` and `/prometheus`; set a different value on the extraction worker
- `EMBEDDING_BACKEND=sentence-transformers`
- `EMBEDDING_MODEL=all-MiniLM-L6-v2` (or a larger model like `all-mpnet-base-v2`)
- `SEARCH_DISTANCE=cosine` (or `l2` to match your index opclass)

Development-only ownership escape hatch (never production): `ACTIVEKG_READYZ_ALLOW_OWNER=true`.

Recommended
- `PGVECTOR_INDEXES=ivfflat,hnsw` (coexist for migration)
- `AUTO_INDEX_ON_STARTUP=false` (prod-like; manage via admin endpoint)
- `RUN_SCHEDULER=true` (exactly one instance)
- `WORKERS=2` (tune up/down based on CPU)
- `TRANSFORMERS_CACHE=/workspace/cache` + attach a persistent volume to avoid re-downloading models
- `EMBEDDING_ASYNC=true` (use Redis queue + worker)
- `EMBEDDING_QUEUE_MAX_DEPTH=5000`
- `EMBEDDING_TENANT_MAX_PENDING=2000`

Security
- Dev: `JWT_SECRET_KEY=<dev-secret>` and `JWT_ALGORITHM=HS256`
- Prod: `JWT_PUBLIC_KEY=<RS256 public>` (preferred) and disable HS256
- Configure extraction-worker provider keys only when extraction is enabled; generic API Q&A is unavailable

Rate Limiting (optional)
- `RATE_LIMIT_ENABLED=true`
- `REDIS_URL=redis://<host>:6379/0`

### Async Embedding Worker (Recommended)

Async embedding requires a **separate worker service** on Railway.

1) Create a new Railway service from the same repo  
2) Set **Railway Config File** to `railway.worker.json`  
3) Set **Start Command**:
```
python -m activekg.embedding.worker
```
4) Set env vars (same DB + Redis as API):
   - `ACTIVEKG_DSN`
   - `REDIS_URL`
   - `EMBEDDING_BACKEND`
   - `EMBEDDING_MODEL`
   - `EMBEDDING_MAX_ATTEMPTS`, `EMBEDDING_RETRY_BASE_SECONDS`, `EMBEDDING_RETRY_MAX_SECONDS`

Note: the worker has no HTTP server, so healthcheck should be disabled (handled by `railway.worker.json`).

### Adopt and release the database

Create one private, manual-only service from the same repository using
`railway.schema-release.json`; keep auto-deploy off, restart policy `NEVER`, no
domain/replicas/healthcheck. It alone receives `ACTIVEKG_MIGRATE_DSN`, the same
target ID/environment and the exact source commit. Existing databases use the
one-time `scripts/adopt_schema_control.py` gate. Thereafter an approved release
temporarily receives `ACTIVEKG_MIGRATION_APPLY=1`, runs
`scripts/init_railway_db.py`, and has the flag removed immediately. Runtime
starts only call read-only schema readiness.

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

Configure each runtime with only the restricted `ACTIVEKG_DSN`, target ID and
production environment. Keep the owner DSN exclusively on the manual release
service. Build ANN indexes via `/admin/indexes` as above.

---

## Multi-Service Architecture (Supported Workers)

Connector ingestion is unavailable. Deploy only the API and the supported embedding/extraction workers; do not
create a connector-worker service or configure storage-provider credentials.

### Service 1: API Server

**Config:** `railway.json`
**Start Command:** leave unset — the Dockerfile runs `scripts/start_railway.sh`
(read-only schema readiness, then Uvicorn). Never bypass readiness in production.

**Environment Variables:**
```bash
# Core runtime identity (see "Configure Environment Variables" above)
ACTIVEKG_DSN=postgresql://activekg_app:...    # restricted runtime role
ACTIVEKG_SCHEMA_TARGET_ID=00000000-0000-4000-8000-000000000000
ACTIVEKG_SCHEMA_ENVIRONMENT=production
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Refresh/purge scheduler (set true on exactly one instance)
RUN_SCHEDULER=true

# Async embeddings (recommended for production)
EMBEDDING_ASYNC=true
REDIS_URL=redis://...  # Required for supported queue workers

# Workers
WORKERS=2  # API server worker processes
```

**Responsibilities:**
- REST API endpoints (`/search`, `/upload`, `/nodes`, etc.); Q&A compatibility routes return HTTP 410
- Connector compatibility routes return HTTP 410 without work
- Background scheduler runs refresh and purge only
- Health checks and metrics

### Service 2: Embedding Worker

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
- Embedding and extraction job queues
- Rate limiting (if `RATE_LIMIT_ENABLED=true`)

---

## Connector Credentials

Do not configure storage-provider credentials. The connector product is unavailable and the API does not read
connector credentials during import or startup.

---

## Deployment Checklist

### Basic Deployment (API Only)
- [ ] API service deployed with `railway.json`
- [ ] PostgreSQL with pgvector provisioned
- [ ] Runtime DSN + target ID/environment set; manual release service is private and auto-deploy-disabled
- [ ] JWT configured and tokens generated
- [ ] ANN indexes created via `/admin/indexes`
- [ ] Public liveness passes (`/health` returns the minimal `alive` response)
- [ ] Readiness passes with the API control-plane bearer (`/readyz` returns `{"status":"ready"}`)

### Full Deployment (Supported Workers)
- [ ] **API service** deployed with `RUN_SCHEDULER=true`
- [ ] **Embedding worker** deployed with `railway.worker.json`
- [ ] **Extraction worker** deployed where extraction queues are enabled
- [ ] **Redis** plugin added and `REDIS_URL` shared across services
- [ ] Connector compatibility routes return HTTP 410

### Monitoring
- [ ] Check API logs for scheduler runs
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
- Scheduler runs only node refresh and purge work

### Workers
- Follow the embedding/extraction worker runbooks for their independent scaling and retry behavior

### Cost Optimization
- Start with the API and only the supported workers required by the deployment
- Scale workers horizontally based on queue depth
- Use smaller embedding models for cost savings (`all-MiniLM-L6-v2` vs `all-mpnet-base-v2`)

---

## Troubleshooting

### "Embeddings not generated"
- Check embedding worker logs
- Verify `EMBEDDING_ASYNC=true` in API
- Check Redis queue: `redis-cli LLEN embedding:default:queue`
