# Production Readiness Validation Checklist

**Version:** 1.0
**Date:** 2025-11-24
**Status:** Pre-Launch Validation

This comprehensive checklist validates all systems before marketing launch and production deployment.

---

## Quick Reference

**Critical Path Items:** 8 categories, 40+ validation points
**Estimated Time:** 4-6 hours for full validation
**Prerequisites:** PostgreSQL + Redis + Real cloud accounts (S3, GCS, Drive)

**Validation Order:** Code & Features → Performance → Security → Operational → Documentation

---

## 1. Code & Features Validation

### 1.1 Connectors (S3, GCS, Google Drive)

**Objective:** Verify end-to-end connector flow: webhook/poller → Redis → worker → DB

#### S3 Connector E2E
- [ ] Register S3 connector via `POST /_admin/connectors/configs`
- [ ] Upload file to S3 bucket (trigger S3 event notification)
- [ ] Verify event appears in Redis queue: `redis-cli LLEN "connector:s3:{tenant}:queue"`
- [ ] Confirm worker processes queue (check worker logs)
- [ ] Verify node created in database:
  ```sql
  SELECT id, props->>'title', created_at
  FROM nodes
  WHERE tenant_id='{tenant}'
  ORDER BY created_at DESC LIMIT 5;
  ```
- [ ] Validate ETag-based change detection (modify file, re-upload)
- [ ] Check connector metrics: `connector_ingest_total{provider="s3"}`

#### GCS Connector E2E
- [ ] Register GCS connector with service account credentials
- [ ] Upload file to GCS bucket
- [ ] Send webhook event to `POST /_webhooks/gcs` (or trigger Pub/Sub)
- [ ] Verify queue processing: `redis-cli LLEN "connector:gcs:{tenant}:queue"`
- [ ] Confirm generation-based versioning works
- [ ] Check connector metrics: `connector_ingest_total{provider="gcs"}`

#### Google Drive Connector E2E ⚠️ **CRITICAL (Deferred)**
- [ ] Register Drive connector with service account
- [ ] Upload file to monitored Drive folder
- [ ] **Run Drive poller manually or via scheduler:**
  ```bash
  # Trigger sync
  curl -X POST http://localhost:8000/_admin/connectors/configs/{config_id}/sync \
    -H "Authorization: Bearer $ADMIN_JWT"
  ```
- [ ] Verify queue population: `redis-cli LLEN "connector:drive:{tenant}:queue"`
- [ ] Confirm worker processes Drive events
- [ ] Validate change detection via `driveId` tracking
- [ ] Check connector metrics: `connector_ingest_total{provider="drive"}`

**Pass Criteria:** All 3 connectors complete full cycle (external source → DB) with metrics incremented

---

### 1.2 Refresh & Drift Detection

**Objective:** Validate refresh scheduler, drift computation, and empty payload handling

#### Test Cases
- [ ] **Non-empty payload refresh:**
  - Create node with `refresh_policy: {interval: "1m", drift_threshold: 0.1}`
  - Wait for refresh cycle (or trigger via `POST /admin/refresh`)
  - Verify `last_refreshed` timestamp updated
  - Check drift score computed: `SELECT drift_score FROM nodes WHERE id=...`
  - Confirm `refreshed` event created in `events` table

- [ ] **Empty/whitespace payload handling:**
  - Create node with empty `props.text` or whitespace-only
  - Trigger refresh
  - Verify warning logged: "Skipping refresh for node {id}: empty or whitespace-only payload"
  - Confirm node NOT refreshed (no new event, no drift update)

- [ ] **Drift threshold behavior:**
  - Modify node content significantly (high drift expected)
  - Trigger refresh
  - Verify `drift_score > threshold`
  - Confirm `refreshed` event emitted with drift payload

- [ ] **Embedding history:**
  - Check `embedding_history` table populated after refresh
  - Verify old embedding stored correctly

**Pass Criteria:** Drift computed correctly, empty payloads skipped with warning, history written

---

### 1.3 Triggers

**Objective:** Validate semantic pattern matching and trigger execution

#### Test Cases
- [ ] Register trigger pattern:
  ```bash
  curl -X POST http://localhost:8000/triggers \
    -H "Authorization: Bearer $ADMIN_JWT" \
    -d '{
      "name": "fraud_detection",
      "example_text": "suspicious wire transfer to offshore account",
      "description": "Detects potential fraud"
    }'
  ```
- [ ] Create node with trigger:
  ```json
  {
    "classes": ["Transaction"],
    "props": {"text": "large wire transfer flagged by system"},
    "triggers": [{"name": "fraud_detection", "threshold": 0.8}]
  }
  ```
- [ ] Verify `trigger_fired` event created when similarity ≥ threshold
- [ ] Confirm trigger runs on refreshed nodes (after drift update)
- [ ] Test trigger with low similarity (below threshold) - no event expected

**Pass Criteria:** Triggers fire correctly on node creation and after refresh when threshold met

---

### 1.4 Search & ANN Indexing

**Objective:** Validate vector search, hybrid search, filters, and ANN index behavior

#### Vector Search
- [ ] **Basic vector search:**
  ```bash
  curl -X POST http://localhost:8000/search \
    -d '{"query": "machine learning", "top_k": 10}'
  ```
- [ ] **Compound filter search:**
  ```bash
  curl -X POST http://localhost:8000/search \
    -d '{
      "query": "AI research",
      "top_k": 10,
      "compound_filter": {"category": "ML", "tags": ["research"]}
    }'
  ```
- [ ] **Drift/recency weighting:**
  - Create nodes with different `drift_score` and `created_at` values
  - Search and verify results weighted correctly

#### Hybrid Search
- [ ] Test hybrid search with BM25 + vector fusion
- [ ] Toggle reranker on/off (`ASK_USE_RERANKER=true/false`)
- [ ] Verify reranker skips when `top hybrid_score >= RERANK_SKIP_TOPSIM` (default 0.80)

#### ANN Index Validation
- [ ] Create IVFFLAT index:
  ```bash
  curl -X POST http://localhost:8000/admin/indexes \
    -H "Authorization: Bearer $ADMIN_JWT" \
    -d '{"action": "ensure", "types": ["ivfflat"], "metric": "cosine"}'
  ```
- [ ] Verify index exists:
  ```sql
  SELECT indexname FROM pg_indexes
  WHERE tablename = 'nodes' AND indexname LIKE '%embedding%';
  ```
- [ ] Test per-query tuning:
  ```sql
  SET LOCAL ivfflat.probes = 10;
  -- Run search query
  RESET ivfflat.probes;
  ```
- [ ] Create HNSW index (if pgvector >= 0.7):
  ```bash
  curl -X POST http://localhost:8000/admin/indexes \
    -d '{"action": "ensure", "types": ["hnsw"], "metric": "cosine"}'
  ```
- [ ] Test HNSW per-query tuning: `SET LOCAL hnsw.ef_search = 80`

**Pass Criteria:** All search modes work, filters apply correctly, ANN indexes used efficiently

---

### 1.5 LLM Paths (Q&A)

**Objective:** Validate LLM provider routing, prompt building, and failure handling

#### Test Cases
- [ ] **Groq backend:**
  - Set `LLM_PROVIDER=groq`, `GROQ_API_KEY=...`, `LLM_MODEL=llama-3.3-70b-versatile`
  - Test `/ask` endpoint
  - Verify response includes citations and sources

- [ ] **OpenAI backend:**
  - Set `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...`, `LLM_MODEL=gpt-4o-mini`
  - Test `/ask` endpoint

- [ ] **Streaming Q&A:**
  ```bash
  curl -N -X POST http://localhost:8000/ask/stream \
    -d '{"question": "What vector databases are discussed?", "max_results": 3}'
  ```
- [ ] **LLM disabled mode:**
  - Set `ASK_ENABLE_LLM=false`
  - Verify `/ask` returns raw snippets without LLM synthesis

- [ ] **Failure handling:**
  - Use invalid API key
  - Verify graceful degradation (error message, no crash)

**Pass Criteria:** All LLM backends work, streaming functions, graceful failure when disabled

---

### 1.6 Auth & Rate Limiting

**Objective:** Validate JWT authentication, RLS, and rate limiting behavior

#### JWT Authentication
- [ ] **JWT-enabled mode:**
  - Set `JWT_ENABLED=true`, `JWT_SECRET_KEY=...`, `JWT_ALGORITHM=HS256`
  - Generate test JWT with tenant claim:
    ```python
    import jwt
    token = jwt.encode(
        {"sub": "user123", "tenant_id": "test_tenant", "exp": ...},
        "secret",
        algorithm="HS256"
    )
    ```
  - Make request with `Authorization: Bearer {token}`
  - Verify tenant context set correctly

- [ ] **JWT-disabled mode:**
  - Set `JWT_ENABLED=false`
  - Verify requests work without token
  - Check default tenant used (`ACTIVEKG_DEFAULT_TENANT`)

#### RLS (Row-Level Security)
- [ ] Enable RLS: `psql -f enable_rls_policies.sql`
- [ ] Set tenant context:
  ```sql
  SELECT set_tenant_context('tenant_a');
  ```
- [ ] Query nodes - verify only `tenant_a` nodes returned
- [ ] Switch tenant context:
  ```sql
  SELECT set_tenant_context('tenant_b');
  ```
- [ ] Verify different nodes returned
- [ ] Test cross-tenant access blocked

#### Rate Limiting
- [ ] **Enable rate limiting:**
  - Set `RATE_LIMIT_ENABLED=true`, `REDIS_URL=redis://localhost:6379/0`
  - Start Redis: `docker run -d -p 6379:6379 redis:7-alpine`

- [ ] **Test `/ask` rate limit (3 req/s, burst 5):**
  ```bash
  for i in {1..10}; do
    curl -X POST http://localhost:8000/ask \
      -H "Authorization: Bearer $TOKEN" \
      -d '{"question": "test"}' &
  done
  ```
  - Verify 429 responses after 5th request
  - Check headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `Retry-After`

- [ ] **Test `/search` rate limit (50 req/s, burst 100)** - similar test

- [ ] **Request size limit:**
  - Send large request body (> 10MB default)
  - Verify 413 response: "Request body too large"

**Pass Criteria:** JWT auth works, RLS isolates tenants, rate limits enforced, size limits work

---

### 1.7 Connector Config Store

**Objective:** Validate encrypted config storage, rotation, and cache invalidation

#### Test Cases
- [ ] **Create connector config:**
  ```bash
  curl -X POST http://localhost:8000/_admin/connectors/configs \
    -H "Authorization: Bearer $ADMIN_JWT" \
    -d '{
      "tenant_id": "test_tenant",
      "provider": "s3",
      "config": {
        "bucket": "test-bucket",
        "region": "us-east-1",
        "access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "enabled": true
      }
    }'
  ```
- [ ] Verify config stored encrypted in database:
  ```sql
  SELECT id, tenant_id, provider, encrypted_config
  FROM connector_configs
  WHERE tenant_id='test_tenant';
  ```
  - Confirm `encrypted_config` is encrypted (not plaintext)

- [ ] **Get config:**
  ```bash
  curl http://localhost:8000/_admin/connectors/configs/{config_id} \
    -H "Authorization: Bearer $ADMIN_JWT"
  ```
  - Verify secrets decrypted correctly

- [ ] **Update config:**
  ```bash
  curl -X PUT http://localhost:8000/_admin/connectors/configs/{config_id} \
    -d '{"config": {"enabled": false}}'
  ```
  - Verify update applied and re-encrypted

- [ ] **Rotate encryption keys:**
  ```bash
  curl -X POST http://localhost:8000/_admin/connectors/rotate_keys \
    -H "Authorization: Bearer $ADMIN_JWT"
  ```
  - Verify configs re-encrypted with new key
  - Confirm old configs still decrypt correctly during rotation

- [ ] **Redis cache invalidation:**
  - Make config change
  - Verify Redis pub/sub message sent
  - Confirm other workers receive cache invalidation

**Pass Criteria:** Configs encrypted at rest, decrypted on read, rotation works, cache invalidates

---

### 1.8 Database Schema & Migrations

**Objective:** Validate database schema matches code expectations

#### Test Cases
- [ ] **Fresh database setup:**
  ```bash
  # Create database
  createdb activekg_test

  # Enable pgvector
  psql activekg_test -c "CREATE EXTENSION IF NOT EXISTS vector;"

  # Run init script
  psql activekg_test -f db/init.sql

  # Run migrations
  psql activekg_test -f db/migrations/add_text_search.sql

  # Enable RLS
  psql activekg_test -f enable_rls_policies.sql
  ```

- [ ] **Verify schema:**
  - Check tables exist: `nodes`, `edges`, `events`, `patterns`, `node_versions`, `embedding_history`, `connector_configs`
  - Verify columns match repository code expectations
  - Check indexes created correctly
  - Confirm RLS policies active:
    ```sql
    SELECT schemaname, tablename, policyname
    FROM pg_policies
    WHERE schemaname = 'public';
    ```

- [ ] **Test repository operations:**
  - Create node via API
  - Create edge via API
  - Query lineage
  - Verify no schema mismatches or missing columns

**Pass Criteria:** Fresh database setup succeeds, all tables/indexes present, RLS active

---

## 2. Performance & Reliability

### 2.1 Embedding Throughput

**Objective:** Measure refresh cycle performance and identify bottlenecks

#### Test Cases
- [ ] **Baseline refresh performance:**
  - Create 100 nodes with 5-minute refresh policy
  - Wait for refresh cycle
  - Measure time to refresh all nodes
  - Check CPU/memory usage during refresh

- [ ] **Large batch test:**
  - Create 1000 nodes
  - Mark all as due for refresh
  - Measure refresh cycle time
  - Note: No per-cycle cap currently implemented ⚠️

- [ ] **Model pre-caching:**
  - Check if models download on first use (~80MB for cross-encoder)
  - Consider pre-caching in Docker image:
    ```dockerfile
    RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
      SentenceTransformer('all-MiniLM-L6-v2'); \
      CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
    ```

**Recommendations:**
- Add per-cycle batch limit: `due = self.repo.find_nodes_due_for_refresh(limit=100)`
- Consider parallelization with thread pool for embedding calls
- Pre-cache models in build images (set `HF_HOME`)

**Pass Criteria:** Refresh completes in reasonable time for expected load (< 5 seconds for 100 nodes)

---

### 2.2 ANN Index State

**Objective:** Validate index configuration matches expectations

#### Test Cases
- [ ] **Check index type:**
  ```sql
  SELECT indexdef FROM pg_indexes
  WHERE tablename = 'nodes' AND indexname LIKE '%embedding%';
  ```
  - Verify IVFFLAT or HNSW as expected

- [ ] **Verify index parameters:**
  - IVFFLAT: Check `lists` value (default 100)
  - HNSW: Check `m` and `ef_construction` (if using HNSW)

- [ ] **Test per-query GUCs:**
  ```sql
  -- Should not error
  SET LOCAL ivfflat.probes = 10;
  SELECT * FROM nodes ORDER BY embedding <=> '[...]' LIMIT 10;
  RESET ivfflat.probes;
  ```

- [ ] **Rebuild index if needed:**
  ```bash
  curl -X POST http://localhost:8000/admin/indexes \
    -d '{"action": "rebuild", "types": ["ivfflat"], "metric": "cosine"}'
  ```

**Pass Criteria:** Indexes exist, match config, per-query tuning works without errors

---

### 2.3 Ingestion Robustness (DLQ/Retry)

**Objective:** Validate Dead Letter Queue and retry behavior

#### Test Cases
- [ ] **Induce transient error:**
  - Temporarily disable database connection
  - Push connector event to Redis queue
  - Verify worker retries (check logs for retry attempts)
  - Restore database connection
  - Confirm event eventually processes

- [ ] **Induce permanent error:**
  - Push malformed event to queue (invalid JSON or missing required fields)
  - Verify worker moves event to DLQ after max retries
  - Check DLQ: `redis-cli LLEN "connector:{provider}:{tenant}:dlq"`

- [ ] **Inspect DLQ:**
  ```bash
  curl http://localhost:8000/_admin/connectors/dlq/inspect/{provider}/{tenant} \
    -H "Authorization: Bearer $ADMIN_JWT"
  ```

- [ ] **Clear DLQ:**
  ```bash
  curl -X POST http://localhost:8000/_admin/connectors/dlq/clear/{provider}/{tenant} \
    -H "Authorization: Bearer $ADMIN_JWT"
  ```

**Pass Criteria:** Transient errors retry successfully, permanent errors go to DLQ, DLQ can be inspected/cleared

---

## 3. Security

### 3.1 Dependency Scans

**Objective:** Validate weekly Safety scan workflow and review findings

#### Test Cases
- [ ] **Verify workflow exists:**
  ```bash
  ls -la .github/workflows/security-scan.yml
  ```

- [ ] **Trigger manual run:**
  ```bash
  gh workflow run security-scan.yml
  ```

- [ ] **Check latest run:**
  ```bash
  gh run list --workflow=security-scan.yml --limit 5
  ```

- [ ] **Download report:**
  - Go to: Actions → Security Scan → Select run
  - Download `safety-report` artifact
  - Extract `safety_report.json`

- [ ] **Review findings:**
  - Check for Critical/High severity CVEs
  - Current known low-severity items:
    - PyPDF2 3.0.1 (no secure version available)
    - transformers 4.57.1 (latest secure version ✓)

**Pass Criteria:** Workflow runs successfully, reports stored, no Critical/High CVEs

---

### 3.2 RLS & JWT Deep Validation

**Objective:** Validate RLS policies match application expectations

#### Test Cases
- [ ] **Verify RLS policies in database:**
  ```sql
  SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual, with_check
  FROM pg_policies
  WHERE schemaname = 'public'
  ORDER BY tablename, policyname;
  ```

- [ ] **Test cross-tenant access blocked:**
  - Create nodes for `tenant_a`
  - Set context to `tenant_b`: `SELECT set_tenant_context('tenant_b')`
  - Query nodes - verify `tenant_a` nodes NOT returned
  - Attempt to update `tenant_a` node - verify blocked

- [ ] **Test JWT claim parsing:**
  - Generate JWT with custom tenant: `{"tenant_id": "custom_tenant"}`
  - Make API request
  - Verify RLS context set to `custom_tenant`
  - Check logs confirm tenant extraction

- [ ] **Test scope-based authorization:**
  - Generate JWT without `admin:refresh` scope
  - Attempt `POST /admin/refresh`
  - Verify 403 Forbidden response

**Pass Criteria:** RLS blocks cross-tenant access, JWT claims parsed correctly, scopes enforced

---

### 3.3 Secrets Handling

**Objective:** Validate secrets encrypted and not logged

#### Test Cases
- [ ] **Check connector config encryption:**
  ```sql
  -- Should show encrypted blob, not plaintext secrets
  SELECT id, tenant_id, provider, encrypted_config
  FROM connector_configs
  LIMIT 1;
  ```

- [ ] **Review logs for leaks:**
  - Tail application logs during config operations
  - Search for plaintext credentials (AWS keys, GCS service account JSON)
  - Verify secrets sanitized/redacted in logs

- [ ] **Test config load logging:**
  - Load connector config
  - Check logs show only metadata (tenant, provider), not secrets

**Pass Criteria:** Secrets encrypted in DB, not logged in plaintext, sanitized in debug output

---

### 3.4 Rate Limiting & Request Limits

**Objective:** Validate limits enforced correctly

#### Test Cases
- [ ] **Rate limit enforcement:**
  - Exceed rate limit for `/ask` endpoint
  - Verify 429 response with correct headers
  - Wait for rate limit window to reset
  - Confirm requests succeed again

- [ ] **Request size limit:**
  - Send 11MB request body (default limit is 10MB)
  - Verify 413 response: "Request body too large"
  - Send 9MB body - confirm accepted

- [ ] **Concurrent request limit:**
  - Test `/ask` concurrent limit (default 3 per tenant)
  - Start 5 concurrent requests
  - Verify some rejected with 429

**Pass Criteria:** Rate limits enforced, size limits work, concurrent limits respected

---

## 4. Operational

### 4.1 Logging & Metrics

**Objective:** Validate logging and Prometheus metrics collection

#### Logging
- [ ] **Check log format:**
  - Verify structured logging (JSON or key=value format)
  - Confirm request IDs present in logs
  - Check tenant ID logged where appropriate

- [ ] **Test log levels:**
  - Set `LOG_LEVEL=DEBUG`
  - Verify debug logs appear
  - Set `LOG_LEVEL=INFO`
  - Verify only INFO and above logged

#### Metrics
- [ ] **Prometheus metrics endpoint:**
  ```bash
  curl http://localhost:8000/prometheus
  ```

- [ ] **Verify key metrics present:**
  - `activekg_refresh_cycles_total` - refresh counter
  - `activekg_search_latency` - search latency histogram
  - `connector_ingest_total{provider="s3|gcs|drive"}` - connector counters
  - `connector_worker_errors_total` - error counter
  - `activekg_access_violations_total{type="..."}` - security metrics

- [ ] **Check label cardinality:**
  - Verify no high-cardinality labels (e.g., node IDs, user IDs as labels)
  - Tenant ID as label is OK (bounded cardinality)

- [ ] **Test metrics in Grafana:**
  - Import dashboard (if exists)
  - Verify graphs populate correctly

**Pass Criteria:** Structured logs with request IDs, all key metrics present, cardinality reasonable

---

### 4.2 CI/CD

**Objective:** Validate all CI workflows green

#### Test Cases
- [ ] **Check CI workflow:**
  ```bash
  gh run list --workflow=ci.yml --limit 5
  ```
  - Verify latest run is green (✓)

- [ ] **Check MkDocs workflow:**
  ```bash
  gh run list --workflow=mkdocs.yml --limit 5
  ```
  - Verify site deploys successfully
  - Check site renders correctly: https://puneetrinity.github.io/active-graph-kg/

- [ ] **Check security scan workflow:**
  ```bash
  gh run list --workflow=security-scan.yml --limit 5
  ```
  - Verify workflow completes (even with || true)
  - Download and review report artifact

- [ ] **Add badges (if needed):**
  - CI badge: Already present ✓
  - Docs badge: Already present ✓
  - Security Scan badge: Already present ✓

- [ ] **Artifact retention:**
  - Review GitHub Actions artifact retention policy (default 90 days)
  - Adjust if needed for security reports

**Pass Criteria:** All workflows green, badges visible, artifacts retained appropriately

---

### 4.3 Build Artifacts

**Objective:** Ensure clean releases and proper .gitignore

#### Test Cases
- [ ] **Verify .gitignore entries:**
  ```bash
  cat .gitignore | grep -E "site/|screenshots/"
  ```
  - Confirm `site/` and `screenshots/` ignored ✓

- [ ] **Check git status:**
  ```bash
  git status
  ```
  - Verify `site/` and `screenshots/` not tracked

- [ ] **Test clean build:**
  ```bash
  mkdocs build
  ls -la site/
  git status  # Should not show site/ as untracked or modified
  ```

**Pass Criteria:** Build artifacts ignored, clean `git status`, releases don't include build output

---

### 4.4 Config Defaults

**Objective:** Review environment variable defaults for production

#### Review Checklist
- [ ] **Database:**
  - `ACTIVEKG_DSN` or `DATABASE_URL` set correctly
  - Connection pool size appropriate

- [ ] **Embeddings:**
  - `EMBEDDING_BACKEND=sentence-transformers` (default OK)
  - `EMBEDDING_MODEL=all-MiniLM-L6-v2` (default OK)

- [ ] **Search:**
  - `SEARCH_DISTANCE=cosine` (default OK)
  - `PGVECTOR_INDEX=ivfflat` or `hnsw` (choose based on scale)
  - `ASK_SIM_THRESHOLD=0.30` (adjust if needed)

- [ ] **LLM:**
  - `LLM_PROVIDER=groq` or `openai`
  - `LLM_MODEL` set to appropriate model
  - `ASK_ENABLE_LLM=true` (or false if LLM-free mode desired)

- [ ] **Auth:**
  - `JWT_ENABLED=true` in production
  - `JWT_ALGORITHM=RS256` (preferred) or `HS256`
  - `JWT_SECRET_KEY` or `JWT_PUBLIC_KEY` set securely

- [ ] **Rate Limiting:**
  - `RATE_LIMIT_ENABLED=true`
  - `REDIS_URL` points to production Redis

- [ ] **Scheduler:**
  - `RUN_SCHEDULER=true` on exactly ONE instance

- [ ] **Reranker:**
  - `ASK_USE_RERANKER=true` (default OK)
  - `RERANK_SKIP_TOPSIM=0.80` (adjust based on accuracy needs)

**Pass Criteria:** Production config values reviewed and appropriate for deployment

---

## 5. Documentation

### 5.1 Support Claims Accuracy

**Objective:** Ensure README and docs match actual capabilities

#### Verification
- [ ] **README.md:**
  - "Supported Connectors" section lists: S3 ✓, GCS ✓, Drive ✓
  - Azure marked as "🚧 Planned"
  - Notion/ATS not mentioned

- [ ] **MkDocs site:**
  - S3 Connector guide: https://puneetrinity.github.io/active-graph-kg/S3_CONNECTOR/
  - GCS Connector guide: https://puneetrinity.github.io/active-graph-kg/GCS_CONNECTOR/
  - Drive Connector guide: https://puneetrinity.github.io/active-graph-kg/DRIVE_CONNECTOR/

- [ ] **Code docstrings:**
  - `activekg/connectors/__init__.py` docstring accurate
  - "Fully supported: S3, GCS, Drive. Planned: Azure."

**Pass Criteria:** All documentation matches actual implementation, no false claims

---

### 5.2 Ops Runbooks

**Objective:** Ensure operational procedures documented

#### Checklist
- [ ] **OPERATIONS.md:**
  - Connector setup instructions
  - Key rotation procedures
  - Admin API reference

- [ ] **Security documentation:**
  - JWT setup guide
  - RLS configuration
  - Rate limiting setup

- [ ] **Connector guides:**
  - S3_CONNECTOR.md: IAM setup, config examples, troubleshooting
  - GCS_CONNECTOR.md: Service account setup, Pub/Sub integration
  - DRIVE_CONNECTOR.md: OAuth setup, folder monitoring

- [ ] **Performance tuning:**
  - Model caching guidance
  - GPU deployment notes (if applicable)
  - Refresh batch limit recommendations

- [ ] **Security scan workflow:**
  - Documented in README Security section ✓
  - Reference to `.github/workflows/security-scan.yml`

**Pass Criteria:** All operational procedures documented, runbooks available

---

### 5.3 Release Notes

**Objective:** Update changelog and readiness reports

#### Tasks
- [ ] **CHANGELOG.md:**
  - Update with latest changes
  - Mark Azure as "Planned"
  - Highlight transformers 4.57.1 upgrade

- [ ] **MARKET_READINESS_FINAL.md:**
  - Current state reflected accurately
  - Deferred items clearly marked (Drive E2E)

- [ ] **Version tags:**
  - Consider tagging v1.0.0 after validation
  - Ensure release notes complete

**Pass Criteria:** Changelog accurate, readiness reports up to date, release notes ready

---

## 6. Validation Execution Plan

### Suggested Order (4-6 hours total)

**Phase 1: Database & Core (45 minutes)**
1. Fresh DB + migrations (15 min)
2. Basic API smoke tests (10 min)
3. RLS and JWT auth (20 min)

**Phase 2: Connectors (2 hours)**
4. S3 connector E2E (30 min)
5. GCS connector E2E (30 min)
6. **Drive connector E2E** ⚠️ (45 min)
7. Config store encryption/rotation (15 min)

**Phase 3: Features (1.5 hours)**
8. Refresh cycle with drift-triggered events (30 min)
9. Trigger engine pattern matching (20 min)
10. Search/hybrid queries with filters and reranker (30 min)
11. LLM Q&A (streaming and non-streaming) (10 min)

**Phase 4: Security & Performance (1 hour)**
12. Rate limiting and size limits (20 min)
13. ANN index presence and per-query tuning (20 min)
14. DLQ/retry behavior (20 min)

**Phase 5: Operational (30 minutes)**
15. CI workflows review (10 min)
16. Metrics/logging validation (10 min)
17. Safety scan output review (10 min)

**Phase 6: Documentation (15 minutes)**
18. Verify all claims accurate (5 min)
19. Review runbooks complete (5 min)
20. Update release notes (5 min)

---

## 7. Drive E2E Operations Checklist

**Priority: CRITICAL** ⚠️
**Estimated Time:** 45 minutes
**Prerequisites:** Google service account, Drive folder, PostgreSQL, Redis

### Setup
1. [ ] Create Google Cloud project (if not exists)
2. [ ] Create service account:
   ```bash
   gcloud iam service-accounts create activekg-drive-test \
     --display-name="ActiveKG Drive Test"
   ```
3. [ ] Grant Drive API access:
   ```bash
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:activekg-drive-test@${PROJECT_ID}.iam.gserviceaccount.com" \
     --role="roles/drive.readonly"
   ```
4. [ ] Create and download key:
   ```bash
   gcloud iam service-accounts keys create drive-key.json \
     --iam-account=activekg-drive-test@${PROJECT_ID}.iam.gserviceaccount.com
   ```
5. [ ] Create test Drive folder and share with service account

### Execution
6. [ ] Start infrastructure:
   ```bash
   docker run -d --name activekg-postgres -e POSTGRES_PASSWORD=test -p 5432:5432 ankane/pgvector
   docker run -d --name activekg-redis -p 6379:6379 redis:7-alpine
   ```
7. [ ] Initialize database:
   ```bash
   export ACTIVEKG_DSN='postgresql://postgres:test@localhost:5432/postgres'
   psql $ACTIVEKG_DSN -c "CREATE EXTENSION vector;"
   psql $ACTIVEKG_DSN -f db/init.sql
   psql $ACTIVEKG_DSN -f enable_rls_policies.sql
   ```
8. [ ] Register Drive connector:
   ```bash
   curl -X POST http://localhost:8000/_admin/connectors/configs \
     -H "Authorization: Bearer $ADMIN_JWT" \
     -H "Content-Type: application/json" \
     -d '{
       "tenant_id": "staging_tenant",
       "provider": "drive",
       "config": {
         "service_account_json_path": "/path/to/drive-key.json",
         "folder_id": "YOUR_FOLDER_ID",
         "enabled": true,
         "poll_interval_seconds": 300
       }
     }'
   ```
9. [ ] Upload test file to Drive folder (PDF, DOCX, or TXT)
10. [ ] Trigger manual sync:
    ```bash
    curl -X POST http://localhost:8000/_admin/connectors/configs/{config_id}/sync \
      -H "Authorization: Bearer $ADMIN_JWT"
    ```
11. [ ] Verify queue populated:
    ```bash
    redis-cli LLEN "connector:drive:staging_tenant:queue"
    # Expected: > 0
    ```
12. [ ] Check worker logs:
    ```bash
    # Should see: "Processing Drive event for file: ..."
    tail -f logs/worker.log
    ```
13. [ ] Verify node in database:
    ```sql
    SELECT id, props->>'title', created_at
    FROM nodes
    WHERE tenant_id='staging_tenant'
    ORDER BY created_at DESC
    LIMIT 5;
    ```
14. [ ] Check metrics:
    ```bash
    curl http://localhost:8000/prometheus | grep 'connector_ingest_total{provider="drive"}'
    # Expected: counter incremented
    ```

### Validation Criteria
- [ ] File appears in Redis queue
- [ ] Worker processes queue without errors
- [ ] Node created in database with correct content
- [ ] Metrics show `connector_ingest_total{provider="drive"}` > 0
- [ ] Change detection works (modify file, re-sync, verify updated)

### Cleanup
15. [ ] Delete test Drive folder
16. [ ] Remove service account key
17. [ ] Stop Docker containers:
    ```bash
    docker stop activekg-postgres activekg-redis
    docker rm activekg-postgres activekg-redis
    ```

---

## 8. Pass/Fail Criteria

### Must Pass (Blocking)
- [ ] All 3 connectors complete E2E cycle
- [ ] Refresh/drift computation works correctly
- [ ] Empty payload guard prevents errors
- [ ] RLS blocks cross-tenant access
- [ ] JWT authentication works
- [ ] Rate limiting enforced
- [ ] ANN indexes exist and query correctly
- [ ] No Critical/High CVEs in dependencies
- [ ] All CI workflows green
- [ ] Documentation claims accurate

### Should Pass (Non-Blocking but Important)
- [ ] Drive E2E validated with real account
- [ ] DLQ/retry behavior tested
- [ ] Secrets encrypted and not logged
- [ ] Metrics/logging comprehensive
- [ ] Model pre-caching implemented
- [ ] Refresh batch limit added

### Nice to Have
- [ ] Performance benchmarks documented
- [ ] Grafana dashboard tested
- [ ] SBOM generated
- [ ] OpenSSF Scorecard badge added

---

## 9. Known Deferred Items

**Post-Launch Validation:**
1. Drive connector E2E with real infrastructure (staging)
2. Load testing with 10K+ nodes
3. GPU deployment validation

**Optional Enhancements:**
1. Refresh batch cap (limit per cycle)
2. Parallel embedding with thread pool
3. Model pre-caching in Docker image
4. PyPDF2 migration to pypdf (v1.1)

---

**Generated:** 2025-11-24
**Document Version:** 1.0
**Status:** Pre-Launch Validation Ready
**Estimated Completion Time:** 4-6 hours full validation
