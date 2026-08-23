# Active Graph KG - Production Deployment Checklist

**Current Status:** Phase 1+ Complete (90% Production Ready)
**Target:** 100% Production Ready
**Last Updated:** 2025-11-24

---

## Pre-Deployment Verification

### ✅ Phase 1+ Features Verified

Run automated verification:
```bash
./verify_phase1_plus.sh
```

Expected output: `✅ ALL CHECKS PASSED - Phase 1+ Complete!` (34/34 checks)

Run comprehensive tests:
```bash
# Phase 1 MVP tests
python tests/test_phase1_complete.py

# Phase 1+ improvement tests
python tests/test_phase1_plus.py

# E2E smoke test (requires API running)
uvicorn activekg.api.main:app --reload &
sleep 5
python scripts/smoke_test.py
```

---

## Database Setup (Production)

### 1. PostgreSQL + pgvector Installation

**Option A: Docker (Recommended for testing)**
```bash
docker run -d \
  --name activekg-postgres \
  --restart unless-stopped \
  -e POSTGRES_USER=activekg \
  -e POSTGRES_PASSWORD=$(openssl rand -base64 32) \
  -e POSTGRES_DB=activekg \
  -v activekg_data:/var/lib/postgresql/data \
  -p 5432:5432 \
  ankane/pgvector:pg16
```

**Option B: Native Installation (Recommended for production)**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y postgresql-16 postgresql-16-pgvector

# macOS (Homebrew)
brew install postgresql@16 pgvector

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### 2. Database Initialization

```bash
# Set connection string
export ACTIVEKG_DSN='postgresql://activekg:YOUR_SECURE_PASSWORD@localhost:5432/activekg'

# Create database and enable pgvector
psql -U postgres -c "CREATE DATABASE activekg;"
psql -U postgres -d activekg -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Create schema
psql $ACTIVEKG_DSN -f db/init.sql

# Enable Row-Level Security (REQUIRED for multi-tenancy)
psql $ACTIVEKG_DSN -f enable_rls_policies.sql

# Create vector index for performance
psql $ACTIVEKG_DSN -f enable_vector_index.sql
```

### 3. Database Tuning (Production)

Add to `postgresql.conf`:
```ini
# Memory settings (adjust based on available RAM)
shared_buffers = 4GB
effective_cache_size = 12GB
work_mem = 64MB
maintenance_work_mem = 1GB

# Connection settings
max_connections = 200

# Write-ahead log
wal_buffers = 16MB
checkpoint_completion_target = 0.9

# Query planner
random_page_cost = 1.1  # SSD
effective_io_concurrency = 200

# Parallel query
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
```

Restart PostgreSQL:
```bash
sudo systemctl restart postgresql
```

### 4. Verify Database

```bash
# Check pgvector extension
psql $ACTIVEKG_DSN -c "SELECT * FROM pg_extension WHERE extname = 'vector';"

# Check tables exist
psql $ACTIVEKG_DSN -c "\dt"

# Check RLS is enabled
psql $ACTIVEKG_DSN -c "SELECT tablename, rowsecurity FROM pg_tables WHERE schemaname = 'public';"

# Check indexes
psql $ACTIVEKG_DSN -c "\di"
```

---

## Application Setup

### 1. Create Production virtualenv

```bash
python3.11 -m venv /opt/activekg/venv
source /opt/activekg/venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 2. Environment Configuration

Create `/opt/activekg/.env`:
```bash
# Database
ACTIVEKG_DSN='postgresql://activekg:YOUR_SECURE_PASSWORD@localhost:5432/activekg'

# Embeddings
EMBEDDING_BACKEND='sentence-transformers'
EMBEDDING_MODEL='all-MiniLM-L6-v2'
EMBEDDING_ASYNC='true'
EMBEDDING_QUEUE_MAX_DEPTH='5000'
EMBEDDING_MAX_ATTEMPTS='5'
EMBEDDING_TENANT_MAX_PENDING='2000'
NODE_BATCH_MAX='200'

# API
ACTIVEKG_VERSION='1.0.0'
WORKERS=4

# Redis (required for async embedding queue)
REDIS_URL='redis://localhost:6379/0'

# Security (ADD THESE FOR 100% PRODUCTION READY)
# JWT_SECRET_KEY='your-256-bit-secret-key'
# JWT_ALGORITHM='HS256'
# JWT_EXPIRATION_HOURS=24

# Rate Limiting
# RATE_LIMIT_PER_MINUTE=60
# RATE_LIMIT_PER_HOUR=1000
```

### 3. Systemd Service (Linux)

Create `/etc/systemd/system/activekg.service`:
```ini
[Unit]
Description=Active Graph KG API Server
After=network.target postgresql.service

[Service]
Type=notify
User=activekg
Group=activekg
WorkingDirectory=/opt/activekg
EnvironmentFile=/opt/activekg/.env
ExecStart=/opt/activekg/venv/bin/uvicorn activekg.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-config logging.conf
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable activekg
sudo systemctl start activekg
sudo systemctl status activekg
```

---

### 4. Embedding Worker Service (Async Queue)

Create `/etc/systemd/system/activekg-embedding-worker.service`:
```ini
[Unit]
Description=Active Graph KG Embedding Worker
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=activekg
Group=activekg
WorkingDirectory=/opt/activekg
EnvironmentFile=/opt/activekg/.env
ExecStart=/opt/activekg/venv/bin/python -m activekg.embedding.worker
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable activekg-embedding-worker
sudo systemctl start activekg-embedding-worker
sudo systemctl status activekg-embedding-worker
```

---

## Security Hardening (Critical for Production)

### 1. Add JWT Authentication (Required for 100%)

Create `activekg/middleware/auth.py`:
```python
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os

security = HTTPBearer()

def verify_jwt(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify JWT token and extract tenant_id."""
    try:
        token = credentials.credentials
        payload = jwt.decode(
            token,
            os.getenv('JWT_SECRET_KEY'),
            algorithms=[os.getenv('JWT_ALGORITHM', 'HS256')]
        )
        return payload  # Should include 'tenant_id', 'user_id', 'scopes'
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

Update `activekg/api/main.py`:
```python
from activekg.middleware.auth import verify_jwt
from fastapi import Depends

@app.post("/search")
def search_nodes(request: KGSearchRequest, auth: dict = Depends(verify_jwt)):
    # Extract tenant_id from JWT claims
    tenant_id = auth.get('tenant_id')

    # Perform search with tenant isolation
    results = repo.vector_search(
        query_embedding=query_vec,
        tenant_id=tenant_id,  # Use JWT tenant_id
        ...
    )
```

### 2. Add Rate Limiting (Required for 100%)

Install slowapi:
```bash
pip install slowapi
```

Update `activekg/api/main.py`:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/search")
@limiter.limit("60/minute")  # 60 requests per minute
def search_nodes(request: Request, ...):
    ...
```

### 3. Content Size Limits

Set `MAX_REQUEST_SIZE_BYTES` for the API-wide body ceiling. Node content is accepted through bounded inline
properties or authenticated multipart upload only; URL, S3 and local-file payload references are unavailable.

### 4. HTTPS/TLS (Required for production)

**Option A: Nginx Reverse Proxy**
```nginx
server {
    listen 443 ssl http2;
    server_name activekg.example.com;

    ssl_certificate /etc/letsencrypt/live/activekg.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/activekg.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Option B: Railway/Render (TLS automatic)**
```bash
# Deploy to Railway
railway login
railway init
railway up
```

---

## Monitoring Setup

### 1. Prometheus Configuration

Create `prometheus.yml`:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'activekg'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/prometheus'
    authorization:
      credentials_file: /run/secrets/activekg_control_plane_token

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']  # postgres_exporter
```

Start Prometheus:
```bash
docker run -d \
  --name prometheus \
  --restart unless-stopped \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

### 2. Grafana Dashboards

Start Grafana:
```bash
docker run -d \
  --name grafana \
  --restart unless-stopped \
  -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  grafana/grafana
```

Add Prometheus data source:
1. Visit http://localhost:3000
2. Configuration → Data Sources → Add Prometheus
3. URL: http://localhost:9090
4. Save & Test

Create dashboard with key metrics:
- `rate(activekg_refresh_cycles_total[5m])` - Refresh rate
- `activekg_search_latency{quantile="0.95"}` - p95 search latency
- `activekg_active_nodes` - Active node count

### 3. Alerts Configuration

Create `alerts.yml`:
```yaml
groups:
  - name: activekg_alerts
    rules:
      - alert: HighDriftRate
        expr: rate(activekg_refresh_cycles_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High drift rate detected"
          description: "Drift rate is {{ $value }} per second"

      - alert: SlowSearchQueries
        expr: activekg_search_latency{quantile="0.95"} > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Search queries are slow"
          description: "p95 latency is {{ $value }}s"

      - alert: APIDown
        expr: up{job="activekg"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Active Graph KG API is down"
```

### 4. Logging

Configure structured logging in `logging.conf`:
```ini
[loggers]
keys=root,activekg

[handlers]
keys=console,file

[formatters]
keys=json

[logger_root]
level=INFO
handlers=console

[logger_activekg]
level=INFO
handlers=console,file
qualname=activekg
propagate=0

[handler_console]
class=StreamHandler
level=INFO
formatter=json
args=(sys.stdout,)

[handler_file]
class=handlers.RotatingFileHandler
level=INFO
formatter=json
args=('/var/log/activekg/app.log', 'a', 100*1024*1024, 5)

[formatter_json]
format={"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}
```

---

## Backup & Disaster Recovery

### 1. Database Backups

Create `/opt/activekg/backup.sh`:
```bash
#!/bin/bash
BACKUP_DIR="/var/backups/activekg"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Full database dump
pg_dump $ACTIVEKG_DSN | gzip > $BACKUP_DIR/activekg_$DATE.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "activekg_*.sql.gz" -mtime +7 -delete

echo "Backup completed: activekg_$DATE.sql.gz"
```

Add cron job:
```bash
crontab -e
# Daily backup at 2 AM
0 2 * * * /opt/activekg/backup.sh >> /var/log/activekg/backup.log 2>&1
```

### 2. Point-in-Time Recovery

Enable WAL archiving in `postgresql.conf`:
```ini
wal_level = replica
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/wal_archive/%f'
```

### 3. Restore Procedure

```bash
# Stop API
sudo systemctl stop activekg

# Drop and recreate database
psql -U postgres -c "DROP DATABASE activekg;"
psql -U postgres -c "CREATE DATABASE activekg;"

# Restore from backup
gunzip -c /var/backups/activekg/activekg_20251104_020000.sql.gz | psql $ACTIVEKG_DSN

# Start API
sudo systemctl start activekg
```

---

## Partitioning Strategies (Optional, for large datasets)

For multi-tenant datasets with high volume, consider table partitioning to improve vacuum/ANALYZE efficiency and reduce bloat.

### Option A: Partition by tenant_id (LIST)

```sql
-- Create a partitioned copy of nodes
CREATE TABLE nodes_partitioned (LIKE nodes INCLUDING ALL) PARTITION BY LIST (tenant_id);

-- Example tenant partition
CREATE TABLE nodes_tenant_acme PARTITION OF nodes_partitioned FOR VALUES IN ('acme');

-- Migrate data (example)
INSERT INTO nodes_partitioned SELECT * FROM nodes WHERE tenant_id = 'acme';

-- Indexes per partition (as needed)
CREATE INDEX idx_nodes_tenant_acme_embedding ON nodes_tenant_acme USING ivfflat (embedding vector_cosine_ops);
```

### Option B: Partition by created_at (RANGE)

```sql
-- Create a time-series partitioned table
CREATE TABLE nodes_timeseries (LIKE nodes INCLUDING ALL) PARTITION BY RANGE (created_at);

-- Monthly partitions
CREATE TABLE nodes_2025_11 PARTITION OF nodes_timeseries FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE nodes_2025_12 PARTITION OF nodes_timeseries FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Indexes per partition
CREATE INDEX idx_nodes_2025_11_embedding ON nodes_2025_11 USING ivfflat (embedding vector_cosine_ops);
```

Notes:
- Keep ANN indexes aligned with SEARCH_DISTANCE (cosine vs l2).
- Adjust autovacuum settings per partition based on write frequency.
- Consider moving partitions to different tablespaces to balance IO.

---

## Performance Optimization

### 1. Vector Index Tuning

For HNSW index (recommended for >100K nodes):
```sql
-- Drop IVFFLAT if it exists
DROP INDEX IF EXISTS idx_nodes_embedding;

-- Create HNSW index
CREATE INDEX idx_nodes_embedding ON nodes
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Analyze for query planner
ANALYZE nodes;
```

### 2. Connection Pooling

Install pgbouncer:
```bash
sudo apt-get install pgbouncer
```

Configure `/etc/pgbouncer/pgbouncer.ini`:
```ini
[databases]
activekg = host=localhost dbname=activekg

[pgbouncer]
listen_addr = 127.0.0.1
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
```

Update application DSN:
```bash
ACTIVEKG_DSN='postgresql://activekg:PASSWORD@localhost:6432/activekg'
```

---

## Production Deployment Checklist

### Critical (Must-Have for Production)

- [ ] PostgreSQL 16+ with pgvector installed
- [ ] Database schema initialized (`db/init.sql`)
- [ ] Row-Level Security enabled (`enable_rls_policies.sql`)
- [ ] Vector index created (`enable_vector_index.sql`)
- [ ] HTTPS/TLS enabled (Nginx or cloud provider)
- [ ] JWT authentication implemented
- [ ] Rate limiting configured
- [ ] Payload size limits enforced
- [ ] Database backups automated (daily)
- [ ] Prometheus metrics scraped
- [ ] Grafana dashboards created
- [ ] Alerts configured (drift, latency, uptime)
- [ ] Logging configured (structured JSON)
- [ ] Environment variables secured (.env with restricted permissions)
- [ ] Systemd service configured (auto-restart)

### Recommended (Nice-to-Have)

- [ ] Connection pooling (pgbouncer)
- [ ] Database tuning (shared_buffers, work_mem)
- [ ] HNSW index for vector search (>100K nodes)
- [ ] Point-in-time recovery (WAL archiving)
- [ ] Load balancing (multiple API instances)
- [ ] CDN for static assets
- [ ] Error tracking (Sentry, Rollbar)
- [ ] APM monitoring (New Relic, DataDog)

### Optional (Future Enhancements)

- [ ] Cron-based refresh policies
- [ ] Multi-region replication
- [ ] Read replicas for scaling
- [ ] Redis caching layer
- [ ] WebSocket event streaming
- [ ] Admin UI dashboard

---

## Post-Deployment Verification

### 1. Health Checks

```bash
# API health
curl https://activekg.example.com/health

# Prometheus metrics (private control plane)
test -n "$ACTIVEKG_CONTROL_PLANE_TOKEN"
curl -H "Authorization: Bearer $ACTIVEKG_CONTROL_PLANE_TOKEN" \
  https://activekg.example.com/prometheus | grep activekg_

# Database connectivity
psql $ACTIVEKG_DSN -c "SELECT count(*) FROM nodes;"
```

### 2. Load Testing

Install locust:
```bash
pip install locust
```

Create `locustfile.py`:
```python
from locust import HttpUser, task, between
import random

class ActiveKGUser(HttpUser):
    wait_time = between(1, 5)

    @task(3)
    def search(self):
        self.client.post("/search", json={
            "query": "machine learning",
            "top_k": 10
        })

    @task(1)
    def create_node(self):
        self.client.post("/nodes", json={
            "classes": ["Document"],
            "props": {"text": f"Test document {random.randint(1, 1000)}"}
        })
```

Run load test:
```bash
locust --host=https://activekg.example.com --users 100 --spawn-rate 10
```

### 3. Security Audit

```bash
# Check for exposed secrets
grep -r "password\|secret\|key" /opt/activekg/ --exclude-dir=venv

# Check file permissions
ls -la /opt/activekg/.env  # Should be 600 or 400

# Check open ports
sudo netstat -tuln | grep LISTEN

# Check SSL certificate
openssl s_client -connect activekg.example.com:443 -servername activekg.example.com
```

---

## Rollback Plan

### Candidate-privacy migration 023 cutover

Migration 023 and the matching API/worker release form one attended,
forward-only cutover. Freeze runtime auto-deploys, stage the complete privacy
configuration, run migration 023 through the manual release service, then
deploy the exact verified code head API → embedding worker → extraction worker.
After the ledger contains 23 rows, old code expects 22 and must not be blindly
redeployed; forward-fix the verified head. A database restore is last-resort
incident recovery, not the normal rollback.

Production placement:

- API: HMAC ring, active key version, intake flag, and exact Flow/Signal service
  authority identifiers.
- Workers: intake flag and authority identifiers only; any privacy HMAC key on
  a worker is a readiness error.
- Manual release service: migration credential only; no HMAC key or runtime
  intake flag is needed.

Ship 1AM with `CANDIDATE_PRIVACY_INTAKE_ENABLED=false`. No production directive
or candidate probe is part of acceptance. Intake stays disabled until the
separately approved Flow, Discover and cross-system replay gates are complete.

If deployment fails:

1. Stop new API instances
2. Restore database from latest backup
3. Revert code to previous version
4. Restart old API instances
5. Verify health checks pass
6. Review logs for root cause

```bash
# Emergency rollback script
#!/bin/bash
set -e

echo "Starting rollback..."

# Stop new service
sudo systemctl stop activekg

# Restore database
LATEST_BACKUP=$(ls -t /var/backups/activekg/*.sql.gz | head -1)
gunzip -c $LATEST_BACKUP | psql $ACTIVEKG_DSN

# Revert code
cd /opt/activekg
git checkout main  # Or previous release tag
source venv/bin/activate
pip install -r requirements.txt

# Restart service
sudo systemctl start activekg

echo "Rollback complete"
```

---

## Support & Escalation

### Runbooks

1. **High Drift Rate** → Check if source data changed → Adjust drift thresholds
2. **Trigger Storm** → Check pattern thresholds → Temporarily disable problematic triggers
3. **Slow Searches** → Check vector index exists → Rebuild index → Add more workers
4. **API Down** → Check logs → Restart service → Check database connectivity

### On-Call Contacts

- **Primary:** [Name] - [Email] - [Phone]
- **Secondary:** [Name] - [Email] - [Phone]
- **Database Admin:** [Name] - [Email] - [Phone]

---

**Status:** Ready for Production Deployment (90% → 100% with JWT + rate limiting)

**Next Steps:**
1. Implement JWT authentication
2. Configure rate limiting
3. Run full test suite
4. Deploy to staging
5. Load test
6. Deploy to production
