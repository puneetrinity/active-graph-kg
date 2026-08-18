# Security Guide - Active Graph KG

**Status**: ✅ Production Ready
**Last Updated:** 2025-11-24

---

## Overview

Active Graph KG implements comprehensive security features for multi-tenant SaaS deployments:

- **JWT Authentication** - Token-based authentication with RS256/HS256 support
- **Row-Level Security (RLS)** - Database-enforced tenant isolation
- **Rate Limiting** - Redis-backed request throttling per tenant/IP
- **Content Admission** - Inline properties and bounded multipart upload only
- **Audit Trail** - Complete actor tracking for all mutations
- **Input Validation** - XSS protection, SQL injection prevention

---

## Table of Contents

1. [Authentication (JWT)](#authentication-jwt)
2. [Multi-Tenancy & RLS](#multi-tenancy-rls)
3. [Rate Limiting](#rate-limiting)
4. [Content Admission](#content-admission-security)
5. [Security Limits](#content-admission-security)
6. [PII Handling](#pii-handling)
7. [Audit Trail](#audit-trail)
8. [API Security](#api-security)
9. [Deployment Checklist](#deployment-checklist)
10. [Testing & Verification](#testing-verification)

---

## Authentication (JWT)

### Configuration

```bash
# Development (HS256)
JWT_ENABLED=true
JWT_SECRET_KEY="dev-secret-key-min-32-chars-long-for-testing"
JWT_ALGORITHM=HS256
JWT_AUDIENCE=activekg
JWT_ISSUER="https://staging-auth.yourcompany.com"
JWT_LEEWAY_SECONDS=30  # Clock skew tolerance

# Production (RS256 - recommended)
JWT_ENABLED=true
JWT_PUBLIC_KEY="-----BEGIN PUBLIC KEY-----\n..."
JWT_ALGORITHM=RS256
JWT_AUDIENCE=activekg
JWT_ISSUER="https://auth.yourcompany.com"
JWT_LEEWAY_SECONDS=30
```

### Token Format

**Required Claims:**
```json
{
  "sub": "user_123",
  "tenant_id": "acme_corp",
  "scopes": ["search:read", "nodes:write", "admin:refresh"],
  "exp": 1699999999,
  "iss": "https://auth.yourcompany.com",
  "aud": "activekg"
}
```

**Scope Format Support:**
- List format: `"scopes": ["search:read", "nodes:write"]`
- String format: `"scope": "search:read nodes:write"` (space-delimited)

### Security Features

#### JWT Leeway (Clock Skew)
```python
# Tolerates 30 seconds of clock difference
JWT_LEEWAY_SECONDS=30
```

**Why**: Prevents false 401s when client/server clocks differ.

#### Scope Validation
```python
# Endpoint requires specific scope
@app.post("/nodes/{id}/refresh")
def refresh_node(
    node_id: str,
    claims: JWTClaims = Depends(get_jwt_claims)
):
    if "admin:refresh" not in claims.scopes:
        raise HTTPException(403, "Requires admin:refresh scope")
```

#### Write Endpoint Protection

**Critical**: Write endpoints extract `tenant_id` from JWT, ignoring client input:

```python
@app.post("/nodes")
def create_node(
    node: Dict[str, Any],
    claims: Optional[JWTClaims] = Depends(get_jwt_claims)
):
    # SECURE: Uses JWT tenant_id in production
    if JWT_ENABLED and claims:
        tenant_id = claims.tenant_id
    else:
        # Dev mode fallback
        tenant_id = node.get("tenant_id")

    n = Node(..., tenant_id=tenant_id)
```

**Protected Endpoints:**
- `POST /nodes` - Creates node with JWT tenant_id
- `POST /edges` - Creates edge with JWT tenant_id
- `POST|GET|DELETE /triggers` - Deliberately unavailable; stable HTTP 410 with no control-plane work
- `POST /nodes/{id}/refresh` - Requires authentication + scope
- `POST /admin/refresh` - Requires admin:refresh scope

---

## Multi-Tenancy & RLS

### Architecture

Row-Level Security (RLS) provides **database-enforced** tenant isolation:

```sql
-- Enable RLS on all tables
ALTER TABLE nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE events ENABLE ROW LEVEL SECURITY;
ALTER TABLE node_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE embedding_history ENABLE ROW LEVEL SECURITY;

-- CRITICAL: Enforce RLS for table owners
ALTER TABLE nodes FORCE ROW LEVEL SECURITY;
ALTER TABLE edges FORCE ROW LEVEL SECURITY;
-- ... (repeat for all tables)
```

### RLS Policies

**Tenant Isolation Policy:**
```sql
CREATE POLICY tenant_isolation ON nodes
    FOR ALL
    USING (
        tenant_id = current_setting('app.current_tenant_id', true)
        OR tenant_id IS NULL  -- Shared data
    );
```

**Admin Bypass Policy:**
```sql
CREATE POLICY admin_all_access ON nodes
    FOR ALL
    TO admin_role
    USING (true);
```

### Application Integration

**Repository Layer (`activekg/graph/repository.py`):**

```python
def _conn(self, tenant_id: Optional[str] = None):
    """Create connection with tenant context for RLS."""
    conn = psycopg.connect(self.dsn)
    register_vector(conn)

    # Set tenant context for RLS
    if tenant_id:
        with conn.cursor() as cur:
            # Transaction-scoped (safe with connection pooling)
            cur.execute("SET LOCAL app.current_tenant_id = %s", (tenant_id,))

    return conn
```

**Read Operations:**
```python
# Vector search (tenant-scoped)
results = repo.vector_search(
    query_embedding=vec,
    tenant_id="acme_corp"  # From JWT
)

# Get node (tenant-scoped)
node = repo.get_node(
    node_id="123",
    tenant_id="acme_corp"  # From JWT
)
```

### Best Practices

#### 1. Always Provide tenant_id

✅ **Good:**
```python
node = Node(
    classes=["Document"],
    props={"text": "Data"},
    tenant_id="acme_corp"  # Explicit
)
```

⚠️ **Avoid:**
```python
node = Node(
    classes=["Document"],
    props={"text": "Data"}
    # No tenant_id - creates NULL tenant (shared data)
)
```

#### 2. Use NULL for Shared Data

```python
# Reference data visible to all tenants
ref_node = Node(
    classes=["Reference"],
    props={"text": "Common abbreviations"},
    tenant_id=None  # Explicitly NULL - shared
)
```

#### 3. Connection Pooling

**Use SET LOCAL (not SET):**
```python
# ✅ Correct - transaction-scoped
cur.execute("SET LOCAL app.current_tenant_id = %s", (tenant_id,))

# ❌ Wrong - session-scoped (leaks with pooling)
cur.execute("SET app.current_tenant_id = %s", (tenant_id,))
```

### Verification

**Test Tenant Isolation:**
```bash
# Create node as tenant_a
curl -X POST http://localhost:8000/nodes \
  -H "Authorization: Bearer $TENANT_A_TOKEN" \
  -d '{"classes":["Test"],"props":{"name":"test"}}'

# Response: {"id": "node_123", "tenant_id": "tenant_a"}

# Try to access as tenant_b (should fail)
curl http://localhost:8000/nodes/node_123 \
  -H "Authorization: Bearer $TENANT_B_TOKEN"

# Response: 404 Not Found (RLS blocked!)
```

---

## Rate Limiting

### Configuration

```bash
# Enable rate limiting
RATE_LIMIT_ENABLED=true
REDIS_URL="redis://localhost:6379/0"

# Proxy trust (only if behind trusted reverse proxy)
TRUST_PROXY=false  # Default: false (secure)
REAL_IP_HEADER=X-Forwarded-For
```

### Rate Limits

**Per-Tenant Limits:**
- `/search`: 100 req/s burst, 10 req/s sustained
- `/nodes` (write): 50 req/s burst, 10 req/s sustained

**Per-IP Limits (unauthenticated):**
- `/search`: 20 req/s burst, 5 req/s sustained

### Security: X-Forwarded-For Trust

**Critical**: Only trust proxy headers when behind a trusted reverse proxy:

```bash
# Without reverse proxy (default - secure)
TRUST_PROXY=false
# Uses request.client.host (cannot be spoofed)

# With reverse proxy (ALB, nginx, CloudFlare)
TRUST_PROXY=true
REAL_IP_HEADER=X-Forwarded-For
# Uses X-Forwarded-For (trust proxy to set correctly)
```

**Why**: Clients can set `X-Forwarded-For` header arbitrarily. Only trust it when your infrastructure (ALB, nginx) sets it.

### Concurrency Limits

**Long-Running Requests:**
- Concurrency cleanup: 600 seconds (10 minutes)
- Allows streaming responses, slow LLM calls
- Explicit cleanup in `finally` blocks

---

## Content admission security

Caller-selected URL, S3 and local-file payload references are unavailable. `NodeCreate.payload_ref` is a null-only
compatibility field and the graph repository contains no generic remote/local/S3 loader. Historical stored values
are inert metadata and are never fetched.

Supported content sources are:

- bounded inline node properties (`text`, `resume_text`, `content`, `body`, `description`);
- authenticated bounded multipart `/upload`.

`GET /_admin/security/limits` reports this closed state without reading or exposing stale allowlists/directories.
The global `MAX_REQUEST_SIZE_BYTES` middleware remains active for declared and chunked request bodies.


## PII Handling

### Recommended Strategies

#### Option A: Field-Level Redaction
```python
PII_FIELDS = {'email', 'phone', 'ssn', 'address'}

def redact_pii(props: dict) -> dict:
    """Redact sensitive fields before storage."""
    redacted = props.copy()
    for key in PII_FIELDS:
        if key in redacted:
            redacted[key] = hashlib.sha256(redacted[key].encode()).hexdigest()
    return redacted
```

#### Option B: Separate PII Store
```sql
CREATE TABLE node_pii (
    node_id UUID PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
    encrypted_data BYTEA NOT NULL,
    encryption_key_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

#### Option C: Retention Policies
```sql
-- Auto-delete old events (GDPR compliance)
CREATE OR REPLACE FUNCTION cleanup_old_events()
RETURNS void AS $$
BEGIN
    DELETE FROM events
    WHERE created_at < now() - INTERVAL '90 days';
END;
$$ LANGUAGE plpgsql;
```

---

## Audit Trail

### Actor Tracking

**Events Table Schema:**
```sql
ALTER TABLE events ADD COLUMN actor_id TEXT;
ALTER TABLE events ADD COLUMN actor_type TEXT;

CREATE INDEX idx_events_actor ON events(actor_id, created_at DESC);
```

**Actor Types:**
- `user` - User-triggered action (from JWT sub claim)
- `api_key` - API key-triggered action
- `scheduler` - Background scheduler
- `trigger` - Trigger engine
- `system` - System-initiated action

**Usage:**
```python
# User-triggered event
repo.append_event(
    node_id=node.id,
    event_type='manual_refresh',
    payload={'reason': 'User requested'},
    tenant_id=node.tenant_id,
    actor_id='user_123',
    actor_type='user'
)

# Scheduler-triggered event
repo.append_event(
    node_id=node.id,
    event_type='refreshed',
    payload={'drift_score': 0.12},
    tenant_id=node.tenant_id,
    actor_id='scheduler',
    actor_type='scheduler'
)
```

---

## API Security

### CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

### Input Validation

**Pydantic Models:**
```python
from pydantic import BaseModel, Field, validator

class NodeCreate(BaseModel):
    classes: List[str] = Field(..., min_items=1)
    props: Dict[str, Any]
    tenant_id: Optional[str] = None

    @validator('props')
    def validate_props(cls, v):
        # Strip HTML tags (XSS protection)
        return sanitize_html(v)
```

### Secrets Management

**Production:**
```python
import boto3
import json

def get_dsn():
    if os.getenv("ENV") == "production":
        client = boto3.client('secretsmanager')
        secret = client.get_secret_value(SecretId='activekg/db-dsn')
        return json.loads(secret['SecretString'])['dsn']
    else:
        return os.getenv("ACTIVEKG_DSN")
```

---

## Deployment Checklist

### Pre-Production

#### Database
- [ ] RLS policies enabled (`enable_rls_policies.sql`)
- [ ] FORCE RLS enabled on all tables
- [ ] Indexes created on tenant_id columns
- [ ] Admin role created
- [ ] App user has limited privileges (not table owner)

#### Authentication
- [ ] JWT_ENABLED=true
- [ ] RS256 keypair generated (production)
- [ ] JWT_PUBLIC_KEY configured
- [ ] JWT_AUDIENCE matches your domain
- [ ] JWT_ISSUER matches your IdP
- [ ] JWT_LEEWAY_SECONDS=30

#### Rate Limiting
- [ ] RATE_LIMIT_ENABLED=true
- [ ] Redis deployed and accessible
- [ ] TRUST_PROXY configured correctly
- [ ] REAL_IP_HEADER matches proxy

#### Content Admission
- [ ] Non-null node `payload_ref` values reject with HTTP 422
- [ ] Inline properties and bounded multipart upload remain available
- [ ] Security limits report remote/local loading disabled

#### API Security
- [ ] CORS configured with specific origins
- [ ] All write endpoints use JWT tenant_id
- [ ] All read endpoints use JWT tenant_id
- [ ] Sensitive endpoints require scopes

### Staging Environment

```bash
export JWT_ENABLED=true
export JWT_SECRET_KEY="dev-secret-key-min-32-chars-long"
export JWT_ALGORITHM=HS256
export JWT_AUDIENCE=activekg
export JWT_ISSUER="https://staging-auth.yourcompany.com"
export JWT_LEEWAY_SECONDS=30

export RATE_LIMIT_ENABLED=true
export REDIS_URL="redis://staging-redis:6379/0"
export TRUST_PROXY=true
export REAL_IP_HEADER=X-Forwarded-For
```

### Production Environment

```bash
export JWT_ENABLED=true
export JWT_PUBLIC_KEY="-----BEGIN PUBLIC KEY-----\n..."
export JWT_ALGORITHM=RS256
export JWT_AUDIENCE=activekg
export JWT_ISSUER="https://auth.yourcompany.com"
export JWT_LEEWAY_SECONDS=30

export RATE_LIMIT_ENABLED=true
export REDIS_URL="redis://prod-redis.internal:6379/0"
export TRUST_PROXY=true
export REAL_IP_HEADER=X-Forwarded-For

# Payload security
export ALLOWED_S3_BUCKETS="prod-bucket-1,prod-bucket-2"
```

---

## Testing & Verification

### Test RLS Tenant Isolation

```python
def test_rls_isolation():
    # Create nodes for two tenants
    node_a = repo.create_node(Node(
        classes=["Doc"],
        props={"text": "A"},
        tenant_id="tenant_a"
    ))

    node_b = repo.create_node(Node(
        classes=["Doc"],
        props={"text": "B"},
        tenant_id="tenant_b"
    ))

    # Search with tenant_a context
    results_a = repo.vector_search(query_vec, tenant_id="tenant_a")
    assert all(n.tenant_id == "tenant_a" for n, _ in results_a)

    # Search with tenant_b context
    results_b = repo.vector_search(query_vec, tenant_id="tenant_b")
    assert all(n.tenant_id == "tenant_b" for n, _ in results_b)

    # Verify cross-tenant access blocked
    node_from_b = repo.get_node(node_a.id, tenant_id="tenant_b")
    assert node_from_b is None  # RLS blocked!
```

### Test JWT Authentication

```bash
# Generate test tokens
TENANT_A_TOKEN=$(python scripts/generate_test_jwt.py \
  --tenant tenant_a \
  --scopes "search:read,nodes:write")

TENANT_B_TOKEN=$(python scripts/generate_test_jwt.py \
  --tenant tenant_b \
  --scopes "search:read,nodes:write")

# Create node as tenant_a
NODE_ID=$(curl -s -X POST http://localhost:8000/nodes \
  -H "Authorization: Bearer $TENANT_A_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"classes":["Test"],"props":{"name":"test"}}' \
  | jq -r '.id')

# Tenant A can access (200)
curl -w "%{http_code}" \
  -H "Authorization: Bearer $TENANT_A_TOKEN" \
  "http://localhost:8000/nodes/$NODE_ID"

# Tenant B cannot access (404)
curl -w "%{http_code}" \
  -H "Authorization: Bearer $TENANT_B_TOKEN" \
  "http://localhost:8000/nodes/$NODE_ID"
```

### Test Rate Limiting

```bash
# Burst test (should allow 100 req/s burst)
for i in {1..120}; do
  curl -s -w "%{http_code}\n" \
    -H "Authorization: Bearer $TOKEN" \
    -X POST http://localhost:8000/search \
    -d '{"query":"test"}' > /dev/null &
done
wait

# Expected: ~100 succeed (200), ~20 rejected (429)
```

### Verify Write Endpoint Protection

```bash
# Attempt tenant impersonation
curl -X POST http://localhost:8000/nodes \
  -H "Authorization: Bearer $TENANT_A_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "classes": ["Test"],
    "props": {"name": "test"},
    "tenant_id": "tenant_b"
  }'

# Verify node created with tenant_a (from JWT), not tenant_b (from body)
```

---

## Security Checklist

### Critical (P0)
- [x] RLS policies enabled + FORCE RLS
- [x] JWT authentication implemented
- [x] Write endpoints use JWT tenant_id
- [x] Read endpoints use JWT tenant_id
- [x] Rate limiting enabled
- [x] Actor tracking in events
- [x] TRUST_PROXY configured correctly

### High Priority (P1)
- [x] Generic URL/S3/file payload loaders removed
- [x] JWT clock skew tolerance
- [x] Scope format support (list + string)
- [x] Concurrency cleanup for long requests

### Medium Priority (P2)
- [ ] PII redaction (depends on use case)
- [ ] Secrets manager integration
- [ ] Encryption at rest
- [ ] CORS specific origins
- [ ] Admin role scope checks

---

## Compliance

### GDPR
- [ ] Right to erasure (DELETE /nodes/{id} with cascade)
- [ ] Right to access (GET /nodes + /events)
- [ ] Data minimization
- [ ] Retention policies (90-day event cleanup)
- [ ] Encryption at rest

### HIPAA
- [ ] Encryption in transit (TLS)
- [ ] Encryption at rest (PostgreSQL TDE)
- [ ] Audit logging (actor_id + events)
- [ ] Access controls (RLS + JWT)
- [ ] BAA with cloud providers

### SOC 2
- [x] Audit trail (events table)
- [x] Change tracking (node_versions)
- [x] Access logging (API middleware)
- [x] Monitoring (Prometheus metrics)

---

## Summary

**Active Graph KG Security Features:**

✅ **JWT Authentication** - RS256/HS256, clock skew tolerance, scope validation
✅ **RLS Tenant Isolation** - Database-enforced, admin bypass, NULL tenant support
✅ **Rate Limiting** - Redis-backed, per-tenant, proxy-aware
✅ **Write Protection** - JWT tenant_id extraction, prevents impersonation
✅ **Payload Security** - Path traversal prevention, size limits, content validation
✅ **Audit Trail** - Actor tracking, event history, tenant-scoped
✅ **API Security** - CORS, input validation, scope checks

**Status**: Production Ready for Multi-Tenant SaaS

**Test Results**: 9/13 tests passing (all critical security tests ✅)

---

## References

- **Original Documentation**:
  - `SECURITY_CONSIDERATIONS.md` - Security architecture overview
  - `SECURITY_FIXES_2025-11-07.md` - Specific security fixes (2025-11-07)
  - `RLS_BEST_PRACTICES.md` - Row-Level Security implementation guide
  - `JWT_RLS_TEST_RESULTS.md` - Integration test results

- **Implementation Files**:
  - `enable_rls_policies.sql` - RLS policies and helper functions
  - `activekg/api/auth.py` - JWT verification and claims extraction
  - `activekg/api/rate_limiter.py` - Rate limiting middleware
  - `activekg/graph/repository.py` - RLS integration in data layer

- **Related Guides**:
  - `docs/operations/deployment.md` - Production deployment guide
  - `docs/development/testing.md` - Security testing procedures
