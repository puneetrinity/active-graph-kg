# Active Graph KG API Reference

**Status:** ✅ Production Ready
**Last Updated:** 2025-11-17

## Overview

Active Graph KG is a drift-aware knowledge graph API built on PostgreSQL and pgvector. It provides semantic search, LLM-powered Q&A with citations, automatic embedding refresh, lineage tracking, and semantic triggers.

**Key Features:**
- Semantic search with hybrid BM25+vector fusion and cross-encoder reranking
- LLM-powered Q&A with grounded citations and confidence scoring
- Drift-aware automatic refresh with configurable policies
- Multi-tenant support with Row-Level Security (RLS)
- JWT authentication and rate limiting
- Lineage tracking with provenance chains
- Semantic trigger patterns
- Prometheus metrics integration
- Dual ANN indexing (IVFFLAT/HNSW)
- DSN fallback for PaaS (DATABASE_URL for Railway/Heroku)

## Authentication

### JWT Authentication (Production)

When `JWT_ENABLED=true`, all endpoints require JWT authentication:

- **Header:** `Authorization: Bearer <token>`
- **Supported Algorithms:** RS256 (public key), HS256 (shared secret)
- **Claims:**
  - `tenant_id` (required): Tenant identifier for RLS
  - `sub` (required): User/service identifier (actor_id)
  - `scopes` (required for protected endpoints): Permission scopes

**Scope Formats** (all accepted):
- `"scopes": ["search:read", "kg:write"]` — array of strings
- `"scopes": "search:read kg:write"` — space-delimited string
- `"scope": "search:read kg:write"` — OAuth2 fallback field (space-delimited)

**Endpoint Scopes:**
| Scope | Endpoints |
|-------|-----------|
| `search:read` | `POST /search` |
| `ask:read` | `POST /ask`, `POST /ask/stream` |
| `kg:write` | `POST /nodes`, `POST /nodes/batch`, `POST /edges`, `POST /upload` |
| `admin:refresh` | `POST /admin/refresh`, debug endpoints |

**Vanta Production Config:**
```bash
JWT_ENABLED=true
JWT_ALGORITHM=RS256
JWT_AUDIENCE=activekg
JWT_ISSUER=vantahire
JWT_PUBLIC_KEY="-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----"
```

### Development Mode

When `JWT_ENABLED=false`, authentication is disabled and `tenant_id` can be provided in request bodies.

**Warning:** Only use development mode for local testing. Always enable JWT in production.

## Base URL

Default: `http://localhost:8000`

Configure via environment variables:
- `ACTIVEKG_DSN`: PostgreSQL connection string (fallback: `DATABASE_URL` for PaaS)
- `EMBEDDING_BACKEND`: Embedding provider (default: `sentence-transformers`)
- `EMBEDDING_MODEL`: Model name (default: `all-MiniLM-L6-v2`)
- `LLM_BACKEND`: LLM provider for `/ask` (default: `groq`)
- `RUN_SCHEDULER`: Run scheduler on exactly one instance (default: `true`)
- `PGVECTOR_INDEXES`: ANN index types (e.g., `ivfflat,hnsw`)
- `SEARCH_DISTANCE`: Distance metric (default: `cosine`)

## Rate Limiting

When `RATE_LIMIT_ENABLED=true`, endpoints are rate-limited per tenant:

**Headers (included in responses):**
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Unix timestamp when limit resets

**HTTP 429 Response:**
```json
{
  "detail": "Rate limit exceeded"
}
```
Header: `Retry-After: <seconds>`

---

## Endpoints

### Health & Metrics

#### GET /health

Health check endpoint with system status.

**Parameters:** None

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-11-24T12:00:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600.0,
  "components": {
    "db": {"status": "unknown"}
  },
  "llm_backend": "groq",
  "llm_model": "llama-3.1-8b-instant"
}
```

**Status Codes:**
- `200 OK`: Service healthy

**Example:**
```bash
curl http://localhost:8000/health
```

---

#### GET /metrics

Get metrics in JSON format.

**Parameters:** None

**Response:**
```json
{
  "counters": {
    "search_requests_total": 1234.0,
    "ask_requests_total": 567.0
  },
  "gauges": {
    "embedding_coverage_ratio": 0.95
  },
  "histograms": {
    "search_latency_ms": {
      "count": 1234,
      "sum": 45678.9,
      "p50": 35.2,
      "p95": 120.5,
      "p99": 250.3
    }
  },
  "timestamp": "2025-11-11T12:00:00Z"
}
```

**Status Codes:**
- `200 OK`: Metrics retrieved

**Example:**
```bash
curl http://localhost:8000/metrics
```

---

#### GET /prometheus

Get metrics in Prometheus exposition format.

**Parameters:** None

**Response:** Plain text in Prometheus format

**Status Codes:**
- `200 OK`: Metrics retrieved
- `503 Service Unavailable`: Metrics disabled

**Example:**
```bash
curl http://localhost:8000/prometheus
```

---

### Nodes

#### POST /nodes

Create a new knowledge graph node.

**Authentication:** Required when JWT enabled

**Request Body:**
```json
{
  "classes": ["Job", "Posting"],
  "props": {
    "text": "Senior ML Engineer position requiring PyTorch expertise...",
    "title": "Senior ML Engineer",
    "location": "San Francisco"
  },
  "payload_ref": "s3://bucket/job_123.json",
  "metadata": {
    "source": "linkedin",
    "posted_date": "2025-11-01"
  },
  "refresh_policy": {
    "interval_seconds": 86400,
    "drift_threshold": 0.1
  },
  "triggers": ["ml_engineer_pattern"],
  "tenant_id": "acme_corp"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `classes` | array[string] | Yes | Node class labels (1-10, max 100 chars each) |
| `props` | object | Yes | Node properties (arbitrary JSON) |
| `payload_ref` | string | No | External payload reference (URL, S3 key, max 500 chars) |
| `metadata` | object | No | Additional metadata (arbitrary JSON) |
| `refresh_policy` | object | No | Auto-refresh configuration |
| `triggers` | array[string] | No | Trigger pattern names to activate |
| `tenant_id` | string | No | Tenant ID (dev mode only, max 100 chars) |
| `extract_before_embed` | bool | No | If true, extract structured fields before embedding (requires `EXTRACTION_ENABLED=true`) |

**Response:**
```json
{
  "id": "01234567-89ab-cdef-0123-456789abcdef"
}
```

**Status Codes:**
- `200 OK`: Node created
- `400 Bad Request`: Invalid input
- `401 Unauthorized`: Missing/invalid JWT
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Creation failed

**Example:**
```bash
curl -X POST http://localhost:8000/nodes \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "classes": ["Job"],
    "props": {"text": "ML Engineer position", "title": "ML Engineer"},
    "metadata": {"source": "linkedin"}
  }'
```

**Notes:**
- If `AUTO_EMBED_ON_CREATE=true` and `EMBEDDING_ASYNC=true`, embedding is queued to Redis and processed by the embedding worker.
- If `EMBEDDING_ASYNC=false`, embedding runs in-process via background task.
- `tenant_id` from JWT overrides request body in production
- Node ID is auto-generated UUID

**Extraction (when `EXTRACTION_ENABLED=true`):**
- `extract_before_embed=true`: Queue extraction first, embedding runs after extraction completes. Best quality (embedding includes extracted fields).
- `extract_before_embed=false` (or omitted): Embed immediately, extraction runs async. Faster ingestion.
- Default behavior controlled by `EXTRACTION_MODE` env var (`async` or `sync`).
- Response includes `extraction_status` and `extraction_job_id` when extraction is queued.

---

#### POST /nodes/batch

Create multiple nodes in a single request.

**Authentication:** Required when JWT enabled

**Request Body:**
```json
{
  "tenant_id": "default",
  "continue_on_error": true,
  "nodes": [
    {
      "classes": ["Resume"],
      "props": {"text": "Candidate text", "external_id": "1"},
      "metadata": {"source": "bulk_upload"}
    }
  ]
}
```

**Response:**
```json
{
  "created": 1,
  "failed": 0,
  "results": [
    {"id": "uuid", "tenant_id": "default", "embedding_status": "queued", "job_id": "uuid"}
  ]
}
```

**Notes:**
- Uses the same embedding mode as `POST /nodes`
- Max batch size controlled by `NODE_BATCH_MAX`

---

#### GET /nodes/{node_id}

Retrieve a node by ID.

**Authentication:** Required when JWT enabled

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `node_id` | string | Yes | Node UUID |

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tenant_id` | string | No | Tenant ID (dev mode only, ignored in production) |

**Response:**
```json
{
  "id": "01234567-89ab-cdef-0123-456789abcdef",
  "classes": ["Job"],
  "props": {
    "text": "Senior ML Engineer position...",
    "title": "Senior ML Engineer"
  },
  "payload_ref": "s3://bucket/job_123.json",
  "metadata": {
    "source": "linkedin"
  },
  "refresh_policy": {
    "interval_seconds": 86400,
    "drift_threshold": 0.1
  },
  "triggers": ["ml_engineer_pattern"],
  "version": 1
}
```

**Status Codes:**
- `200 OK`: Node found
- `401 Unauthorized`: Missing/invalid JWT
- `404 Not Found`: Node not found or not visible to tenant
- `429 Too Many Requests`: Rate limit exceeded

**Example:**
```bash
curl http://localhost:8000/nodes/01234567-89ab-cdef-0123-456789abcdef \
  -H "Authorization: Bearer <token>"
```

---

#### POST /nodes/{node_id}/refresh

Manually refresh a node's embedding.

**Authentication:** Required when JWT enabled

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `node_id` | string | Yes | Node UUID to refresh |

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tenant_id` | string | No | Tenant ID (dev mode only, ignored in production) |

**Response:**
```json
{
  "id": "01234567-89ab-cdef-0123-456789abcdef",
  "drift_score": 0.12,
  "last_refreshed": "2025-11-11T12:00:00Z",
  "event_id": "event_123"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Node ID |
| `drift_score` | float | Cosine distance from previous embedding (0.0-1.0) |
| `last_refreshed` | string | ISO 8601 timestamp |
| `event_id` | string | Event ID if drift exceeded threshold, null otherwise |

**Status Codes:**
- `200 OK`: Refresh completed
- `401 Unauthorized`: Missing/invalid JWT
- `404 Not Found`: Node not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Refresh failed

**Example:**
```bash
curl -X POST http://localhost:8000/nodes/01234567-89ab-cdef-0123-456789abcdef/refresh \
  -H "Authorization: Bearer <token>"
```

**Notes:**
- Computes drift vs previous embedding using cosine similarity
- Emits `refreshed` event if drift > `refresh_policy.drift_threshold`
- Updates `embedding`, `drift_score`, and `last_refreshed` fields
- Writes to `embedding_history` table

---

#### GET /nodes/{node_id}/versions

Get embedding version history for a node.

**Authentication:** Required when JWT enabled

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `node_id` | string | Yes | Node UUID |

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `limit` | integer | No | Max versions to return (default: 10, max: 100) |

**Response:**
```json
{
  "node_id": "01234567-89ab-cdef-0123-456789abcdef",
  "versions": [
    {
      "version_index": 3,
      "drift_score": 0.12,
      "created_at": "2025-11-11T12:00:00Z",
      "embedding_ref": "s3://bucket/job_123.json"
    },
    {
      "version_index": 2,
      "drift_score": 0.08,
      "created_at": "2025-11-10T12:00:00Z",
      "embedding_ref": "s3://bucket/job_123.json"
    }
  ],
  "count": 2
}
```

**Status Codes:**
- `200 OK`: Versions retrieved
- `400 Bad Request`: Invalid limit
- `401 Unauthorized`: Missing/invalid JWT
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Query failed

**Example:**
```bash
curl http://localhost:8000/nodes/01234567-89ab-cdef-0123-456789abcdef/versions?limit=20 \
  -H "Authorization: Bearer <token>"
```

---

#### POST /upload

Upload PDF/DOCX files for text extraction, chunking, and embedding.

**Authentication:** Required when JWT enabled

**Content-Type:** `multipart/form-data`

**Form Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `files` | file(s) | Yes | - | One or more files (PDF, DOCX, HTML, TXT). Max 50 files. |
| `tenant_id` | string | No | "default" | Tenant ID (dev mode only, ignored when JWT enabled) |
| `classes` | string | No | "Document,Resume" | Comma-separated node classes for created nodes |

**Supported File Types:**
- `.pdf` (application/pdf)
- `.docx` (application/vnd.openxmlformats-officedocument.wordprocessingml.document)
- `.doc` (application/msword)
- `.html` / `.htm` (text/html)
- `.txt` (text/plain)

**Response:**
```json
{
  "uploaded": 2,
  "skipped": 1,
  "chunks_created": 5,
  "embeddings_queued": 5,
  "files": [
    {"filename": "resume.pdf", "chunks": 3, "status": "ok"},
    {"filename": "cover_letter.docx", "chunks": 2, "status": "ok"},
    {"filename": "empty.pdf", "chunks": 0, "status": "skipped"}
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `uploaded` | integer | Number of files successfully processed |
| `skipped` | integer | Number of files skipped (empty text or errors) |
| `chunks_created` | integer | Total chunk nodes created across all files |
| `embeddings_queued` | integer | Total embedding jobs enqueued |
| `files` | array | Per-file status details |

**Status Codes:**
- `200 OK`: Upload processed
- `400 Bad Request`: No files provided or too many files (max 50)
- `401 Unauthorized`: Missing/invalid JWT
- `429 Too Many Requests`: Rate limit or embedding queue capacity exceeded
- `503 Service Unavailable`: Embedding queue unavailable (Redis not configured)

**Example:**
```bash
curl -X POST http://localhost:8000/upload \
  -H "Authorization: Bearer <token>" \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.docx" \
  -F "classes=Document,Resume"
```

**Notes:**
- Each file is text-extracted, split into chunks via `create_chunk_nodes()`, and embedding jobs are enqueued per chunk
- Parent nodes are tagged with `source: "manual_upload"` in props for easy querying
- External ID format: `upload:{tenant}:{filename}:{content_hash}` (deterministic, re-uploading the same file overwrites)
- Content type is detected from the upload header or inferred from file extension
- `tenant_id` from JWT overrides the form field in production
- Max files per request controlled by `UPLOAD_MAX_FILES` env var (default: 50)

---

### Edges

#### POST /edges

Create a relationship between two nodes.

**Authentication:** Required when JWT enabled

**Request Body:**
```json
{
  "src": "node_123",
  "dst": "node_456",
  "rel": "DERIVED_FROM",
  "props": {
    "confidence": 0.95,
    "timestamp": "2025-11-11T12:00:00Z"
  },
  "tenant_id": "acme_corp"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `src` | string | Yes | Source node ID (max 100 chars) |
| `dst` | string | Yes | Target node ID (max 100 chars) |
| `rel` | string | Yes | Relationship type (max 100 chars) |
| `props` | object | No | Edge properties (arbitrary JSON) |
| `tenant_id` | string | No | Tenant ID (dev mode only, max 100 chars) |

**Common Relationship Types:**
- `DERIVED_FROM`: Provenance/lineage (used by `/lineage` endpoint)
- `WORKS_WITH`: Collaboration
- `REPORTS_TO`: Hierarchy
- `SIMILAR_TO`: Similarity

**Response:**
```json
{
  "status": "created",
  "src": "node_123",
  "rel": "DERIVED_FROM",
  "dst": "node_456"
}
```

**Status Codes:**
- `200 OK`: Edge created
- `400 Bad Request`: Invalid input
- `401 Unauthorized`: Missing/invalid JWT
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Creation failed

**Example:**
```bash
curl -X POST http://localhost:8000/edges \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "src": "node_123",
    "dst": "node_456",
    "rel": "DERIVED_FROM",
    "props": {"confidence": 0.95}
  }'
```

---

### Search

#### POST /search

Semantic search across knowledge graph nodes.

**Authentication:** Required when JWT enabled

**Request Body:**
```json
{
  "query": "ML engineer with PyTorch experience",
  "top_k": 10,
  "metadata_filters": {
    "source": "linkedin"
  },
  "compound_filter": {
    "metadata": {"job_type": "full_time"}
  },
  "tenant_id": "acme_corp",
  "use_weighted_score": true,
  "use_hybrid": true,
  "use_reranker": true,
  "decay_lambda": 0.01,
  "drift_beta": 0.1
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query text (1-2000 chars) |
| `top_k` | integer | No | 10 | Number of results (1-100) |
| `metadata_filters` | object | No | null | Simple equality filters (key-value pairs) |
| `compound_filter` | object | No | null | JSONB containment filter for nested queries |
| `tenant_id` | string | No | null | Tenant ID (dev mode only, max 100 chars) |
| `use_weighted_score` | boolean | No | false | Apply recency/drift weighting |
| `use_hybrid` | boolean | No | false | Use BM25+vector fusion (recommended) |
| `use_reranker` | boolean | No | true | Apply cross-encoder reranking (hybrid only) |
| `decay_lambda` | float | No | 0.01 | Age decay rate (0.0-1.0) |
| `drift_beta` | float | No | 0.1 | Drift penalty weight (0.0-1.0) |

**Search Modes:**

1. **Vector-only** (`use_hybrid=false`): Pure semantic similarity using embeddings
2. **Hybrid** (`use_hybrid=true`): BM25 + vector fusion with optional reranking
   - RRF fusion (default): Reciprocal rank fusion, scores 0.01-0.04
   - Weighted fusion (`HYBRID_RRF_ENABLED=false`): Linear combination, scores 0.0-1.0

**Weighted Scoring Formula** (when `use_weighted_score=true`):
```
score = similarity * exp(-decay_lambda * age_days) * (1 - drift_beta * drift_score)
```

**Response:**
```json
{
  "query": "ML engineer with PyTorch experience",
  "results": [
    {
      "id": "node_123",
      "classes": ["Resume"],
      "props": {
        "text": "Experienced ML engineer specializing in PyTorch...",
        "name": "Jane Doe"
      },
      "payload_ref": "s3://bucket/resume_123.pdf",
      "metadata": {
        "source": "linkedin"
      },
      "similarity": 0.8542,
      "text": "Experienced ML engineer specializing in PyTorch..."
    }
  ],
  "count": 10
}
```

**Status Codes:**
- `200 OK`: Search completed
- `400 Bad Request`: Invalid query
- `401 Unauthorized`: Missing/invalid JWT
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Search failed

**Example (Vector-only):**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "query": "ML engineer with PyTorch",
    "top_k": 5
  }'
```

**Example (Hybrid with reranking):**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "query": "ML engineer with PyTorch",
    "top_k": 10,
    "use_hybrid": true,
    "use_reranker": true
  }'
```

**Notes:**
- Hybrid search automatically falls back to vector-only if BM25 index unavailable
- Reranker uses cross-encoder model for higher precision (slower)
- `tenant_id` from JWT overrides request body in production
- Empty results may indicate missing embeddings (check `/debug/embed_info`)

---

### Ask (LLM Q&A)

#### POST /ask

LLM-powered Q&A with grounded citations from knowledge graph.

**Authentication:** Required when JWT enabled

**Request Body:**
```json
{
  "question": "What ML frameworks does the ML engineer position require?",
  "max_results": 5,
  "tenant_id": "acme_corp",
  "use_weighted_score": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | string | Yes | - | Question to answer (1-1000 chars) |
| `max_results` | integer | No | 5 | Max context nodes to retrieve (1-20) |
| `tenant_id` | string | No | null | Tenant ID (dev mode only, max 100 chars) |
| `use_weighted_score` | boolean | No | true | Use recency/drift weighting |

**Response:**
```json
{
  "answer": "The ML engineer position requires PyTorch and TensorFlow [0], along with experience in scikit-learn [1].",
  "citations": [
    {
      "node_id": "job_123",
      "classes": ["Job"],
      "drift_score": 0.08,
      "age_days": 1.2,
      "lineage": [
        {
          "ancestor": "linkedin_scrape_456",
          "depth": 1
        }
      ]
    }
  ],
  "confidence": 0.92,
  "metadata": {
    "searched_nodes": 20,
    "filtered_nodes": 3,
    "cited_nodes": 2,
    "top_similarity": 0.854,
    "top_vector_similarity": 0.854,
    "max_vector_similarity": 0.862,
    "gating_score": 0.854,
    "gating_score_type": "rrf_fused",
    "first_citation_idx": 0,
    "citation_at_1_precision": 1.0,
    "llm_path": "fast",
    "routing_reason": "high_confidence_sim=0.854",
    "intent_detected": "entity_job",
    "intent_type": "entity_job",
    "classes_filter": ["Job"],
    "must_have_terms": ["machine learning engineer"],
    "structured_results_count": 0
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | LLM-generated answer with citation markers [0], [1], etc. |
| `citations` | array | Cited nodes with lineage and freshness metadata |
| `confidence` | float | Answer confidence score (0.0-1.0) |
| `metadata` | object | Search diagnostics and routing info |

**Citation Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | string | Cited node UUID |
| `classes` | array[string] | Node class labels |
| `drift_score` | float | Latest drift score (0.0-1.0) |
| `age_days` | float | Days since last refresh |
| `lineage` | array | Provenance chain (DERIVED_FROM edges) |

**Metadata Fields:**

**Interpretation Note:**  
`top_similarity` reflects the **gating score** used for routing/guardrails (RRF/weighted/cosine depending on mode).  
Use `top_vector_similarity` / `max_vector_similarity` for the **true cosine similarity** between query and document embeddings.

| Field | Type | Description |
|-------|------|-------------|
| `searched_nodes` | integer | Total nodes retrieved |
| `filtered_nodes` | integer | Nodes after similarity filtering |
| `cited_nodes` | integer | Nodes actually cited in answer |
| `top_similarity` | float | Gating score for the top result (RRF/weighted/cosine depending on mode). This is the score used for routing and low-similarity guardrails. |
| `top_vector_similarity` | float | Cosine similarity for the top-ranked result (true vector similarity). |
| `max_vector_similarity` | float | Max cosine similarity among candidate results (true vector similarity). |
| `gating_score` | float | Score used for gating decision |
| `gating_score_type` | string | Score type: `rrf_fused`, `weighted_fusion`, or `cosine` |
| `first_citation_idx` | integer | Index of first citation (0-based) |
| `citation_at_1_precision` | float | 1.0 if first citation is top result, 0.0 otherwise |
| `llm_path` | string | LLM used: `fast` or `fallback` |
| `routing_reason` | string | Why fast/fallback was chosen |
| `intent_detected` | string | Detected query intent type |
| `intent_type` | string | Intent category (e.g., `entity_job`, `open_positions`) |
| `classes_filter` | array[string] | Node classes filtered by intent |
| `must_have_terms` | array[string] | Required terms for intent-based filtering |

**Scoring Modes (Hybrid Search):**

- **`HYBRID_RRF_ENABLED=true` (default):**  
  Uses Reciprocal Rank Fusion (RRF) over vector similarity rank and text rank.  
  Scores are **small (≈0.01–0.04)** and should not be compared directly to cosine.  
  Formula: `rrf = 1/(k + rank_vec) + 1/(k + rank_text)` where `k = HYBRID_RRF_K` (default 60).

- **`HYBRID_RRF_ENABLED=false`:**  
  Uses weighted fusion of vector similarity and text rank.  
  Formula: `hybrid = vector_weight * vec_sim + text_weight * (ts_rank / max_ts_rank)`  
  (default `vector_weight=0.7`, `text_weight=0.3`).

**How `top_similarity` is computed:**  
`top_similarity` is the **top gating score** produced by the hybrid search stage:
- RRF mode → RRF score  
- Weighted mode → weighted hybrid score  
- Vector-only mode → cosine similarity

**Recommended thresholds (defaults):**
- `ASK_SIM_THRESHOLD=0.30` (low-confidence fallback threshold in /ask)  
- `RRF_LOW_SIM_THRESHOLD=0.01` (extremely low similarity guardrail when RRF is enabled)  
- `RAW_LOW_SIM_THRESHOLD=0.15` (extremely low similarity guardrail when RRF is disabled)

**Example (RRF score vs cosine):**
```json
{
  "top_similarity": 0.033,
  "top_vector_similarity": 0.583,
  "max_vector_similarity": 0.583,
  "gating_score_type": "rrf_fused"
}
```
In RRF mode, `top_similarity` is the fused ranking score (small values), while
`top_vector_similarity` reflects the true cosine similarity of the top-ranked result.

**Intent Detection:**

The `/ask` endpoint detects structured query intents and applies specialized retrieval:

- **`entity_job`**: Job posting queries → filters to `Job` class
- **`entity_resume`**: Resume/experience queries → filters to `Resume` class
- **`entity_article`**: Article/knowledge queries → filters to `Article` class
- **`open_positions`**: Open positions queries → uses structured SQL query
- **`performance_issues`**: Performance issue queries → uses structured SQL query

**Hybrid Routing:**

When `HYBRID_ROUTING_ENABLED=true`, the system routes to fast or fallback LLM:

- **Fast path** (`llama-3.1-8b-instant`): High-confidence queries (top_sim >= 0.70)
- **Fallback path** (`gpt-4o-mini`): Complex queries, low confidence, reasoning

**Gating & Quality:**

- **Extremely low similarity**: Returns "I don't have enough information" if top_sim < threshold
- **Ambiguity gating**: Rejects if top 3 results are too similar (< 0.02 gap)
- **Low similarity fallback**: Limits to top-1 result, caps confidence at 0.6
- **Citation quality**: Tracks first citation precision (is top result cited?)

**Status Codes:**
- `200 OK`: Question answered (even if low confidence)
- `400 Bad Request`: Invalid question
- `401 Unauthorized`: Missing/invalid JWT
- `429 Too Many Requests`: Rate limit exceeded (includes `Retry-After` header)
- `503 Service Unavailable`: LLM disabled (`LLM_ENABLED=false`)
- `500 Internal Server Error`: Processing failed

**Example:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "question": "What ML frameworks does the position require?",
    "max_results": 5
  }'
```

**Example (Low Confidence Response):**
```json
{
  "answer": "I don't have enough information to answer this question confidently.",
  "citations": [],
  "confidence": 0.2,
  "metadata": {
    "searched_nodes": 5,
    "cited_nodes": 0,
    "filtered_nodes": 0,
    "top_similarity": 0.12,
    "gating_score": 0.12,
    "gating_score_type": "rrf_fused",
    "reason": "extremely_low_similarity"
  }
}
```

**Notes:**
- Uses hybrid search (BM25+vector) with cross-encoder reranking by default
- Citations include up to 3 ancestors in lineage chain
- Confidence calculated from citation coverage, similarity, and intent match
- Cached responses (TTL: 600s) for identical questions
- Max concurrency: 3 concurrent requests per tenant
- `tenant_id` from JWT overrides request body in production

---

#### POST /ask/stream

Server-Sent Events streaming for LLM Q&A.

**Authentication:** Required when JWT enabled

**Request Body:** Same as `/ask`

**Response:** Server-Sent Events stream with three event types:

1. **`context` event** (initial):
```
event: context
data: {"type":"context","node_ids":["node_123","node_456"],"top_similarity":0.854,"count":3}
```

2. **`token` events** (streaming):
```
event: token
data: {"type":"token","text":"The"}

event: token
data: {"type":"token","text":" position"}
```

3. **`final` event** (last):
```
event: final
data: {"type":"final","answer":"The position requires PyTorch [0]","citations":[...],"confidence":0.92,"metadata":{...}}
```

4. **`error` event** (on failure):
```
event: error
data: {"type":"error","message":"LLM generation failed"}
```

**Status Codes:**
- `200 OK`: Stream started
- `401 Unauthorized`: Missing/invalid JWT
- `429 Too Many Requests`: Rate limit exceeded
- `503 Service Unavailable`: LLM disabled

**Example:**
```bash
curl -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"question": "What are the ML frameworks?"}' \
  --no-buffer
```

**Notes:**
- Max concurrency: 2 concurrent `/ask/stream` requests per tenant
- Stricter rate limits than `/ask`
- Use `--no-buffer` with curl to see streaming tokens

---

### Triggers & Patterns

#### POST /triggers

Register a semantic trigger pattern.

**Authentication:** Required when JWT enabled

**Request Body:**
```json
{
  "name": "ml_engineer_pattern",
  "example_text": "machine learning engineer position requiring PyTorch and TensorFlow",
  "description": "Trigger for ML engineer job postings"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Pattern name (unique identifier) |
| `example_text` | string | Yes | Example text to embed as pattern |
| `description` | string | No | Human-readable description |

**Response:**
```json
{
  "status": "registered",
  "name": "ml_engineer_pattern",
  "description": "Trigger for ML engineer job postings"
}
```

**Status Codes:**
- `200 OK`: Pattern registered
- `400 Bad Request`: Missing required fields
- `401 Unauthorized`: Missing/invalid JWT
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Registration failed

**Example:**
```bash
curl -X POST http://localhost:8000/triggers \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "name": "ml_engineer_pattern",
    "example_text": "machine learning engineer position",
    "description": "ML engineer jobs"
  }'
```

**Notes:**
- Pattern embedding generated from `example_text`
- Patterns are global (not tenant-scoped)
- Triggers fire when node embeddings are similar to pattern

---

#### GET /triggers

List all registered trigger patterns.

**Authentication:** Optional (rate limited)

**Parameters:** None

**Response:**
```json
{
  "patterns": [
    {
      "name": "ml_engineer_pattern",
      "description": "ML engineer jobs",
      "created_at": "2025-11-10T12:00:00Z"
    }
  ],
  "count": 1
}
```

**Status Codes:**
- `200 OK`: Patterns listed
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Listing failed

**Example:**
```bash
curl http://localhost:8000/triggers \
  -H "Authorization: Bearer <token>"
```

---

#### DELETE /triggers/{name}

Delete a trigger pattern by name.

**Authentication:** Required when JWT enabled

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Pattern name to delete |

**Response:**
```json
{
  "status": "deleted",
  "name": "ml_engineer_pattern"
}
```

**Status Codes:**
- `200 OK`: Pattern deleted
- `401 Unauthorized`: Missing/invalid JWT
- `404 Not Found`: Pattern not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Deletion failed

**Example:**
```bash
curl -X DELETE http://localhost:8000/triggers/ml_engineer_pattern \
  -H "Authorization: Bearer <token>"
```

---

### Events

#### GET /events

List events with optional filtering.

**Authentication:** Required when JWT enabled

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `node_id` | string | No | Filter by node ID |
| `event_type` | string | No | Filter by event type |
| `tenant_id` | string | No | Tenant ID (dev mode only, ignored in production) |
| `limit` | integer | No | Max events to return (default: 100, max: 1000) |

**Event Types:**
- `refreshed`: Node embedding refreshed (drift > threshold)
- `trigger_fired`: Semantic trigger matched
- `created`: Node created
- `updated`: Node updated

**Response:**
```json
{
  "events": [
    {
      "id": "event_123",
      "node_id": "node_456",
      "type": "refreshed",
      "payload": {
        "drift_score": 0.12,
        "last_refreshed": "2025-11-11T12:00:00Z",
        "manual_trigger": true
      },
      "created_at": "2025-11-11T12:00:00Z"
    }
  ],
  "count": 1
}
```

**Status Codes:**
- `200 OK`: Events retrieved
- `401 Unauthorized`: Missing/invalid JWT
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Listing failed

**Example:**
```bash
curl "http://localhost:8000/events?node_id=node_123&limit=50" \
  -H "Authorization: Bearer <token>"
```

**Notes:**
- Events are ordered by `created_at DESC`
- `tenant_id` from JWT applies RLS filtering in production

---

### Lineage

#### GET /lineage/{node_id}

Retrieve provenance lineage for a node.

**Authentication:** Required when JWT enabled

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `node_id` | string | Yes | Node UUID |

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `max_depth` | integer | No | Max lineage depth (default: 5) |
| `tenant_id` | string | No | Tenant ID (dev mode only, ignored in production) |

**Response:**
```json
{
  "node_id": "node_123",
  "ancestors": [
    {
      "id": "node_456",
      "depth": 1,
      "edge_props": {
        "confidence": 0.95
      }
    },
    {
      "id": "node_789",
      "depth": 2,
      "edge_props": {}
    }
  ],
  "depth": 2
}
```

**Status Codes:**
- `200 OK`: Lineage retrieved
- `401 Unauthorized`: Missing/invalid JWT
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Retrieval failed

**Example:**
```bash
curl http://localhost:8000/lineage/node_123?max_depth=10 \
  -H "Authorization: Bearer <token>"
```

**Notes:**
- Traverses `DERIVED_FROM` edges recursively
- `depth=1` means direct parent, `depth=2` means grandparent, etc.
- `tenant_id` from JWT applies RLS filtering in production

---

### Admin

#### POST /admin/refresh

Trigger on-demand refresh cycle.

**Authentication:** Required (scope: `admin:refresh`)

**Request Body:**

Option 1 (specific nodes):
```json
{
  "node_ids": ["node_123", "node_456"]
}
```

Option 2 (array shorthand):
```json
["node_123", "node_456"]
```

Option 3 (all due nodes):
```json
null
```

**Response (specific nodes):**
```json
{
  "status": "completed",
  "mode": "specific_nodes",
  "requested": 2,
  "refreshed": 2
}
```

**Response (all nodes):**
```json
{
  "status": "completed",
  "mode": "all_due_nodes",
  "message": "Check logs for refresh count"
}
```

**Status Codes:**
- `200 OK`: Refresh completed
- `401 Unauthorized`: Missing/invalid JWT
- `403 Forbidden`: Missing `admin:refresh` scope
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Refresh failed

**Example (specific nodes):**
```bash
curl -X POST http://localhost:8000/admin/refresh \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"node_ids": ["node_123", "node_456"]}'
```

**Example (all due nodes):**
```bash
curl -X POST http://localhost:8000/admin/refresh \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d 'null'
```

**Notes:**
- Requires `admin:refresh` scope in JWT claims
- Emits `refreshed` event if drift > threshold
- Writes to `embedding_history` table
- `tenant_id` from JWT applies RLS filtering

---

#### GET /admin/embedding/status

Get embedding queue depth (Redis) and embedding status counts (DB).

**Authentication:** Required (scope: `admin:refresh`)

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tenant_id` | string | No | Tenant ID (dev mode only; ignored when JWT enabled) |

**Response:**
```json
{
  "tenant_id": "default",
  "status_counts": {"queued": 120, "processing": 4, "ready": 950, "failed": 2},
  "queue": {"queue": 124, "retry": 3, "dlq": 1}
}
```

---

#### POST /admin/embedding/requeue

Requeue failed embeddings (moves nodes back to queue).

**Authentication:** Required (scope: `admin:refresh`)

**Request Body:**
```json
{
  "tenant_id": "default",
  "node_ids": ["node_123", "node_456"],
  "status": "queued",
  "only_missing_embedding": true,
  "backfill_ready": true,
  "limit": 2000
}
```

**Notes:**
- If `node_ids` is omitted, nodes are selected by `status` (default: `failed`).
- Use `status: "all"` to ignore status filtering.
- `only_missing_embedding=true` requeues only nodes with `embedding IS NULL`.
- `backfill_ready=true` marks nodes with embeddings as `ready` before requeueing.

**Example:**
```bash
curl -X POST http://localhost:8000/admin/embedding/requeue \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"tenant_id":"default","status":"queued","only_missing_embedding":true,"backfill_ready":true,"limit":2000}'
```

---

#### GET /admin/extraction/status

Get extraction queue depth (Redis) and extraction status counts (DB).

**Authentication:** Required (scope: `admin:refresh`)

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tenant_id` | string | No | Tenant ID (dev mode only; ignored when JWT enabled) |

**Response:**
```json
{
  "enabled": true,
  "mode": "async",
  "tenant_id": "default",
  "status_counts": {"ready": 67, "none": 100, "failed": 2},
  "queue": {"queue": 5, "retry": 1, "dlq": 0}
}
```

**Notes:**
- `status_counts` shows nodes grouped by `props->>'extraction_status'`
- `"none"` indicates nodes that have never had extraction queued
- `queue` shows Redis queue depths (main queue, retry ZSET, dead letter queue)

---

#### POST /admin/extraction/requeue

Requeue extraction jobs for nodes.

**Authentication:** Required (scope: `admin:refresh`)

**Request Body:**
```json
{
  "tenant_id": "default",
  "node_ids": ["node_123", "node_456"],
  "status": "failed",
  "only_null_status": true,
  "limit": 2000
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tenant_id` | string | null | Tenant ID to filter nodes |
| `node_ids` | list[str] | null | Specific node IDs to requeue |
| `status` | string | null | Filter by extraction_status ("failed", "queued", etc.) |
| `only_null_status` | bool | true | Only requeue nodes that never had extraction |
| `limit` | int | 2000 | Maximum nodes to requeue |

**Notes:**
- If `node_ids` is provided, only those nodes are requeued (still filtered by tenant_id)
- If `status` is provided, `only_null_status` is auto-disabled
- Use `status: "null"` to explicitly match nodes with no extraction_status

**Example — Requeue nodes that never had extraction:**
```bash
curl -X POST http://localhost:8000/admin/extraction/requeue \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"tenant_id":"my_tenant","only_null_status":true,"limit":2000}'
```

**Example — Requeue failed extractions:**
```bash
curl -X POST http://localhost:8000/admin/extraction/requeue \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"tenant_id":"my_tenant","status":"failed","limit":500}'
```

**Response:**
```json
{
  "requested": 67,
  "enqueued": 67,
  "tenant_id": "my_tenant"
}
```

---

### Connector Admin API

#### GET /_admin/connectors/cache/health

Check connector config cache subscriber health status.

**Authentication:** JWT required when `JWT_ENABLED=true`

**Parameters:** None

**Response:**
```json
{
  "status": "ok",
  "subscriber": {
    "connected": true,
    "last_message_ts": "2025-11-11T15:08:16.411197Z",
    "reconnects": 0
  }
}
```

**Status Codes:**
- `200 OK`: Health check successful

**Example:**
```bash
curl http://localhost:8000/_admin/connectors/cache/health
```

**Notes:**
- `status: "ok"` when subscriber connected and operational
- `status: "degraded"` when subscriber down or disconnected
- See [OPERATIONS.md](operations/OPERATIONS.md) for complete operations guide

---

#### POST /_admin/connectors/rotate_keys

Rotate encryption keys for connector configurations.

**Authentication:** JWT required when `JWT_ENABLED=true`

**Request Body:**
```json
{
  "providers": ["s3", "gcs"],
  "tenants": ["acme", "corp"],
  "batch_size": 100,
  "dry_run": false
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `providers` | array | No | null | Filter by provider names |
| `tenants` | array | No | null | Filter by tenant IDs |
| `batch_size` | integer | No | 100 | Rows per batch |
| `dry_run` | boolean | No | false | Count only, no changes |

**Response:**
```json
{
  "rotated": 42,
  "skipped": 0,
  "errors": 0,
  "dry_run": false
}
```

**Dry-Run Response:**
```json
{
  "rotated": 0,
  "skipped": 0,
  "errors": 0,
  "candidates": 42,
  "dry_run": true
}
```

**Status Codes:**
- `200 OK`: Rotation completed
- `401 Unauthorized`: Missing/invalid JWT
- `500 Internal Server Error`: Rotation failed

**Examples:**

Dry-run to preview candidates:
```bash
curl -X POST http://localhost:8000/_admin/connectors/rotate_keys \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true}'
```

Rotate all configs:
```bash
curl -X POST http://localhost:8000/_admin/connectors/rotate_keys \
  -H "Content-Type: application/json" \
  -d '{"dry_run": false}'
```

Rotate specific provider:
```bash
curl -X POST http://localhost:8000/_admin/connectors/rotate_keys \
  -H "Content-Type: application/json" \
  -d '{"providers": ["s3"], "dry_run": false}'
```

With JWT authentication:
```bash
curl -X POST http://localhost:8000/_admin/connectors/rotate_keys \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true}'
```

**Notes:**
- Selects rows where `key_version != ACTIVE_VERSION`
- Decrypts with old key, re-encrypts with active key
- Invalidates cache and publishes Redis pub/sub notification
- Per-row error handling (one failure doesn't stop batch)
- See [OPERATIONS.md](operations/OPERATIONS.md) for complete runbook

---

#### POST /_admin/connectors/{provider}/ingest

Queue files from a connector (S3/GCS/Drive) for async ingestion with safeguards.

**Authentication:** JWT required with `super_admin` scope when `JWT_ENABLED=true`

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `provider` | string | Yes | Connector provider: `s3`, `gcs`, or `drive` |

**Request Body:**
```json
{
  "tenant_id": "default",
  "dry_run": true,
  "max_items": 1000,
  "batch_size": 100,
  "skip_existing": true,
  "cursor": null
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `tenant_id` | string | No | "default" | Tenant ID |
| `dry_run` | boolean | No | true | Preview only, don't actually queue |
| `max_items` | integer | No | 1000 | Max total items (1-10000) |
| `batch_size` | integer | No | 100 | Items per batch (1-500) |
| `skip_existing` | boolean | No | true | Skip already queued/processed items |
| `cursor` | string | No | null | Pagination cursor (from previous response) |

**Response (dry_run=true):**
```json
{
  "status": "dry_run",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "dry_run": true,
  "would_queue": 42,
  "skipped_count": 8,
  "total_found": 50,
  "queue_key": "connector:gcs:default:queue",
  "next_cursor": null
}
```

**Response (dry_run=false):**
```json
{
  "status": "queued",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "dry_run": false,
  "queued_count": 42,
  "skipped_count": 8,
  "total_found": 50,
  "queue_key": "connector:gcs:default:queue",
  "next_cursor": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "dry_run" or "queued" |
| `job_id` | string | Unique identifier for this ingest job |
| `dry_run` | boolean | Whether this was a preview only |
| `queued_count` / `would_queue` | integer | Files queued or would be queued |
| `skipped_count` | integer | Files skipped (already processed/queued) |
| `total_found` | integer | Total files found in source |
| `queue_key` | string | Redis queue key for monitoring |
| `next_cursor` | string | Cursor for next page (null if done) |

**Status Codes:**
- `200 OK`: Success (dry run or queued)
- `400 Bad Request`: Connector not registered or unsupported provider
- `401 Unauthorized`: Missing/invalid JWT
- `403 Forbidden`: Missing `super_admin` scope
- `500 Internal Server Error`: Operation failed

**Example (dry run first):**
```bash
# Preview what would be queued
curl -X POST http://localhost:8000/_admin/connectors/gcs/ingest \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "default", "dry_run": true}'

# Actually queue files
curl -X POST http://localhost:8000/_admin/connectors/gcs/ingest \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "default", "dry_run": false, "max_items": 500}'
```

**Notes:**
- Default is `dry_run=true` to prevent accidental bulk operations
- `skip_existing=true` dedupes against Redis queue and processed nodes in DB
- `max_items` caps total work; `batch_size` chunks processing
- Requires the ConnectorWorker to be running (`python -m activekg.connectors.worker`)
- If `EMBEDDING_ASYNC=true`, also requires EmbeddingWorker (`python -m activekg.embedding.worker`)

---

#### GET /_admin/connectors/{provider}/queue-status

Get current queue depth for a connector.

**Authentication:** JWT required with `super_admin` scope when `JWT_ENABLED=true`

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `provider` | string | Yes | Connector provider: `s3`, `gcs`, or `drive` |

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tenant_id` | string | Yes | Tenant ID |

**Response:**
```json
{
  "tenant_id": "default",
  "provider": "gcs",
  "queue_key": "connector:gcs:default:queue",
  "pending": 15
}
```

| Field | Type | Description |
|-------|------|-------------|
| `tenant_id` | string | Tenant ID |
| `provider` | string | Connector provider |
| `queue_key` | string | Redis queue key |
| `pending` | integer | Number of files waiting to be processed |

**Status Codes:**
- `200 OK`: Status retrieved
- `401 Unauthorized`: Missing/invalid JWT
- `403 Forbidden`: Missing `super_admin` scope

**Example:**
```bash
curl "http://localhost:8000/_admin/connectors/gcs/queue-status?tenant_id=default"
```

**Notes:**
- When `pending` reaches 0, all queued files have been processed
- Use with `/ingest` endpoint to monitor bulk ingestion progress

---

#### GET /admin/anomalies

Detect operational anomalies in the knowledge graph.

**Authentication:** Optional (rate limited)

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `types` | string | No | all | Comma-separated anomaly types |
| `lookback_hours` | integer | No | 24 | Hours to look back |
| `drift_spike_threshold` | float | No | 2.0 | Drift multiplier threshold |
| `trigger_storm_threshold` | integer | No | 50 | Min trigger events for storm |
| `scheduler_lag_multiplier` | float | No | 2.0 | Lag multiplier for overdue |
| `tenant_id` | string | No | null | Tenant ID filter |

**Anomaly Types:**
- `drift_spike`: Nodes with drift > 2x mean for 3+ consecutive refreshes
- `trigger_storm`: >50 trigger events in 1 hour (runaway triggers)
- `scheduler_lag`: Nodes overdue for refresh (>2x expected interval)

**Response:**
```json
{
  "anomalies": {
    "drift_spike": [
      {
        "node_id": "node_123",
        "avg_drift": 0.45,
        "recent_drifts": [0.42, 0.48, 0.44],
        "threshold": 0.20
      }
    ],
    "trigger_storm": [
      {
        "trigger_name": "ml_engineer_pattern",
        "event_count": 75,
        "time_window_hours": 1.0
      }
    ],
    "scheduler_lag": [
      {
        "node_id": "node_456",
        "expected_interval_seconds": 86400,
        "actual_interval_seconds": 180000,
        "lag_multiplier": 2.08
      }
    ]
  },
  "summary": {
    "total": 3,
    "by_type": {
      "drift_spike": 1,
      "trigger_storm": 1,
      "scheduler_lag": 1
    },
    "lookback_hours": 24
  }
}
```

**Status Codes:**
- `200 OK`: Anomalies detected
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Detection failed

**Example:**
```bash
curl "http://localhost:8000/admin/anomalies?types=drift_spike,trigger_storm&lookback_hours=48" \
  -H "Authorization: Bearer <token>"
```

---

### Debug Endpoints

Debug endpoints require `admin:refresh` scope when JWT enabled.

#### GET /debug/dbinfo

Inspect database and tenant context.

**Authentication:** Required (scope: `admin:refresh`) when JWT enabled

**Response:**
```json
{
  "database": "activekg",
  "tenant_context": "acme_corp",
  "server_host": "10.0.1.5",
  "server_port": 5432
}
```

**Example:**
```bash
curl http://localhost:8000/debug/dbinfo \
  -H "Authorization: Bearer <token>"
```

---

#### GET /debug/search_sanity

Retrieval sanity checks for diagnosing empty search results.

**Authentication:** Required (scope: `admin:refresh`) when JWT enabled

**Response:**
```json
{
  "tenant_id": "acme_corp",
  "total_nodes": 1000,
  "nodes_with_embeddings": 950,
  "nodes_with_text_search": 950,
  "embedding_coverage_pct": 95.0,
  "text_search_coverage_pct": 95.0,
  "sample_nodes_with_embedding": [
    {"id": "node_123", "classes": ["Job"], "has_text": true}
  ],
  "sample_nodes_without_embedding": [
    {"id": "node_456", "classes": ["Resume"], "has_text": false}
  ]
}
```

**Example:**
```bash
curl http://localhost:8000/debug/search_sanity \
  -H "Authorization: Bearer <token>"
```

---

#### POST /debug/search_explain

Detailed search result triage with similarity scores and snippets.

**Authentication:** Required (scope: `admin:refresh`) when JWT enabled

**Request Body:**
```json
{
  "query": "ML engineer",
  "use_hybrid": true,
  "top_k": 10
}
```

**Response:**
```json
{
  "query": "ML engineer",
  "mode": "hybrid",
  "score_type": "rrf_fused",
  "score_range": "0.01-0.04 (low)",
  "result_count": 10,
  "results": [
    {
      "node_id": "node_123",
      "similarity": 0.0342,
      "score_type": "rrf_fused",
      "classes": ["Job"],
      "snippet": "Senior ML Engineer position requiring...",
      "metadata": {"source": "linkedin"},
      "has_embedding": true,
      "has_text_search": true
    }
  ],
  "threshold_info": {
    "recommended_min": 0.15,
    "recommended_max": 0.28,
    "top_similarity": 0.0342,
    "bottom_similarity": 0.0089
  },
  "scoring_notes": {
    "rrf_fused": "RRF scores range 0.01-0.04 (rank-based fusion of vector+BM25)",
    "weighted_fusion": "Weighted scores range 0.0-1.0 (linear combination of vector+BM25)",
    "cosine": "Cosine similarity range 0.0-1.0 (vector-only)"
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/debug/search_explain \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"query": "ML engineer", "use_hybrid": true, "top_k": 10}'
```

---

#### GET /debug/embed_info

Inspect embedding configuration and stored vectors.

**Authentication:** Required (scope: `admin:refresh`) when JWT enabled

**Response:**
```json
{
  "embedding_backend": "sentence-transformers",
  "embedding_model": "all-MiniLM-L6-v2",
  "counts": {
    "total_nodes": 1000,
    "with_embedding": 950,
    "without_embedding": 50
  },
  "vector_dimension": {
    "db_type": "vector(384)",
    "db_dim": 384,
    "sampled_dims": [384]
  },
  "sample": {
    "n": 100,
    "norm_min": 0.998,
    "norm_max": 1.002,
    "norm_mean": 1.0,
    "example_ids": ["node_123", "node_456"]
  },
  "last_refreshed": {
    "count": 950,
    "age_seconds": {
      "min": 120.5,
      "avg": 43200.2,
      "max": 86400.8
    }
  }
}
```

**Example:**
```bash
curl http://localhost:8000/debug/embed_info \
  -H "Authorization: Bearer <token>"
```

---

#### GET /debug/intent

Test intent detection without running full `/ask`.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `q` | string | Yes | Query to test |

**Response:**
```json
{
  "query": "What ML frameworks does the position require",
  "normalized": "what machine learning frameworks does the position require",
  "intent_type": "entity_job",
  "params": {
    "expected_classes": ["Job"],
    "must_have_terms": ["machine learning engineer"]
  }
}
```

**Example:**
```bash
curl "http://localhost:8000/debug/intent?q=What%20ML%20frameworks%20does%20the%20position%20require"
```

---

### Demo Console

#### GET /demo

HTML demo console for testing API functionality.

**Authentication:** None

**Response:** HTML page with interactive forms for:
- Search
- Trigger management
- Event listing
- Lineage exploration
- Anomaly detection

**Example:**
```bash
open http://localhost:8000/demo
```

---

## Error Responses

All endpoints return consistent error responses:

```json
{
  "detail": "Error message"
}
```

Common status codes:
- `400 Bad Request`: Invalid input (validation error)
- `401 Unauthorized`: Missing or invalid JWT token
- `403 Forbidden`: Insufficient permissions (missing scope)
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded (includes `Retry-After` header)
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service disabled (e.g., LLM_ENABLED=false)

---

## Configuration

Key environment variables:

### Database & Embedding
- `ACTIVEKG_DSN`: PostgreSQL connection string
- `EMBEDDING_BACKEND`: `sentence-transformers`, `openai`, `cohere`
- `EMBEDDING_MODEL`: Model name (default: `all-MiniLM-L6-v2`)

### LLM (Q&A)
- `LLM_ENABLED`: Enable `/ask` endpoint (default: `true`)
- `LLM_BACKEND`: `groq`, `openai`, `litellm`
- `LLM_MODEL`: Model name (default: `llama-3.1-8b-instant`)

### Hybrid Routing
- `HYBRID_ROUTING_ENABLED`: Enable fast/fallback routing (default: `false`)
- `ASK_FAST_BACKEND`: Fast LLM backend (default: `groq`)
- `ASK_FAST_MODEL`: Fast model (default: `llama-3.1-8b-instant`)
- `ASK_FALLBACK_BACKEND`: Fallback backend (default: `openai`)
- `ASK_FALLBACK_MODEL`: Fallback model (default: `gpt-4o-mini`)

### Search & Retrieval
- `WEIGHTED_SEARCH_CANDIDATE_FACTOR`: Candidate multiplier for weighted search (default: `2.0`)
- `ASK_SIM_THRESHOLD`: Similarity cutoff for `/ask` (default: `0.30`)
- `ASK_USE_RERANKER`: Enable cross-encoder reranking (default: `true`)
- `RERANK_SKIP_TOPSIM`: Skip reranking if top_sim >= threshold (default: `0.80`)

### Authentication & Rate Limiting
- `JWT_ENABLED`: Enable JWT authentication (default: `false`)
- `JWT_SECRET_KEY`: Shared secret for HS256 (required if HS256)
- `JWT_PUBLIC_KEY_PATH`: Public key for RS256 (required if RS256)
- `RATE_LIMIT_ENABLED`: Enable rate limiting (default: `false`)
- `REDIS_URL`: Redis URL for rate limiting (required if enabled)

### Operations
- `AUTO_EMBED_ON_CREATE`: Auto-embed new nodes (default: `true`)
- `RUN_SCHEDULER`: Start refresh scheduler (default: `true`)
- `METRICS_ENABLED`: Enable Prometheus metrics (default: `true`)

---

## Rate Limits

Default rate limits per tenant (when `RATE_LIMIT_ENABLED=true`):

| Endpoint | Limit | Window | Concurrency |
|----------|-------|--------|-------------|
| `/search` | 100 req | 1 min | - |
| `/ask` | 20 req | 1 min | 3 concurrent |
| `/ask/stream` | 10 req | 1 min | 2 concurrent |
| `/nodes` | 50 req | 1 min | - |
| `/edges` | 50 req | 1 min | - |
| `/triggers` | 20 req | 1 min | - |
| `/admin/refresh` | 10 req | 1 min | - |
| default | 100 req | 1 min | - |

Concurrency limits prevent resource exhaustion from parallel requests.

---

## Best Practices

### 1. Use Hybrid Search with Reranking

For best retrieval quality:
```json
{
  "query": "ML engineer",
  "use_hybrid": true,
  "use_reranker": true
}
```

### 2. Monitor Embedding Coverage

Check `/debug/embed_info` regularly to ensure high coverage:
```bash
curl http://localhost:8000/debug/embed_info -H "Authorization: Bearer <token>"
```

Target: >95% embedding coverage for optimal search quality.

### 3. Set Appropriate Refresh Policies

For frequently changing content:
```json
{
  "refresh_policy": {
    "interval_seconds": 3600,
    "drift_threshold": 0.05
  }
}
```

### 4. Use Lineage for Provenance

Always link derived nodes to sources:
```bash
curl -X POST http://localhost:8000/edges \
  -d '{"src": "resume_v2", "dst": "resume_v1", "rel": "DERIVED_FROM"}'
```

### 5. Monitor Anomalies

Check for drift spikes and trigger storms:
```bash
curl http://localhost:8000/admin/anomalies
```

### 6. Inspect Low-Confidence Answers

Use `/ask` metadata to diagnose quality issues:
- `top_similarity < 0.30`: Likely irrelevant context
- `cited_nodes = 0`: No citations found (trust accordingly)
- `ambiguity_reason`: Results too similar (query needs refinement)

---

## Changelog

### v0.1.0 (Current)
- Initial release with core KG functionality
- Hybrid search with BM25+vector fusion
- LLM Q&A with grounded citations
- Multi-tenant RLS support
- JWT authentication
- Rate limiting with Redis
- Prometheus metrics
- Drift-aware refresh scheduler
- Semantic triggers
- Lineage tracking

---

## Support

For issues and questions:
- **GitHub Issues:** [Active Graph KG](https://github.com/puneetrinity/active-graph-kg/issues)
- **GitHub Discussions:** [Community](https://github.com/puneetrinity/active-graph-kg/discussions)
- **Documentation:** [Active Graph KG Docs](https://puneetrinity.github.io/active-graph-kg/)

---

**Generated:** 2025-11-24
**Version:** 1.0.0
**License:** MIT
