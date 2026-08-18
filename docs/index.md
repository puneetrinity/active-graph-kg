# Active Graph KG Documentation

Welcome to the **Active Graph KG** documentation — a knowledge graph system with tenant-scoped semantic search,
lineage, refresh and extraction. Generic grounded Q&A is unavailable for launch.

---

## Quick Navigation

<div class="grid cards" markdown>

-   :material-rocket-launch: **Quickstart Guide**

    ---

    Get running in 5 minutes with PostgreSQL + pgvector

    [:octicons-arrow-right-24: See setup below](#getting-started)

-   :material-api: **[API Reference](api-reference.md)**

    ---

    Complete reference for all 24 endpoints with examples

    [:octicons-arrow-right-24: Browse API](api-reference.md)

-   :material-shield-lock: **[Security Guide](operations/security.md)**

    ---

    JWT authentication, RLS, rate limiting, and security best practices

    [:octicons-arrow-right-24: Secure your deployment](operations/security.md)

-   :material-monitor-dashboard: **[Monitoring Setup](operations/monitoring.md)**

    ---

    Prometheus metrics, Grafana dashboards, and alerting rules

    [:octicons-arrow-right-24: Setup monitoring](operations/monitoring.md)

-   :material-cloud-upload: **[Production Deployment](operations/deployment.md)**

    ---

    Production deployment checklist, database tuning, and best practices

    [:octicons-arrow-right-24: Deploy to production](operations/deployment.md)

-   :material-test-tube: **[Testing Guide](development/testing.md)**

    ---

    Comprehensive testing guide with setup, execution, and troubleshooting

    [:octicons-arrow-right-24: Run tests](development/testing.md)

-   :material-floor-plan: **[Architecture](development/architecture.md)**

    ---

    System architecture with component details and code locations

    [:octicons-arrow-right-24: Understand the system](development/architecture.md)

-   :material-file-document: **Implementation Status**

    ---

    Complete feature inventory with exact code locations

    [:octicons-arrow-right-24: See features below](#key-features)

</div>

---

## What is Active Graph KG?

Active Graph KG is a **self-improving knowledge graph** that combines:

- **Semantic Search** - pgvector-powered vector search with hybrid ranking
- **Bounded extraction** - Provider-backed structured extraction through the deployed worker
- **Self-Refreshing** - Automatic drift detection and scheduled refreshes
- **Multi-Tenant** - Row-level security (RLS) with per-tenant isolation
- **Production-Ready** - JWT auth, rate limiting, Prometheus metrics, comprehensive testing

### Key Features

✅ **REST API endpoints** - Health, nodes, edges, direct search, events and bounded admin operations
✅ **Hybrid search** - Vector + text search with RRF (Reciprocal Rank Fusion) reranking  
✅ **Strict citations** - LLM answers cite source nodes with [0], [1], [2] references  
💤 **Triggers & patterns** - Dormant design; CRUD and evaluation are unavailable for launch
✅ **Row-Level Security** - PostgreSQL RLS policies for tenant isolation  
✅ **Comprehensive testing** - 50+ tests covering unit, integration, E2E, security  
✅ **Observability** - Prometheus metrics, Grafana dashboards, debug endpoints

---

## Getting Started

### 1. Quick Setup (5 minutes)

```bash
# Clone and setup
git clone https://github.com/puneetrinity/active-graph-kg.git
cd active-graph-kg
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Initialize database
export ACTIVEKG_DSN="postgresql://user:pass@localhost:5432/activekg"
psql $ACTIVEKG_DSN -f db/init.sql
psql $ACTIVEKG_DSN -f enable_rls_policies.sql

# Start API server
export GROQ_API_KEY="your-key-here"
uvicorn activekg.api.main:app --host 0.0.0.0 --port 8000
```

See the setup instructions above for detailed configuration options.

### 2. Create Your First Node

```bash
curl -X POST http://localhost:8000/nodes \
  -H "Content-Type: application/json" \
  -d '{
    "id": "python-guide",
    "classes": ["Document"],
    "props": {
      "title": "Python Best Practices",
      "text": "Use type hints for better code clarity. Follow PEP 8 style guide."
    }
  }'
```

### 3. Search the Graph

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python best practices",
    "top_k": 5,
    "use_hybrid": true
  }'
```

Response:
```json
{
  "results": [
    {
      "id": "python-guide",
      "classes": ["Document"],
      "similarity": 0.92
    }
  ],
  "count": 1
}
```

---

## Operations Guides

Production deployment, security, and monitoring documentation.

- **[Security Guide](operations/security.md)** - JWT authentication, RLS, rate limiting, payload security
- **[Monitoring Setup](operations/monitoring.md)** - Prometheus metrics, Grafana dashboards, alerting rules
- **[Production Deployment](operations/deployment.md)** - Deployment checklist, database tuning, best practices

---

## Development Guides

Documentation for developers working with the codebase.

- **[API Reference](api-reference.md)** - All 24 endpoints with authentication, examples, error codes
- **[Testing Guide](development/testing.md)** - Setup, test execution, results, troubleshooting
- **[Architecture](development/architecture.md)** - System components, data flow, code locations

---

## Additional Resources

- **Phase 1+ Summary** - Executive summary with architecture overview
- **Phase 1+ Improvements** - Detailed implementation guide
- **Production Optimization Guide** - 7-phase optimization plan
- **Implementation Status** - Complete feature inventory with code locations

See the repository root for these additional documents.

---

## Support & Community

- **Questions?** Check the [Getting Started](#getting-started) section above
- **API Issues?** See the [API Reference](api-reference.md)
- **Deployment Issues?** Check [Deployment Guide](operations/deployment.md)
- **Security Questions?** Review [Security Guide](operations/security.md)
- **Bugs/Features?** Open an issue on GitHub

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         FastAPI REST API                     │
│  /health · private /readyz + metrics · /nodes /search /events│
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                      Graph Repository                        │
│  • Nodes & Edges CRUD    • Vector Search    • Hybrid Search │
│  • Dormant rule schema   • Events & Lineage • RLS Isolation │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                  PostgreSQL + pgvector + RLS                 │
│  • Nodes (JSONB + vector)  • Edges  • Events  • Triggers    │
│  • Row-Level Security (tenant_id)   • Full-text search      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                   Background Scheduler                        │
│  • Scheduled refreshes   • Drift monitor                      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                     Extraction Worker                        │
│  • Bounded structured extraction   • Re-embedding handoff    │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                   Redis (Rate Limits & Cache)                 │
│  • Per-tenant rate limiting                                  │
└──────────────────────────────────────────────────────────────┘
```

See **[Architecture Guide](development/architecture.md)** for detailed component documentation.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](https://github.com/puneetrinity/active-graph-kg/blob/main/LICENSE) file for details.

For commercial use and enterprise support, see [LICENSE-ENTERPRISE.md](https://github.com/puneetrinity/active-graph-kg/blob/main/LICENSE-ENTERPRISE.md).

---

**Last Updated:** 2025-11-24  
**Documentation Version:** 1.0.0
