# Active Graph KG Documentation

Welcome to the **Active Graph KG** documentation — a production-ready knowledge graph system with semantic search, LLM-powered Q&A, and self-improving capabilities.

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
- **LLM Q&A** - Citation-backed answers using retrieval-augmented generation
- **Self-Improving** - Automatic drift detection, scheduled refreshes, and trigger-based updates
- **Multi-Tenant** - Row-level security (RLS) with per-tenant isolation
- **Production-Ready** - JWT auth, rate limiting, Prometheus metrics, comprehensive testing

### Key Features

✅ **24 REST API endpoints** - Health, nodes, edges, search, ask, triggers, events, admin  
✅ **Hybrid search** - Vector + text search with RRF (Reciprocal Rank Fusion) reranking  
✅ **Strict citations** - LLM answers cite source nodes with [0], [1], [2] references  
✅ **Triggers & patterns** - Auto-refresh nodes on conditions, schedule-based updates  
✅ **Row-Level Security** - PostgreSQL RLS policies for tenant isolation  
✅ **Comprehensive testing** - 50+ tests covering unit, integration, E2E, security  
✅ **Observability** - Prometheus metrics, Grafana dashboards, debug endpoints

---

## Getting Started

### 1. Quick Setup (5 minutes)

```bash
# Clone and setup
git clone https://github.com/yourusername/active-graph-kg.git
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

### 3. Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are Python best practices?"
  }'
```

Response:
```json
{
  "answer": "Python best practices include using type hints for better code clarity and following the PEP 8 style guide [0].",
  "sources": [
    {
      "node_id": "python-guide",
      "title": "Python Best Practices",
      "similarity": 0.92
    }
  ]
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
│  /health /nodes /search /ask /triggers /events /admin /_prom │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                      Graph Repository                        │
│  • Nodes & Edges CRUD    • Vector Search    • Hybrid Search │
│  • Triggers & Patterns   • Events & Lineage • RLS Isolation │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────────────┐
│                  PostgreSQL + pgvector + RLS                 │
│  • Nodes (JSONB + vector)  • Edges  • Events  • Triggers    │
│  • Row-Level Security (tenant_id)   • Full-text search      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                   Background Scheduler                        │
│  • Trigger polling   • Scheduled refreshes   • Drift monitor │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                   LLM Provider (Groq/OpenAI)                  │
│  • Citation-backed answers   • Intent detection   • Embeddings│
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                   Redis (Rate Limits & Cache)                 │
│  • Per-tenant rate limiting   • /ask response cache          │
└──────────────────────────────────────────────────────────────┘
```

See **[Architecture Guide](development/architecture.md)** for detailed component documentation.

---

## License

[Your License Here - e.g., MIT, Apache 2.0]

---

**Last Updated:** 2025-11-11  
**Documentation Version:** 1.0.0
