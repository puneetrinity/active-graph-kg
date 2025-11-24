# Changelog

All notable changes to Active Graph KG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Fast unit tests job in CI (runs without database in ~30s)
- Makefile targets: `make test-unit` and `make test-all`
- Grafana error tracking panels (API Errors by Type, Top 5 Error Endpoints)
- MkDocs documentation site with Material theme
- CHANGELOG.md, SECURITY.md, CODE_OF_CONDUCT.md
- CODEOWNERS file for automatic PR review assignment
- CI status badges in README

### Changed
- Switched documentation theme from ReadTheDocs to Material
- Improved import sorting and code formatting (ruff)

### Fixed
- MkDocs configuration (license display, repository links)
- Test mode support to avoid database connection during unit tests
- Import organization and linting errors

## [1.0.0] - 2025-11-17

### Added
- **Self-Refreshing Nodes** with drift detection and automatic updates
- **Semantic Triggers** with pattern matching (exact, prefix, suffix, contains, regex)
- **JWT Authentication** with RS256/HS256 support and tenant isolation
- **Row-Level Security (RLS)** with PostgreSQL policies for multi-tenancy
- **Rate Limiting** per tenant with Redis backend
- **Prometheus Metrics** with 15+ custom metrics
- **Grafana Dashboards** with 15 pre-built panels
- **Hybrid Search** combining vector and text search with RRF reranking
- **LLM Q&A** with strict citations using RAG
- **24 REST API Endpoints** covering all core functionality
- **Comprehensive Testing** with 31+ test files (unit, integration, E2E)
- **Production Deployment Guides** for Docker, Railway, and manual deployment
- **Security Hardening** (SSRF protection, file access controls, request size limits)
- **Observability** with structured logging and performance tracking
- **Scheduler** with SLA tracking and failure recovery
- **Connector Framework** supporting S3, GCS, Azure, Google Drive
- **Dual Indexing Strategy** (IVFFLAT + HNSW) for vector search
- **Connection Pooling** with psycopg-pool for scalability
- **Pre-commit Hooks** for code quality enforcement

### Changed
- Migrated from single IVFFLAT index to dual IVFFLAT/HNSW strategy
- Upgraded to PostgreSQL 16 with pgvector 0.4.1
- Improved error categorization in metrics (validation, auth, rate_limit, etc.)
- Enhanced logger with automatic tenant_id/request_id context injection

### Fixed
- Race condition in scheduler with multiple instances
- Memory leak in long-running refresh operations
- Timezone handling in scheduled jobs (now uses UTC)
- Connection pool exhaustion under high load

### Security
- Implemented SSRF protection with IP blocklists
- Added request body size limits (10MB default)
- Enforced file access restrictions with base directory allowlists
- Added JWT signature verification with proper algorithm enforcement
- Implemented secure credential encryption for connector configs

## [0.2.0] - 2025-10-15

### Added
- Vector search with pgvector integration
- Basic node and edge CRUD operations
- Simple search endpoint with similarity matching
- Docker Compose setup for local development
- Initial test suite covering core functionality

### Changed
- Refactored repository layer for better testability
- Improved error handling across API endpoints

### Fixed
- Vector embedding consistency issues
- SQL injection vulnerabilities in search queries

## [0.1.0] - 2025-09-01

### Added
- Initial MVP release
- Basic knowledge graph structure with nodes and edges
- PostgreSQL backend with vector extension support
- FastAPI REST API with 12 core endpoints
- Basic authentication (API key)
- Simple documentation and README

---

## Version History

- **1.0.0** (2025-11-17) - Production release with full feature set
- **0.2.0** (2025-10-15) - Vector search and testing improvements
- **0.1.0** (2025-09-01) - Initial MVP release

## Migration Guides

### Migrating from 0.2.0 to 1.0.0

**Breaking Changes:**
- Authentication now requires JWT tokens (API key auth removed)
- Environment variable changes: `API_KEY` → `JWT_SECRET_KEY`
- Database schema migration required (run `db/migrations/add_tenant_id.sql`)

**Steps:**
1. Backup your database
2. Run database migrations: `psql $ACTIVEKG_DSN -f db/migrations/*.sql`
3. Update environment variables (see docs/operations/deployment.md)
4. Generate JWT tokens using `scripts/generate_test_jwt.py`
5. Update API clients to use `Authorization: Bearer <token>` header

### Migrating from 0.1.0 to 0.2.0

**Breaking Changes:**
- Search endpoint now requires vector embeddings
- Node creation requires `content` field for embedding generation

**Steps:**
1. Upgrade pgvector: `CREATE EXTENSION IF NOT EXISTS vector;`
2. Run schema migration: `psql $ACTIVEKG_DSN -f db/init.sql`
3. Re-index existing nodes: `POST /admin/reindex`

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on our development process and how to submit pull requests.

## Links

- [Documentation](https://puneetrinity.github.io/active-graph-kg/)
- [GitHub Repository](https://github.com/puneetrinity/active-graph-kg)
- [Issue Tracker](https://github.com/puneetrinity/active-graph-kg/issues)
- [Release Notes](https://github.com/puneetrinity/active-graph-kg/releases)
