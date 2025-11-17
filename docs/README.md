# Active Graph KG Documentation

This directory contains the complete documentation for Active Graph KG.

## Quick Navigation

### Operations Guides (Production)
Production deployment, security, and monitoring documentation.

- **[operations/OPERATIONS.md](operations/OPERATIONS.md)** - Complete operations runbook with admin API reference and key rotation procedures
- **[security.md](operations/security.md)** - Complete security guide including JWT authentication, Row-Level Security (RLS), rate limiting, and payload security
- **[monitoring.md](operations/monitoring.md)** - Prometheus metrics, Grafana dashboards, and alerting rules for production monitoring
- **[deployment.md](operations/deployment.md)** - Production deployment checklist, database tuning, and best practices

### Development Guides
Documentation for developers working with the codebase.

- **[api-reference.md](api-reference.md)** - Complete API reference with all 24 endpoints, authentication, request/response examples, and error codes
- **[testing.md](development/testing.md)** - Comprehensive testing guide covering setup, test execution, results, and troubleshooting
- **[architecture.md](development/architecture.md)** - System architecture documentation with component details and code locations

## Additional Resources

### Root Documentation
Key documentation files located in the project root (outside docs/):

- **QUICKSTART.md** - 5-minute setup guide to get started quickly
- **IMPLEMENTATION_STATUS.md** - Complete feature inventory with code locations
- **PHASE1_PLUS_SUMMARY.md** - Executive summary with architecture overview
- **PHASE1_PLUS_IMPROVEMENTS.md** - Detailed implementation guide for Phase 1+ features

### Archive
Historical documentation preserved in `../archive/`:

- `progress/` - Daily/weekly progress summaries (15 files)
- `implementation/` - Security and Prometheus implementation details (8 files)
- `assessments/` - System assessments and reviews (4 files)
- `marketing/` - Marketing materials, pitch decks, whitepapers (5 files)
- `setup/` - Alternative setup guides (1 file)
- `testing/` - Testing documentation history (3 files)

## Documentation Organization

```
docs/
├── README.md (you are here)
├── api-reference.md
├── operations/
│   ├── security.md
│   ├── monitoring.md
│   └── deployment.md
└── development/
    ├── testing.md
    └── architecture.md
```

## Contributing to Documentation

When adding new documentation:

1. Place operational/production docs in `operations/`
2. Place developer-focused docs in `development/`
3. Update this README.md to include the new documentation
4. Update the main project README.md if it's a major addition
5. Use clear headings, code examples, and cross-references to other docs

## Getting Help

- **Questions?** Check the main README.md in the project root
- **API Issues?** See the [API Reference](api-reference.md)
- **Deployment Issues?** Check [Deployment Guide](operations/deployment.md)
- **Security Questions?** Review [Security Guide](operations/security.md)

---

## Static Site Generation

✅ **MkDocs setup complete!** - See `mkdocs.yml` and `.github/workflows/mkdocs.yml`

**Local development:**
```bash
pip install mkdocs mkdocs-material pymdown-extensions
mkdocs serve
# Open http://127.0.0.1:8000
```

**GitHub Pages deployment:**
- Ensure repository settings: Settings → Pages → Source = "GitHub Actions"
- Push to main branch triggers automatic deployment
- Site will be available at `https://yourusername.github.io/active-graph-kg`

**Optional Material theme:**
- Uncomment `theme.name: material` in `mkdocs.yml`
- Configure palette, icons, and features as needed

---

## Future Enhancements

Planned improvements for the documentation system:

1. ✅ **Static Site Generation** - MkDocs configuration complete with GitHub Pages CI/CD
2. **CI Lint Checks** - Add automated link checking and anchor validation in CI pipeline to catch broken references
3. **API Documentation** - Consider auto-generating API docs from OpenAPI spec using mkdocstrings
4. **Interactive Examples** - Add runnable code examples with syntax highlighting
5. **Versioning** - Add version-specific documentation for major releases using `mike`

---

**Last Updated:** 2025-11-06
**Documentation Version:** 1.0.0
