# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| 0.2.x   | :x:                |
| 0.1.x   | :x:                |

**Note:** Only the latest minor version receives security updates.

---

## Reporting a Vulnerability

We take the security of Active Graph KG seriously. If you discover a security vulnerability, please follow these steps:

### 1. **Do NOT open a public issue**

Security vulnerabilities should be reported privately to prevent exploitation before a fix is available.

### 2. **Email us directly**

Send your report to: **security@[yourdomain.com]** (replace with actual email)

Include the following information:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact and severity assessment
- Suggested fix (if any)
- Your contact information for follow-up

### 3. **Response Timeline**

- **Initial Response:** Within 48 hours of submission
- **Status Update:** Within 5 business days
- **Fix Timeline:** Depends on severity (see below)

### 4. **Severity Levels**

| Severity | Response Time | Examples |
|----------|--------------|----------|
| **Critical** | 24-48 hours | Authentication bypass, SQL injection, RCE |
| **High** | 1 week | XSS, CSRF, information disclosure |
| **Medium** | 2-4 weeks | DoS, authorization issues |
| **Low** | 4-8 weeks | Minor information leaks |

---

## Security Best Practices

When deploying Active Graph KG in production, follow these security guidelines:

### Authentication & Authorization

✅ **Enable JWT Authentication**
```bash
export JWT_ENABLED=true
export JWT_SECRET_KEY="<secure-256-bit-key>"  # Use RS256 for production
export JWT_ALGORITHM=RS256
```

✅ **Use Strong Secrets**
- Generate JWT secrets with: `openssl rand -base64 32`
- Use RS256 (public/private keys) instead of HS256 for production
- Rotate secrets every 90 days

✅ **Enable Row-Level Security**
```sql
-- Verify RLS is enabled
SELECT tablename, rowsecurity FROM pg_tables WHERE schemaname = 'public';
```

### Rate Limiting

✅ **Configure Redis-backed rate limiting**
```bash
export RATE_LIMIT_ENABLED=true
export REDIS_URL=redis://localhost:6379/0
```

✅ **Set appropriate limits per tenant**
- Default: 100 requests/minute per tenant
- Adjust based on your use case

### Network Security

✅ **SSRF Protection**
```bash
# Block internal IPs (enabled by default)
export ACTIVEKG_URL_ALLOWLIST="trusted-api.com,example.com"
export ACTIVEKG_FILE_BASEDIRS="/opt/data,/mnt/uploads"
```

✅ **Request Size Limits**
```bash
export ACTIVEKG_MAX_REQUEST_BODY_BYTES=10485760  # 10MB
export ACTIVEKG_MAX_FILE_BYTES=1048576          # 1MB
```

### Database Security

✅ **Use connection pooling**
```bash
export ACTIVEKG_DSN="postgresql://user:pass@host/db?pool_min_size=2&pool_max_size=10"
```

✅ **Encrypt credentials at rest**
```bash
# Generate KEK (Key Encryption Key)
export CONNECTOR_KEK_V1=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
```

✅ **Regular backups**
```bash
# Automated backups with encryption
pg_dump $ACTIVEKG_DSN | gzip | openssl enc -aes-256-cbc -out backup.sql.gz.enc
```

### Monitoring & Logging

✅ **Enable structured logging**
```bash
export LOG_LEVEL=INFO  # Use DEBUG only in development
```

✅ **Monitor security metrics**
- Track failed authentication attempts
- Monitor rate limit violations
- Alert on access control violations

✅ **Set up Prometheus alerts**
```yaml
# See observability/prometheus-alerts.yml
- alert: HighAuthFailureRate
  expr: rate(activekg_auth_failures_total[5m]) > 10
  annotations:
    summary: "High authentication failure rate detected"
```

### Deployment

✅ **Use HTTPS only**
```bash
# Ensure your reverse proxy (nginx, Caddy) enforces HTTPS
# Never expose port 8000 directly to the internet
```

✅ **Run with least privilege**
```dockerfile
# Use non-root user in Docker
USER activekg:activekg
```

✅ **Keep dependencies updated**
```bash
# Check for security vulnerabilities
pip install safety
safety check -r requirements.txt
```

---

## Known Security Considerations

### 1. LLM Provider API Keys

Active Graph KG makes requests to LLM providers (OpenAI, Groq, etc.) on behalf of users. Ensure:

- API keys are stored securely (environment variables, not in code)
- Rate limiting is configured to prevent API abuse
- LLM responses are sanitized before returning to users

### 2. Vector Search Privacy

Vector embeddings can potentially leak information about node content. Consider:

- Using tenant isolation (RLS) to prevent cross-tenant embedding exposure
- Implementing additional access controls for sensitive data
- Encrypting embeddings at rest if required by compliance

### 3. SQL Injection Prevention

All SQL queries use parameterized queries via psycopg3. However:

- Be cautious when adding custom SQL in extensions
- Never interpolate user input directly into SQL strings
- Use the repository layer's prepared statements

### 4. Dependency Vulnerabilities

We pin all dependencies to specific versions. To check for vulnerabilities:

```bash
pip install safety bandit
safety check -r requirements.txt
bandit -r activekg/ -f json -o bandit-report.json
```

---

## Security Audit History

| Date | Auditor | Scope | Findings |
|------|---------|-------|----------|
| 2025-11-17 | Internal | Full codebase review | 0 critical, 2 medium (resolved) |
| 2025-10-01 | Automated (Bandit) | Python code scanning | 0 high, 3 low (accepted risk) |

---

## Security Features

Active Graph KG includes the following security features:

✅ **Authentication**
- JWT with RS256/HS256 algorithms
- Token expiration and validation
- Tenant-based access control

✅ **Authorization**
- Row-Level Security (RLS) in PostgreSQL
- Tenant isolation for all operations
- Admin-only endpoints (`/_admin/*`)

✅ **Input Validation**
- Pydantic models for all API requests
- Size limits on request bodies and files
- Content-Type validation

✅ **Network Security**
- SSRF protection with IP blocklists
- URL allowlists for external requests
- File access restricted to base directories

✅ **Data Protection**
- Encrypted connector credentials (Fernet)
- Connection pooling with secure defaults
- Structured logging (no PII in logs)

✅ **Monitoring**
- Prometheus metrics for security events
- Failed authentication tracking
- Rate limit violation monitoring
- Access control violation alerts

---

## Compliance

Active Graph KG is designed to support compliance with common security standards:

- **OWASP Top 10** - Mitigations for all major web vulnerabilities
- **CWE Top 25** - Protection against common software weaknesses
- **GDPR** - Tenant isolation and data deletion capabilities
- **SOC 2** - Audit logging and access controls

For compliance documentation, see `docs/operations/security.md`.

---

## Security Checklist for Production

Before deploying to production, verify:

- [ ] JWT authentication enabled (`JWT_ENABLED=true`)
- [ ] Strong JWT secret configured (RS256 with key pair)
- [ ] Rate limiting enabled with Redis backend
- [ ] RLS policies applied to all tables
- [ ] SSRF protection configured
- [ ] Request size limits set appropriately
- [ ] HTTPS enforced (no HTTP access)
- [ ] Database backups configured and tested
- [ ] Monitoring and alerting set up
- [ ] Dependencies scanned for vulnerabilities
- [ ] Secrets stored in secure vault (not environment variables in production)
- [ ] Logs configured to exclude sensitive data
- [ ] Connection pooling configured
- [ ] File access restrictions configured

---

## Contact

For security inquiries:
- **Email:** security@[yourdomain.com]
- **Response Time:** Within 48 hours
- **PGP Key:** [Link to public key] (optional)

For general questions:
- **GitHub Issues:** https://github.com/puneetrinity/active-graph-kg/issues
- **Documentation:** https://puneetrinity.github.io/active-graph-kg/

---

**Last Updated:** 2025-11-24
