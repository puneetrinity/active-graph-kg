# Contributing to actvgraph-kg

Thanks for your interest in contributing! This project is open‑source and Postgres‑native. We welcome issues, discussions, and PRs.

## Quick Start
- Fork the repo and create a feature branch
- Run tests locally: `./verify_phase1_plus.sh && python tests/test_phase1_plus.py`
- For API changes, run the server (`uvicorn activekg.api.main:app --reload`) and use the evaluation harness (`evaluation/run_all.sh`)

## Code Style & Scope
- Keep changes minimal and focused; avoid unrelated refactors
- Follow existing patterns and naming in each module
- Prefer small, reviewable PRs (<300 lines when possible)
- Don’t add new dependencies unless necessary; justify in PR description

## Tests & Docs
- Add or update tests for new logic (unit or E2E)
- Update docs if behavior or env knobs change:
  - `README.md`, `QUICKSTART.md`, `IMPLEMENTATION_STATUS.md`
  - Add/adjust env variables and examples

## Commit Messages
- Use clear, imperative style: `add hybrid reranker knob`, `fix search response metadata`
- Reference issues when applicable: `fixes #123`

## PR Checklist
- [ ] Feature is scoped and documented
- [ ] Tests added/updated and passing locally
- [ ] README/QUICKSTART updated (if user‑facing behavior changes)
- [ ] No unrelated diffs (formatting, renames) bundled in

## Issue Labels
- `bug`, `enhancement`, `documentation`, `good first issue`, `help wanted`

## Security & Responsible Disclosure
- Don’t post secrets or real PII in issues
- For sensitive disclosures, open a minimal issue and request a maintainer contact

## Development Notes
- DB: PostgreSQL + pgvector 384‑dim vectors; enable via `db/init.sql`
- Hybrid text search: apply `db/migrations/add_text_search.sql`
- Generic `/ask`, `/ask/stream`, and `/debug/search_explain` are unavailable compatibility routes; do not revive
  them without a separately approved design

Happy hacking! 🚀
