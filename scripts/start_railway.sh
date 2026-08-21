#!/usr/bin/env sh
set -eu

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-2}"

# Ordinary runtime startup is strictly read-only.
python /app/scripts/schema_ready.py

# Drop privileged credentials before the app starts: Uvicorn workers must
# only ever hold the runtime DSN, never the owner/migration credential.
# DATABASE_URL is Railway's auto-injected plugin DSN and is owner-valued —
# when a dedicated runtime DSN exists it must not remain as a fallback.
unset ACTIVEKG_MIGRATE_DSN ACTIVEKG_RUNTIME_PASSWORD ACTIVEKG_MIGRATION_APPLY \
    ACTIVEKG_SCHEMA_ADOPT_EXISTING ACTIVEKG_SCHEMA_FRESH_INIT \
    ACTIVEKG_SCHEMA_SOURCE_COMMIT ACTIVEKG_ALLOW_MIGRATION_DRIFT || true
if [ "${ACTIVEKG_SCHEMA_ENVIRONMENT:-}" = "production" ] && [ -z "${ACTIVEKG_DSN:-}" ]; then
    echo "ERROR: production runtime requires ACTIVEKG_DSN" >&2
    exit 1
fi
unset DATABASE_URL || true

# Start the application
echo "Starting application server..."
exec uvicorn activekg.api.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
