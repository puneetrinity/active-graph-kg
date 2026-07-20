#!/usr/bin/env sh
set -eu

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-2}"

# Run database initialization/migrations before starting the app
echo "Running database initialization..."
python /app/scripts/init_railway_db.py

# Drop privileged credentials before the app starts: Uvicorn workers must
# only ever hold the runtime DSN, never the owner/migration credential.
# DATABASE_URL is Railway's auto-injected plugin DSN and is owner-valued —
# when a dedicated runtime DSN exists it must not remain as a fallback.
unset ACTIVEKG_MIGRATE_DSN ACTIVEKG_RUNTIME_PASSWORD || true
if [ -n "${ACTIVEKG_DSN:-}" ]; then
    unset DATABASE_URL || true
else
    echo "WARNING: ACTIVEKG_DSN not set — the app will fall back to DATABASE_URL."
    echo "         That credential is typically the database owner; production must"
    echo "         set ACTIVEKG_DSN to the restricted runtime role instead."
fi

# Start the application
echo "Starting application server..."
exec uvicorn activekg.api.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
