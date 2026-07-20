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
unset ACTIVEKG_MIGRATE_DSN ACTIVEKG_RUNTIME_PASSWORD || true

# Start the application
echo "Starting application server..."
exec uvicorn activekg.api.main:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
