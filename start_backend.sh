#!/bin/bash
cd /home/ews/active-graph-kg
source venv/bin/activate

# PostgreSQL connection (port 5433)
export ACTIVEKG_DSN='postgresql://ews@localhost:5433/activekg'

# Enable sentence-transformers embeddings
export EMBEDDING_BACKEND='sentence-transformers'
export EMBEDDING_MODEL='all-MiniLM-L6-v2'

# Development settings
export RUN_SCHEDULER=false
export JWT_ENABLED=false
export RATE_LIMIT_ENABLED=false

exec uvicorn activekg.api.main:app --host 0.0.0.0 --port 8000
