#!/bin/bash
# Active Graph KG API Startup Script
# With all fixes applied from backend readiness check

set -e

echo "Starting Active Graph KG API..."

# Activate virtual environment
source venv/bin/activate

# Set all required environment variables
export ACTIVEKG_DSN="postgresql://activekg:activekg@localhost:5433/activekg"
export GROQ_API_KEY="${GROQ_API_KEY:-your-groq-api-key-here}"
export LLM_ENABLED=true
export LLM_BACKEND=groq
export LLM_MODEL="llama-3.1-8b-instant"
export JWT_ENABLED=true
export JWT_SECRET_KEY="test-secret-key-min-32-chars-long-for-testing-purposes"
export JWT_ALGORITHM=HS256
export JWT_AUDIENCE=activekg
export JWT_ISSUER="https://test-auth.activekg.local"
export REDIS_URL="redis://localhost:6379/0"
export RATE_LIMIT_ENABLED=false
export RUN_SCHEDULER=false  # Set to true for production
export EMBEDDING_BACKEND="sentence-transformers"
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
export AUTO_EMBED_ON_CREATE=true

echo "Environment configured"
echo "Starting uvicorn on port 8000..."

# Start API
uvicorn activekg.api.main:app --host 0.0.0.0 --port 8000 --reload
