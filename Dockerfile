# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/usr/local/bin:$PATH

WORKDIR /app

# System deps (psycopg, build utils)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Exact public artifacts are acquired at build time, never on index startup.
# Keep this cache separate from an operator home and copy every referenced blob.
ENV HF_HUB_CACHE=/opt/ealana-models/hub
COPY scripts/candidate_index_artifacts.json scripts/provision_candidate_index_artifacts.py /app/scripts/
RUN python /app/scripts/provision_candidate_index_artifacts.py

COPY . /app

RUN python /app/scripts/provision_candidate_index_artifacts.py --verify-only

# Default runtime env (override in Railway/production)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=2 \
    ACTIVEKG_VERSION=1.0.0 \
    AUTO_INDEX_ON_STARTUP=false

EXPOSE 8000

# Ordinary API startup performs read-only schema readiness. Schema writes run
# only through the manual railway.schema-release.json one-shot service.
CMD ["sh", "/app/scripts/start_railway.sh"]
