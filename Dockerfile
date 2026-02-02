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

COPY . /app

# Default runtime env (override in Railway/production)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=2 \
    ACTIVEKG_VERSION=1.0.0 \
    AUTO_INDEX_ON_STARTUP=false

EXPOSE 8000

CMD bash -c "uvicorn activekg.api.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --workers ${WORKERS:-2}"

