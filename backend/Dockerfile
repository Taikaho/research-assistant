FROM python:3.12-slim

# libpq-dev is needed to compile psycopg2; gcc for any C extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies before copying source for better layer caching.
# A minimal app/__init__.py stub satisfies hatchling's package discovery.
COPY pyproject.toml .
RUN mkdir -p app && touch app/__init__.py \
    && pip install --no-cache-dir -e ".[dev]"

# Copy application source (overwrites the stub)
COPY . .

RUN mkdir -p /app/storage

EXPOSE 8000

# Run migrations then start the server.
# Migrations are idempotent; re-running against an already-migrated DB is safe.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]
