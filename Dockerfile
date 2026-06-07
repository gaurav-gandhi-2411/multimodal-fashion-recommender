# ── Stage 1: dependency resolver ─────────────────────────────────────────────
# Cache-efficient: only pyproject.toml + uv.lock are copied here so Docker
# rebuilds this layer only when dependencies change, not on source edits.
FROM python:3.11-slim AS deps
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
# Install main deps only (no ml extras, no dev); skip installing the project
# itself since source is copied in the next stage.
RUN uv sync --frozen --no-dev --no-install-project

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime
WORKDIR /app

# Copy the populated virtualenv from the deps stage
COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Application source (no ML pipeline scripts, no notebooks)
COPY src/ src/
COPY app/__init__.py app/
COPY app/api/ app/api/
COPY app/brands/ app/brands/
COPY brands/ brands/
COPY config.yaml .

# Non-root user (UID 1001 avoids collision with common host UIDs)
RUN adduser --disabled-password --no-create-home --uid 1001 appuser
USER appuser

# Data artifacts (checkpoints/, data/processed/) must be mounted at runtime:
#   docker run -v $(pwd)/checkpoints:/app/checkpoints \
#              -v $(pwd)/data:/app/data \
#              -e HM_API_KEY=... fashion-rec:latest
EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
