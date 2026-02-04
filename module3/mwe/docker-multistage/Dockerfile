# Multi-Stage Dockerfile for ML Inference Service
# Demonstrates Docker best practices: multi-stage builds, layer caching, security

# ============================================
# Stage 1: Builder
# Purpose: Install dependencies with build tools
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (only needed for compilation)
# These won't be in the final image, saving ~200MB
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
# Changes to app code won't invalidate this layer
COPY requirements.txt .

# Install Python packages to user directory for easy copying
# --no-cache-dir reduces image size by not caching pip downloads
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Runtime
# Purpose: Minimal image with only runtime dependencies
# ============================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create non-root user for security (CIS Docker Benchmark 4.1)
RUN useradd --create-home --shell /bin/bash appuser

# Copy only the installed packages from builder stage
# This excludes build-essential, gcc, and other build tools (~200MB saved)
COPY --from=builder /root/.local /home/appuser/.local

# Update PATH to include user-installed packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code with proper ownership
COPY --chown=appuser:appuser app.py .

# Switch to non-root user before running
USER appuser

# Document the port (informational, doesn't publish)
EXPOSE 8000

# Health check for container orchestration (Kubernetes, Docker Compose)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the inference service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================
# Build Commands:
#   docker build -t ml-service:v1 .
#   docker build --target builder -t ml-service:builder .  # Debug builder stage
#
# Run Command:
#   docker run -p 8000:8000 ml-service:v1
#
# Size Comparison (typical):
#   Single-stage with python:3.11      : ~1.0 GB
#   Single-stage with python:3.11-slim : ~500 MB
#   Multi-stage with python:3.11-slim  : ~200 MB
# ============================================