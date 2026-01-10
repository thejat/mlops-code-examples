# Multi-Stage Dockerfile for ML Inference Service
# Demonstrates Module 3 best practices: multi-stage builds, layer caching, security

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (only needed for compilation)
# These won't be in the final image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
# Changes to code won't invalidate this layer
COPY requirements.txt .

# Install Python packages to user directory for easy copying
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim

WORKDIR /app

# Create non-root user for security (CIS Docker Benchmark)
RUN useradd --create-home --shell /bin/bash appuser

# Copy only the installed packages from builder stage
# This excludes build-essential and other build tools
COPY --from=builder /root/.local /home/appuser/.local

# Update PATH to include user-installed packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code with proper ownership
COPY --chown=appuser:appuser . .

# Switch to non-root user before running
USER appuser

# Document the port for container orchestration
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the inference service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================
# Build Commands:
#   docker build -t ml-service:v1 .
#   docker build --target builder -t ml-service:builder .  # Debug builder stage
#
# Size Comparison:
#   Single-stage (python:3.11): ~1.2 GB
#   Multi-stage (python:3.11-slim): ~400 MB
#   Savings: ~65% reduction
# ============================================