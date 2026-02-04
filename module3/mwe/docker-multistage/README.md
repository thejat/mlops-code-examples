# Docker Multistage Build MWE

**Pattern:** Use multistage Docker builds to create smaller, more secure container images by separating build-time dependencies from runtime.

## Prerequisites

- Python 3.9+
- Docker (optional - the analyzer works without Docker)
- Linux, macOS, or Windows

## Quick Start (3 Steps)

```bash
# 1. Clone and navigate
git clone <repo-url>
cd module3/mwe/docker-multistage

# 2. Run the analyzer (no dependencies needed)
python main.py

# 3. (Optional) Build and run the container
docker build -t ml-service:v1 .
docker run -p 8000:8000 ml-service:v1
```

## Expected Output

```
============================================================
DOCKER MULTISTAGE BUILD - Analysis & Demonstration
============================================================

📄 Analyzing: Dockerfile
   Found 2 build stage(s)

🔄 Build Stage Flow:
============================================================

  ┌──────────────────────────────────────────────────────┐
  │ Stage 1: builder                                      │
  │ Base: python:3.11-slim                                │
  │ Instructions: WORKDIR, RUN, COPY                      │
  └──────────────────────────────────────────────────────┘
           │
           │  COPY --from=builder
           ▼
  ┌──────────────────────────────────────────────────────┐
  │ Stage 2: runtime                                      │
  │ Base: python:3.11-slim                                │
  │ Instructions: WORKDIR, RUN, COPY, ENV, USER, EXPOSE   │
  └──────────────────────────────────────────────────────┘

🔍 Best Practices Validation:
------------------------------------------------------------
   ✅ Multistage Build: Uses named stages for smaller final image
   ✅ Slim Base Image: Uses slim/alpine variant for smaller size
   ✅ Non-root User: Runs as non-root user (security best practice)
   ✅ Layer Caching: Requirements copied before app code
   ✅ Health Check: Includes HEALTHCHECK for orchestration
   ✅ Pip Cache: Uses --no-cache-dir to reduce image size
   ✅ Stage Copying: Uses COPY --from to transfer artifacts

   Score: 7/7 checks passed

📊 Estimated Size Comparison:
------------------------------------------------------------
   python:3.11 (single-stage)     : ~1000 MB
   python:3.11-slim (single-stage): ~500 MB
   python:3.11-slim (multistage)  : ~200 MB

   💾 Savings: ~800 MB (80% reduction)

   Excluded from final image:
   • build-essential (~150 MB)
   • gcc, g++ (~100 MB)
   • pip cache (~50 MB)
   • development headers (~50 MB)

🛠️  Build Commands:
------------------------------------------------------------
   # Build the full image
   docker build -t ml-service:v1 .

   # Build only the builder stage (for debugging)
   docker build --target builder -t ml-service:builder .

   # Run the container
   docker run -p 8000:8000 ml-service:v1

   # Test the health endpoint
   curl http://localhost:8000/health

============================================================
📚 Key Takeaways:
============================================================
   1. Multistage builds separate build-time from runtime dependencies
   2. COPY --from=builder transfers only needed artifacts
   3. Non-root USER improves container security
   4. Copy requirements.txt first for better layer caching
   5. HEALTHCHECK enables container orchestration integration
```

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Multistage builds | `FROM python:3.11-slim AS builder` + `FROM python:3.11-slim AS runtime` |
| Layer caching | `COPY requirements.txt .` before `COPY . .` |
| Non-root user | `RUN useradd appuser` + `USER appuser` |
| Artifact transfer | `COPY --from=builder /root/.local /home/appuser/.local` |
| Health checks | `HEALTHCHECK --interval=30s` for orchestration |
| Slim base images | `python:3.11-slim` instead of `python:3.11` |

## Files Included

| File | Purpose |
|------|---------|
| [`Dockerfile`](Dockerfile) | Multistage Dockerfile demonstrating best practices |
| [`main.py`](main.py) | Dockerfile analyzer and validator (no Docker required) |
| [`app.py`](app.py) | Sample FastAPI application to containerize |
| [`requirements.txt`](requirements.txt) | Python dependencies for the container |

## Project Structure

```
docker-multistage/
├── Dockerfile           # Multistage build definition
├── main.py              # Analyzer script
├── app.py               # FastAPI application
├── requirements.txt     # Dependencies
├── expected_output/
│   └── sample_output.txt
└── README.md
```

## Understanding Multistage Builds

### Why Multistage?

```
┌─────────────────────────────────────────────────────────┐
│                    SINGLE-STAGE BUILD                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  python:3.11 (900 MB)                           │   │
│  │  + build-essential (150 MB)                     │   │
│  │  + pip packages (50 MB)                         │   │
│  │  + your app (1 MB)                              │   │
│  │  ─────────────────────────────────              │   │
│  │  = ~1.1 GB final image                          │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   MULTISTAGE BUILD                      │
│                                                         │
│  Stage 1 (builder):          Stage 2 (runtime):        │
│  ┌───────────────────┐       ┌───────────────────┐     │
│  │ python:3.11-slim  │       │ python:3.11-slim  │     │
│  │ + build-essential │  ───► │ + pip packages    │     │
│  │ + pip packages    │ COPY  │ + your app        │     │
│  └───────────────────┘       └───────────────────┘     │
│       (discarded)             = ~200 MB final image    │
└─────────────────────────────────────────────────────────┘
```

### Security Benefits

1. **Non-root user**: The container runs as `appuser`, not `root`
2. **Minimal attack surface**: No compilers or build tools in production
3. **Read-only filesystem**: Can be run with `--read-only` flag

## Running the Container

```bash
# Build the image
docker build -t ml-service:v1 .

# Run the container
docker run -p 8000:8000 ml-service:v1

# Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```

## Extension Challenge

🔧 **Add build arguments and labels:**

1. Modify the Dockerfile to accept build arguments:
   ```dockerfile
   ARG VERSION=dev
   LABEL version=$VERSION
   LABEL maintainer="your-email@example.com"
   ```

2. Build with version:
   ```bash
   docker build --build-arg VERSION=1.0.0 -t ml-service:1.0.0 .
   ```

3. Add a `/version` endpoint in `app.py` that reads from environment variable

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot connect to Docker daemon` | Start Docker Desktop or Docker service |
| `COPY failed: file not found` | Run docker build from the directory containing Dockerfile |
| `Port already in use` | Change port: `docker run -p 8001:8000 ml-service:v1` |
| Health check fails | Wait for `--start-period` (5s) before checking |

## Related Materials

- Docker Build Best Practices
- Container Security Fundamentals
- Kubernetes Pod Security Standards