# PySpark Docker Compose - Minimum Working Example

**Pattern:** Run a distributed PySpark cluster using Docker Compose with the official Apache Spark image.

---

## Prerequisites

- Docker Desktop (or Docker Engine + Docker Compose)
- ~2GB disk space for Spark images (apache/spark-py image)
- Linux, macOS, or Windows with WSL2

---

## Setup (3 Steps)

### 1. Clone and Navigate

```bash
git clone <repository-url>
cd module5/mwe/pyspark-docker
```

### 2. Start the Spark Cluster

```bash
docker compose up -d
```

This starts:
- 1 Spark master (port 8080 for Web UI, port 7077 for Spark protocol)
- 2 Spark workers (1 core, 1GB each)

### 3. Submit the Word Count Job

```bash
docker exec -it spark-master /opt/spark/bin/spark-submit /opt/spark/apps/word_count.py
```

---

## Expected Output

```
============================================================
PYSPARK WORD COUNT: Distributed Word Count Example
============================================================

[1/4] Connecting to Spark cluster...
  Connected to: spark://spark-master:7077
  App ID: app-20260304184523-0001

[2/4] Processing 5 text batches...

[3/4] Counting words across workers...

[4/4] Collecting results...

============================================================
RESULTS: Top 10 Most Frequent Words
============================================================
  the               6 ██████
  fox               2 ██
  dog               2 ██
  and               2 ██
  distributed       2 ██
  for               2 ██
  quick             1 █
  brown             1 █
  jumps             1 █
  over              1 █

Total unique words: 30
Total word occurrences: 40

Spark session stopped.
============================================================
```

---

## Web UI Access

| Service | URL | Description |
|---------|-----|-------------|
| Spark Master | http://localhost:8080 | Cluster overview, running applications |
| Worker 1 | http://localhost:8081 | Worker 1 details |
| Worker 2 | http://localhost:8082 | Worker 2 details |

---

## Key Concepts Demonstrated

| Concept | Where in Code |
|---------|---------------|
| Spark Cluster Setup | `docker-compose.yml` - master/worker services |
| Service Networking | `sparknet` network connects all containers |
| Volume Mounting | `./work:/opt/spark/apps` shares code with cluster |
| SparkSession Creation | `word_count.py` - connects to master URL |
| DataFrame API | `explode()`, `split()`, `groupBy()`, `agg()` |
| Distributed Collection | `word_counts.collect()` gathers results |

---

## Cluster Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Compose Network (sparknet)            │
│                                                                 │
│   ┌───────────────────────────────────────────────────────┐    │
│   │              spark-master                              │    │
│   │              Port 7077 (Spark protocol)               │    │
│   │              Port 8080 (Web UI)                       │    │
│   └───────────────────────────────────────────────────────┘    │
│                    ▲                    ▲                       │
│                    │                    │                       │
│        ┌───────────┘                    └───────────┐          │
│        │                                            │          │
│   ┌────┴────────────────┐          ┌───────────────┴────┐     │
│   │   spark-worker-1    │          │   spark-worker-2    │     │
│   │   1 core / 1GB      │          │   1 core / 1GB      │     │
│   │   Port 8081 (UI)    │          │   Port 8082 (UI)    │     │
│   └─────────────────────┘          └────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Volume mount: ./work:/opt/spark/apps
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                        Host Machine                              │
│                                                                  │
│   ./work/word_count.py  ─────►  /opt/spark/apps/word_count.py   │
│                                                                  │
│   $ docker exec -it spark-master /opt/spark/bin/spark-submit /opt/spark/apps/word_count.py
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Comparison with Ray MWE

This MWE uses the **same sample text and output format** as the Ray MWE (`module5/mwe/ray-map-reduce/`) to enable direct comparison:

| Aspect | Ray MWE | PySpark MWE |
|--------|---------|-------------|
| **Infrastructure** | Local process | Docker Compose cluster |
| **Setup** | `pip install ray` | `docker compose up` |
| **Parallelism Model** | `@ray.remote` tasks | DataFrame partitions |
| **Driver Location** | Same process | Inside master container |
| **Web UI** | Ray Dashboard | Spark Master UI (port 8080) |
| **Code Style** | Python decorators | DataFrame method chaining |
| **Image** | N/A (local Python) | apache/spark-py:latest |

### Code Comparison

**Ray (ray-map-reduce/main.py):**
```python
@ray.remote
def map_word_count(text_batch: str, batch_id: int) -> dict:
    words = text_batch.lower().split()
    return dict(Counter(words))

# Submit tasks in parallel
futures = [map_word_count.remote(batch, i) for i, batch in enumerate(batches)]
results = ray.get(futures)
```

**PySpark (work/word_count.py):**
```python
word_counts = df \
    .select(explode(split(lower(col("text")), " ")).alias("word")) \
    .groupBy("word") \
    .agg(count("*").alias("count")) \
    .orderBy(col("count").desc())

results = word_counts.collect()
```

---

## Files Included

| File | Purpose |
|------|---------|
| [`docker-compose.yml`](docker-compose.yml) | Spark cluster definition (1 master + 2 workers) |
| [`work/word_count.py`](work/word_count.py) | PySpark job using same data as Ray MWE |
| [`main.py`](main.py) | Local analyzer script (no Docker required) |
| [`requirements.txt`](requirements.txt) | Python dependencies for main.py |

---

## Project Structure

```
pyspark-docker/
├── docker-compose.yml       # Spark cluster definition
├── work/
│   └── word_count.py        # PySpark job script
├── main.py                  # Local analyzer (validates setup)
├── requirements.txt         # Dependencies for main.py
├── expected_output/
│   └── sample_output.txt
└── README.md
```

---

## Commands Reference

```bash
# Start the cluster (detached)
docker compose up -d

# Check cluster status
docker compose ps

# View Spark Master UI
open http://localhost:8080    # macOS
xdg-open http://localhost:8080  # Linux

# Submit the word count job
docker exec -it spark-master /opt/spark/bin/spark-submit /opt/spark/apps/word_count.py

# Follow cluster logs
docker compose logs -f spark-master

# Open shell in master container
docker exec -it spark-master bash

# Stop the cluster
docker compose down

# Stop and remove volumes
docker compose down -v
```

---

## Local Analyzer

Run the analyzer script without Docker to validate your setup:

```bash
# Install dependencies (optional, for YAML parsing)
pip install -r requirements.txt

# Run analyzer
python main.py
```

This validates the docker-compose.yml configuration and displays the cluster topology.

---

## Extension Challenges

### 1. Add More Workers

Edit `docker-compose.yml` to add `spark-worker-3`:

```yaml
spark-worker-3:
  image: apache/spark-py:latest
  container_name: spark-worker-3
  depends_on: [spark-master]
  environment:
    - SPARK_MODE=worker
    - SPARK_MASTER=spark://spark-master:7077
    - SPARK_WORKER_CORES=1
    - SPARK_WORKER_MEMORY=1G
    - SPARK_WORKER_WEBUI_PORT=8083
  ports:
    - "8083:8083"
  networks: [sparknet]
  volumes:
    - ./work:/opt/spark/apps
  command: /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker spark://spark-master:7077
```

### 2. Increase Resources

Modify worker environment variables:

```yaml
environment:
  - SPARK_WORKER_CORES=2
  - SPARK_WORKER_MEMORY=2G
```

### 3. Write Your Own Job

Create a new script in `./work/`:

```python
# ./work/my_analysis.py
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("MyAnalysis") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# Your Spark code here
df = spark.range(0, 1_000_000)
print(f"Count: {df.count()}")

spark.stop()
```

Submit it:

```bash
docker exec -it spark-master /opt/spark/bin/spark-submit /opt/spark/apps/my_analysis.py
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot connect to Docker daemon` | Start Docker Desktop or Docker service |
| `Port 8080 already in use` | Change port in docker-compose.yml: `"8090:8080"` |
| Worker not connecting | Check `SPARK_MASTER_URL` matches master container name |
| Job hangs on submit | Ensure workers are running: `docker compose ps` |
| Out of memory | Reduce `SPARK_WORKER_MEMORY` or data size |
| Container exits immediately | Check logs: `docker compose logs spark-master` |

---

## Resources

- [Apache Spark Docker Image](https://hub.docker.com/r/apache/spark-py)
- [Apache Spark DataFrame Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/index.html)
- [Docker Compose Documentation](https://docs.docker.com/compose/)