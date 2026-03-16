# Kafka Streaming Docker - Minimum Working Example

**Pattern:** Run a Kafka broker in Docker, produce transaction events, and consume them with sliding window aggregation, deduplication, and consumer lag tracking.

---

## Prerequisites

- Docker Desktop (or Docker Engine + Docker Compose)
- Python 3.9+ with `kafka-python` and `pyyaml`
- ~500MB disk space for Kafka image
- Linux, macOS, or Windows with WSL2

---

## Setup (4 Steps)

### 1. Clone and Navigate

```bash
git clone <repository-url>
cd module5/mwe/kafka-streaming-docker
```

### 2. Install Python Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start the Kafka Broker

```bash
docker compose up -d
```

This starts:
- 1 Kafka broker in **KRaft mode** (no Zookeeper required)
- 1 init container that creates the `transactions` topic with **3 partitions**

Wait for the init container to finish:

```bash
docker compose logs kafka-init
# Should see: ✅ Topic [transactions] created with 3 partitions
```

### 4. Run the Demo

```bash
# Terminal 1: Send events
python producer.py

# Terminal 2: Consume with windowing
python consumer.py
```

Or run the local analyzer (no Docker required):

```bash
python main.py
```

---

## Expected Output

### `main.py` (no Docker required)

```
============================================================
KAFKA STREAMING DOCKER - Analysis & Simulation
============================================================

📄 Analyzing: docker-compose.yml
   Found 2 service(s): kafka, kafka-init

🔍 Configuration Validation:
------------------------------------------------------------
   ✅ Kafka Broker: Broker service configured
   ✅ KRaft Mode: Running without Zookeeper (modern)
   ✅ Topic Init: Topic 'transactions' with 3 partitions
   ✅ Partitioning: 3 partitions enable parallel consumption
   ✅ Init Container: Separate init service for topic creation

   Score: 5/5 checks passed

🔄 Simulating Sliding Window (no Kafka required)
------------------------------------------------------------
  Event 1: user=u000, amount=$1.27
  Event 2: user=u001, amount=$7.51
  ...
  Event 11: evt-0005 → DUPLICATE (skipped)

  --- Final Window Stats ---
  u000: count=4, sum=$58.03, avg=$14.51
  u001: count=6, sum=$315.33, avg=$52.55
  ...

  Deduplication: 1 duplicate(s) skipped
  ⚠  In-memory only — not persisted across restarts
```

### `producer.py` (requires Docker)

```
============================================================
KAFKA PRODUCER: Transaction Event Generator
============================================================

[1/3] Connecting to Kafka broker...
      Connected to: localhost:9092

[2/3] Sending 100 events to 'transactions' topic (20.0 events/sec)...
      → Event 1: user_id=u042, amount=$23.50, category=food → partition 1
      → Event 2: user_id=u017, amount=$156.20, category=electronics → partition 0
      ...
      → Event 100: user_id=u091, amount=$12.34, category=food → partition 1

[3/3] All events sent. Flushing...
      100 events produced in 5.12s
      Partition distribution: {0: 31, 1: 35, 2: 34}
```

### `consumer.py` (requires Docker)

```
============================================================
KAFKA CONSUMER: Windowed Aggregation + Deduplication
============================================================

[1/4] Connecting to Kafka broker (group: feature-pipeline)...

[2/4] Consuming from 'transactions' topic...

--- Window Stats after 25 events (1.2s) ---
  u003: count=3, sum=$72.15, avg=$24.05
  u042: count=4, sum=$95.30, avg=$23.83
  ...

[3/4] Deduplication stats (in-memory, this run only):
      → 100 events received, 0 duplicates skipped
      ⚠  Note: IDs tracked in memory only; not persisted across restarts

[4/4] Consumer lag: 0 messages behind
```

---

## Key Concepts Demonstrated

| Concept | Where in Code |
|---------|---------------|
| Kafka broker (KRaft) | `docker-compose.yml` — single broker, no Zookeeper |
| Explicit topic creation | `docker-compose.yml` — `kafka-init` service with `--partitions 3` |
| Producer with JSON serialization | `producer.py` — `KafkaProducer` with `value_serializer` |
| Key-based partition routing | `producer.py` — `user_id` as message key |
| Consumer group | `consumer.py` — `group_id='feature-pipeline'` |
| At-least-once delivery | `consumer.py` — `enable_auto_commit=False` + manual commit |
| Sliding window aggregation | `consumer.py` — `SlidingWindowAggregator` class |
| Deduplication (single-run) | `consumer.py` — `DeduplicationTracker` with in-memory set |
| Consumer lag tracking | `consumer.py` — committed offset vs end offset comparison |
| Backpressure demo | `producer.py --burst` sends faster than consumer processes |

---

## Deduplication Limitations

The `consumer.py` deduplication uses an **in-memory `processed_ids` set**:

| Behavior | Status |
|----------|--------|
| Skips duplicate events within one consumer run | ✅ Works |
| Persists IDs across consumer restarts | ❌ Lost on exit |
| Bounds memory usage | ❌ Set grows without limit |
| True exactly-once semantics | ❌ Not guaranteed |

This is a **teaching simplification**. Production exactly-once requires:
- External state store (Redis, database) for processed IDs
- Kafka's transactional API (`isolation_level='read_committed'`)
- Idempotent writes to downstream systems

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                  Docker Compose Network (kafkanet)                │
│                                                                   │
│   ┌────────────────────────────────────────────────────────┐     │
│   │              kafka-broker (KRaft mode)                  │     │
│   │              Port 9092 (external)                       │     │
│   │                                                         │     │
│   │   Topic: transactions                                   │     │
│   │   Partitions: [P0 | P1 | P2]                           │     │
│   └────────────────────────────────────────────────────────┘     │
│                    ▲                    │                          │
│                    │                    ▼                          │
│          ┌─────────┴───┐      ┌────────┴────────┐                │
│          │ producer.py │      │  consumer.py    │                │
│          │ (events →)  │      │  (→ windowing)  │                │
│          └─────────────┘      └─────────────────┘                │
│                                                                   │
│   ┌────────────────────────────────────────────────────────┐     │
│   │   kafka-init (runs once at startup)                     │     │
│   │   Creates topic 'transactions' with 3 partitions        │     │
│   └────────────────────────────────────────────────────────┘     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              │ Host: localhost:9092
                              │
┌─────────────────────────────┴────────────────────────────────────┐
│                        Host Machine                               │
│                                                                   │
│   python producer.py   → sends events to Kafka                   │
│   python consumer.py   → consumes with windowing                 │
│   python main.py       → analyzes config (no Docker needed)      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Files Included

| File | Purpose | Requires Docker? |
|------|---------|:---:|
| [`docker-compose.yml`](docker-compose.yml) | Kafka broker + topic init | Yes (to run) |
| [`producer.py`](producer.py) | Send transaction events to Kafka | Yes |
| [`consumer.py`](consumer.py) | Consume with windowing + deduplication | Yes |
| [`main.py`](main.py) | Local analyzer + windowing simulator | **No** |
| [`requirements.txt`](requirements.txt) | Python dependencies | N/A |

---

## Project Structure

```
kafka-streaming-docker/
├── docker-compose.yml       # Kafka broker (KRaft) + init container
├── producer.py              # Event producer with partition tracking
├── consumer.py              # Consumer with windowing + deduplication
├── main.py                  # Local analyzer (no Docker required)
├── requirements.txt         # Dependencies (kafka-python, pyyaml, numpy)
├── expected_output/
│   └── sample_output.txt
└── README.md
```

---

## Commands Reference

```bash
# Start Kafka broker
docker compose up -d

# Check broker and topic are ready
docker compose ps
docker compose logs kafka-init

# Send 100 events (default)
python producer.py

# Send 500 events at 50/sec
python producer.py --n_events 500 --rate 50

# Backpressure demo: burst all events at once
python producer.py --burst --n_events 500

# Consume with default settings
python consumer.py

# Consume with 10-second timeout
python consumer.py --timeout 10

# Run local analyzer (no Docker needed)
python main.py

# View broker logs
docker compose logs -f kafka

# Stop Kafka
docker compose down

# Stop and remove volumes
docker compose down -v
```

---

## Comparison with PySpark MWE

| Aspect | PySpark MWE | Kafka MWE |
|--------|-------------|-----------|
| **Processing model** | Batch (bounded data) | Streaming (continuous) |
| **Docker services** | Master + 2 workers | Broker + init |
| **Data input** | Static text batches | Live event stream |
| **State management** | Stateless transforms | Stateful windowing |
| **Output** | One-time result | Continuous aggregation |
| **Delivery** | N/A (batch) | At-least-once + dedup |

---

## Extension Challenges

### 1. Add a Second Consumer Group

Run two consumers with different group IDs to see independent consumption:

```bash
python consumer.py --group_id analytics-team &
python consumer.py --group_id ml-pipeline &
python producer.py
```

### 2. Increase Partitions

Edit `docker-compose.yml` to create the topic with more partitions:

```yaml
command: |
  "kafka-topics.sh ... --partitions 6 ..."
```

### 3. Add Tumbling Windows

Modify `consumer.py` to implement non-overlapping tumbling windows:

```python
class TumblingWindowAggregator:
    def __init__(self, window_size_sec: int = 60):
        self.window_size_sec = window_size_sec
        self.current_window_start = None
        self.window_data = defaultdict(list)
```

### 4. Persistent Deduplication

Replace the in-memory set with a file-based or SQLite store:

```python
import sqlite3

class PersistentDeduplicator:
    def __init__(self, db_path="processed.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS seen (event_id TEXT PRIMARY KEY)")
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot connect to Docker daemon` | Start Docker Desktop or Docker service |
| `NoBrokersAvailable` | Ensure Kafka is running: `docker compose ps` |
| `Topic not found` | Wait for kafka-init: `docker compose logs kafka-init` |
| Consumer gets no messages | Run producer first; check `--group_id` hasn't consumed all |
| Port 9092 in use | Change port in docker-compose.yml: `"9093:9092"` |
| Consumer hangs | Increase `--timeout`; check broker is healthy |
| `ModuleNotFoundError: kafka` | Activate venv and install: `pip install kafka-python` |

---

## Resources

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [kafka-python Library](https://kafka-python.readthedocs.io/)
- [KRaft Mode Guide](https://kafka.apache.org/documentation/#kraft)
- [Kafka Docker Image](https://hub.docker.com/r/apache/kafka)
