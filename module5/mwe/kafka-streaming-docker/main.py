#!/usr/bin/env python3
"""
Kafka Streaming Docker Compose - Analyzer and Simulator.

This script parses the docker-compose.yml to explain Kafka concepts,
validate configuration, and simulate the windowing logic locally
without requiring Docker or a running Kafka broker.

Run with: python main.py
"""

import re
import time
from collections import defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def parse_docker_compose(compose_path: str) -> dict:
    """Parse docker-compose.yml and extract Kafka configuration.

    Args:
        compose_path: Path to docker-compose.yml.

    Returns:
        Dictionary with parsed configuration.
    """
    content = Path(compose_path).read_text()

    if yaml:
        return yaml.safe_load(content)
    else:
        return {"raw_content": content}


def extract_kafka_config(config: dict) -> dict:
    """Extract Kafka broker and topic configuration from compose config.

    Args:
        config: Parsed docker-compose configuration.

    Returns:
        Dictionary with broker and topic details.
    """
    result = {
        "broker": {"image": "unknown", "port": "9092", "mode": "unknown"},
        "topic": {"name": "unknown", "partitions": 0},
        "services": [],
    }

    if "raw_content" in config:
        content = config["raw_content"]
        # Extract from raw text
        image_match = re.search(r"image:\s*(apache/kafka\S*)", content)
        if image_match:
            result["broker"]["image"] = image_match.group(1)

        port_match = re.search(r'"(\d+):9092"', content)
        if port_match:
            result["broker"]["port"] = port_match.group(1)

        if "KAFKA_PROCESS_ROLES" in content:
            result["broker"]["mode"] = "KRaft (no Zookeeper)"

        partition_match = re.search(r"--partitions\s+(\d+)", content)
        if partition_match:
            result["topic"]["partitions"] = int(partition_match.group(1))

        topic_match = re.search(r"--topic\s+(\w+)", content)
        if topic_match:
            result["topic"]["name"] = topic_match.group(1)

        # Find service names (only under the 'services:' block)
        services_section = re.search(
            r"^services:\s*\n((?:\s{2}\S.*\n|\s{4,}.*\n|\s*\n)*)",
            content, re.MULTILINE
        )
        if services_section:
            svc_block = services_section.group(1)
            for svc_match in re.finditer(r"^\s{2}(\w[\w-]*):", svc_block, re.MULTILINE):
                result["services"].append(svc_match.group(1))
    else:
        services = config.get("services", {})
        for name, svc in services.items():
            result["services"].append(name)
            env = svc.get("environment", {})

            if isinstance(env, dict) and "KAFKA_PROCESS_ROLES" in env:
                result["broker"]["mode"] = "KRaft (no Zookeeper)"
                result["broker"]["image"] = svc.get("image", "unknown")
            elif isinstance(env, list):
                for item in env:
                    if "KAFKA_PROCESS_ROLES" in str(item):
                        result["broker"]["mode"] = "KRaft (no Zookeeper)"
                        result["broker"]["image"] = svc.get("image", "unknown")

            # Check init container for topic config
            cmd = svc.get("command", "")
            if isinstance(cmd, str):
                partition_match = re.search(r"--partitions\s+(\d+)", cmd)
                if partition_match:
                    result["topic"]["partitions"] = int(partition_match.group(1))
                topic_match = re.search(r"--topic\s+(\w+)", cmd)
                if topic_match:
                    result["topic"]["name"] = topic_match.group(1)

            ports = svc.get("ports", [])
            if ports:
                result["broker"]["port"] = str(ports[0]).split(":")[0]

    return result


def print_architecture_diagram(kafka_config: dict) -> None:
    """Print a visual diagram of the Kafka streaming architecture."""
    topic = kafka_config["topic"]["name"]
    n_parts = kafka_config["topic"]["partitions"]
    mode = kafka_config["broker"]["mode"]

    partitions_str = " | ".join([f"P{i}" for i in range(n_parts)])

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │                Docker Compose Network (kafkanet)            │
  │                                                             │
  │   ┌───────────────────────────────────────────────────┐    │
  │   │              kafka-broker ({mode})            │    │
  │   │              Port {kafka_config['broker']['port']} (external)                 │    │
  │   │                                                   │    │
  │   │   Topic: {topic:<20}                  │    │
  │   │   Partitions: [{partitions_str}]{'':>{30 - len(partitions_str)}}│    │
  │   └───────────────────────────────────────────────────┘    │
  │                    ▲                    │                    │
  │                    │                    ▼                    │
  │          ┌─────────┴───┐      ┌────────┴────────┐          │
  │          │ producer.py │      │  consumer.py    │          │
  │          │ (events →)  │      │  (→ windowing)  │          │
  │          └─────────────┘      └─────────────────┘          │
  │                                                             │
  │   ┌───────────────────────────────────────────────────┐    │
  │   │   kafka-init (runs once)                          │    │
  │   │   Creates topic '{topic}' with {n_parts} partitions      │    │
  │   └───────────────────────────────────────────────────┘    │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
""")


def validate_configuration(kafka_config: dict) -> list[tuple[str, str, str]]:
    """Validate Kafka configuration from docker-compose.

    Returns:
        List of (status, check_name, message) tuples.
    """
    results = []

    # Check 1: Has broker
    if "kafka" in kafka_config["services"] or "kafka-broker" in str(kafka_config):
        results.append(("✅", "Kafka Broker", "Broker service configured"))
    else:
        results.append(("❌", "Kafka Broker", "No broker service found"))

    # Check 2: KRaft mode
    if "KRaft" in kafka_config["broker"]["mode"]:
        results.append(("✅", "KRaft Mode", "Running without Zookeeper (modern)"))
    else:
        results.append(("⚠️", "Broker Mode", f"Mode: {kafka_config['broker']['mode']}"))

    # Check 3: Topic creation
    if kafka_config["topic"]["partitions"] > 0:
        results.append(("✅", "Topic Init",
                        f"Topic '{kafka_config['topic']['name']}' "
                        f"with {kafka_config['topic']['partitions']} partitions"))
    else:
        results.append(("❌", "Topic Init", "No explicit topic creation found"))

    # Check 4: Multiple partitions
    if kafka_config["topic"]["partitions"] >= 2:
        results.append(("✅", "Partitioning",
                        f"{kafka_config['topic']['partitions']} partitions enable parallel consumption"))
    else:
        results.append(("⚠️", "Partitioning", "Consider using multiple partitions"))

    # Check 5: Init container
    if "kafka-init" in kafka_config["services"]:
        results.append(("✅", "Init Container", "Separate init service for topic creation"))
    else:
        results.append(("⚠️", "Init Container", "Consider explicit topic creation"))

    return results


def simulate_windowing() -> None:
    """Simulate the sliding window logic locally without Kafka.

    Demonstrates the same windowing algorithm used in consumer.py
    with synthetic events.
    """
    print("\n🔄 Simulating Sliding Window (no Kafka required)")
    print("-" * 60)

    # Simulate events
    import random
    random.seed(42)

    user_events: dict[str, list[tuple[float, float]]] = defaultdict(list)
    window_size = 300  # 5 minutes
    base_time = time.time()

    users = [f"u{i:03d}" for i in range(5)]
    categories = ["food", "electronics", "clothing"]
    n_events = 20

    processed_ids: set[str] = set()
    duplicate_count = 0

    print(f"  Generating {n_events} synthetic events (window={window_size}s)...\n")

    for i in range(n_events):
        user_id = random.choice(users)
        amount = round(random.expovariate(1 / 50.0), 2)
        ts = base_time + random.uniform(0, 60)  # Within 1 minute
        event_id = f"evt-{i:04d}"

        # Simulate a duplicate at event 10
        if i == 10:
            event_id = "evt-0005"  # duplicate of event 5

        # Deduplication check
        if event_id in processed_ids:
            duplicate_count += 1
            print(f"  Event {i + 1}: {event_id} → DUPLICATE (skipped)")
            continue
        processed_ids.add(event_id)

        # Add to window
        user_events[user_id].append((ts, amount))

        # Cleanup old events
        cutoff = ts - window_size
        user_events[user_id] = [(t, a) for t, a in user_events[user_id] if t >= cutoff]

        if i < 5 or i == n_events - 1:
            print(f"  Event {i + 1}: user={user_id}, amount=${amount:.2f}")

        if i == 5:
            print(f"  ... ({n_events - 6} more events) ...")

    # Print final window stats
    print(f"\n  --- Final Window Stats ---")
    for uid in sorted(user_events.keys()):
        events = user_events[uid]
        if events:
            total = sum(a for _, a in events)
            print(f"  {uid}: count={len(events)}, "
                  f"sum=${total:.2f}, avg=${total / len(events):.2f}")

    print(f"\n  Deduplication: {duplicate_count} duplicate(s) skipped")
    print(f"  ⚠  In-memory only — not persisted across restarts")


def main():
    compose_path = Path(__file__).parent / "docker-compose.yml"

    print("=" * 60)
    print("KAFKA STREAMING DOCKER - Analysis & Simulation")
    print("=" * 60)

    # Check if docker-compose.yml exists
    if not compose_path.exists():
        print(f"\n❌ Error: docker-compose.yml not found at {compose_path}")
        return

    print(f"\n📄 Analyzing: {compose_path.name}")

    # Parse and extract config
    config = parse_docker_compose(compose_path)
    kafka_config = extract_kafka_config(config)
    print(f"   Found {len(kafka_config['services'])} service(s): "
          f"{', '.join(kafka_config['services'])}")

    # Display architecture
    print("\n🏗️  Kafka Architecture:")
    print("=" * 60)
    print_architecture_diagram(kafka_config)

    # Validate
    print("🔍 Configuration Validation:")
    print("-" * 60)
    validations = validate_configuration(kafka_config)
    for status, check, message in validations:
        print(f"   {status} {check}: {message}")

    passed = sum(1 for v in validations if v[0] == "✅")
    total = len(validations)
    print(f"\n   Score: {passed}/{total} checks passed")

    # Simulate windowing locally
    simulate_windowing()

    # Kafka concepts
    print("\n" + "=" * 60)
    print("📚 Kafka Streaming Concepts:")
    print("=" * 60)
    print("""
   ┌───────────────────┬────────────────────────────────────────────┐
   │ Concept           │ Implementation in This MWE                │
   ├───────────────────┼────────────────────────────────────────────┤
   │ Producer          │ producer.py — KafkaProducer + JSON         │
   │ Consumer          │ consumer.py — KafkaConsumer + group_id     │
   │ Partitions        │ docker-compose.yml — 3 partitions          │
   │ Key-based routing │ producer.py — user_id as message key       │
   │ Sliding window    │ consumer.py — SlidingWindowAggregator      │
   │ At-least-once     │ consumer.py — manual commit after process  │
   │ Deduplication     │ consumer.py — in-memory processed_ids set  │
   │ Consumer lag      │ consumer.py — offset vs end-offset check   │
   └───────────────────┴────────────────────────────────────────────┘
""")

    # Commands
    print("🛠️  Commands:")
    print("-" * 60)
    print("   # Start Kafka broker")
    print("   docker compose up -d")
    print()
    print("   # Verify topic created (wait for kafka-init to finish)")
    print("   docker compose logs kafka-init")
    print()
    print("   # Send events")
    print("   python producer.py")
    print()
    print("   # Consume with windowing")
    print("   python consumer.py")
    print()
    print("   # Backpressure demo: burst mode")
    print("   python producer.py --burst --n_events 500")
    print("   python consumer.py --timeout 10")
    print()
    print("   # Stop Kafka")
    print("   docker compose down")
    print()


if __name__ == "__main__":
    main()
