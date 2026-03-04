#!/usr/bin/env python3
"""
PySpark Docker Compose Analyzer and Demonstrator.

This script parses the docker-compose.yml to explain Spark cluster concepts,
validate configuration, and display useful commands without requiring Docker.

Run with: python main.py
"""

import re
from pathlib import Path
from dataclasses import dataclass

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class SparkService:
    """Represents a Spark service in docker-compose."""
    name: str
    mode: str  # master or worker
    image: str
    cores: int
    memory: str
    ui_port: int
    depends_on: list


def parse_docker_compose(compose_path: str) -> dict:
    """
    Parse docker-compose.yml and extract Spark cluster configuration.
    
    Args:
        compose_path: Path to docker-compose.yml
        
    Returns:
        Dictionary with cluster configuration
    """
    content = Path(compose_path).read_text()
    
    if yaml:
        config = yaml.safe_load(content)
        return config
    else:
        # Basic parsing without PyYAML
        return {"raw_content": content}


def extract_spark_services(config: dict) -> list[SparkService]:
    """
    Extract Spark service definitions from docker-compose config.
    
    Args:
        config: Parsed docker-compose configuration
        
    Returns:
        List of SparkService objects
    """
    services = []
    
    if "raw_content" in config:
        # Fallback regex parsing if PyYAML not available
        content = config["raw_content"]
        
        # Find services section
        service_blocks = re.findall(
            r'(\w+-(?:master|worker-\d+)):\s*\n((?:[ ]+.*\n)*)', 
            content
        )
        
        for name, block in service_blocks:
            mode = "master" if "master" in name else "worker"
            
            # Extract values using regex
            cores_match = re.search(r'SPARK_WORKER_CORES=(\d+)', block)
            memory_match = re.search(r'SPARK_WORKER_MEMORY=(\w+)', block)
            port_match = re.search(r'"(\d+):(\d+)"', block)
            image_match = re.search(r'image:\s*(\S+)', block)
            
            services.append(SparkService(
                name=name,
                mode=mode,
                image=image_match.group(1) if image_match else "bitnami/spark:3.5.0",
                cores=int(cores_match.group(1)) if cores_match else (0 if mode == "master" else 1),
                memory=memory_match.group(1) if memory_match else ("N/A" if mode == "master" else "1G"),
                ui_port=int(port_match.group(1)) if port_match else 8080,
                depends_on=["spark-master"] if mode == "worker" else []
            ))
    else:
        # Parse from YAML structure
        for name, svc in config.get("services", {}).items():
            env = svc.get("environment", [])
            env_dict = {}
            for item in env:
                if "=" in item:
                    k, v = item.split("=", 1)
                    env_dict[k] = v
            
            mode = env_dict.get("SPARK_MODE", "unknown")
            ports = svc.get("ports", [])
            ui_port = int(ports[0].split(":")[0]) if ports else 8080
            
            services.append(SparkService(
                name=name,
                mode=mode,
                image=svc.get("image", "unknown"),
                cores=int(env_dict.get("SPARK_WORKER_CORES", 0)),
                memory=env_dict.get("SPARK_WORKER_MEMORY", "N/A"),
                ui_port=ui_port,
                depends_on=svc.get("depends_on", [])
            ))
    
    return services


def print_cluster_diagram(services: list[SparkService]) -> None:
    """Print a visual diagram of the Spark cluster topology."""
    print("\n🔄 Spark Cluster Topology:")
    print("=" * 60)
    
    master = next((s for s in services if s.mode == "master"), None)
    workers = [s for s in services if s.mode == "worker"]
    
    if master:
        print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │                  Docker Compose Network                 │
  │                                                         │
  │   ┌───────────────────────────────────────────────┐    │
  │   │              {master.name:<22}          │    │
  │   │              Port 7077 (Spark protocol)        │    │
  │   │              Port {master.ui_port} (Web UI)              │    │
  │   └───────────────────────────────────────────────┘    │
  │                    ▲                    ▲               │""")
        
        if len(workers) >= 2:
            print(f"""  │                    │                    │               │
  │        ┌──────────┘                    └──────────┐    │
  │        │                                          │    │
  │   ┌────┴────────────────┐    ┌────────────────────┴─┐  │
  │   │   {workers[0].name:<15}  │    │   {workers[1].name:<15}   │  │
  │   │   {workers[0].cores} core / {workers[0].memory:<8}   │    │   {workers[1].cores} core / {workers[1].memory:<8}    │  │
  │   │   Port {workers[0].ui_port} (UI)      │    │   Port {workers[1].ui_port} (UI)       │  │
  │   └─────────────────────┘    └──────────────────────┘  │""")
        elif len(workers) == 1:
            print(f"""  │                    │                                  │
  │        ┌──────────┘                                    │
  │        │                                               │
  │   ┌────┴────────────────┐                             │
  │   │   {workers[0].name:<15}  │                             │
  │   │   {workers[0].cores} core / {workers[0].memory:<8}   │                             │
  │   │   Port {workers[0].ui_port} (UI)      │                             │
  │   └─────────────────────┘                             │""")
        
        print("  │                                                         │")
        print("  └─────────────────────────────────────────────────────────┘")
        print("                              │")
        print("                              │ Volume mount: ./work:/opt/work")
        print("                              │")
        print("  ┌─────────────────────────────────────────────────────────┐")
        print("  │                      Host Machine                        │")
        print("  │                                                          │")
        print("  │   ./work/word_count.py  ────►  /opt/work/word_count.py  │")
        print("  │                                                          │")
        print("  └──────────────────────────────────────────────────────────┘")
    
    print()


def validate_configuration(services: list[SparkService]) -> list[tuple[str, str, str]]:
    """
    Validate Spark cluster configuration.
    
    Returns:
        List of (status, check_name, message) tuples
    """
    results = []
    
    # Check 1: Has master
    has_master = any(s.mode == "master" for s in services)
    if has_master:
        results.append(("✅", "Spark Master", "Master node configured"))
    else:
        results.append(("❌", "Spark Master", "No master node found"))
    
    # Check 2: Has workers
    workers = [s for s in services if s.mode == "worker"]
    if len(workers) >= 1:
        results.append(("✅", "Spark Workers", f"{len(workers)} worker(s) configured"))
    else:
        results.append(("❌", "Spark Workers", "No workers found"))
    
    # Check 3: Workers connect to master
    master_url_ok = all(s.depends_on for s in workers)
    if master_url_ok:
        results.append(("✅", "Worker Dependencies", "Workers depend on master"))
    else:
        results.append(("⚠️", "Worker Dependencies", "Workers should depend on master"))
    
    # Check 4: Resource allocation
    total_cores = sum(s.cores for s in workers)
    if total_cores >= 1:
        results.append(("✅", "CPU Resources", f"Total worker cores: {total_cores}"))
    else:
        results.append(("⚠️", "CPU Resources", "Consider allocating CPU cores to workers"))
    
    # Check 5: Memory allocation
    has_memory = all(s.memory != "N/A" for s in workers)
    if has_memory:
        results.append(("✅", "Memory Resources", "Memory allocated to workers"))
    else:
        results.append(("⚠️", "Memory Resources", "Consider setting SPARK_WORKER_MEMORY"))
    
    # Check 6: Unique UI ports
    ui_ports = [s.ui_port for s in services]
    if len(ui_ports) == len(set(ui_ports)):
        results.append(("✅", "UI Ports", "Each service has unique UI port"))
    else:
        results.append(("❌", "UI Ports", "UI ports must be unique"))
    
    return results


def print_comparison_with_ray() -> None:
    """Print comparison between PySpark and Ray approaches."""
    print("\n📊 PySpark vs Ray Comparison:")
    print("-" * 60)
    print("""
   ┌─────────────────────┬─────────────────────┬──────────────────────┐
   │ Aspect              │ Ray MWE             │ PySpark MWE          │
   ├─────────────────────┼─────────────────────┼──────────────────────┤
   │ Infrastructure      │ Local process       │ Docker Compose       │
   │ Setup               │ pip install ray     │ docker compose up    │
   │ Parallelism model   │ @ray.remote tasks   │ DataFrame/RDD        │
   │ Driver location     │ Same process        │ Inside container     │
   │ Web UI              │ Ray Dashboard       │ Spark Master UI      │
   │ Same sample data    │ ✓ 5 text batches    │ ✓ 5 text batches     │
   │ Same output format  │ ✓ Top 10 words      │ ✓ Top 10 words       │
   └─────────────────────┴─────────────────────┴──────────────────────┘
""")


def main():
    """Run the docker-compose analysis demonstration."""
    compose_path = Path(__file__).parent / "docker-compose.yml"
    work_dir = Path(__file__).parent / "work"
    
    print("=" * 60)
    print("PYSPARK DOCKER COMPOSE - Analysis & Demonstration")
    print("=" * 60)
    
    # Check if docker-compose.yml exists
    if not compose_path.exists():
        print(f"\n❌ Error: docker-compose.yml not found at {compose_path}")
        return
    
    print(f"\n📄 Analyzing: {compose_path.name}")
    
    # Parse docker-compose
    config = parse_docker_compose(compose_path)
    services = extract_spark_services(config)
    print(f"   Found {len(services)} service(s)")
    
    # Display cluster diagram
    print_cluster_diagram(services)
    
    # Validate configuration
    print("🔍 Configuration Validation:")
    print("-" * 60)
    
    validations = validate_configuration(services)
    for status, check, message in validations:
        print(f"   {status} {check}: {message}")
    
    passed = sum(1 for v in validations if v[0] == "✅")
    total = len(validations)
    print(f"\n   Score: {passed}/{total} checks passed")
    
    # Check for work directory and scripts
    print("\n📁 Work Directory Contents:")
    print("-" * 60)
    if work_dir.exists():
        for f in work_dir.iterdir():
            print(f"   • {f.name}")
    else:
        print("   ⚠️  ./work directory not found")
    
    # Print comparison with Ray
    print_comparison_with_ray()
    
    # Cluster commands
    print("🛠️  Cluster Commands:")
    print("-" * 60)
    print("   # Start the Spark cluster")
    print("   docker compose up -d")
    print()
    print("   # Check cluster status")
    print("   docker compose ps")
    print()
    print("   # View Spark Master UI")
    print("   open http://localhost:8080")
    print()
    print("   # Submit the word count job")
    print("   docker exec -it spark-master spark-submit /opt/work/word_count.py")
    print()
    print("   # View cluster logs")
    print("   docker compose logs -f spark-master")
    print()
    print("   # Stop the cluster")
    print("   docker compose down")
    
    # Key takeaways
    print("\n" + "=" * 60)
    print("📚 Key Takeaways:")
    print("=" * 60)
    print("   1. Bitnami Spark image provides master/worker modes via env vars")
    print("   2. Workers connect to master using SPARK_MASTER_URL")
    print("   3. Shared volume ./work:/opt/work enables code sharing")
    print("   4. spark-submit runs jobs inside the cluster")
    print("   5. Same word count example as Ray MWE for comparison")
    print()


if __name__ == "__main__":
    main()