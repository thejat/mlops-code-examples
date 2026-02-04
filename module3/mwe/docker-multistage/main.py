"""
Docker Multistage Build Analyzer and Demonstrator.

This script parses the Dockerfile to explain multistage build concepts,
validate best practices, and demonstrate the benefits without requiring Docker.
"""

import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DockerStage:
    """Represents a stage in a multistage Dockerfile."""
    name: str
    base_image: str
    line_number: int
    instructions: list[str]


def parse_dockerfile(dockerfile_path: str) -> list[DockerStage]:
    """
    Parse a Dockerfile and extract stages.
    
    Args:
        dockerfile_path: Path to the Dockerfile
        
    Returns:
        List of DockerStage objects representing each build stage
    """
    content = Path(dockerfile_path).read_text()
    lines = content.split('\n')
    
    stages = []
    current_stage = None
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Skip comments and empty lines for instruction tracking
        if stripped.startswith('#') or not stripped:
            continue
            
        # Detect FROM instruction (start of new stage)
        from_match = re.match(r'^FROM\s+(\S+)(?:\s+AS\s+(\S+))?', stripped, re.IGNORECASE)
        if from_match:
            if current_stage:
                stages.append(current_stage)
            
            base_image = from_match.group(1)
            stage_name = from_match.group(2) or f"stage_{len(stages)}"
            current_stage = DockerStage(
                name=stage_name,
                base_image=base_image,
                line_number=i,
                instructions=[]
            )
        elif current_stage:
            # Extract instruction type
            instruction_match = re.match(r'^(\w+)\s+', stripped)
            if instruction_match:
                current_stage.instructions.append(instruction_match.group(1))
    
    if current_stage:
        stages.append(current_stage)
    
    return stages


def validate_best_practices(dockerfile_path: str) -> list[tuple[str, str, str]]:
    """
    Validate Dockerfile against best practices.
    
    Args:
        dockerfile_path: Path to the Dockerfile
        
    Returns:
        List of (status, check_name, message) tuples
    """
    content = Path(dockerfile_path).read_text()
    results = []
    
    # Check 1: Multistage build
    if ' AS ' in content.upper():
        results.append(("✅", "Multistage Build", "Uses named stages for smaller final image"))
    else:
        results.append(("⚠️", "Multistage Build", "Consider using multistage builds"))
    
    # Check 2: Slim base image
    if '-slim' in content or '-alpine' in content:
        results.append(("✅", "Slim Base Image", "Uses slim/alpine variant for smaller size"))
    else:
        results.append(("⚠️", "Slim Base Image", "Consider python:3.x-slim for smaller images"))
    
    # Check 3: Non-root user
    if re.search(r'\bUSER\s+(?!root)', content):
        results.append(("✅", "Non-root User", "Runs as non-root user (security best practice)"))
    else:
        results.append(("❌", "Non-root User", "Add USER instruction for security"))
    
    # Check 4: Layer caching order
    req_pos = content.find('requirements.txt')
    copy_app_pos = content.find('COPY --chown') if 'COPY --chown' in content else content.rfind('COPY')
    if req_pos != -1 and copy_app_pos != -1 and req_pos < copy_app_pos:
        results.append(("✅", "Layer Caching", "Requirements copied before app code"))
    else:
        results.append(("⚠️", "Layer Caching", "Copy requirements.txt before application code"))
    
    # Check 5: HEALTHCHECK
    if 'HEALTHCHECK' in content:
        results.append(("✅", "Health Check", "Includes HEALTHCHECK for orchestration"))
    else:
        results.append(("⚠️", "Health Check", "Add HEALTHCHECK for container orchestration"))
    
    # Check 6: --no-cache-dir
    if '--no-cache-dir' in content:
        results.append(("✅", "Pip Cache", "Uses --no-cache-dir to reduce image size"))
    else:
        results.append(("⚠️", "Pip Cache", "Add --no-cache-dir to pip install"))
    
    # Check 7: COPY --from for multistage
    if 'COPY --from=' in content:
        results.append(("✅", "Stage Copying", "Uses COPY --from to transfer artifacts"))
    else:
        results.append(("⚠️", "Stage Copying", "No COPY --from found"))
    
    return results


def estimate_size_savings() -> dict:
    """
    Estimate image size savings from multistage builds.
    
    Returns:
        Dictionary with size estimates and savings
    """
    return {
        "single_stage_full": {"image": "python:3.11", "size_mb": 1000},
        "single_stage_slim": {"image": "python:3.11-slim", "size_mb": 500},
        "multistage_slim": {"image": "python:3.11-slim (multistage)", "size_mb": 200},
        "excluded_packages": [
            "build-essential (~150 MB)",
            "gcc, g++ (~100 MB)",
            "pip cache (~50 MB)",
            "development headers (~50 MB)"
        ]
    }


def print_stage_diagram(stages: list[DockerStage]) -> None:
    """Print a visual diagram of the build stages."""
    print("\n🔄 Build Stage Flow:")
    print("=" * 60)
    
    for i, stage in enumerate(stages):
        is_last = i == len(stages) - 1
        
        print(f"\n  ┌{'─' * 50}┐")
        print(f"  │ Stage {i + 1}: {stage.name:<42} │")
        print(f"  │ Base: {stage.base_image:<44} │")
        print(f"  │ Instructions: {', '.join(set(stage.instructions)):<35} │")
        print(f"  └{'─' * 50}┘")
        
        if not is_last:
            print("           │")
            print("           │  COPY --from=" + stage.name)
            print("           ▼")
    
    print()


def main():
    """Run the Dockerfile analysis demonstration."""
    dockerfile_path = Path(__file__).parent / "Dockerfile"
    
    print("=" * 60)
    print("DOCKER MULTISTAGE BUILD - Analysis & Demonstration")
    print("=" * 60)
    
    # Check if Dockerfile exists
    if not dockerfile_path.exists():
        print(f"\n❌ Error: Dockerfile not found at {dockerfile_path}")
        return
    
    print(f"\n📄 Analyzing: {dockerfile_path.name}")
    
    # Parse stages
    stages = parse_dockerfile(dockerfile_path)
    print(f"   Found {len(stages)} build stage(s)")
    
    # Display stage diagram
    print_stage_diagram(stages)
    
    # Validate best practices
    print("🔍 Best Practices Validation:")
    print("-" * 60)
    
    validations = validate_best_practices(dockerfile_path)
    for status, check, message in validations:
        print(f"   {status} {check}: {message}")
    
    passed = sum(1 for v in validations if v[0] == "✅")
    total = len(validations)
    print(f"\n   Score: {passed}/{total} checks passed")
    
    # Size estimation
    print("\n📊 Estimated Size Comparison:")
    print("-" * 60)
    
    sizes = estimate_size_savings()
    print(f"   python:3.11 (single-stage)     : ~{sizes['single_stage_full']['size_mb']} MB")
    print(f"   python:3.11-slim (single-stage): ~{sizes['single_stage_slim']['size_mb']} MB")
    print(f"   python:3.11-slim (multistage)  : ~{sizes['multistage_slim']['size_mb']} MB")
    print(f"\n   💾 Savings: ~{sizes['single_stage_full']['size_mb'] - sizes['multistage_slim']['size_mb']} MB ({((sizes['single_stage_full']['size_mb'] - sizes['multistage_slim']['size_mb']) / sizes['single_stage_full']['size_mb'] * 100):.0f}% reduction)")
    
    print("\n   Excluded from final image:")
    for pkg in sizes['excluded_packages']:
        print(f"   • {pkg}")
    
    # Build commands
    print("\n🛠️  Build Commands:")
    print("-" * 60)
    print("   # Build the full image")
    print("   docker build -t ml-service:v1 .")
    print()
    print("   # Build only the builder stage (for debugging)")
    print("   docker build --target builder -t ml-service:builder .")
    print()
    print("   # Run the container")
    print("   docker run -p 8000:8000 ml-service:v1")
    print()
    print("   # Test the health endpoint")
    print("   curl http://localhost:8000/health")
    
    # Key takeaways
    print("\n" + "=" * 60)
    print("📚 Key Takeaways:")
    print("=" * 60)
    print("   1. Multistage builds separate build-time from runtime dependencies")
    print("   2. COPY --from=builder transfers only needed artifacts")
    print("   3. Non-root USER improves container security")
    print("   4. Copy requirements.txt first for better layer caching")
    print("   5. HEALTHCHECK enables container orchestration integration")
    print()


if __name__ == "__main__":
    main()