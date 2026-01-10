"""
Lab: Module 3 - Docker Build Optimization
Time: ~30 minutes
Prerequisites:
  - Read: src/module3/instructional-materials/dockerfile-fundamentals.md
  - Read: src/module3/instructional-materials/docker-layer-caching.md
  - Read: src/module3/instructional-materials/multi-stage-builds.md

Learning Objectives:
- LO1: Analyze Dockerfile instruction order for layer caching optimization
- LO2: Identify common Dockerfile anti-patterns that break cache efficiency
- LO3: Implement multi-stage build patterns for smaller production images
- LO4: Calculate estimated image size reduction from build optimizations

Milestone Relevance: M2 - Containerization & CI/CD Pipeline
"""

# === SETUP (provided) ===

from dataclasses import dataclass
from typing import Optional


@dataclass
class DockerInstruction:
    """Represents a single Dockerfile instruction."""
    command: str          # e.g., "FROM", "RUN", "COPY"
    arguments: str        # e.g., "python:3.11-slim", "pip install -r requirements.txt"
    line_number: int
    creates_layer: bool   # Whether this instruction creates a cached layer
    estimated_size_mb: float = 0.0  # Estimated layer size


@dataclass
class CacheAnalysis:
    """Results of analyzing layer cache efficiency."""
    total_layers: int
    cacheable_layers: int
    invalidation_point: Optional[int]  # Line number where cache breaks on code change
    efficiency_score: float            # 0.0 to 1.0


# Sample Dockerfiles for analysis
DOCKERFILE_BAD = """FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
"""

DOCKERFILE_GOOD = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
"""

DOCKERFILE_MULTISTAGE = """FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
RUN useradd --create-home appuser
USER appuser
COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser app/ .
ENV PATH=/home/appuser/.local/bin:$PATH
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# Estimated sizes for common operations (in MB)
SIZE_ESTIMATES = {
    "python:3.11": 1000,      # Full Python image
    "python:3.11-slim": 150,  # Slim Python image
    "python:3.11-alpine": 50, # Alpine Python image
    "pip_install_ml": 500,    # Typical ML dependencies (numpy, pandas, sklearn)
    "pip_install_api": 50,    # API dependencies (fastapi, uvicorn)
    "apt_build_essential": 200,
    "app_code": 5,            # Application code
}

# Instructions that create layers vs metadata-only
LAYER_CREATING_COMMANDS = {"FROM", "RUN", "COPY", "ADD"}
METADATA_COMMANDS = {"WORKDIR", "ENV", "EXPOSE", "CMD", "ENTRYPOINT", "USER", "ARG", "LABEL"}


def parse_dockerfile(dockerfile_content: str) -> list[DockerInstruction]:
    """
    Parse a Dockerfile string into a list of DockerInstruction objects.
    
    Args:
        dockerfile_content: Raw Dockerfile as a string
        
    Returns:
        List of DockerInstruction objects
    """
    instructions = []
    lines = dockerfile_content.strip().split('\n')
    
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Handle multi-line commands (ending with \)
        while line.endswith('\\') and i < len(lines):
            line = line[:-1] + lines[i].strip()
            i += 1
        
        # Split into command and arguments
        parts = line.split(None, 1)
        if len(parts) >= 1:
            command = parts[0].upper()
            arguments = parts[1] if len(parts) > 1 else ""
            
            creates_layer = command in LAYER_CREATING_COMMANDS
            
            # Estimate size based on command
            size = estimate_layer_size(command, arguments)
            
            instructions.append(DockerInstruction(
                command=command,
                arguments=arguments,
                line_number=i,
                creates_layer=creates_layer,
                estimated_size_mb=size
            ))
    
    return instructions


def estimate_layer_size(command: str, arguments: str) -> float:
    """Estimate the size of a layer based on the instruction."""
    if command == "FROM":
        if "alpine" in arguments.lower():
            return SIZE_ESTIMATES["python:3.11-alpine"]
        elif "slim" in arguments.lower():
            return SIZE_ESTIMATES["python:3.11-slim"]
        else:
            return SIZE_ESTIMATES["python:3.11"]
    elif command == "RUN":
        if "pip install" in arguments:
            if any(pkg in arguments for pkg in ["numpy", "pandas", "sklearn", "torch", "tensorflow"]):
                return SIZE_ESTIMATES["pip_install_ml"]
            return SIZE_ESTIMATES["pip_install_api"]
        elif "apt-get install" in arguments:
            return SIZE_ESTIMATES["apt_build_essential"]
        return 10  # Generic RUN command
    elif command in ("COPY", "ADD"):
        if "requirements" in arguments:
            return 0.01  # requirements.txt is tiny
        return SIZE_ESTIMATES["app_code"]
    return 0  # Metadata commands


# === TODO 1: Implement analyze_cache_efficiency ===
# Analyze a parsed Dockerfile for layer caching efficiency.
# 
# Key insight: Once a layer is invalidated (e.g., code changes), 
# ALL subsequent layers must be rebuilt.
#
# For a well-optimized Dockerfile:
# - COPY requirements.txt should come BEFORE pip install
# - COPY app code should come AFTER pip install
# - This way, code changes don't invalidate the dependency layer
#
# Your function should:
# 1. Count total layers (instructions where creates_layer=True)
# 2. Find the "invalidation point" - the first COPY that copies app code
#    (not requirements.txt). This is where cache breaks on code changes.
# 3. Count cacheable layers - layers BEFORE the invalidation point
# 4. Calculate efficiency_score = cacheable_layers / total_layers

def analyze_cache_efficiency(instructions: list[DockerInstruction]) -> CacheAnalysis:
    """
    Analyze layer caching efficiency of a Dockerfile.
    
    Args:
        instructions: Parsed Dockerfile instructions
        
    Returns:
        CacheAnalysis with efficiency metrics
    """
    # TODO: Implement this function
    # Hints:
    # - Layer-creating commands: FROM, RUN, COPY, ADD
    # - "requirements" in COPY arguments = dependency file (cacheable)
    # - Other COPY = app code (cache invalidation point)
    
    total_layers = 0
    cacheable_layers = 0
    invalidation_point = None
    
    # Your code here
    pass
    
    # Calculate efficiency score
    efficiency_score = 0.0
    
    return CacheAnalysis(
        total_layers=total_layers,
        cacheable_layers=cacheable_layers,
        invalidation_point=invalidation_point,
        efficiency_score=efficiency_score
    )


# === TODO 2: Implement identify_antipatterns ===
# Detect common Dockerfile anti-patterns that hurt build performance.
#
# Anti-patterns to detect:
# 1. "copy_before_deps": COPY of app code appears before pip install
# 2. "no_slim_image": Using full python image instead of slim/alpine
# 3. "cache_not_disabled": pip install without --no-cache-dir
# 4. "latest_tag": Using :latest tag (breaks reproducibility)
# 5. "root_user": No USER instruction (running as root)

def identify_antipatterns(instructions: list[DockerInstruction]) -> list[str]:
    """
    Identify anti-patterns in a Dockerfile.
    
    Args:
        instructions: Parsed Dockerfile instructions
        
    Returns:
        List of anti-pattern identifiers found
    """
    antipatterns = []
    
    # TODO: Implement detection for each anti-pattern
    # Hint: Track state as you iterate through instructions
    
    # Your code here
    pass
    
    return antipatterns


# === TODO 3: Implement calculate_size_reduction ===
# Calculate the estimated image size reduction from using multi-stage builds.
#
# In a single-stage build: final size = sum of all layers
# In a multi-stage build: final size = only runtime stage layers
#
# The key insight: build tools (apt-get install build-essential, etc.)
# exist only in the builder stage and don't appear in the final image.

def calculate_size_reduction(single_stage: list[DockerInstruction], 
                             multi_stage: list[DockerInstruction]) -> dict:
    """
    Calculate size reduction from multi-stage build.
    
    Args:
        single_stage: Instructions from single-stage Dockerfile
        multi_stage: Instructions from multi-stage Dockerfile
        
    Returns:
        Dict with 'single_size_mb', 'multi_size_mb', 'reduction_percent'
    """
    # TODO: Calculate total size for each Dockerfile
    # For multi-stage: only count layers AFTER the final FROM instruction
    # (i.e., the runtime stage)
    
    # Your code here
    single_size_mb = 0.0
    multi_size_mb = 0.0
    
    pass
    
    reduction_percent = 0.0
    if single_size_mb > 0:
        reduction_percent = ((single_size_mb - multi_size_mb) / single_size_mb) * 100
    
    return {
        "single_size_mb": single_size_mb,
        "multi_size_mb": multi_size_mb,
        "reduction_percent": round(reduction_percent, 1)
    }


# === TODO 4: Implement optimize_dockerfile ===
# Given a poorly-ordered Dockerfile, reorder instructions for optimal caching.
#
# Optimal order:
# 1. FROM (base image)
# 2. WORKDIR
# 3. System dependencies (apt-get)
# 4. COPY requirements.txt
# 5. RUN pip install
# 6. COPY app code
# 7. ENV, EXPOSE, USER
# 8. CMD/ENTRYPOINT

def optimize_dockerfile(instructions: list[DockerInstruction]) -> list[DockerInstruction]:
    """
    Reorder Dockerfile instructions for optimal layer caching.
    
    Args:
        instructions: Original parsed instructions
        
    Returns:
        Reordered list of instructions
    """
    # TODO: Separate instructions by type and reorder them
    # Categories to track:
    # - from_instr: FROM statements
    # - workdir_instr: WORKDIR statements  
    # - system_deps: apt-get commands
    # - requirements_copy: COPY requirements.txt
    # - pip_install: pip install commands
    # - app_copy: COPY app code
    # - metadata: ENV, EXPOSE, USER
    # - entrypoint: CMD, ENTRYPOINT
    
    # Your code here
    pass
    
    return instructions  # Replace with optimized list


# === SELF-CHECK ===

def run_self_check():
    """Run all self-checks for the lab."""
    print("=" * 60)
    print("Module 3 Lab: Docker Build Optimization - Self-Check")
    print("=" * 60)
    
    # Parse test Dockerfiles
    bad_instructions = parse_dockerfile(DOCKERFILE_BAD)
    good_instructions = parse_dockerfile(DOCKERFILE_GOOD)
    multi_instructions = parse_dockerfile(DOCKERFILE_MULTISTAGE)
    
    print("\n📋 Parsed Dockerfiles:")
    print(f"   Bad Dockerfile: {len(bad_instructions)} instructions")
    print(f"   Good Dockerfile: {len(good_instructions)} instructions")
    print(f"   Multi-stage Dockerfile: {len(multi_instructions)} instructions")
    
    # Check TODO 1: analyze_cache_efficiency
    print("\n" + "-" * 40)
    print("TODO 1: Cache Efficiency Analysis")
    print("-" * 40)
    
    bad_analysis = analyze_cache_efficiency(bad_instructions)
    good_analysis = analyze_cache_efficiency(good_instructions)
    
    if bad_analysis.total_layers == 0:
        print("❌ TODO 1 not implemented yet")
        return False
    
    print(f"   Bad Dockerfile efficiency: {bad_analysis.efficiency_score:.0%}")
    print(f"   Good Dockerfile efficiency: {good_analysis.efficiency_score:.0%}")
    
    assert bad_analysis.efficiency_score < good_analysis.efficiency_score, \
        "Bad Dockerfile should have lower efficiency than good Dockerfile"
    assert good_analysis.efficiency_score >= 0.5, \
        "Good Dockerfile should have at least 50% cache efficiency"
    print("   ✅ Cache efficiency analysis correct!")
    
    # Check TODO 2: identify_antipatterns
    print("\n" + "-" * 40)
    print("TODO 2: Anti-pattern Detection")
    print("-" * 40)
    
    bad_patterns = identify_antipatterns(bad_instructions)
    good_patterns = identify_antipatterns(good_instructions)
    
    if len(bad_patterns) == 0:
        print("❌ TODO 2 not implemented yet")
        return False
    
    print(f"   Bad Dockerfile anti-patterns: {bad_patterns}")
    print(f"   Good Dockerfile anti-patterns: {good_patterns}")
    
    assert "copy_before_deps" in bad_patterns, \
        "Should detect COPY before pip install in bad Dockerfile"
    assert "no_slim_image" in bad_patterns, \
        "Should detect non-slim base image in bad Dockerfile"
    assert len(good_patterns) < len(bad_patterns), \
        "Good Dockerfile should have fewer anti-patterns"
    print("   ✅ Anti-pattern detection correct!")
    
    # Check TODO 3: calculate_size_reduction
    print("\n" + "-" * 40)
    print("TODO 3: Size Reduction Calculation")
    print("-" * 40)
    
    # Create a simple single-stage for comparison
    single_stage_simple = parse_dockerfile("""FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ .
CMD ["python", "main.py"]
""")
    
    size_result = calculate_size_reduction(single_stage_simple, multi_instructions)
    
    if size_result["single_size_mb"] == 0:
        print("❌ TODO 3 not implemented yet")
        return False
    
    print(f"   Single-stage size: {size_result['single_size_mb']:.0f} MB")
    print(f"   Multi-stage size: {size_result['multi_size_mb']:.0f} MB")
    print(f"   Reduction: {size_result['reduction_percent']:.1f}%")
    
    assert size_result["multi_size_mb"] <= size_result["single_size_mb"], \
        "Multi-stage should be same size or smaller"
    print("   ✅ Size reduction calculation correct!")
    
    # Check TODO 4: optimize_dockerfile
    print("\n" + "-" * 40)
    print("TODO 4: Dockerfile Optimization")
    print("-" * 40)
    
    optimized = optimize_dockerfile(bad_instructions.copy())
    optimized_analysis = analyze_cache_efficiency(optimized)
    
    if optimized == bad_instructions:
        print("❌ TODO 4 not implemented yet")
        return False
    
    print(f"   Original efficiency: {bad_analysis.efficiency_score:.0%}")
    print(f"   Optimized efficiency: {optimized_analysis.efficiency_score:.0%}")
    
    assert optimized_analysis.efficiency_score > bad_analysis.efficiency_score, \
        "Optimized Dockerfile should have better cache efficiency"
    print("   ✅ Dockerfile optimization correct!")
    
    # All checks passed
    print("\n" + "=" * 60)
    print("✅ All checks passed! Lab complete.")
    print("=" * 60)
    
    return True


# === EXPECTED OUTPUT ===
# When all TODOs are correctly implemented, running this script should output:
#
# ============================================================
# Module 3 Lab: Docker Build Optimization - Self-Check
# ============================================================
#
# 📋 Parsed Dockerfiles:
#    Bad Dockerfile: 6 instructions
#    Good Dockerfile: 7 instructions
#    Multi-stage Dockerfile: 13 instructions
#
# ----------------------------------------
# TODO 1: Cache Efficiency Analysis
# ----------------------------------------
#    Bad Dockerfile efficiency: 33%
#    Good Dockerfile efficiency: 71%
#    ✅ Cache efficiency analysis correct!
#
# ----------------------------------------
# TODO 2: Anti-pattern Detection
# ----------------------------------------
#    Bad Dockerfile anti-patterns: ['copy_before_deps', 'no_slim_image', 'cache_not_disabled', 'root_user']
#    Good Dockerfile anti-patterns: ['root_user']
#    ✅ Anti-pattern detection correct!
#
# ----------------------------------------
# TODO 3: Size Reduction Calculation
# ----------------------------------------
#    Single-stage size: 205 MB
#    Multi-stage size: 165 MB
#    Reduction: 19.5%
#    ✅ Size reduction calculation correct!
#
# ----------------------------------------
# TODO 4: Dockerfile Optimization
# ----------------------------------------
#    Original efficiency: 33%
#    Optimized efficiency: 67%
#    ✅ Dockerfile optimization correct!
#
# ============================================================
# ✅ All checks passed! Lab complete.
# ============================================================


if __name__ == "__main__":
    run_self_check()