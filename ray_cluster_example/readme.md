

# Ray Cluster Example using Docker Compose

This project sets up a **Ray cluster** using Docker Compose. The cluster consists of a head node and multiple worker nodes, allowing for distributed computation and parallel task execution with Ray.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Build the Docker Images](#2-build-the-docker-images)
  - [3. Start the Ray Cluster](#3-start-the-ray-cluster)
  - [4. Access the Ray Dashboard](#4-access-the-ray-dashboard)
  - [5. Running a Distributed Task](#5-running-a-distributed-task)
- [Scaling the Cluster](#scaling-the-cluster)
- [Stopping the Cluster](#stopping-the-cluster)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Make sure you have the following installed on your system:

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/thejat/mlops-code-examples.git
cd mlops-code-examples
cd ray_cluster_example
```

### 2. Build the Docker Images

To build the Ray head and worker Docker images:

```bash
docker-compose build
```

### 3. Start the Ray Cluster

To start the Ray cluster, including one head node and two worker nodes:

```bash
docker-compose up -d
```

### 4. Access the Ray Dashboard

You can monitor the Ray cluster through the **Ray Dashboard**. The dashboard will be available at [http://localhost:8265](http://localhost:8265).

### 5. Running a Distributed Task

Once the Ray cluster is running, you can submit jobs or run distributed tasks. For example, you can create a simple script like `example.py` in the `app` directory:

```python
import ray

# Connect to the Ray cluster
ray.init(address='auto')

@ray.remote
def square(x):
    return x * x

if __name__ == "__main__":
    # Distribute tasks across the Ray cluster
    results = ray.get([square.remote(i) for i in range(100)])
    print(results)
```

To run the script inside the head node container:

```bash
docker exec -it ray_cluster_example-ray-head-1 python /app/example.py
```

This will distribute the computation across the Ray cluster and return the results.

## Scaling the Cluster

To scale the number of worker nodes up or down, adjust the `deploy.replicas` value in the `docker-compose.yml` file under the `ray-worker` service:

```yaml
deploy:
  replicas: 4  # Number of worker nodes
```

Then apply the changes:

```bash
docker-compose up -d --scale ray-worker=4
```

## Stopping the Cluster

To stop the cluster and remove the containers:

```bash
docker-compose down
```

## Troubleshooting

- **Port Conflict**: If you encounter a port conflict on `6379` (Redis or Ray head port), edit the `docker-compose.yml` file and change the external port for the Ray head service. For example, use `6380:6379` to avoid conflicts.
  
- **Containers Exiting Immediately**: Ensure that the containers remain running by using `tail -f /dev/null` in the `docker-compose.yml` to keep the head and worker nodes alive after starting Ray.
