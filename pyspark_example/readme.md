# Apache Spark Cluster with PySpark using Docker

This project demonstrates how to set up an Apache Spark cluster with PySpark using Docker and Docker Compose. The setup includes a Spark master node, two Spark worker nodes, and a PySpark client that can submit jobs to the cluster.

## Prerequisites

Ensure you have the following installed on your machine:
- [Docker](https://www.docker.com/products/docker-desktop)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Setup Instructions

1. **Clone this repository** or create the following `docker-compose.yml` file in your project directory:

    ```yaml
    version: '3'
    services:
      spark-master:
        image: bitnami/spark:latest
        container_name: spark-master
        environment:
          - SPARK_MODE=master
          - SPARK_RPC_AUTHENTICATION_ENABLED=no
          - SPARK_RPC_ENCRYPTION_ENABLED=no
          - SPARK_SSL_ENABLED=no
        ports:
          - "8080:8080"
          - "7077:7077"

      spark-worker-1:
        image: bitnami/spark:latest
        container_name: spark-worker-1
        environment:
          - SPARK_MODE=worker
          - SPARK_MASTER_URL=spark://spark-master:7077
          - SPARK_WORKER_MEMORY=1G
          - SPARK_WORKER_CORES=1
        depends_on:
          - spark-master
        ports:
          - "8081:8081"

      spark-worker-2:
        image: bitnami/spark:latest
        container_name: spark-worker-2
        environment:
          - SPARK_MODE=worker
          - SPARK_MASTER_URL=spark://spark-master:7077
          - SPARK_WORKER_MEMORY=1G
          - SPARK_WORKER_CORES=1
        depends_on:
          - spark-master
        ports:
          - "8082:8082"

      spark-pyspark:
        image: bitnami/spark:latest
        container_name: spark-pyspark
        environment:
          - SPARK_MODE=client
        depends_on:
          - spark-master
        volumes:
          - ./app:/app
        command: "spark-submit --master spark://spark-master:7077 /app/example.py"
    ```

2. **Create the PySpark Script**:

    Create a directory called `app`, and inside it, create a PySpark script named `example.py` with the following content:

    ```python
    # your_script.py
    from pyspark.sql import SparkSession

    # Create a Spark session
    spark = SparkSession.builder.appName("PySpark Example").getOrCreate()

    # Create a simple DataFrame
    data = [("John", 30), ("Jane", 25), ("Sam", 35)]
    df = spark.createDataFrame(data, ["Name", "Age"])

    # Show the DataFrame
    df.show()

    # Stop the Spark session
    spark.stop()
    ```

3. **Start the Spark Cluster**:

    Run the following command to start the Spark master and worker nodes, as well as the PySpark client:

    ```bash
    docker-compose up -d
    ```

    This will launch:
    - Spark Master on `http://localhost:8080`
    - Two Spark Workers on `http://localhost:8081` and `http://localhost:8082`
    - PySpark client that runs the job defined in `your_script.py`.

4. **Check the Spark UI**:

    - **Spark Master UI**: Visit `http://localhost:8080` to monitor the Spark master.
    - **Worker UIs**: Workers are available at `http://localhost:8081` and `http://localhost:8082`.

5. **Submit the PySpark Job**:

    The PySpark job (`example.py`) is automatically submitted when you start the containers. To check the logs of the PySpark job, run:

    ```bash
    docker logs spark-pyspark
    ```

6. **Stop the Cluster**:

    To stop the Spark cluster, use:

    ```bash
    docker-compose down
    ```

## Useful Commands

- **Start the cluster**: `docker-compose up -d`
- **Stop the cluster**: `docker-compose down`
- **Check PySpark job logs**: `docker logs spark-pyspark`
- **Check running containers**: `docker ps`
- **Restart the cluster**: `docker-compose restart`

## Directory Structure

. ├── docker-compose.yml └── app └── example.py


## Troubleshooting

- **Ports Conflict**: Ensure that ports `8080`, `7077`, `8081`, and `8082` are not being used by other services on your machine.
- **Logs**: Check logs for more detailed error messages using:

    ```bash
    docker-compose logs
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
