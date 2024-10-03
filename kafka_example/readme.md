# Kafka with Docker and Zookeeper

This repository demonstrates how to set up Apache Kafka with Zookeeper using Docker and Docker Compose.

## Prerequisites

Ensure you have the following installed on your machine:
- [Docker](https://www.docker.com/products/docker-desktop)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Setup Instructions

1. **Clone the repository** or create the following `docker-compose.yml` file in your project directory:

    ```yaml
    services:
      zookeeper:
        image: 'confluentinc/cp-zookeeper:latest'
        environment:
          ZOOKEEPER_CLIENT_PORT: 2181
          ZOOKEEPER_TICK_TIME: 2000
        ports:
          - "2181:2181"

      kafka:
        image: 'confluentinc/cp-kafka:latest'
        depends_on:
          - zookeeper
        environment:
          KAFKA_BROKER_ID: 1
          KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
          KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
          KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
        ports:
          - "9092:9092"
    ```

2. **Start the Services**:

    Run the following command to bring up Kafka and Zookeeper:

    ```bash
    docker-compose up -d
    ```

    This will start the Kafka broker on port `9092` and Zookeeper on port `2181`.

3. **Verify Kafka Setup**:

    - To **list the topics**, use:

      ```bash
      docker exec -it <kafka_container_name> kafka-topics --bootstrap-server localhost:9092 --list
      ```

    - To **create a new topic**, use:

      ```bash
      docker exec -it <kafka_container_name> kafka-topics --bootstrap-server localhost:9092 --create --topic test-topic --partitions 1 --replication-factor 1
      ```

      Replace `<kafka_container_name>` with the actual name of the Kafka container. You can find the container name by running:

      ```bash
      docker ps
      ```

4. **Kafka CLI Examples**:

    - **Producing messages** to a topic:

      ```bash
      docker exec -it <kafka_container_name> kafka-console-producer --bootstrap-server localhost:9092 --topic test-topic
      ```

      Type your message and hit Enter to send it.

    - **Consuming messages** from a topic:

      ```bash
      docker exec -it <kafka_container_name> kafka-console-consumer --bootstrap-server localhost:9092 --topic test-topic --from-beginning
      ```

## Useful Commands

- **Stop services**:

    ```bash
    docker-compose down
    ```

- **Restart services**:

    ```bash
    docker-compose restart
    ```

## Troubleshooting

- If Kafka fails to start, ensure no other service is using port `9092` or `2181`.
- You can check logs using:

    ```bash
    docker-compose logs
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
