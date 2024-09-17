# Docker Compose Example

## Description

This project provides examples of using Docker Compose to orchestrate containerized applications.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, follow these steps:

1. Clone the repository.
2. Install Docker and Docker Compose.
3. Run `docker-compose up` to start the application.

## Usage

To connect to the Docker container and manage your PostgreSQL database, you can use pgAdmin 4. Follow these steps:

1. Open a web browser and visit `http://localhost:5050`.
2. Log in to pgAdmin 4 using the default credentials (username: `pgadmin4@pgadmin.org`, password: `admin`).
3. Click on "Add New Server" in the "Quick Links" section.
4. Enter a name for the server and switch to the "Connection" tab.
5. In the "Host name/address" field, enter the name of the Docker container running PostgreSQL (e.g., `postgres`).
6. Set the "Port" to `5432`.
7. Enter the username and password for the PostgreSQL database.
8. Click "Save" to connect to the Docker container.

Now you can use pgAdmin 4 to manage your PostgreSQL database running in the Docker container.


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This example is licensed under the [MIT License](LICENSE).