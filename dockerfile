
# Use the official Python image as a base
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    CONFIG_PATH=/app/config.json \
    UVICORN_PORT=8000 \
    UVICORN_LOG_LEVEL=info

# Create and set the working directory
WORKDIR /app

# Copy source to container
COPY . /app

# Install mcpo from source
RUN pip install .

# Install git, curl, and npm (and clean up)
RUN apt-get update && \
    apt-get install -y git curl npm && \
    rm -rf /var/lib/apt/lists/*

# Copy the entrypoint script into the container
COPY docker-entrypoint.sh /usr/local/bin/

# Copy the example configuration JSON into the container in case the user does not supply one
COPY example.config.json /usr/local/bin/

# Make sure the entrypoint script is executable
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Expose the port mcpo will run on
EXPOSE $UVICORN_PORT

# Use the entrypoint script
ENTRYPOINT ["docker-entrypoint.sh"]

# Command to run mcpo with the specified configuration
CMD mcpo --config "$CONFIG_PATH"