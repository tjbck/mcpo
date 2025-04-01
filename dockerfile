
# Use the official Python image as a base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    CONFIG_PATH=/app/config.json \
    PORT=8000

# Create and set the working directory
WORKDIR /app

# Install git (and clean up)
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Create servers directory
RUN mkdir /servers

# Install mcpo and uv for managing Python packages
RUN pip install uvx npx mcpo

# Copy the entrypoint script into the container
COPY docker-entrypoint.sh /usr/local/bin/

# Copy the example configuration JSON into the container in case the user does not supply one
COPY example.config.json /usr/local/bin/

# Make sure the entrypoint script is executable
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Expose the port mcpo will run on
EXPOSE $PORT

# Use the entrypoint script
ENTRYPOINT ["docker-entrypoint.sh"]

# Command to run mcpo with the specified configuration
CMD mcpo --config "$CONFIG_PATH"