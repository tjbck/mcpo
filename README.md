# ⚡️ mcpo

Expose any MCP tool as an OpenAPI-compatible HTTP server—instantly.

mcpo is a dead-simple proxy that takes an MCP server command and makes it accessible via standard RESTful OpenAPI, so your tools "just work" with LLM agents and apps expecting OpenAPI servers.

No custom protocol. No glue code. No hassle.

## 🚀 Quick Usage

We recommend using uv for lightning-fast startup and zero config.

```bash
uvx mcpo --port 8000 -- your_mcp_server_command
```

Or, if you’re using Python:

```bash
pip install mcpo
mcpo --port 8000 -- your_mcp_server_command
```

Example:

```bash
uvx mcpo --port 8000 -- uvx mcp-server-time --local-timezone=America/New_York
```

That’s it. Your MCP tool is now available at http://localhost:8000 with a generated OpenAPI schema — test it live at [http://localhost:8000/docs](http://localhost:8000/docs).


### 🔄 Using a Config File

You can serve multiple MCP tools via a single config file that follows the [Claude Desktop](https://modelcontextprotocol.io/quickstart/user) format:

Start via:

```bash
mcpo --config /path/to/config.json
```

Example config.json:

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "time": {
      "command": "uvx",
      "args": ["mcp-server-time", "--local-timezone=America/New_York"]
    }
  }
}
```

Each tool will be accessible under its own unique route, e.g.:
- http://localhost:8000/memory
- http://localhost:8000/time

Each with a dedicated OpenAPI schema and proxy handler. Access full schema UI at: `http://localhost:8000/<tool>/docs`  (e.g. /memory/docs, /time/docs)

## 🐋 Docker

mcpo can be run in Docker with the enclosed dockerfile. Mount a volume containing config.json to /app and add servers to install at /servers. Each time the container launches it will look for new tools it has not yet installed (in the /servers directory) and install them. Use the pip installation method when adding new tools to the config.json for use in this container.

If no servers are added and/or if config.json doesn't exist, the container will load the example config and clone "time" and "fetch" from github.

Example docker-compose:

```yaml
services:
  mcpo:
    build:
      context: .
      dockerfile: dockerfile
    container_name: mcpo
    ports:
      - "8080:8000"
    volumes:
      - ./app:/app
      - ./servers:/servers
```

## 🔧 Requirements

- Python 3.8+
- uv (optional, but highly recommended for performance + packaging)

## 🪪 License

MIT