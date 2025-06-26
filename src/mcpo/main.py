import json
import os
import socket
import tempfile
import shutil
import asyncio
import random
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from starlette.routing import Mount

logger = logging.getLogger(__name__)

from mcpo.utils.main import get_model_fields, get_tool_handler
from mcpo.utils.auth import get_verify_api_key, APIKeyMiddleware


async def create_dynamic_endpoints(app: FastAPI, api_dependency=None):
    session: ClientSession = app.state.session
    if not session:
        logger.warning("Session is not initialized in the app state, skipping endpoint creation")
        return

    # Get server name
    server_name = getattr(app.state, "server_name", getattr(app, 'title', 'unknown'))

    try:
        result = await session.initialize()
        server_info = getattr(result, "serverInfo", None)
        if server_info:
            app.title = server_info.name or app.title
            server_name = server_info.name or server_name
            app.description = (
                f"{server_info.name} MCP Server" if server_info.name else app.description
            )
            app.version = server_info.version or app.version

        instructions = getattr(result, "instructions", None)
        if instructions:
            app.description = instructions

        tools_result = await session.list_tools()
        tools = tools_result.tools

        logger.info(f"[{server_name}] Found {len(tools)} tools from MCP server")

        successful_endpoints = 0
        failed_endpoints = 0

        for tool in tools:
            try:
                endpoint_name = tool.name
                endpoint_description = tool.description

                logger.debug(f"[{server_name}] Processing tool: {endpoint_name}")

                inputSchema = tool.inputSchema
                outputSchema = getattr(tool, "outputSchema", None)

                form_model_fields = get_model_fields(
                    f"{endpoint_name}_form_model",
                    inputSchema.get("properties", {}),
                    inputSchema.get("required", []),
                    inputSchema.get("$defs", {}),
                    server_name,
                )

                response_model_fields = None
                if outputSchema:
                    response_model_fields = get_model_fields(
                        f"{endpoint_name}_response_model",
                        outputSchema.get("properties", {}),
                        outputSchema.get("required", []),
                        outputSchema.get("$defs", {}),
                        server_name,
                    )

                tool_handler = get_tool_handler(
                    session,
                    endpoint_name,
                    form_model_fields,
                    response_model_fields,
                )

                app.post(
                    f"/{endpoint_name}",
                    summary=endpoint_name.replace("_", " ").title(),
                    description=endpoint_description,
                    response_model_exclude_none=True,
                    dependencies=[Depends(api_dependency)] if api_dependency else [],
                )(tool_handler)

                logger.debug(f"[{server_name}] Successfully created endpoint: /{endpoint_name}")
                successful_endpoints += 1

            except Exception as e:
                logger.error(f"[{server_name}] Failed to create endpoint for tool '{tool.name}': {e}")
                failed_endpoints += 1
                continue

        logger.info(
            f"[{server_name}] Endpoint creation completed: {successful_endpoints} successful, {failed_endpoints} failed")

        if successful_endpoints == 0 and failed_endpoints > 0:
            logger.warning(f"[{server_name}] No endpoints were created successfully")

    except Exception as e:
        logger.error(f"[{server_name}] Failed to initialize MCP session or list tools: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Main lifespan function, handles single server mode and main application"""
    worker_id = os.getpid()
    logger.info(f"Worker {worker_id} starting up...")

    server_type = getattr(app.state, "server_type", "stdio")
    command = getattr(app.state, "command", None)
    args = getattr(app.state, "args", [])

    if (server_type == "stdio" and not command) or (
        server_type == "sse" and not args[0]
    ):
        # Main app lifespan (when config_path is provided)
        logger.info(f"Worker {worker_id} handling main app lifespan")
        try:
            async with AsyncExitStack() as stack:
                for route in app.routes:
                    if isinstance(route, Mount) and isinstance(route.app, FastAPI):
                        try:
                            await stack.enter_async_context(
                                route.app.router.lifespan_context(route.app),
                            )
                            logger.debug(f"Worker {worker_id} successfully started sub-app: {route.path}")
                        except Exception as e:
                            logger.error(f"Worker {worker_id} failed to start sub-app {route.path}: {e}")
                            # Continue processing other routes
                            continue
                yield
        except Exception as e:
            logger.error(f"Worker {worker_id} main app lifespan failed: {e}")
            # Even if there are errors, let the main application continue running
            yield
    else:
        # Single server mode, use safe lifespan
        async with safe_lifespan(app):
            yield

    logger.info(f"Worker {worker_id} shutting down...")


def create_app(**kwargs):
    """Create FastAPI application instance"""
    # Read configuration from environment variables (for multi-worker mode)
    if not kwargs:
        import pickle
        import base64

        encoded_config = os.environ.get("MCPO_CONFIG")
        if encoded_config:
            try:
                kwargs = pickle.loads(base64.b64decode(encoded_config))
            except Exception as e:
                logger.error(f"Failed to decode config from environment: {e}")
                kwargs = {}

    # Server API Key
    api_key = kwargs.get("api_key", "")
    api_dependency = get_verify_api_key(api_key) if api_key else None
    strict_auth = kwargs.get("strict_auth", False)

    # MCP Server
    server_type = kwargs.get("server_type")
    server_command = kwargs.get("server_command")

    # MCP Config
    config_path = kwargs.get("config_path")

    # mcpo server
    name = kwargs.get("name") or "MCP OpenAPI Proxy"
    description = (
            kwargs.get("description") or "Automatically generated API from MCP Tool Schemas"
    )
    version = kwargs.get("version") or "1.0"

    ssl_certfile = kwargs.get("ssl_certfile")
    ssl_keyfile = kwargs.get("ssl_keyfile")
    path_prefix = kwargs.get("path_prefix") or "/"

    main_app = FastAPI(
        title=name,
        description=description,
        version=version,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        lifespan=lifespan,
    )

    main_app.add_middleware(
        CORSMiddleware,
        allow_origins=kwargs.get("cors_allow_origins", ["*"]) or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add middleware to protect also documentation and spec
    if api_key and strict_auth:
        main_app.add_middleware(APIKeyMiddleware, api_key=api_key)

    headers = kwargs.get("headers")
    if headers and isinstance(headers, str):
        try:
            headers = json.loads(headers)
        except json.JSONDecodeError:
            print("Warning: Invalid JSON format for headers. Headers will be ignored.")
            headers = None

    if server_type == "sse":
        main_app.state.server_type = "sse"
        main_app.state.args = server_command[0]
        main_app.state.api_dependency = api_dependency
        main_app.state.headers = headers
    elif server_type == "streamablehttp" or server_type == "streamable_http":
        main_app.state.server_type = "streamablehttp"
        main_app.state.args = server_command[0]
        main_app.state.api_dependency = api_dependency
        main_app.state.headers = headers
    elif server_command:  # This handles stdio
        logger.info(
            f"Configuring for a single Stdio MCP Server with command: {' '.join(server_command)}"
        )
        main_app.state.server_type = "stdio"  # Explicitly set type
        main_app.state.command = server_command[0]
        main_app.state.args = server_command[1:]
        main_app.state.env = os.environ.copy()
        main_app.state.api_dependency = api_dependency
    elif config_path:
        logger.info(f"Loading MCP server configurations from: {config_path}")
        with open(config_path, "r") as f:
            config_data = json.load(f)

        mcp_servers = config_data.get("mcpServers", {})
        if not mcp_servers:
            logger.error(f"No 'mcpServers' found in config file: {config_path}")
            raise ValueError("No 'mcpServers' found in config file.")

        logger.info("Configured MCP Servers:")
        for server_name_cfg, server_cfg_details in mcp_servers.items():
            if server_cfg_details.get("command"):
                args_info = (
                    f" with args: {server_cfg_details['args']}"
                    if server_cfg_details.get("args")
                    else ""
                )
                logger.info(
                    f"  Configuring Stdio MCP Server '{server_name_cfg}' with command: {server_cfg_details['command']}{args_info}"
                )
            elif server_cfg_details.get("type") == "sse" and server_cfg_details.get(
                "url"
            ):
                logger.info(
                    f"  Configuring SSE MCP Server '{server_name_cfg}' with URL: {server_cfg_details['url']}"
                )
            elif (
                server_cfg_details.get("type") == "streamablehttp"
                or server_cfg_details.get("type") == "streamable_http"
            ) and server_cfg_details.get("url"):
                logger.info(
                    f"  Configuring StreamableHTTP MCP Server '{server_name_cfg}' with URL: {server_cfg_details['url']}"
                )
            elif server_cfg_details.get("url"):  # Fallback for old SSE config
                logger.info(
                    f"  Configuring SSE (fallback) MCP Server '{server_name_cfg}' with URL: {server_cfg_details['url']}"
                )
            else:
                logger.warning(
                    f"  Unknown configuration for MCP server: {server_name_cfg}"
                )

        main_app.description += "\n\n- **available tools**："

        # Track successful and failed servers
        successful_servers = []
        failed_servers = []

        for server_name, server_cfg in mcp_servers.items():
            try:
                logger.info(f"Attempting to register MCP server: {server_name}")

                # Create sub-application
                sub_app = FastAPI(
                    title=f"{server_name}",
                    description=f"{server_name} MCP Server\n\n- [back to tool list](/docs)",
                    version="1.0",
                    lifespan=safe_lifespan,  # Use safe lifespan
                )

                sub_app.add_middleware(
                    CORSMiddleware,
                    allow_origins=kwargs.get("cors_allow_origins", ["*"]) or ["*"],
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                )

                # Configure server parameters
                if server_cfg.get("command"):
                    # stdio
                    sub_app.state.server_type = "stdio"
                    sub_app.state.command = server_cfg["command"]
                    sub_app.state.args = server_cfg.get("args", [])
                    sub_app.state.env = {**os.environ, **server_cfg.get("env", {})}

                server_config_type = server_cfg.get("type")
                if server_config_type == "sse" and server_cfg.get("url"):
                    sub_app.state.server_type = "sse"
                    sub_app.state.args = server_cfg["url"]
                    sub_app.state.headers = server_cfg.get("headers")

                elif (
                        server_config_type == "streamablehttp"
                        or server_config_type == "streamable_http"
                ) and server_cfg.get("url"):
                    url = server_cfg["url"]
                    if not url.endswith("/"):
                        url = f"{url}/"
                    sub_app.state.server_type = "streamablehttp"
                    sub_app.state.args = url
                    sub_app.state.headers = server_cfg.get("headers")

                elif not server_config_type and server_cfg.get("url"):
                    sub_app.state.server_type = "sse"
                    sub_app.state.args = server_cfg["url"]
                    sub_app.state.headers = server_cfg.get("headers")

                # Add middleware to protect also documentation and spec
                if api_key and strict_auth:
                    sub_app.add_middleware(APIKeyMiddleware, api_key=api_key)

                sub_app.state.api_dependency = api_dependency
                sub_app.state.server_name = server_name  # Add server name for logging

                # Try to mount sub-application
                main_app.mount(f"{path_prefix}{server_name}", sub_app)
                successful_servers.append(server_name)
                logger.info(f"Successfully registered MCP server: {server_name}")

            except Exception as e:
                logger.error(f"Failed to register MCP server '{server_name}': {e}")
                failed_servers.append((server_name, str(e)))
                # Continue processing next server, don't stop because of one failure

        # Update main app description, only include successful servers
        if successful_servers:
            for server_name in successful_servers:
                main_app.description += f"\n    - [{server_name}](/{server_name}/docs)"

        # Log statistics
        logger.info(f"MCP server registration completed:")
        logger.info(f"  - Successful: {len(successful_servers)} servers")
        logger.info(f"  - Failed: {len(failed_servers)} servers")

        if successful_servers:
            logger.info(f"  - Active servers: {', '.join(successful_servers)}")

        if failed_servers:
            logger.warning(f"  - Failed servers:")
            for server_name, error in failed_servers:
                logger.warning(f"    - {server_name}: {error}")

        if not successful_servers:
            logger.warning("No MCP servers were successfully registered!")
            main_app.description += "\n\n⚠️ **No MCP servers are currently available**"
    else:
        logger.error("MCPO server_command or config_path must be provided.")
        raise ValueError("You must provide either server_command or config.")

    return main_app


@asynccontextmanager
async def safe_lifespan(app: FastAPI):
    """Safe lifespan function that won't crash the entire application on failure"""
    worker_id = os.getpid()
    server_name = getattr(app.state, "server_name", getattr(app, 'title', 'unknown'))

    logger.info(f"Worker {worker_id} starting MCP server: {server_name}")

    server_type = getattr(app.state, "server_type", "stdio")
    command = getattr(app.state, "command", None)
    args = getattr(app.state, "args", [])
    env = getattr(app.state, "env", {})

    args = args if isinstance(args, list) else [args]
    api_dependency = getattr(app.state, "api_dependency", None)

    temp_dir = None
    session_established = False

    # Add random startup delay to avoid all workers starting simultaneously
    startup_delay = random.uniform(0, 2)
    await asyncio.sleep(startup_delay)

    try:
        if server_type == "stdio":
            # Create independent temporary directory for each worker
            temp_dir = tempfile.mkdtemp(prefix=f"mcpo-{server_name}-{worker_id}-")
            worker_npm_cache = os.path.join(temp_dir, ".npm")
            worker_npx_cache = os.path.join(temp_dir, ".npx")

            os.makedirs(worker_npm_cache, exist_ok=True)
            os.makedirs(worker_npx_cache, exist_ok=True)

            worker_env = {**env}
            logger.info(f"worker_env after copy ({worker_env} items):")
            worker_env.update({
                "NPM_CONFIG_CACHE": worker_npm_cache,
                "npm_config_cache": worker_npm_cache,
                "NPX_CACHE": worker_npx_cache,
                "TMPDIR": temp_dir,
                "TMP": temp_dir,
                "TEMP": temp_dir,
                "XDG_CACHE_HOME": temp_dir,
                "HOME": temp_dir,
            })

            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=worker_env,
            )

            logger.info(f"Worker {worker_id} attempting to connect to {server_name}")

            # Remove complex timeout mechanism, let MCP library handle timeout itself
            async with stdio_client(server_params) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    app.state.session = session
                    session_established = True

                    # Try to create endpoints
                    try:
                        await create_dynamic_endpoints(app, api_dependency=api_dependency)
                        logger.info(f"Worker {worker_id} successfully initialized {server_name}")
                    except Exception as e:
                        logger.error(f"Worker {worker_id} failed to create endpoints for {server_name}: {e}")
                        # Even if endpoint creation fails, keep session alive

                    yield

        elif server_type == "sse":
            async with sse_client(url=args[0], sse_read_timeout=None) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    app.state.session = session
                    session_established = True

                    try:
                        await create_dynamic_endpoints(app, api_dependency=api_dependency)
                        logger.info(f"Worker {worker_id} successfully initialized SSE {server_name}")
                    except Exception as e:
                        logger.error(f"Worker {worker_id} failed to create endpoints for SSE {server_name}: {e}")

                    yield

        elif server_type == "streamablehttp" or server_type == "streamable_http":
            url = args[0]
            if not url.endswith("/"):
                url = f"{url}/"

            async with streamablehttp_client(url=url) as (reader, writer, _):
                async with ClientSession(reader, writer) as session:
                    app.state.session = session
                    session_established = True

                    try:
                        await create_dynamic_endpoints(app, api_dependency=api_dependency)
                        logger.info(f"Worker {worker_id} successfully initialized StreamableHTTP {server_name}")
                    except Exception as e:
                        logger.error(
                            f"Worker {worker_id} failed to create endpoints for StreamableHTTP {server_name}: {e}")

                    yield

    except Exception as e:
        logger.error(f"Worker {worker_id} failed to initialize {server_name}: {e}")

        # If connection fails, create an empty application state
        app.state.session = None

        # Add error endpoints
        @app.get("/")
        async def server_unavailable():
            return {
                "error": f"MCP server '{server_name}' is currently unavailable",
                "reason": str(e),
                "status": "failed"
            }

        @app.get("/health")
        async def health_check():
            return {
                "status": "unhealthy",
                "server": server_name,
                "error": str(e)
            }

        yield  # Important: yield even on failure to let the application continue running

    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"Worker {worker_id} cleaned up temp directory for {server_name}")
            except Exception as e:
                logger.warning(f"Worker {worker_id} failed to clean temp directory for {server_name}: {e}")

    if session_established:
        logger.info(f"Worker {worker_id} shutting down {server_name}")
    else:
        logger.warning(f"Worker {worker_id} {server_name} was never properly initialized")


# Module-level application instance for multi-worker mode
app = None


def get_app():
    """Application factory function for multi-worker mode"""
    global app
    if app is None:
        app = create_app()
    return app


async def run(
    host: str = "127.0.0.1",
    port: int = 8000,
    workers: int = 1,
    api_key: Optional[str] = "",
    cors_allow_origins=["*"],
    **kwargs,
):
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [PID:%(process)d] - %(message)s",
    )

    logger.info("Starting MCPO Server...")
    logger.info(f"  Name: {kwargs.get('name', 'MCP OpenAPI Proxy')}")
    logger.info(f"  Version: {kwargs.get('version', '1.0')}")
    logger.info(
        f"  Description: {kwargs.get('description', 'Automatically generated API from MCP Tool Schemas')}"
    )
    logger.info(f"  Hostname: {socket.gethostname()}")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Workers: {workers}")
    logger.info(f"  API Key: {'Provided' if api_key else 'Not Provided'}")
    logger.info(f"  CORS Allowed Origins: {cors_allow_origins}")

    ssl_certfile = kwargs.get("ssl_certfile")
    ssl_keyfile = kwargs.get("ssl_keyfile")
    if ssl_certfile:
        logger.info(f"  SSL Certificate File: {ssl_certfile}")
    if ssl_keyfile:
        logger.info(f"  SSL Key File: {ssl_keyfile}")
    logger.info(f"  Path Prefix: {kwargs.get('path_prefix', '/')}")

    # Prepare application configuration
    app_config = {
        "api_key": api_key,
        "cors_allow_origins": cors_allow_origins,
        **kwargs,
    }

    # If only one worker, run directly
    if workers == 1:
        main_app = create_app(**app_config)

        logger.info("Starting single worker mode...")
        config = uvicorn.Config(
            app=main_app,
            host=host,
            port=port,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()
    else:
        # Multi-worker mode - serialize configuration to environment variable
        logger.info(f"Starting multi-worker mode with {workers} workers...")

        import pickle
        import base64

        # Serialize configuration and save to environment variable
        encoded_config = base64.b64encode(pickle.dumps(app_config)).decode()
        os.environ["MCPO_CONFIG"] = encoded_config

        # Use uvicorn.run() instead of uvicorn.Config + uvicorn.Server
        uvicorn.run(
            "mcpo.main:get_app",
            host=host,
            port=port,
            workers=workers,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            log_level="info",
            factory=True,  # Tell uvicorn this is a factory function
        )


# For compatibility with direct import, create a default application instance
try:
    app = get_app()
except:
    # If creation fails (e.g., missing configuration), create an empty application
    app = FastAPI(title="MCP OpenAPI Proxy")
