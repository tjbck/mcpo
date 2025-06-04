import json
import os
import logging
import socket
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
        raise ValueError("Session is not initialized in the app state.")

    result = await session.initialize()
    server_info = getattr(result, "serverInfo", None)
    if server_info:
        app.title = server_info.name or app.title
        app.description = (
            f"{server_info.name} MCP Server" if server_info.name else app.description
        )
        app.version = server_info.version or app.version

    instructions = getattr(result, "instructions", None)
    if instructions:
        app.description = instructions

    tools_result = await session.list_tools()
    tools = tools_result.tools

    for tool in tools:
        endpoint_name = tool.name
        endpoint_description = tool.description

        inputSchema = tool.inputSchema
        outputSchema = getattr(tool, "outputSchema", None)

        form_model_fields = get_model_fields(
            f"{endpoint_name}_form_model",
            inputSchema.get("properties", {}),
            inputSchema.get("required", []),
            inputSchema.get("$defs", {}),
        )

        response_model_fields = None
        if outputSchema:
            response_model_fields = get_model_fields(
                f"{endpoint_name}_response_model",
                outputSchema.get("properties", {}),
                outputSchema.get("required", []),
                outputSchema.get("$defs", {}),
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Get worker process ID for logging identification
    worker_id = os.getpid()
    logger.info(f"Worker {worker_id} starting up...")

    server_type = getattr(app.state, "server_type", "stdio")
    command = getattr(app.state, "command", None)
    args = getattr(app.state, "args", [])
    env = getattr(app.state, "env", {})

    args = args if isinstance(args, list) else [args]
    api_dependency = getattr(app.state, "api_dependency", None)

    if (server_type == "stdio" and not command) or (
        server_type == "sse" and not args[0]
    ):
        # Main app lifespan (when config_path is provided)
        async with AsyncExitStack() as stack:
            for route in app.routes:
                if isinstance(route, Mount) and isinstance(route.app, FastAPI):
                    await stack.enter_async_context(
                        route.app.router.lifespan_context(route.app),  # noqa
                    )
            yield
    else:
        if server_type == "stdio":
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env={**os.environ, **env},
            )

            async with stdio_client(server_params) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    app.state.session = session
                    await create_dynamic_endpoints(app, api_dependency=api_dependency)
                    logger.info(f"Worker {worker_id} MCP connections established")
                    yield
        if server_type == "sse":
            headers = getattr(app.state, "headers", None)
            async with sse_client(
                url=args[0], sse_read_timeout=None, headers=headers
            ) as (
                reader,
                writer,
            ):
                async with ClientSession(reader, writer) as session:
                    app.state.session = session
                    await create_dynamic_endpoints(app, api_dependency=api_dependency)
                    logger.info(f"Worker {worker_id} MCP connections established")
                    yield
        if server_type == "streamablehttp" or server_type == "streamable_http":
            headers = getattr(app.state, "headers", None)

            # Ensure URL has trailing slash to avoid redirects
            url = args[0]
            if not url.endswith("/"):
                url = f"{url}/"

            # Connect using streamablehttp_client from the SDK, similar to sse_client
            async with streamablehttp_client(url=url, headers=headers) as (
                reader,
                writer,
                _,  # get_session_id callback not needed for ClientSession
            ):
                async with ClientSession(reader, writer) as session:
                    app.state.session = session
                    await create_dynamic_endpoints(app, api_dependency=api_dependency)
                    logger.info(f"Worker {worker_id} MCP connections established")
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
        main_app.state.args = server_command[0]  # Expects URL as the first element
        main_app.state.api_dependency = api_dependency
        main_app.state.headers = headers
    elif server_type == "streamablehttp" or server_type == "streamable_http":
        main_app.state.server_type = "streamablehttp"
        main_app.state.args = server_command[0]  # Expects URL as the first element
        main_app.state.api_dependency = api_dependency
        main_app.state.headers = headers
    elif server_command:  # This handles stdio
        main_app.state.server_type = "stdio"  # Explicitly set type
        main_app.state.command = server_command[0]
        main_app.state.args = server_command[1:]
        main_app.state.env = os.environ.copy()
        main_app.state.api_dependency = api_dependency
    elif config_path:
        with open(config_path, "r") as f:
            config_data = json.load(f)

        mcp_servers = config_data.get("mcpServers", {})
        if not mcp_servers:
            raise ValueError("No 'mcpServers' found in config file.")

        main_app.description += "\n\n- **available tools**："
        for server_name, server_cfg in mcp_servers.items():
            sub_app = FastAPI(
                title=f"{server_name}",
                description=f"{server_name} MCP Server\n\n- [back to tool list](/docs)",
                version="1.0",
                lifespan=lifespan,
            )

            sub_app.add_middleware(
                CORSMiddleware,
                allow_origins=kwargs.get("cors_allow_origins", ["*"]) or ["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

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
                # Store the URL with trailing slash to avoid redirects
                url = server_cfg["url"]
                if not url.endswith("/"):
                    url = f"{url}/"
                sub_app.state.server_type = "streamablehttp"
                sub_app.state.args = url
                sub_app.state.headers = server_cfg.get("headers")

            elif not server_config_type and server_cfg.get(
                "url"
            ):  # Fallback for old SSE config
                sub_app.state.server_type = "sse"
                sub_app.state.args = server_cfg["url"]
                sub_app.state.headers = server_cfg.get("headers")

            # Add middleware to protect also documentation and spec
            if api_key and strict_auth:
                sub_app.add_middleware(APIKeyMiddleware, api_key=api_key)

            sub_app.state.api_dependency = api_dependency

            main_app.mount(f"{path_prefix}{server_name}", sub_app)
            main_app.description += f"\n    - [{server_name}](/{server_name}/docs)"
    else:
        raise ValueError("You must provide either server_command or config.")

    return main_app


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
