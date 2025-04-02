from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from pydantic import create_model
from contextlib import AsyncExitStack, asynccontextmanager

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from typing import Dict, Any, Callable
import uvicorn
import json
import os


def get_python_type(param_type: str):
    if param_type == "string":
        return str
    elif param_type == "integer":
        return int
    elif param_type == "boolean":
        return bool
    elif param_type == "number":
        return float
    elif param_type == "object":
        return Dict[str, Any]
    elif param_type == "array":
        return list
    else:
        return str  # Fallback
    # Expand as needed. PRs welcome!


async def create_dynamic_endpoints(app: FastAPI):
    session = app.state.session
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

    tools_result = await session.list_tools()
    tools = tools_result.tools

    for tool in tools:
        endpoint_name = tool.name
        endpoint_description = tool.description
        schema = tool.inputSchema

        # Build Pydantic model
        model_fields = {}
        required_fields = schema.get("required", [])
        for param_name, param_schema in schema["properties"].items():
            param_type = param_schema.get("type", "string")
            param_desc = param_schema.get("description", "")
            python_type = get_python_type(param_type)
            default_value = ... if param_name in required_fields else None
            model_fields[param_name] = (
                python_type,
                Body(default_value, description=param_desc),
            )

        FormModel = create_model(f"{endpoint_name}_form_model", **model_fields)

        def make_endpoint_func(endpoint_name: str, FormModel, session: ClientSession):
            async def tool_endpoint(form_data: FormModel):
                args = form_data.model_dump()
                print(f"Calling {endpoint_name} with arguments:", args)
                result = await session.call_tool(endpoint_name, arguments=args)
                response = []
                for content in result.content:
                    text = content.text
                    if isinstance(text, str):
                        try:
                            text = json.loads(text)
                        except json.JSONDecodeError:
                            pass
                    response.append(text)
                return response

            return tool_endpoint

        tool = make_endpoint_func(endpoint_name, FormModel, session)

        app.post(
            f"/{endpoint_name}",
            summary=endpoint_name.replace("_", " ").title(),
            description=endpoint_description,
        )(tool)


@asynccontextmanager
async def lifespan(app: FastAPI):
    command = getattr(app.state, "command", None)
    args = getattr(app.state, "args", [])
    env = getattr(app.state, "env", {})

    if not command:
        async with AsyncExitStack() as stack:
            for route in app.routes:
                if isinstance(route, Mount) and isinstance(route.app, FastAPI):
                    await stack.enter_async_context(
                        route.app.router.lifespan_context(route.app),  # noqa
                    )
            yield

    else:
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env={**env},
        )

        async with stdio_client(server_params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                app.state.session = session
                await create_dynamic_endpoints(app)
                yield


async def run(host: str = "127.0.0.1", port: int = None, **kwargs):
    if port is None:
        port = int(os.getenv("UVICORN_PORT", 8000))  # Use PORT from env, or fallback to 8000
    config_path = kwargs.get("config")
    server_command = kwargs.get("server_command")
    name = kwargs.get("name") or "MCP OpenAPI Proxy"
    description = (
        kwargs.get("description") or "Automatically generated API from MCP Tool Schemas"
    )
    version = kwargs.get("version") or "1.0"

    main_app = FastAPI(
        title=name, description=description, version=version, lifespan=lifespan
    )

    main_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if server_command:
        main_app.state.command = server_command[0]
        main_app.state.args = server_command[1:]
        main_app.state.env = os.environ.copy()
    elif config_path:
        with open(config_path, "r") as f:
            config_data = json.load(f)
        mcp_servers = config_data.get("mcpServers", {})

        if not mcp_servers:
            raise ValueError("No 'mcpServers' found in config file.")

        for server_name, server_cfg in mcp_servers.items():
            sub_app = FastAPI(
                title=f"{server_name}",
                description=f"{server_name} MCP Server",
                version="1.0",
                lifespan=lifespan,
            )

            sub_app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            sub_app.state.command = server_cfg["command"]
            sub_app.state.args = server_cfg.get("args", [])
            sub_app.state.env = {**os.environ, **server_cfg.get("env", {})}

            main_app.mount(f"/{server_name}", sub_app)

    else:
        raise ValueError("You must provide either server_command or config.")
    log_level = os.getenv("UVICORN_LOG_LEVEL", "info")  # Use UVICORN_LOG_LEVEL from env, or fallback to info
    config = uvicorn.Config(app=main_app, host=host, port=port, log_level=log_level)
    server = uvicorn.Server(config)

    await server.serve()
