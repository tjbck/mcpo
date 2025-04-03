from fastapi import FastAPI, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from pydantic import create_model
from contextlib import AsyncExitStack, asynccontextmanager

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from typing import Dict, Any, Optional
import uvicorn
import json
import os

from mcpo.utils.auth import get_verify_api_key


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


async def create_dynamic_endpoints(app: FastAPI, api_dependency=None):
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
                args = form_data.model_dump(exclude_none=True)
                print(f"Calling {endpoint_name} with arguments:", args)
                result = await session.call_tool(endpoint_name, arguments=args)
                response = []
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        text = content.text
                        if isinstance(text, str):
                            try:
                                text = json.loads(text)
                            except json.JSONDecodeError:
                                pass
                        response.append(text)
                    elif isinstance(content, types.ImageContent):
                        image_data = content.data
                        image_data = f"data:{content.mimeType};base64,{image_data}"
                        response.append(image_data)
                    elif isinstance(content, types.EmbeddedResource):
                        # TODO: Handle embedded resources
                        response.append("Embedded resource not supported yet.")

                return response

            return tool_endpoint

        tool = make_endpoint_func(endpoint_name, FormModel, session)

        app.post(
            f"/{endpoint_name}",
            summary=endpoint_name.replace("_", " ").title(),
            description=endpoint_description,
            dependencies=[Depends(api_dependency)] if api_dependency else [],
        )(tool)


@asynccontextmanager
async def lifespan(app: FastAPI):
    command = getattr(app.state, "command", None)
    args = getattr(app.state, "args", [])
    env = getattr(app.state, "env", {})

    api_dependency = getattr(app.state, "api_dependency", None)

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
                await create_dynamic_endpoints(app, api_dependency=api_dependency)
                yield


async def run(
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: Optional[str] = "",
    cors_allow_origins=["*"],
    **kwargs,
):

    # Server API Key
    api_dependency = get_verify_api_key(api_key) if api_key else None

    # MCP Config
    config_path = kwargs.get("config")
    server_command = kwargs.get("server_command")
    name = kwargs.get("name") or "MCP OpenAPI Proxy"
    description = (
        kwargs.get("description") or "Automatically generated API from MCP Tool Schemas"
    )
    version = kwargs.get("version") or "1.0"
    ssl_certfile = kwargs.get("ssl_certfile")
    ssl_keyfile= kwargs.get("ssl_keyfile")

    main_app = FastAPI(
        title=name, description=description, version=version, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile, lifespan=lifespan
    )

    main_app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if server_command:

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

        for server_name, server_cfg in mcp_servers.items():
            sub_app = FastAPI(
                title=f"{server_name}",
                description=f"{server_name} MCP Server",
                version="1.0",
                lifespan=lifespan,
            )

            sub_app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_allow_origins or ["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            sub_app.state.command = server_cfg["command"]
            sub_app.state.args = server_cfg.get("args", [])
            sub_app.state.env = {**os.environ, **server_cfg.get("env", {})}

            sub_app.state.api_dependency = api_dependency

            main_app.mount(f"/{server_name}", sub_app)

    else:
        raise ValueError("You must provide either server_command or config.")

    config = uvicorn.Config(app=main_app, host=host, port=port, ssl_certfile=ssl_certfile , ssl_keyfile=ssl_keyfile ,log_level="info")
    server = uvicorn.Server(config)

    await server.serve()
