from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from pydantic import create_model, Field
from contextlib import AsyncExitStack, asynccontextmanager

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from typing import Dict, Any, Callable
import uvicorn
import json
import os
import asyncio


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


def handle_union_schema(schema: Dict[str, Any], tool_name: str = "") -> Dict[str, Any]:
    """Handle anyOf/oneOf schemas by flattening them for FastAPI/Pydantic"""
    if "anyOf" in schema or "oneOf" in schema:
        union_key = "anyOf" if "anyOf" in schema else "oneOf"
        union_types = schema[union_key]
        
        flattened = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        discriminator = schema.get("discriminator", {}).get("propertyName")
        if not discriminator:
            potential_discriminators = {}
            
            for variant in union_types:
                if variant.get("type") == "object" and "properties" in variant:
                    for prop_name, prop_schema in variant.get("properties", {}).items():
                        if "const" in prop_schema:
                            potential_discriminators.setdefault(prop_name, 0)
                            potential_discriminators[prop_name] += 1
            
            if potential_discriminators:
                max_count = max(potential_discriminators.values())
                for prop_name, count in potential_discriminators.items():
                    if count == max_count:
                        discriminator = prop_name
                        break
        
        variant_map = {}
        discriminator_values = []
        
        for variant in union_types:
            if variant.get("type") == "object" and "properties" in variant:
                for prop_name, prop_schema in variant.get("properties", {}).items():
                    if prop_name not in flattened["properties"] or prop_name == discriminator:
                        flattened["properties"][prop_name] = prop_schema.copy()
                
                if discriminator and discriminator in variant.get("properties", {}):
                    disc_prop = variant["properties"][discriminator]
                    if "const" in disc_prop:
                        disc_value = disc_prop["const"]
                        discriminator_values.append(disc_value)
                        variant_map[disc_value] = variant
        
        if discriminator and discriminator_values and discriminator in flattened["properties"]:
            flattened["properties"][discriminator] = {
                "type": "string",
                "enum": discriminator_values,
                "description": flattened["properties"][discriminator].get("description", f"Operation type: {', '.join(discriminator_values)}")
            }
            
            if discriminator not in flattened["required"]:
                flattened["required"].append(discriminator)
                
            flattened["x-enumValueMappings"] = {}
            
            for value, variant in variant_map.items():
                variant_required = variant.get("required", [])
                if variant_required:
                    flattened["x-enumValueMappings"][value] = {
                        "required": variant_required
                    }
        
        if "description" in schema:
            flattened["description"] = schema["description"]
            
        return flattened
    
    # If not a union schema, return the original schema
    return schema


async def create_dynamic_endpoints(app: FastAPI):
    session = app.state.session
    if not session:
        raise ValueError("Session is not initialized in the app state.")

    try:
        result = await session.initialize()
        
        server_info = getattr(result, "serverInfo", None)
        if server_info:
            app.title = server_info.name or app.title
            app.description = (
                f"{server_info.name} MCP Server" if server_info.name else app.description
            )
            app.version = server_info.version or app.version
    except Exception as e:
        raise ValueError(f"Error initializing MCP session: {str(e)}")

    try:
        tools_result = await session.list_tools()
        tools = tools_result.tools
    except Exception as e:
        raise ValueError(f"Error listing tools: {str(e)}")

    for tool in tools:
        try:
            endpoint_name = tool.name
            endpoint_description = tool.description
            original_schema = tool.inputSchema
            
            schema = handle_union_schema(original_schema, endpoint_name)
            
			# Build Pydantic model
            model_fields = {}
            required_fields = schema.get("required", [])
            discriminator = None
            enum_mappings = schema.get("x-enumValueMappings", {})
            
            for param_name, param_schema in schema.get("properties", {}).items():
                if "enum" in param_schema and param_name in required_fields:
                    discriminator = param_name
                    break
            
            for param_name, param_schema in schema.get("properties", {}).items():
                param_type = param_schema.get("type", "string")
                param_desc = param_schema.get("description", "")
                python_type = get_python_type(param_type)
                
                if "enum" in param_schema:
                    enum_values = param_schema["enum"]
                    param_desc += f" Allowed values: {', '.join(map(str, enum_values))}"
                
                is_required = param_name in required_fields
                    
                if "const" in param_schema:
                    const_value = param_schema["const"]
                    model_fields[param_name] = (
                        python_type,
                        Field(default=const_value, description=param_desc),
                    )
                else:
                    default_value = ... if is_required else None
                    model_fields[param_name] = (
                        python_type,
                        Field(default=default_value, description=param_desc),
                    )

            if not model_fields:
                model_fields = {
                    "params": (Dict[str, Any], Field(default=..., description="Tool parameters"))
                }

            FormModel = create_model(f"{endpoint_name}_form_model", **model_fields)

            def make_endpoint_func(endpoint_name: str, FormModel, session: ClientSession, enum_mappings=None, discriminator=None):
                async def tool_endpoint(form_data: FormModel):
                    # Convert form_data to dict for sending to MCP
                    if hasattr(form_data, "params") and len(model_fields) == 1 and "params" in model_fields:
                        args = form_data.params
                    else:
                        args = form_data.model_dump(exclude_unset=True)
                    
                    if discriminator and discriminator in args and enum_mappings:
                        disc_value = args[discriminator]
                        if disc_value in enum_mappings:
                            mapping = enum_mappings[disc_value]
                            required_fields = mapping.get("required", [])
                            
                            missing = [field for field in required_fields if field not in args]
                            if missing:
                                error_msg = f"When operation is '{disc_value}', the following fields are required: {', '.join(missing)}"
                                return [{"success": False, "error": error_msg, "guidance": f"Please provide values for: {', '.join(missing)}"}]
                    
                    try:
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
                    except Exception as e:
                        return [{"success": False, "error": str(e), "guidance": "Please check your parameters and try again."}]

                return tool_endpoint

            tool = make_endpoint_func(endpoint_name, FormModel, session, enum_mappings, discriminator)

            app.post(
                f"/{endpoint_name}",
                summary=endpoint_name.replace("_", " ").title(),
                description=endpoint_description,
            )(tool)
        except Exception:
            continue


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

        try:
            async with stdio_client(server_params) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    app.state.session = session
                    
                    try:
                        await asyncio.wait_for(create_dynamic_endpoints(app), timeout=30)
                    except asyncio.TimeoutError:
                        pass
                    
                    yield
        except Exception:
            yield


async def run(host: str = "127.0.0.1", port: int = 8000, **kwargs):
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

    config = uvicorn.Config(app=main_app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    await server.serve()
