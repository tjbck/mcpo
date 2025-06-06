import asyncio
import json
from typing import Any, Dict, ForwardRef, List, Optional, Type, Union

from fastapi import HTTPException

from mcp import ClientSession, types
from mcp.types import (
    CallToolResult,
    PARSE_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)

from mcp.shared.exceptions import McpError

from pydantic import Field, create_model
from pydantic.fields import FieldInfo

MCP_ERROR_TO_HTTP_STATUS = {
    PARSE_ERROR: 400,
    INVALID_REQUEST: 400,
    METHOD_NOT_FOUND: 404,
    INVALID_PARAMS: 422,
    INTERNAL_ERROR: 500,
}


async def wait_list_tools(session, timeout: int = 15, interval: int = 1) -> list:
    """
    Polls the MCP server until the tool list is available or timeout is reached.
    Raises TimeoutError if tools are not available within the timeout.
    """
    start = asyncio.get_event_loop().time()
    last_error = None
    while True:
        try:
            tools_result = await session.list_tools()
            if getattr(tools_result, "tools", None):
                return tools_result
        except Exception as e:
            last_error = e
        if asyncio.get_event_loop().time() - start > timeout:
            raise TimeoutError(f"Timed out waiting for MCP server tools: {last_error}")
        await asyncio.sleep(interval)


def process_tool_response(result: CallToolResult) -> list:
    """Universal response processor for all tool endpoints"""
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
            image_data = f"data:{content.mimeType};base64,{content.data}"
            response.append(image_data)
        elif isinstance(content, types.EmbeddedResource):
            # TODO: Handle embedded resources
            response.append("Embedded resource not supported yet.")
    return response


def _process_schema_property(
    _model_cache: Dict[str, Type],
    prop_schema: Dict[str, Any],
    model_name_prefix: str,
    prop_name: str,
    is_required: bool,
    schema_defs: Optional[Dict] = None,
) -> tuple[Union[Type, List, ForwardRef, Any], FieldInfo]:
    """
    Recursively processes a schema property to determine its Python type hint
    and Pydantic Field definition.

    Returns:
        A tuple containing (python_type_hint, pydantic_field).
        The pydantic_field contains default value and description.
    """
    if "$ref" in prop_schema:
        ref = prop_schema["$ref"]
        ref = ref.split("/")[-1]
        assert ref in schema_defs, "Custom field not found"
        prop_schema = schema_defs[ref]

    prop_type = prop_schema.get("type")
    prop_desc = prop_schema.get("description", "")

    default_value = ... if is_required else prop_schema.get("default", None)
    pydantic_field = Field(default=default_value, description=prop_desc)

    # Handle the case where prop_type is missing but 'anyOf' key exists
    # In this case, use data type from 'anyOf' to determine the type hint
    if "anyOf" in prop_schema:
        type_hints = []
        for i, schema_option in enumerate(prop_schema["anyOf"]):
            type_hint, _ = _process_schema_property(
                _model_cache,
                schema_option,
                f"{model_name_prefix}_{prop_name}",
                f"choice_{i}",
                False,
            )
            type_hints.append(type_hint)
        return Union[tuple(type_hints)], pydantic_field

    # Handle the case where prop_type is a list of types, e.g. ['string', 'number']
    if isinstance(prop_type, list):
        # Create a Union of all the types
        type_hints = []
        for type_option in prop_type:
            # Create a temporary schema with the single type and process it
            temp_schema = dict(prop_schema)
            temp_schema["type"] = type_option
            type_hint, _ = _process_schema_property(
                _model_cache, temp_schema, model_name_prefix, prop_name, False
            )
            type_hints.append(type_hint)

        # Return a Union of all possible types
        return Union[tuple(type_hints)], pydantic_field

    if prop_type == "object":
        nested_properties = prop_schema.get("properties", {})
        nested_required = prop_schema.get("required", [])
        nested_fields = {}

        nested_model_name = f"{model_name_prefix}_{prop_name}_model".replace(
            "__", "_"
        ).rstrip("_")

        if nested_model_name in _model_cache:
            return _model_cache[nested_model_name], pydantic_field

        for name, schema in nested_properties.items():
            is_nested_required = name in nested_required
            nested_type_hint, nested_pydantic_field = _process_schema_property(
                _model_cache,
                schema,
                nested_model_name,
                name,
                is_nested_required,
                schema_defs,
            )

            nested_fields[name] = (nested_type_hint, nested_pydantic_field)

        if not nested_fields:
            return Dict[str, Any], pydantic_field

        NestedModel = create_model(nested_model_name, **nested_fields)
        _model_cache[nested_model_name] = NestedModel

        return NestedModel, pydantic_field

    elif prop_type == "array":
        items_schema = prop_schema.get("items")
        if not items_schema:
            # Default to list of anything if items schema is missing
            return List[Any], pydantic_field

        # Recursively determine the type of items in the array
        item_type_hint, _ = _process_schema_property(
            _model_cache,
            items_schema,
            f"{model_name_prefix}_{prop_name}",
            "item",
            False,  # Items aren't required at this level,
            schema_defs,
        )
        list_type_hint = List[item_type_hint]
        return list_type_hint, pydantic_field

    elif prop_type == "string":
        return str, pydantic_field
    elif prop_type == "integer":
        return int, pydantic_field
    elif prop_type == "boolean":
        return bool, pydantic_field
    elif prop_type == "number":
        return float, pydantic_field
    elif prop_type == "null":
        return None, pydantic_field
    else:
        return Any, pydantic_field


def get_model_fields(form_model_name, properties, required_fields, schema_defs=None):
    model_fields = {}

    _model_cache: Dict[str, Type] = {}

    for param_name, param_schema in properties.items():
        is_required = param_name in required_fields
        python_type_hint, pydantic_field_info = _process_schema_property(
            _model_cache,
            param_schema,
            form_model_name,
            param_name,
            is_required,
            schema_defs,
        )
        # Use the generated type hint and Field info
        model_fields[param_name] = (python_type_hint, pydantic_field_info)
    return model_fields


def get_tool_handler(
    session,
    endpoint_name,
    form_model_fields,
    response_model_fields=None,
):
    if form_model_fields:
        FormModel = create_model(f"{endpoint_name}_form_model", **form_model_fields)
        ResponseModel = (
            create_model(f"{endpoint_name}_response_model", **response_model_fields)
            if response_model_fields
            else Any
        )

        def make_endpoint_func(
            endpoint_name: str, FormModel, session: ClientSession
        ):  # Parameterized endpoint
            async def tool(form_data: FormModel) -> ResponseModel:
                args = form_data.model_dump(exclude_none=True)
                print(f"Calling endpoint: {endpoint_name}, with args: {args}")
                try:
                    result = await session.call_tool(endpoint_name, arguments=args)

                    if result.isError:
                        error_message = "Unknown tool execution error"
                        error_data = None  # Initialize error_data
                        if result.content:
                            if isinstance(result.content[0], types.TextContent):
                                error_message = result.content[0].text
                        detail = {"message": error_message}
                        if error_data is not None:
                            detail["data"] = error_data
                        raise HTTPException(
                            status_code=500,
                            detail=detail,
                        )

                    response_data = process_tool_response(result)
                    final_response = (
                        response_data[0] if len(response_data) == 1 else response_data
                    )
                    return final_response

                except McpError as e:
                    print(f"MCP Error calling {endpoint_name}: {e.error}")
                    status_code = MCP_ERROR_TO_HTTP_STATUS.get(e.error.code, 500)
                    raise HTTPException(
                        status_code=status_code,
                        detail=(
                            {"message": e.error.message, "data": e.error.data}
                            if e.error.data is not None
                            else {"message": e.error.message}
                        ),
                    )
                except Exception as e:
                    print(f"Unexpected error calling {endpoint_name}: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail={"message": "Unexpected error", "error": str(e)},
                    )

            return tool

        tool_handler = make_endpoint_func(endpoint_name, FormModel, session)
    else:

        def make_endpoint_func_no_args(
            endpoint_name: str, session: ClientSession
        ):  # Parameterless endpoint
            async def tool():  # No parameters
                print(f"Calling endpoint: {endpoint_name}, with no args")
                try:
                    result = await session.call_tool(
                        endpoint_name, arguments={}
                    )  # Empty dict

                    if result.isError:
                        error_message = "Unknown tool execution error"
                        if result.content:
                            if isinstance(result.content[0], types.TextContent):
                                error_message = result.content[0].text
                        detail = {"message": error_message}
                        raise HTTPException(
                            status_code=500,
                            detail=detail,
                        )

                    response_data = process_tool_response(result)
                    final_response = (
                        response_data[0] if len(response_data) == 1 else response_data
                    )
                    return final_response

                except McpError as e:
                    print(f"MCP Error calling {endpoint_name}: {e.error}")
                    status_code = MCP_ERROR_TO_HTTP_STATUS.get(e.error.code, 500)
                    # Propagate the error received from MCP as an HTTP exception
                    raise HTTPException(
                        status_code=status_code,
                        detail=(
                            {"message": e.error.message, "data": e.error.data}
                            if e.error.data is not None
                            else {"message": e.error.message}
                        ),
                    )
                except Exception as e:
                    print(f"Unexpected error calling {endpoint_name}: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail={"message": "Unexpected error", "error": str(e)},
                    )

            return tool

        tool_handler = make_endpoint_func_no_args(endpoint_name, session)

    return tool_handler
