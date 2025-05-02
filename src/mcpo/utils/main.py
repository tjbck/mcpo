import json
import logging
from typing import Any, Dict, ForwardRef, List, Optional, Type, Union, Tuple, get_type_hints

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

from pydantic import Field, create_model, BaseModel
from pydantic.fields import FieldInfo

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MCP_ERROR_TO_HTTP_STATUS = {
    PARSE_ERROR: 400,
    INVALID_REQUEST: 400,
    METHOD_NOT_FOUND: 404,
    INVALID_PARAMS: 422,
    INTERNAL_ERROR: 500,
}

_model_cache = {}

def _process_schema_property(
        model_cache: Dict[str, Any],
        schema: Dict[str, Any],
        model_name: str,
        field_name: str,
        is_required: bool,
        schema_defs: Optional[Dict[str, Any]] = None,
        _processing_refs: Optional[set] = None
) -> Tuple[Any, FieldInfo]:
    """Process a schema property and return its type information and field metadata."""
    if _processing_refs is None:
        _processing_refs = set()

    field_kwargs = {
        "json_schema_extra": {"metadata": {}},
        "description": "",
    }
    if not is_required:
        field_kwargs["default"] = None
    if "description" in schema:
        field_kwargs["description"] = schema["description"]

    # Handle $ref
    if "$ref" in schema:
        ref_path = schema["$ref"]
        if ref_path.startswith("#/definitions/") or ref_path.startswith("#/$defs/"):
            ref_name = ref_path.split("/")[-1]

            # Handle circular references
            if ref_name in _processing_refs:
                return ForwardRef(ref_name), Field(**field_kwargs)

            # Handle special types
            if ref_name.startswith("_"):
                return Dict[str, Any], Field(**field_kwargs)

            # Handle schema definitions
            if schema_defs is not None and ref_name in schema_defs:
                _processing_refs.add(ref_name)
                ref_schema = schema_defs[ref_name]

                # Check model cache
                if ref_name in model_cache:
                    if model_cache[ref_name] is None:  # Circular reference detected
                        _processing_refs.remove(ref_name)
                        return ForwardRef(ref_name), Field(**field_kwargs)
                    return model_cache[ref_name], Field(**field_kwargs)

                try:
                    model_cache[ref_name] = None  # Placeholder for circular references
                    type_info, _ = _process_schema_property(
                        model_cache,
                        ref_schema,
                        ref_name,
                        field_name,
                        is_required,
                        schema_defs,
                        _processing_refs
                    )
                    _processing_refs.remove(ref_name)

                    if isinstance(type_info, ForwardRef):
                        type_info = create_model(ref_name, __base__=BaseModel)
                        model_cache[ref_name] = type_info

                        # Process fields for the model
                        fields = {}
                        required_fields = ref_schema.get("required", [])
                        for prop_name, prop_schema in ref_schema.get("properties", {}).items():
                            prop_type, prop_field = _process_schema_property(
                                model_cache,
                                prop_schema,
                                ref_name,
                                prop_name,
                                prop_name in required_fields,
                                schema_defs,
                                set()  # Use a new set to allow self-references
                            )
                            fields[prop_name] = (prop_type, prop_field)

                        # Update model fields
                        for name, (type_, field) in fields.items():
                            type_info.model_fields[name] = field
                            type_info.model_fields[name].annotation = type_

                    model_cache[ref_name] = type_info
                    return type_info, Field(**field_kwargs)
                except Exception as e:
                    logger.error(f"Error processing reference '{ref_name}': {str(e)}")
                    _processing_refs.remove(ref_name)
                    return Dict[str, Any], Field(**field_kwargs)
            else:
                # If we can't find the reference or schema_defs is None, use Dict[str, Any]
                return Dict[str, Any], Field(**field_kwargs)

    # Handle special types that might be used in the server's schema
    if "anyOf" in schema:
        # For special types like _Not, _And, _Or, etc.
        # Create a model that allows any of the special types
        special_types = {}
        types = []
        for i, sub_schema in enumerate(schema["anyOf"]):
            if "$ref" in sub_schema:
                ref_name = sub_schema["$ref"].split("/")[-1]
                if ref_name.startswith("_"):
                    special_types[ref_name] = Optional[Dict[str, Any]]
                else:
                    type_info, _ = _process_schema_property(
                        model_cache,
                        sub_schema,
                        model_name,
                        field_name,
                        True,
                        schema_defs,
                        _processing_refs
                    )
                    types.append(type_info)
            elif "type" in sub_schema:
                if sub_schema["type"] == "null":
                    types.append(None)
                elif sub_schema["type"] == "string":
                    if "enum" in sub_schema:
                        types.append(str)
                    else:
                        types.append(str)
                elif sub_schema["type"] == "integer":
                    types.append(int)
                elif sub_schema["type"] == "number":
                    types.append(float)
                elif sub_schema["type"] == "boolean":
                    types.append(bool)
                elif sub_schema["type"] == "object":
                    # Create a unique model name for each object type
                    unique_model_name = f"{model_name}_{field_name}_type_{i}"
                    type_info, _ = _process_schema_property(
                        model_cache,
                        sub_schema,
                        unique_model_name,
                        field_name,
                        True,
                        schema_defs,
                        _processing_refs
                    )
                    types.append(type_info)
                elif sub_schema["type"] == "array":
                    type_info, _ = _process_schema_property(
                        model_cache,
                        sub_schema,
                        model_name,
                        field_name,
                        True,
                        schema_defs,
                        _processing_refs
                    )
                    types.append(type_info)
            else:
                type_info, _ = _process_schema_property(
                    model_cache,
                    sub_schema,
                    model_name,
                    field_name,
                    True,
                    schema_defs,
                    _processing_refs
                )
                types.append(type_info)

        if special_types:
            special_model = create_model(
                f"{model_name}_{field_name}_model",
                **special_types,
                __base__=BaseModel
            )
            return special_model, Field(**field_kwargs)
        elif types:
            # Remove duplicates while preserving order
            unique_types = []
            for t in types:
                if t not in unique_types:
                    unique_types.append(t)
            if len(unique_types) == 1:
                return unique_types[0], Field(**field_kwargs)
            return Union[tuple(unique_types)], Field(**field_kwargs)

    # Handle special types
    if "type" in schema and schema["type"] == "object" and "properties" not in schema:
        # This is likely a custom type
        return Dict[str, Any], Field(**field_kwargs)

    # Handle special filter types
    if field_name == "filters":
        # Match GraphQL: orFilters: [AndFilterInput!]
        AndFilterInput = create_model(
            "AndFilterInput",
            field=(str, ...),
            condition=(str, ...),
            values=(List[str], ...),
            __base__=BaseModel
        )
        return List[AndFilterInput], Field(**field_kwargs)

    # Handle schema composition
    if "allOf" in schema:
        # Merge all schemas in allOf
        merged_schema = {"type": "object", "properties": {}, "required": []}
        for sub_schema in schema["allOf"]:
            if "properties" in sub_schema:
                merged_schema["properties"].update(sub_schema["properties"])
            if "required" in sub_schema:
                merged_schema["required"].extend(sub_schema["required"])
        return _process_schema_property(
            model_cache,
            merged_schema,
            model_name,
            field_name,
            is_required,
            schema_defs,
            _processing_refs
        )

    if "anyOf" in schema or "oneOf" in schema:
        # Create a union of all possible types
        sub_schemas = schema.get("anyOf", []) or schema.get("oneOf", [])
        types = []
        for i, sub_schema in enumerate(sub_schemas):
            type_info, _ = _process_schema_property(
                model_cache,
                sub_schema,
                f"{model_name}_{field_name}_{i}",
                "item",
                True,
                schema_defs,
                _processing_refs
            )
            if type_info not in types:  # Avoid duplicate types
                types.append(type_info)
        return Union[tuple(types)], Field(**field_kwargs)

    # Handle primitive types
    property_type = schema.get("type")
    if isinstance(property_type, list):
        # Handle multiple types
        types = []
        for t in property_type:
            if t == "string":
                types.append(str)
            elif t == "integer":
                types.append(int)
            elif t == "number":
                types.append(float)
            elif t == "boolean":
                types.append(bool)
            elif t == "null":
                types.append(None)
            elif t == "array":
                types.append(List[Any])
            elif t == "object":
                types.append(Dict[str, Any])
        return Union[tuple(types)], Field(**field_kwargs)

    if property_type == "string":
        if "minLength" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["min_length"] = schema["minLength"]
        if "maxLength" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["max_length"] = schema["maxLength"]
        if "pattern" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["pattern"] = schema["pattern"]
        if "format" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["format"] = schema["format"]
        if "enum" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["enum"] = schema["enum"]
        if "const" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["const"] = schema["const"]
        if "default" in schema:
            field_kwargs["default"] = schema["default"]
        return str, Field(**field_kwargs)

    elif property_type == "integer":
        if "minimum" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["minimum"] = schema["minimum"]
        if "maximum" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["maximum"] = schema["maximum"]
        if "exclusiveMinimum" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["exclusive_minimum"] = schema["exclusiveMinimum"]
        if "exclusiveMaximum" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["exclusive_maximum"] = schema["exclusiveMaximum"]
        if "multipleOf" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["multiple_of"] = schema["multipleOf"]
        if "default" in schema:
            field_kwargs["default"] = schema["default"]
        return int, Field(**field_kwargs)

    elif property_type == "number":
        if "minimum" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["minimum"] = schema["minimum"]
        if "maximum" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["maximum"] = schema["maximum"]
        if "exclusiveMinimum" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["exclusive_minimum"] = schema["exclusiveMinimum"]
        if "exclusiveMaximum" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["exclusive_maximum"] = schema["exclusiveMaximum"]
        if "multipleOf" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["multiple_of"] = schema["multipleOf"]
        if "default" in schema:
            field_kwargs["default"] = schema["default"]
        return float, Field(**field_kwargs)

    elif property_type == "boolean":
        if "default" in schema:
            field_kwargs["default"] = schema["default"]
        return bool, Field(**field_kwargs)

    elif property_type == "null":
        return None, Field(**field_kwargs)

    elif property_type == "array":
        if "items" in schema:
            item_type, _ = _process_schema_property(
                model_cache,
                schema["items"],
                model_name,
                field_name,
                True,
                schema_defs,
                _processing_refs
            )
            if "minItems" in schema:
                field_kwargs["json_schema_extra"]["metadata"]["min_items"] = schema["minItems"]
            if "maxItems" in schema:
                field_kwargs["json_schema_extra"]["metadata"]["max_items"] = schema["maxItems"]
            if "uniqueItems" in schema:
                field_kwargs["json_schema_extra"]["metadata"]["unique_items"] = schema["uniqueItems"]
            if "default" in schema:
                field_kwargs["default"] = schema["default"]
            return List[item_type], Field(**field_kwargs)
        return List[Any], Field(**field_kwargs)

    elif property_type == "object":
        if not schema.get("properties"):
            return Dict[str, Any], Field(**field_kwargs)

        # Generate a unique model name
        full_model_name = f"{model_name}_{field_name}_model"
        if full_model_name in model_cache:
            return model_cache[full_model_name], Field(**field_kwargs)

        fields = {}
        required_fields = schema.get("required", [])
        for prop_name, prop_schema in schema["properties"].items():
            prop_type, prop_field = _process_schema_property(
                model_cache,
                prop_schema,
                model_name,
                prop_name,
                prop_name in required_fields,
                schema_defs,
                _processing_refs
            )
            fields[prop_name] = (prop_type, prop_field)

        if "minProperties" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["min_properties"] = schema["minProperties"]
        if "maxProperties" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["max_properties"] = schema["maxProperties"]
        if "additionalProperties" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["additional_properties"] = schema["additionalProperties"]
        if "default" in schema:
            field_kwargs["default"] = schema["default"]

        model = create_model(full_model_name, **fields)
        model_cache[full_model_name] = model
        return model, Field(**field_kwargs)

    # Default to Any for unknown types
    return Any, Field(**field_kwargs)

def get_model_fields(
        name: str,
        properties: Dict[str, Any],
        required: List[str],
        schema_defs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[Type, Field]]:
    """Get Pydantic model fields from a schema."""
    fields = {}
    for prop_name, prop_schema in properties.items():
        try:
            type_hint, field_info = _process_schema_property(
                _model_cache,
                prop_schema,
                name,
                prop_name,
                prop_name in required,
                schema_defs,
                )
            fields[prop_name] = (type_hint, field_info)
        except Exception as e:
            logger.error(f"Error processing parameter {prop_name}: {str(e)}")
            continue

    return fields

def get_tool_handler(
        session: Any,
        endpoint_name: str,
        form_model_fields: Dict[str, Tuple[Type, Field]],
        response_model_fields: Optional[Dict[str, Tuple[Type, Field]]] = None,
):
    """Get a FastAPI endpoint handler for a tool."""
    # Create form model with proper field definitions
    form_model = create_model(
        f"{endpoint_name}_form_model",
        __base__=BaseModel,
        **{k: (v[0], v[1]) for k, v in form_model_fields.items()}
    )
    form_model.model_rebuild()

    # Create response model if fields are provided
    response_model = None
    if response_model_fields:
        response_model = create_model(
            f"{endpoint_name}_response_model",
            __base__=BaseModel,
            **{k: (v[0], v[1]) for k, v in response_model_fields.items()}
        )
        response_model.model_rebuild()

    async def handler(form_data: form_model):
        try:
            # Transform filters for search endpoint
            params = form_data.dict()

            logger.debug(f"Executing tool {endpoint_name} with params: {params}")
            result = await session.execute_tool(endpoint_name, params)
            logger.debug(f"Tool {endpoint_name} result type: {type(result)}")
            logger.debug(f"Tool {endpoint_name} result: {result}")

            if isinstance(result, CallToolResult):
                logger.debug(f"CallToolResult fields: {result.model_fields}")
                logger.debug(f"CallToolResult isError: {getattr(result, 'isError', None)}")
                logger.debug(f"CallToolResult content: {getattr(result, 'content', None)}")
                logger.debug(f"CallToolResult meta: {getattr(result, 'meta', None)}")

                if result.isError:
                    raise HTTPException(status_code=400, detail="Tool execution failed")
                # Return the raw result for now, let the tool handle its own response format
                return result
            return result
        except Exception as e:
            logger.error(f"Error executing tool {endpoint_name}: {str(e)}", exc_info=True)
            raise

    handler.__name__ = endpoint_name
    return handler

async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
    """Execute a tool with the given parameters."""
    try:
        logger.debug(f"Calling tool {tool_name} with params: {params}")
        result = await self.call_tool(tool_name, params)
        logger.debug(f"Tool {tool_name} call result type: {type(result)}")
        logger.debug(f"Tool {tool_name} call result: {result}")

        if isinstance(result, CallToolResult):
            logger.debug(f"CallToolResult fields: {result.model_fields}")
            logger.debug(f"CallToolResult isError: {getattr(result, 'isError', None)}")
            logger.debug(f"CallToolResult content: {getattr(result, 'content', None)}")
            logger.debug(f"CallToolResult meta: {getattr(result, 'meta', None)}")
            # Return the result directly if it's a CallToolResult
            return result
        return result
    except McpError as e:
        status_code = MCP_ERROR_TO_HTTP_STATUS.get(e.code, 500)
        raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Monkey patch the ClientSession class to add the execute_tool method
ClientSession.execute_tool = execute_tool