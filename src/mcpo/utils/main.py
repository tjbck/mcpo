from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Tuple, Set, Union, ForwardRef, Callable, Awaitable

from fastapi import HTTPException
from mcp import ClientSession
from mcp.types import (
    CallToolResult,
    PARSE_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)
from mcpo.utils.common_logging import logger
from pydantic import Field, create_model, BaseModel
from pydantic.fields import FieldInfo

MCP_ERROR_TO_HTTP_STATUS = {
    PARSE_ERROR: 400,
    INVALID_REQUEST: 400,
    METHOD_NOT_FOUND: 404,
    INVALID_PARAMS: 422,
    INTERNAL_ERROR: 500,
}
        
@dataclass
class ModelCache:
    """
    A simple cache for storing and retrieving dynamically created models
    during schema processing to avoid redundant model creation and handle
    circular references.
    """ 
    cache: Dict[str, Any] = field(default_factory=dict)

    def clear(self):
        self.cache.clear()
        
    def len(self):
        return len(self.cache)

    def get(self, key: str):
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        self.cache[key] = value

    def __contains__(self, key: str):
        return key in self.cache

    def __getitem__(self, key: str):
        return self.cache[key]


def _handle_all_of(schemas, model_cache, model_name, field_name, is_required, schema_defs, _processing_refs, field_kwargs) -> Tuple[Any, FieldInfo]:
    merged = {"type": "object", "properties": {}, "required": []}
    for sub in schemas:
        if "properties" in sub:
            merged["properties"].update(sub["properties"])
        if "required" in sub:
            merged["required"].extend(sub["required"])
    return _process_schema_property(
        model_cache, merged, model_name, field_name, is_required, schema_defs, _processing_refs
    )


def _handle_union(schemas, union_name, model_cache, field_name, schema_defs, _processing_refs, field_kwargs) -> Tuple[Any, FieldInfo]:
    types = []
    for i, sub_schema in enumerate(schemas):
        type_info, _ = _process_schema_property(
            model_cache, sub_schema, union_name, f"{field_name}_u{i}", True, schema_defs, _processing_refs
        )
        if type_info not in types:
            types.append(type_info)
    if len(types) == 1:
        return types[0], Field(**field_kwargs)
    return Union[tuple(types)], Field(**field_kwargs)


def _process_schema_property(
        model_cache: ModelCache,
        schema: Dict[str, Any],
        model_name: str,
        field_name: str,
        is_required: bool,
        schema_defs: Optional[Dict[str, Any]] = None,
        _processing_refs: Optional[Set[str]] = None
    ) -> Tuple[Any, FieldInfo]:
    """
    Recursively process a JSON schema property and return the corresponding
    Python type and Pydantic FieldInfo for use in dynamic model creation.

    Args:
        model_cache: ModelCache instance for caching models.
        schema: The JSON schema property definition.
        model_name: Name of the parent model.
        field_name: Name of the field being processed.
        is_required: Whether the field is required.
        schema_defs: Optional dictionary of schema definitions for $ref resolution.
        _processing_refs: Internal set to track references and avoid circular refs.

    Returns:
        Tuple of (type, FieldInfo) for use in Pydantic model creation.
    """
    if _processing_refs is None:
        _processing_refs = set()

    field_kwargs = {
        "json_schema_extra": {"metadata": {}},
        "description": schema.get("description", ""),
    }

    if "default" in schema:
        field_kwargs["default"] = schema["default"]
    elif not is_required:
        field_kwargs["default"] = None
    else:
        field_kwargs["default"] = ...

    def handle_ref(ref_path: str):
        ref_name = ref_path.split("/")[-1]
        if ref_name in _processing_refs:
            return ForwardRef(ref_name), Field(**field_kwargs)
        if ref_name.startswith("_"):
            return Dict[str, Any], Field(**field_kwargs)
        if schema_defs and ref_name in schema_defs:
            _processing_refs.add(ref_name)
            ref_schema = schema_defs[ref_name]
            if ref_name in model_cache and model_cache.get(ref_name) is not None:
                _processing_refs.remove(ref_name)
                return model_cache[ref_name], Field(**field_kwargs)
            model_cache.set(ref_name, None)  # placeholder for circular refs
            type_info, _ = _process_schema_property(
                model_cache, ref_schema, ref_name, ref_name, True, schema_defs, _processing_refs
            )
            _processing_refs.remove(ref_name)
            if isinstance(type_info, ForwardRef):
                type_info = create_model(ref_name, __base__=BaseModel)
            model_cache.set(ref_name, type_info)
            return type_info, Field(**field_kwargs)
        return Dict[str, Any], Field(**field_kwargs)

    if "$ref" in schema:
        return handle_ref(schema["$ref"])

    if "allOf" in schema:
        return _handle_all_of(
            schema["allOf"], model_cache, model_name, field_name, is_required, schema_defs, _processing_refs, field_kwargs
        )

    if "anyOf" in schema or "oneOf" in schema:
        sub_schemas = schema.get("anyOf") or schema.get("oneOf")
        return _handle_union(
            sub_schemas, f"{model_name}_{field_name}_union", model_cache, field_name, schema_defs, _processing_refs, field_kwargs
        )
    
    if schema.get("type") == "object":
        if not schema.get("properties"):
            return Dict[str, Any], Field(**field_kwargs)
        full_model_name = f"{model_name}_{field_name}_model"
        if full_model_name in model_cache and model_cache.get(full_model_name) is not None:
            return model_cache[full_model_name], Field(**field_kwargs)
        fields = {}
        required_fields = schema.get("required", [])
        for prop_name, prop_schema in schema["properties"].items():
            prop_type, prop_field = _process_schema_property(
                model_cache, prop_schema, full_model_name, prop_name, prop_name in required_fields, schema_defs, _processing_refs
            )
            fields[prop_name] = (prop_type, prop_field)
        model = create_model(full_model_name, **fields)
        model_cache.set(full_model_name, model)
        if "minProperties" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["min_properties"] = schema["minProperties"]
        if "maxProperties" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["max_properties"] = schema["maxProperties"]
        if "additionalProperties" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["additional_properties"] = schema["additionalProperties"]
        return model, Field(**field_kwargs)

    if schema.get("type") == "array":
        if "items" in schema:
            item_type, _ = _process_schema_property(
                model_cache, schema["items"], model_name, f"{field_name}_item", True, schema_defs, _processing_refs
            )
            type_info = List[item_type]
        else:
            type_info = List[Any]
        if "minItems" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["min_items"] = schema["minItems"]
        if "maxItems" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["max_items"] = schema["maxItems"]
        if "uniqueItems" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["unique_items"] = schema["uniqueItems"]
        return type_info, Field(**field_kwargs)

    t = schema.get("type")
    type_info = Any
    if t == "string":
        type_info = str
        if "minLength" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["min_length"] = schema["minLength"]
        if "maxLength" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["max_length"] = schema["maxLength"]
        if "pattern" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["pattern"] = schema["pattern"]
        if "format" in schema:
            field_kwargs["json_schema_extra"]["metadata"]["format"] = schema["format"]
    elif t == "integer":
        type_info = int
    elif t == "number":
        type_info = float
    elif t == "boolean":
        type_info = bool
    elif t == "null":
        type_info = None
    elif isinstance(t, list):
        types = []
        for typ in t:
            if typ == "string":
                types.append(str)
            elif typ == "integer":
                types.append(int)
            elif typ == "number":
                types.append(float)
            elif typ == "boolean":
                types.append(bool)
            elif typ == "null":
                types.append(type(None))
            elif typ == "array":
                types.append(List[Any])
            elif typ == "object":
                types.append(Dict[str, Any])
        if len(types) == 1:
            type_info = types[0]
        else:
            type_info = Union[tuple(types)]
    elif t is not None:
        type_info = Any

    if t in ("number", "integer"):
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

    if "enum" in schema:
        field_kwargs["json_schema_extra"]["metadata"]["enum"] = schema["enum"]
    if "const" in schema:
        field_kwargs["json_schema_extra"]["metadata"]["const"] = schema["const"]

    return type_info, Field(**field_kwargs)


def get_model_fields(
        name: str,
        properties: Dict[str, Any],
        required: List[str],
        schema_defs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[Type, Field]]:
    """
    Generate a dictionary of Pydantic model fields from a JSON schema.

    Args:
        name: Name of the model.
        properties: Dictionary of property schemas.
        required: List of required property names.
        schema_defs: Optional dictionary of schema definitions for $ref resolution.

    Returns:
        Dictionary mapping property names to (type, Field) tuples.
    """
    fields = {}
    model_cache = ModelCache()
    for prop_name, prop_schema in properties.items():
        try:
            type_hint, field_info = _process_schema_property(
                model_cache,
                prop_schema,
                name,
                prop_name,
                prop_name in required,
                schema_defs,
                )
            fields[prop_name] = (type_hint, field_info)
        except Exception as e:
            logger.error(f"Error processing parameter {prop_name}: {str(e)}", exc_info=True)
            continue

    return fields


def get_tool_handler(
        session: ClientSession,
        endpoint_name: str,
        form_model_fields: Dict[str, Tuple[Type, Field]],
        response_model_fields: Optional[Dict[str, Tuple[Type, Field]]] = None,
) -> Tuple[Callable[[BaseModel], Awaitable[Any]], Optional[Type[BaseModel]]]:
    """
    Dynamically creates a FastAPI endpoint handler and response model for a given tool.

    Args:
        session: The MCP client session used to execute the tool.
        endpoint_name: The name of the endpoint/tool.
        form_model_fields: Dictionary mapping request field names to (type, Field) tuples for the request model.
        response_model_fields: Optional dictionary mapping response field names to (type, Field) tuples for the response model.

    Returns:
        A tuple containing:
            - An async handler function for FastAPI, accepting the generated request model.
            - The generated Pydantic response model class, or None if not provided.
    """
    # Create form model with proper field definitions
    form_model = create_model(
        f"{endpoint_name}_form_model",
        __base__=BaseModel,
        **{k: (v[0], v[1]) for k, v in form_model_fields.items()}
    )

    # Create response model if fields are provided
    response_model = None
    if response_model_fields:
        response_model = create_model(
            f"{endpoint_name}_response_model",
            __base__=BaseModel,
            **{k: (v[0], v[1]) for k, v in response_model_fields.items()}
        )

    async def handler(form_data: form_model):
        try:
            # Transform filters for search endpoint
            params = form_data.model_dump()

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
        except HTTPException:
            raise  # Let FastAPI handle HTTPException properly       
        except Exception as e:
            logger.error(f"Error executing tool {endpoint_name}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

    handler.__name__ = endpoint_name
    return handler, response_model