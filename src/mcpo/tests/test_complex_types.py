import pytest
from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from mcpo.utils.main import _process_schema_property, _model_cache

@pytest.fixture(autouse=True)
def clear_model_cache():
    _model_cache.clear()
    yield
    _model_cache.clear()

def test_array_type():
    """Test processing of array type."""
    # Array of strings
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "type": "array",
            "items": {"type": "string"},
            "title": "Test Array"
        },
        "test_model",
        "test_field",
        True
    )
    assert type_info == List[str]
    assert isinstance(field_info, FieldInfo)

    # Array with validation
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10,
            "uniqueItems": True
        },
        "test_model",
        "test_field",
        True
    )
    assert type_info == List[str]
    assert isinstance(field_info, FieldInfo)

    # Array of objects
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "value": {"type": "number"}
                },
                "required": ["id"]
            }
        },
        "test_model",
        "test_field",
        True
    )
    assert str(type_info).startswith("typing.List[")
    item_type = type_info.__args__[0]
    assert issubclass(item_type, BaseModel)
    assert "id" in item_type.__annotations__
    assert "value" in item_type.__annotations__

def test_object_type():
    """Test processing of object type."""
    # Simple object
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"],
            "title": "Test Object"
        },
        "test_model",
        "test_field",
        True
    )
    assert isinstance(type_info, type)
    assert issubclass(type_info, BaseModel)
    assert "name" in type_info.__annotations__
    assert "age" in type_info.__annotations__
    assert isinstance(field_info, FieldInfo)

    # Empty object
    type_info, field_info = _process_schema_property(
        _model_cache,
        {"type": "object", "properties": {}},
        "test_model",
        "test_field",
        True
    )
    assert type_info == Dict[str, Any]
    assert isinstance(field_info, FieldInfo)

    # Object with patternProperties
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "type": "object",
            "patternProperties": {
                "^S_": {"type": "string"},
                "^I_": {"type": "integer"}
            }
        },
        "test_model",
        "test_field",
        True
    )
    assert type_info == Dict[str, Any]
    assert isinstance(field_info, FieldInfo)

    # Object with additionalProperties
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": {"type": "string"}
        },
        "test_model",
        "test_field",
        True
    )
    assert isinstance(type_info, type)
    assert issubclass(type_info, BaseModel)
    assert "name" in type_info.__annotations__

def test_nested_objects():
    """Test processing of nested object types."""
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "roles": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name"]
            },
            "settings": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "count": {"type": "integer"}
                }
            }
        },
        "required": ["user"]
    }
    
    type_info, field_info = _process_schema_property(
        _model_cache, schema, "test_model", "test_field", True
    )
    assert isinstance(type_info, type)
    assert issubclass(type_info, BaseModel)
    assert "user" in type_info.__annotations__
    assert "settings" in type_info.__annotations__
    
    user_type = type_info.__annotations__["user"]
    assert issubclass(user_type, BaseModel)
    assert "name" in user_type.__annotations__
    assert "roles" in user_type.__annotations__
    
    settings_type = type_info.__annotations__["settings"]
    assert issubclass(settings_type, BaseModel)
    assert "enabled" in settings_type.__annotations__
    assert "count" in settings_type.__annotations__ 