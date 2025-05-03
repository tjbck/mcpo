from typing import Union

import pytest
from mcpo.utils.main import _process_schema_property, ModelCache
from pydantic import BaseModel
from pydantic.fields import FieldInfo

_model_cache = ModelCache()

@pytest.fixture(autouse=True)
def clear_model_cache():
    _model_cache.clear()
    yield
    _model_cache.clear()

def test_any_of_type():
    """Test processing of schema with anyOf."""
    # Simple anyOf with primitive types
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"}
            ],
            "title": "Test Union"
        },
        "test_model",
        "test_field",
        True
    )
    assert type_info.__origin__ == Union
    assert len(type_info.__args__) == 2
    assert str in type_info.__args__
    assert int in type_info.__args__
    assert isinstance(field_info, FieldInfo)

    # anyOf with object types
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name"]
                },
                {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "value": {"type": "number"}
                    },
                    "required": ["id"]
                }
            ]
        },
        "test_model",
        "test_field",
        True
    )
    assert type_info.__origin__ == Union
    assert len(type_info.__args__) == 2
    for arg in type_info.__args__:
        assert issubclass(arg, BaseModel)

def test_all_of_type():
    """Test processing of schema with allOf."""
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "allOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                },
                {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer"}
                    },
                    "required": ["age"]
                }
            ]
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

def test_one_of_type():
    """Test processing of schema with oneOf."""
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"]
                },
                {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"}
                    },
                    "required": ["id"]
                }
            ]
        },
        "test_model",
        "test_field",
        True
    )
    assert type_info.__origin__ == Union
    assert len(type_info.__args__) == 2
    for arg in type_info.__args__:
        assert issubclass(arg, BaseModel)
    assert isinstance(field_info, FieldInfo)

def test_mixed_composition():
    """Test processing of schema with mixed composition types."""
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "allOf": [
                {
                    "type": "object",
                    "properties": {
                        "base": {"type": "string"}
                    },
                    "required": ["base"]
                },
                {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "option1": {"type": "string"}
                            }
                        },
                        {
                            "type": "object",
                            "properties": {
                                "option2": {"type": "integer"}
                            }
                        }
                    ]
                }
            ]
        },
        "test_model",
        "test_field",
        True
    )
    assert isinstance(type_info, type)
    assert issubclass(type_info, BaseModel)
    assert "base" in type_info.__annotations__
    assert isinstance(field_info, FieldInfo) 