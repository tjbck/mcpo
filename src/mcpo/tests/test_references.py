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

def test_simple_reference():
    """Test processing of simple schema references."""
    schema_defs = {
        "Person": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
    }

    type_info, field_info = _process_schema_property(
        _model_cache,
        {"$ref": "#/definitions/Person"},
        "test_model",
        "test_field",
        True,
        schema_defs
    )
    assert isinstance(type_info, type)
    assert issubclass(type_info, BaseModel)
    assert "name" in type_info.__annotations__
    assert "age" in type_info.__annotations__
    assert isinstance(field_info, FieldInfo)

def test_nested_references():
    """Test processing of nested schema references."""
    schema_defs = {
        "Address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"}
            },
            "required": ["street", "city"]
        },
        "Person": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {"$ref": "#/definitions/Address"}
            },
            "required": ["name", "address"]
        }
    }

    type_info, field_info = _process_schema_property(
        _model_cache,
        {"$ref": "#/definitions/Person"},
        "test_model",
        "test_field",
        True,
        schema_defs
    )
    assert isinstance(type_info, type)
    assert issubclass(type_info, BaseModel)
    assert "name" in type_info.__annotations__
    assert "address" in type_info.__annotations__
    assert isinstance(field_info, FieldInfo)

def test_circular_reference():
    """Test processing of circular schema references."""
    schema_defs = {
        "Node": {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "next": {"$ref": "#/definitions/Node"}
            },
            "required": ["value"]
        }
    }

    type_info, field_info = _process_schema_property(
        _model_cache,
        {"$ref": "#/definitions/Node"},
        "test_model",
        "test_field",
        True,
        schema_defs
    )
    assert isinstance(type_info, type)
    assert issubclass(type_info, BaseModel)
    assert "value" in type_info.__annotations__
    assert "next" in type_info.__annotations__
    assert isinstance(field_info, FieldInfo)

def test_mutual_circular_reference():
    """Test processing of mutual circular schema references."""
    schema_defs = {
        "Person": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "spouse": {"$ref": "#/definitions/Person"}
            },
            "required": ["name"]
        }
    }

    type_info, field_info = _process_schema_property(
        _model_cache,
        {"$ref": "#/definitions/Person"},
        "test_model",
        "test_field",
        True,
        schema_defs
    )
    assert isinstance(type_info, type)
    assert issubclass(type_info, BaseModel)
    assert "name" in type_info.__annotations__
    assert "spouse" in type_info.__annotations__
    assert isinstance(field_info, FieldInfo)

def test_complex_circular_reference():
    """Test processing of complex circular schema references."""
    schema_defs = {
        "Department": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "employees": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/Employee"}
                }
            },
            "required": ["name"]
        },
        "Employee": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "department": {"$ref": "#/definitions/Department"}
            },
            "required": ["name"]
        }
    }

    type_info, field_info = _process_schema_property(
        _model_cache,
        {"$ref": "#/definitions/Department"},
        "test_model",
        "test_field",
        True,
        schema_defs
    )
    assert isinstance(type_info, type)
    assert issubclass(type_info, BaseModel)
    assert "name" in type_info.__annotations__
    assert "employees" in type_info.__annotations__
    assert isinstance(field_info, FieldInfo) 