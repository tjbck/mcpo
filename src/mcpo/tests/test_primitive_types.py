import pytest
from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from mcpo.utils.main import _process_schema_property, ModelCache

_model_cache = ModelCache()

@pytest.fixture(autouse=True)
def clear_model_cache():
    _model_cache.clear()
    yield
    _model_cache.clear()

def test_string_type():
    """Test processing of string type."""
    # Required string
    type_info, field_info = _process_schema_property(
        _model_cache,
        {"type": "string", "description": "A simple string"},
        "test_model",
        "test_field",
        True
    )
    assert type_info == str
    assert isinstance(field_info, FieldInfo)
    assert field_info.description == "A simple string"

    # Optional string with default
    type_info, field_info = _process_schema_property(
        _model_cache,
        {"type": "string", "default": "default_value"},
        "test_model",
        "test_field",
        False
    )
    assert type_info == str
    assert isinstance(field_info, FieldInfo)
    assert field_info.default == "default_value"

    # String with format
    type_info, field_info = _process_schema_property(
        _model_cache,
        {"type": "string", "format": "email"},
        "test_model",
        "test_field",
        True
    )
    assert type_info == str
    assert isinstance(field_info, FieldInfo)

def test_number_types():
    """Test processing of number types."""
    # Integer type
    type_info, field_info = _process_schema_property(
        _model_cache,
        {"type": "integer"},
        "test_model",
        "test_field",
        True
    )
    assert type_info == int
    assert isinstance(field_info, FieldInfo)

    # Number type
    type_info, field_info = _process_schema_property(
        _model_cache,
        {"type": "number"},
        "test_model",
        "test_field",
        True
    )
    assert type_info == float
    assert isinstance(field_info, FieldInfo)

    # Number with validation
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "exclusiveMinimum": True,
            "exclusiveMaximum": True,
            "multipleOf": 0.5
        },
        "test_model",
        "test_field",
        True
    )
    assert type_info == float
    assert isinstance(field_info, FieldInfo)

def test_boolean_type():
    """Test processing of boolean type."""
    # Required boolean
    type_info, field_info = _process_schema_property(
        _model_cache,
        {"type": "boolean"},
        "test_model",
        "test_field",
        True
    )
    assert type_info == bool
    assert isinstance(field_info, FieldInfo)

    # Optional boolean with default
    type_info, field_info = _process_schema_property(
        _model_cache,
        {"type": "boolean", "default": True},
        "test_model",
        "test_field",
        False
    )
    assert type_info == bool
    assert isinstance(field_info, FieldInfo)
    assert field_info.default == True

def test_null_type():
    """Test processing of null type."""
    type_info, field_info = _process_schema_property(
        _model_cache,
        {"type": "null"},
        "test_model",
        "test_field",
        True
    )
    assert type_info == None
    assert isinstance(field_info, FieldInfo)

def test_unknown_type():
    """Test processing of unknown type."""
    type_info, field_info = _process_schema_property(
        _model_cache,
        {"type": "unknown"},
        "test_model",
        "test_field",
        True
    )
    assert type_info == Any
    assert isinstance(field_info, FieldInfo) 