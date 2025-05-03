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

def test_string_validation():
    """Test string validation constraints."""
    # Test minLength and maxLength
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "type": "string",
            "minLength": 3,
            "maxLength": 10
        },
        "test_model",
        "test_field",
        True
    )
    assert type_info == str
    assert isinstance(field_info, FieldInfo)
    assert field_info.json_schema_extra["metadata"]["min_length"] == 3
    assert field_info.json_schema_extra["metadata"]["max_length"] == 10

    # Test pattern
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "type": "string",
            "pattern": "^[A-Za-z]+$"
        },
        "test_model",
        "test_field",
        True
    )
    assert type_info == str
    assert isinstance(field_info, FieldInfo)
    assert field_info.json_schema_extra["metadata"]["pattern"] == "^[A-Za-z]+$"

    # Test format
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "type": "string",
            "format": "email"
        },
        "test_model",
        "test_field",
        True
    )
    assert type_info == str
    assert isinstance(field_info, FieldInfo)
    assert field_info.json_schema_extra["metadata"]["format"] == "email"

def test_number_validation():
    """Test number validation constraints."""
    # Test minimum and maximum
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "exclusiveMinimum": True,
            "exclusiveMaximum": True
        },
        "test_model",
        "test_field",
        True
    )
    assert type_info == float
    assert isinstance(field_info, FieldInfo)
    assert field_info.json_schema_extra["metadata"]["minimum"] == 0
    assert field_info.json_schema_extra["metadata"]["maximum"] == 100
    assert field_info.json_schema_extra["metadata"]["exclusive_minimum"] == True
    assert field_info.json_schema_extra["metadata"]["exclusive_maximum"] == True

    # Test multipleOf
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "type": "number",
            "multipleOf": 0.5
        },
        "test_model",
        "test_field",
        True
    )
    assert type_info == float
    assert isinstance(field_info, FieldInfo)
    assert field_info.json_schema_extra["metadata"]["multiple_of"] == 0.5

def test_array_validation():
    """Test array validation constraints."""
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
            "uniqueItems": True
        },
        "test_model",
        "test_field",
        True
    )
    assert type_info.__origin__ == list
    assert type_info.__args__[0] == str
    assert isinstance(field_info, FieldInfo)
    assert field_info.json_schema_extra["metadata"]["min_items"] == 1
    assert field_info.json_schema_extra["metadata"]["max_items"] == 5
    assert field_info.json_schema_extra["metadata"]["unique_items"] == True

def test_object_validation():
    """Test object validation constraints."""
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"],
            "minProperties": 1,
            "maxProperties": 2,
            "additionalProperties": False
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
    assert field_info.json_schema_extra["metadata"]["min_properties"] == 1
    assert field_info.json_schema_extra["metadata"]["max_properties"] == 2
    assert field_info.json_schema_extra["metadata"]["additional_properties"] == False

def test_enum_validation():
    """Test enum validation."""
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "type": "string",
            "enum": ["red", "green", "blue"]
        },
        "test_model",
        "test_field",
        True
    )
    assert type_info == str
    assert isinstance(field_info, FieldInfo)
    assert field_info.json_schema_extra["metadata"]["enum"] == ["red", "green", "blue"]

def test_const_validation():
    """Test const validation."""
    type_info, field_info = _process_schema_property(
        _model_cache,
        {
            "type": "string",
            "const": "fixed_value"
        },
        "test_model",
        "test_field",
        True
    )
    assert type_info == str
    assert isinstance(field_info, FieldInfo)
    assert field_info.json_schema_extra["metadata"]["const"] == "fixed_value" 