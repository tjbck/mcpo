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

def test_complex_schema_processing():
    """Test processing of a complex schema with multiple features."""
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 100
                    },
                    "email": {
                        "type": "string",
                        "format": "email"
                    },
                    "age": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 150
                    },
                    "preferences": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["dark_mode", "notifications", "language"]
                        },
                        "uniqueItems": True
                    }
                },
                "required": ["name", "email"]
            },
            "settings": {
                "type": "object",
                "properties": {
                    "theme": {
                        "type": "string",
                        "enum": ["light", "dark", "system"]
                    },
                    "notifications": {
                        "type": "boolean"
                    }
                },
                "required": ["theme"]
            }
        },
        "required": ["user"]
    }

    type_info, field_info = _process_schema_property(
        _model_cache,
        schema,
        "test_model",
        "test_field",
        True
    )
    assert isinstance(type_info, type)
    assert issubclass(type_info, BaseModel)
    assert "user" in type_info.__annotations__
    assert "settings" in type_info.__annotations__
    assert isinstance(field_info, FieldInfo)

def test_schema_with_all_features():
    """Test processing of a schema that includes all major features."""
    schema_defs = {
        "Address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
                "zip": {
                    "type": "string",
                    "pattern": "^\\d{5}(-\\d{4})?$"
                }
            },
            "required": ["street", "city", "zip"]
        }
    }

    schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "format": "uuid"
            },
            "name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 100
            },
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 150
            },
            "email": {
                "type": "string",
                "format": "email"
            },
            "address": {"$ref": "#/definitions/Address"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 10,
                "uniqueItems": True
            },
            "metadata": {
                "type": "object",
                "additionalProperties": {
                    "type": "string"
                }
            },
            "status": {
                "type": "string",
                "enum": ["active", "inactive", "pending"]
            },
            "score": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
                "multipleOf": 0.5
            }
        },
        "required": ["id", "name", "email", "address"]
    }

    type_info, field_info = _process_schema_property(
        _model_cache,
        schema,
        "test_model",
        "test_field",
        True,
        schema_defs
    )
    assert isinstance(type_info, type)
    assert issubclass(type_info, BaseModel)
    assert "id" in type_info.__annotations__
    assert "name" in type_info.__annotations__
    assert "age" in type_info.__annotations__
    assert "email" in type_info.__annotations__
    assert "address" in type_info.__annotations__
    assert "tags" in type_info.__annotations__
    assert "metadata" in type_info.__annotations__
    assert "status" in type_info.__annotations__
    assert "score" in type_info.__annotations__
    assert isinstance(field_info, FieldInfo)

def test_schema_with_composition():
    """Test processing of a schema with composition features."""
    schema = {
        "type": "object",
        "properties": {
            "person": {
                "allOf": [
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
                            "email": {"type": "string", "format": "email"},
                            "phone": {"type": "string", "pattern": "^\\+?[1-9]\\d{1,14}$"}
                        },
                        "required": ["email"]
                    }
                ]
            },
            "preferences": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "theme": {"type": "string", "enum": ["light", "dark"]}
                        }
                    },
                    {
                        "type": "object",
                        "properties": {
                            "notifications": {"type": "boolean"}
                        }
                    }
                ]
            }
        },
        "required": ["person"]
    }

    type_info, field_info = _process_schema_property(
        _model_cache,
        schema,
        "test_model",
        "test_field",
        True
    )
    assert isinstance(type_info, type)
    assert issubclass(type_info, BaseModel)
    assert "person" in type_info.__annotations__
    assert "preferences" in type_info.__annotations__
    assert isinstance(field_info, FieldInfo) 