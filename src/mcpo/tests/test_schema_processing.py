from typing import List, Union

import pytest
from mcpo.utils.main import _process_schema_property, get_model_fields, ModelCache
from pydantic import BaseModel
from pydantic.fields import FieldInfo

_model_cache = ModelCache()

@pytest.fixture(autouse=True)
def clear_model_cache():
    _model_cache.clear()
    yield
    _model_cache.clear()

def test_process_primitive_types():
    """Test processing of primitive schema types."""
    # Test string type
    type_info, field_info = _process_schema_property(
        _model_cache,
        {"type": "string", "title": "Test String"},
        "test_model",
        "test_field",
        True
    )
    assert type_info == str
    assert isinstance(field_info, FieldInfo)

    # Test integer type
    type_info, field_info = _process_schema_property(
        _model_cache,
        {"type": "integer", "title": "Test Integer"},
        "test_model",
        "test_field",
        False
    )
    assert type_info == int
    assert isinstance(field_info, FieldInfo)

    # Test boolean type
    type_info, field_info = _process_schema_property(
        _model_cache,
        {"type": "boolean", "title": "Test Boolean"},
        "test_model",
        "test_field",
        True
    )
    assert type_info == bool
    assert isinstance(field_info, FieldInfo)

def test_process_array_type():
    """Test processing of array schema type."""
    schema = {
        "type": "array",
        "items": {
            "type": "string"
        },
        "title": "Test Array"
    }
    type_info, field_info = _process_schema_property(
        _model_cache, schema, "test_model", "test_field", True
    )
    assert type_info == List[str]
    assert isinstance(field_info, FieldInfo)

def test_process_object_type():
    """Test processing of object schema type."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "title": "Test Object"
    }
    type_info, field_info = _process_schema_property(
        _model_cache, schema, "test_model", "test_field", True
    )
    assert isinstance(type_info, type)
    assert isinstance(field_info, FieldInfo)

def test_process_ref_type():
    """Test processing of schema with $ref."""
    schema = {
        "$ref": "#/$defs/TestType"
    }
    schema_defs = {
        "TestType": {
            "type": "string",
            "title": "Test Type"
        }
    }
    type_info, field_info = _process_schema_property(
        _model_cache, schema, "test_model", "test_field", True, schema_defs
    )
    assert type_info == str
    assert isinstance(field_info, FieldInfo)

def test_process_any_of_type():
    """Test processing of schema with anyOf."""
    schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"}
        ],
        "title": "Test Union"
    }
    type_info, field_info = _process_schema_property(
        _model_cache, schema, "test_model", "test_field", True
    )
    assert type_info.__origin__ == Union
    assert isinstance(field_info, FieldInfo)

def test_process_circular_ref():
    """Test handling of circular references."""
    schema = {
        "$ref": "#/$defs/CircularType"
    }
    schema_defs = {
        "CircularType": {
            "type": "object",
            "properties": {
                "self": {"$ref": "#/$defs/CircularType"}
            }
        }
    }
    type_info, field_info = _process_schema_property(
        _model_cache, schema, "test_model", "test_field", True, schema_defs
    )
    assert isinstance(type_info, type)
    assert issubclass(type_info, BaseModel)
    assert "self" in type_info.__annotations__
    assert isinstance(field_info, FieldInfo)

def test_get_model_fields():
    """Test the get_model_fields function."""
    properties = {
        "name": {"type": "string", "title": "Name"},
        "age": {"type": "integer", "title": "Age"}
    }
    required = ["name"]
    fields = get_model_fields("TestModel", properties, required, {})
    
    assert "name" in fields
    assert "age" in fields
    assert fields["name"][0] == str
    assert fields["age"][0] == int
    assert isinstance(fields["name"][1], FieldInfo)
    assert isinstance(fields["age"][1], FieldInfo)

def test_complex_schema_processing():
    """Test processing of a complex schema with nested objects and arrays."""
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
    assert "user" in type_info.__annotations__
    assert "settings" in type_info.__annotations__
    assert isinstance(field_info, FieldInfo)

def test_schema_with_default_values():
    """Test processing of schema with default values."""
    schema = {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "default": 10
            },
            "enabled": {
                "type": "boolean",
                "default": True
            }
        }
    }
    
    type_info, field_info = _process_schema_property(
        _model_cache, schema, "test_model", "test_field", False
    )
    assert isinstance(type_info, type)
    assert "count" in type_info.__annotations__
    assert "enabled" in type_info.__annotations__
    assert isinstance(field_info, FieldInfo)

def test_schema_with_enum():
    """Test processing of schema with enum values."""
    schema = {
        "type": "string",
        "enum": ["option1", "option2", "option3"],
        "title": "Test Enum"
    }
    
    type_info, field_info = _process_schema_property(
        _model_cache, schema, "test_model", "test_field", True
    )
    assert isinstance(type_info, type)
    assert isinstance(field_info, FieldInfo)

def test_recursive_object_schema():
    """Test processing of a schema with recursive object references."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "children": {
                "type": "array",
                "items": {
                    "$ref": "#/$defs/Node"
                }
            }
        },
        "required": ["name"],
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "children": {
                        "type": "array",
                        "items": {
                            "$ref": "#/$defs/Node"
                        }
                    }
                },
                "required": ["name"]
            }
        }
    }

    fields = get_model_fields("Node", schema["properties"], schema["required"], schema["$defs"])
    assert "name" in fields
    assert "children" in fields
    assert fields["name"][0] == str
    assert fields["children"][0].__origin__ == list

def test_mutually_recursive_schemas():
    """Test processing of mutually recursive schemas."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "parent": {
                "$ref": "#/$defs/Parent"
            }
        },
        "required": ["name"],
        "$defs": {
            "Parent": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "children": {
                        "type": "array",
                        "items": {
                            "$ref": "#/$defs/Child"
                        }
                    }
                },
                "required": ["name"]
            },
            "Child": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "parent": {
                        "$ref": "#/$defs/Parent"
                    }
                },
                "required": ["name"]
            }
        }
    }

    fields = get_model_fields("Parent", schema["properties"], schema["required"], schema["$defs"])
    assert "name" in fields
    assert "parent" in fields
    assert fields["name"][0] == str

def test_complex_recursive_schema():
    """Test processing of a complex schema with multiple recursive references."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"$ref": "#/$defs/Item"},
                        {"$ref": "#/$defs/Container"}
                    ]
                }
            }
        },
        "required": ["name"],
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "parent": {
                        "$ref": "#/$defs/Container"
                    }
                },
                "required": ["name"]
            },
            "Container": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"$ref": "#/$defs/Item"},
                                {"$ref": "#/$defs/Container"}
                            ]
                        }
                    }
                },
                "required": ["name"]
            }
        }
    }

    fields = get_model_fields("Container", schema["properties"], schema["required"], schema["$defs"])
    assert "name" in fields
    assert "items" in fields
    assert fields["name"][0] == str
    assert fields["items"][0].__origin__ == list

def test_recursive_schema_with_optional_fields():
    """Test processing of recursive schemas with optional fields."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "next": {
                "anyOf": [
                    {"$ref": "#/$defs/Node"},
                    {"type": "null"}
                ]
            }
        },
        "required": ["name"],
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "next": {
                        "anyOf": [
                            {"$ref": "#/$defs/Node"},
                            {"type": "null"}
                        ]
                    }
                },
                "required": ["name"]
            }
        }
    }

    fields = get_model_fields("Node", schema["properties"], schema["required"], schema["$defs"])
    assert "name" in fields
    assert "next" in fields
    assert fields["name"][0] == str
    assert fields["next"][0].__origin__ == Union

def test_deeply_nested_recursive_schema():
    """Test processing of deeply nested recursive schemas."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "nested": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "deeper": {
                        "$ref": "#/$defs/DeepNested"
                    }
                },
                "required": ["value"]
            }
        },
        "required": ["name"],
        "$defs": {
            "DeepNested": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "nested": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "deeper": {
                                "$ref": "#/$defs/DeepNested"
                            }
                        },
                        "required": ["value"]
                    }
                },
                "required": ["value"]
            }
        }
    }

    fields = get_model_fields("DeepNested", schema["properties"], schema["required"], schema["$defs"])
    assert "name" in fields
    assert "nested" in fields
    assert fields["name"][0] == str
    assert isinstance(fields["nested"][0], type(BaseModel))

def test_recursive_schema_with_mixed_types():
    """Test processing of recursive schemas with mixed type references."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {"$ref": "#/$defs/ComplexItem"},
                        {"type": "null"}
                    ]
                }
            }
        },
        "required": ["name"],
        "$defs": {
            "ComplexItem": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "string"},
                                {"$ref": "#/$defs/ComplexItem"},
                                {"type": "null"}
                            ]
                        }
                    }
                },
                "required": ["name"]
            }
        }
    }

    fields = get_model_fields("ComplexItem", schema["properties"], schema["required"], schema["$defs"])
    assert "name" in fields
    assert "items" in fields
    assert fields["name"][0] == str
    assert fields["items"][0].__origin__ == list 