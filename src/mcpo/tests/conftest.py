import pytest
from typing import Dict, Any

@pytest.fixture
def sample_schema_definitions() -> Dict[str, Any]:
    """Fixture providing sample schema definitions for testing."""
    return {
        "TestType": {
            "type": "string",
            "title": "Test Type"
        },
        "CircularType": {
            "type": "object",
            "properties": {
                "self": {"$ref": "#/$defs/CircularType"}
            }
        },
        "ComplexType": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "integer"}
            },
            "required": ["name"]
        }
    }

@pytest.fixture
def complex_schema() -> Dict[str, Any]:
    """Fixture providing a complex schema for testing."""
    return {
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