import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi import FastAPI
from mcpo.main import lifespan


@pytest.mark.asyncio
async def test_sse_client_with_headers():
    """Test that headers are correctly passed to the sse_client function."""
    # Create test app
    app = FastAPI()
    app.state.server_type = "sse"
    app.state.args = "http://example.com/sse"
    app.state.headers = {"Authorization": "Bearer test-token", "X-Test": "value"}
    
    # Mock the sse_client context manager and ClientSession
    mock_reader = AsyncMock()
    mock_writer = AsyncMock()
    mock_session = AsyncMock()
    
    # Mock client session, create_dynamic_endpoints, and sse_client
    with patch("mcpo.main.sse_client", return_value=AsyncMock()) as mock_sse_client, \
         patch("mcpo.main.ClientSession", return_value=mock_session) as mock_client_session, \
         patch("mcpo.main.create_dynamic_endpoints") as mock_create_endpoints:
        
        # Configure the mocks
        mock_sse_client.return_value.__aenter__.return_value = (mock_reader, mock_writer)
        mock_client_session.return_value.__aenter__.return_value = mock_session
        
        # Create and enter the lifespan context
        async with lifespan(app):
            # Verify sse_client was called with the correct headers
            mock_sse_client.assert_called_once_with(
                url="http://example.com/sse", 
                sse_read_timeout=None, 
                headers={"Authorization": "Bearer test-token", "X-Test": "value"}
            )


@pytest.mark.asyncio
async def test_sse_client_without_headers():
    """Test that the sse_client works correctly when no headers are provided."""
    # Create test app
    app = FastAPI()
    app.state.server_type = "sse"
    app.state.args = "http://example.com/sse"
    app.state.headers = None
    
    # Mock the sse_client context manager and ClientSession
    mock_reader = AsyncMock()
    mock_writer = AsyncMock()
    mock_session = AsyncMock()
    
    # Mock client session, create_dynamic_endpoints, and sse_client
    with patch("mcpo.main.sse_client", return_value=AsyncMock()) as mock_sse_client, \
         patch("mcpo.main.ClientSession", return_value=mock_session) as mock_client_session, \
         patch("mcpo.main.create_dynamic_endpoints") as mock_create_endpoints:
        
        # Configure the mocks
        mock_sse_client.return_value.__aenter__.return_value = (mock_reader, mock_writer)
        mock_client_session.return_value.__aenter__.return_value = mock_session
        
        # Create and enter the lifespan context
        async with lifespan(app):
            # Verify sse_client was called without headers
            mock_sse_client.assert_called_once_with(
                url="http://example.com/sse", 
                sse_read_timeout=None, 
                headers=None
            )


@pytest.mark.asyncio
async def test_parse_headers_from_json_string():
    """Test that JSON string headers are correctly parsed."""
    # Create a FastAPI app with the run function
    with patch("mcpo.main.json.loads") as mock_json_loads:
        from mcpo.main import run
        
        mock_json_loads.return_value = {"Authorization": "Bearer token", "X-Test": "test-value"}
        
        # Call run with a JSON string for headers
        with patch("mcpo.main.uvicorn.Server.serve") as mock_serve:
            await run(
                host="localhost",
                port=8000,
                server_type="sse",
                server_command=["http://example.com/sse"],
                headers='{"Authorization": "Bearer token", "X-Test": "test-value"}'
            )
            
            # Verify json.loads was called with the correct string
            mock_json_loads.assert_called_once_with('{"Authorization": "Bearer token", "X-Test": "test-value"}')


@pytest.mark.asyncio
async def test_invalid_json_headers():
    """Test error handling for invalid JSON in headers."""
    # Create a FastAPI app with the run function
    with patch("mcpo.main.json.loads") as mock_json_loads, \
         patch("builtins.print") as mock_print:
        from mcpo.main import run
        
        # Simulate a JSON decode error
        mock_json_loads.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        # Call run with an invalid JSON string for headers
        with patch("mcpo.main.uvicorn.Server.serve") as mock_serve:
            await run(
                host="localhost",
                port=8000,
                server_type="sse",
                server_command=["http://example.com/sse"],
                headers='{"invalid json'
            )
            
            # Verify warning message was printed
            mock_print.assert_called_with("Warning: Invalid JSON format for headers. Headers will be ignored.")
