import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcpo.utils.sse import sse_client_loop
from fastapi import FastAPI

@pytest.mark.asyncio
async def test_sse_client_loop():
    # Mock dependencies
    app = FastAPI()
    api_dependency = MagicMock()
    create_dynamic_endpoints = AsyncMock()
    
    # Mock the sse_client context manager
    mock_reader = AsyncMock()
    mock_writer = AsyncMock()
    mock_session = AsyncMock()
    
    # Set up the mock reader to return a message once and then raise a CancelledError
    mock_reader.receive.side_effect = [MagicMock(), asyncio.CancelledError()]

    with patch('mcpo.utils.sse.sse_client') as mock_sse_client, \
         patch('mcpo.utils.sse.ClientSession') as mock_client_session:
        
        mock_sse_client.return_value.__aenter__.return_value = (mock_reader, mock_writer)
        mock_client_session.return_value.__aenter__.return_value = mock_session

        # Run the sse_client_loop
        await sse_client_loop("http://test-url", {}, api_dependency, create_dynamic_endpoints, app)

    # Verify that the functions were called as expected
    create_dynamic_endpoints.assert_awaited_once_with(app, api_dependency=api_dependency)
    mock_reader.receive.assert_awaited()
    assert mock_reader.receive.await_count == 2  # Once for the message, once for the CancelledError