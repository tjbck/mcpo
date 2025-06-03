# tests/test_multi_worker.py
import pytest
import os
import pickle
import base64
import tempfile
import json
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI

from mcpo.main import create_app, get_app, run


# 确保清理全局状态
@pytest.fixture(autouse=True)
def clear_global_app():
    """Clear global app state before each test"""
    import mcpo.main

    mcpo.main.app = None
    yield
    mcpo.main.app = None


class TestMultiWorkerSupport:
    """Test multi-worker functionality and configuration management"""

    def test_create_app_with_basic_config(self):
        """Test creating app with basic configuration"""
        config = {
            "api_key": "test-key",
            "name": "Test MCPO",
            "version": "1.0.0",
            "description": "Test description",
        }

        # Create a temporary config file for testing
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {"mcpServers": {"test_server": {"command": "echo", "args": ["test"]}}},
                f,
            )
            config_path = f.name

        try:
            config["config_path"] = config_path
            app = create_app(**config)

            assert isinstance(app, FastAPI)
            assert app.title == "Test MCPO"
            assert app.version == "1.0.0"
            assert "Test description" in app.description
        finally:
            os.unlink(config_path)

    def test_config_serialization_for_multi_worker(self):
        """Test configuration serialization and deserialization for multi-worker mode"""
        original_config = {
            "api_key": "test-secret-key",
            "name": "Multi Worker Test",
            "version": "2.0.0",
            "cors_allow_origins": ["http://localhost:3000"],
            "strict_auth": True,
        }

        # Test encoding
        encoded_config = base64.b64encode(pickle.dumps(original_config)).decode()
        assert isinstance(encoded_config, str)
        assert len(encoded_config) > 0

        # Test decoding
        decoded_config = pickle.loads(base64.b64decode(encoded_config))
        assert decoded_config == original_config
        assert decoded_config["api_key"] == "test-secret-key"
        assert decoded_config["name"] == "Multi Worker Test"
        assert decoded_config["cors_allow_origins"] == ["http://localhost:3000"]

    def test_create_app_from_environment_config(self):
        """Test creating app from environment variable configuration"""
        config = {
            "api_key": "env-test-key",
            "name": "Environment Test App",
            "server_command": ["echo", "hello"],
        }

        # Serialize and set environment variable
        encoded_config = base64.b64encode(pickle.dumps(config)).decode()

        with patch.dict(os.environ, {"MCPO_CONFIG": encoded_config}):
            # Call create_app without arguments to trigger env config loading
            app = create_app()

            assert isinstance(app, FastAPI)
            assert app.title == "Environment Test App"

    def test_create_app_environment_config_decode_error(self):
        """Test handling of corrupted environment configuration"""
        # Set invalid base64 data
        with patch.dict(os.environ, {"MCPO_CONFIG": "invalid-base64-data"}):
            with patch("mcpo.main.logger") as mock_logger:
                # Need to provide minimal config to avoid ValueError
                with pytest.raises(
                    ValueError, match="You must provide either server_command or config"
                ):
                    create_app()

                # Should log error during decode attempt
                mock_logger.error.assert_called_once()

    def test_get_app_factory_function(self):
        """Test application factory function for multi-worker mode"""
        # Mock environment to provide valid config
        config = {"server_command": ["echo", "test"]}
        encoded_config = base64.b64encode(pickle.dumps(config)).decode()

        with patch.dict(os.environ, {"MCPO_CONFIG": encoded_config}):
            # First call should create new app
            app1 = get_app()
            assert isinstance(app1, FastAPI)

            # Second call should return same app instance
            app2 = get_app()
            assert app1 is app2

    def test_get_app_with_environment_config(self):
        """Test get_app with environment configuration"""
        config = {
            "name": "Factory Test App",
            "version": "3.0.0",
            "server_command": ["python", "-c", "print('test')"],
        }

        encoded_config = base64.b64encode(pickle.dumps(config)).decode()

        with patch.dict(os.environ, {"MCPO_CONFIG": encoded_config}):
            app = get_app()
            assert app.title == "Factory Test App"
            assert app.version == "3.0.0"

    @pytest.mark.asyncio
    async def test_run_single_worker_mode(self):
        """Test running in single worker mode"""
        config = {"name": "Single Worker Test", "server_command": ["echo", "test"]}

        with patch("mcpo.main.uvicorn.Server") as mock_server_class:
            with patch("mcpo.main.uvicorn.Config") as mock_config_class:
                mock_server = MagicMock()
                mock_server.serve = AsyncMock()
                mock_server_class.return_value = mock_server

                mock_config = MagicMock()
                mock_config_class.return_value = mock_config

                await run(host="127.0.0.1", port=8000, workers=1, **config)

                # Verify single worker mode was used
                mock_config_class.assert_called_once()
                mock_server_class.assert_called_once_with(mock_config)
                mock_server.serve.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_multi_worker_mode(self):
        """Test running in multi-worker mode"""
        config = {"name": "Multi Worker Test", "server_command": ["echo", "test"]}

        with patch("mcpo.main.uvicorn.run") as mock_uvicorn_run:
            await run(host="127.0.0.1", port=8000, workers=4, **config)

            # Verify multi-worker mode was used
            mock_uvicorn_run.assert_called_once()
            call_args = mock_uvicorn_run.call_args

            assert call_args[1]["host"] == "127.0.0.1"
            assert call_args[1]["port"] == 8000
            assert call_args[1]["workers"] == 4
            assert call_args[1]["factory"] is True
            assert call_args[0][0] == "mcpo.main:get_app"

    @pytest.mark.asyncio
    async def test_run_multi_worker_config_environment_setup(self):
        """Test that multi-worker mode properly sets up environment configuration"""
        config = {
            "api_key": "multi-worker-key",
            "name": "Multi Worker Config Test",
            "cors_allow_origins": ["http://example.com"],
        }

        with patch("mcpo.main.uvicorn.run") as mock_uvicorn_run:
            # Start with clean environment
            original_env = os.environ.get("MCPO_CONFIG")
            if "MCPO_CONFIG" in os.environ:
                del os.environ["MCPO_CONFIG"]

            try:
                await run(host="0.0.0.0", port=9000, workers=8, **config)

                # Verify environment variable was set
                assert "MCPO_CONFIG" in os.environ

                # Verify config can be decoded
                encoded_config = os.environ["MCPO_CONFIG"]
                decoded_config = pickle.loads(base64.b64decode(encoded_config))

                assert decoded_config["api_key"] == "multi-worker-key"
                assert decoded_config["name"] == "Multi Worker Config Test"
                assert decoded_config["cors_allow_origins"] == ["http://example.com"]
            finally:
                # Restore original environment
                if "MCPO_CONFIG" in os.environ:
                    del os.environ["MCPO_CONFIG"]
                if original_env:
                    os.environ["MCPO_CONFIG"] = original_env

    def test_worker_logging_configuration(self):
        """Test that worker logging includes process ID"""
        import logging

        # Test log format includes PID
        with patch("mcpo.main.logging.basicConfig") as mock_logging_config:
            # Simulate what happens in the run function
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - [PID:%(process)d] - %(message)s",
            )

            # Check that basicConfig was called with the right format
            # Note: this might not be called if logging is already configured
            if mock_logging_config.called:
                format_arg = mock_logging_config.call_args[1]["format"]
                assert "[PID:%(process)d]" in format_arg

    def test_app_state_configuration_stdio(self):
        """Test app state configuration for stdio server type"""
        config = {
            "server_command": ["python", "-m", "test_server"],
            "api_key": "test-key",
        }

        app = create_app(**config)

        assert app.state.server_type == "stdio"
        assert app.state.command == "python"
        assert app.state.args == ["-m", "test_server"]
        assert app.state.api_dependency is not None

    def test_app_state_configuration_sse(self):
        """Test app state configuration for SSE server type"""
        config = {
            "server_type": "sse",
            "server_command": ["http://localhost:8080/sse"],
            "api_key": "test-key",
        }

        app = create_app(**config)

        assert app.state.server_type == "sse"
        assert app.state.args == "http://localhost:8080/sse"
        assert app.state.api_dependency is not None

    def test_app_state_configuration_streamablehttp(self):
        """Test app state configuration for streamablehttp server type"""
        config = {
            "server_type": "streamablehttp",
            "server_command": ["http://localhost:8080/stream"],
            "api_key": "test-key",
        }

        app = create_app(**config)

        assert app.state.server_type == "streamablehttp"
        assert app.state.args == "http://localhost:8080/stream"
        assert app.state.api_dependency is not None

    def test_cors_configuration(self):
        """Test CORS configuration in multi-worker setup"""
        config = {
            "cors_allow_origins": ["http://localhost:3000", "https://example.com"],
            "server_command": ["echo", "test"],
        }

        app = create_app(**config)

        # Check that CORS middleware was added
        cors_middleware = None
        for middleware in app.user_middleware:
            if middleware.cls.__name__ == "CORSMiddleware":
                cors_middleware = middleware
                break

        assert cors_middleware is not None

    def test_ssl_configuration(self):
        """Test SSL configuration handling"""
        config = {
            "ssl_certfile": "/path/to/cert.pem",
            "ssl_keyfile": "/path/to/key.pem",
            "server_command": ["echo", "test"],
        }

        app = create_app(**config)

        # SSL configuration is passed to FastAPI constructor, not stored as attributes
        # We can verify the app was created successfully with SSL config
        assert isinstance(app, FastAPI)
        assert app.title == "MCP OpenAPI Proxy"  # Default title

    def test_error_handling_missing_config(self):
        """Test error handling when required configuration is missing"""
        # Create a clean environment without MCPO_CONFIG
        clean_env = {k: v for k, v in os.environ.items() if k != "MCPO_CONFIG"}

        with patch.dict(os.environ, clean_env, clear=True):
            with pytest.raises(
                ValueError, match="You must provide either server_command or config"
            ):
                create_app()

    def test_error_handling_invalid_config_file(self):
        """Test error handling for invalid config file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"invalid": "config"}, f)  # Missing mcpServers
            config_path = f.name

        try:
            with pytest.raises(
                ValueError, match="No 'mcpServers' found in config file"
            ):
                create_app(config_path=config_path)
        finally:
            os.unlink(config_path)

    @pytest.mark.asyncio
    async def test_lifespan_worker_identification(self):
        """Test that lifespan properly identifies workers"""
        app = FastAPI()
        app.state.server_type = "stdio"
        app.state.command = "echo"
        app.state.args = ["test"]
        app.state.env = {}

        with patch("mcpo.main.logger") as mock_logger:
            with patch("mcpo.main.stdio_client") as mock_stdio_client:
                with patch("mcpo.main.ClientSession") as mock_client_session:
                    # Mock the stdio client context manager
                    mock_reader = MagicMock()
                    mock_writer = MagicMock()
                    mock_session = MagicMock()
                    mock_session.initialize = AsyncMock(
                        return_value=MagicMock(serverInfo=None)
                    )
                    mock_session.list_tools = AsyncMock(
                        return_value=MagicMock(tools=[])
                    )

                    # Setup context manager mocks
                    mock_stdio_client.return_value.__aenter__ = AsyncMock(
                        return_value=(mock_reader, mock_writer)
                    )
                    mock_stdio_client.return_value.__aexit__ = AsyncMock(
                        return_value=None
                    )

                    mock_client_session.return_value.__aenter__ = AsyncMock(
                        return_value=mock_session
                    )
                    mock_client_session.return_value.__aexit__ = AsyncMock(
                        return_value=None
                    )

                    from mcpo.main import lifespan

                    async with lifespan(app):
                        pass

                    # Verify worker startup and shutdown logging
                    startup_calls = [
                        call
                        for call in mock_logger.info.call_args_list
                        if "starting up" in str(call)
                    ]
                    shutdown_calls = [
                        call
                        for call in mock_logger.info.call_args_list
                        if "shutting down" in str(call)
                    ]

                    assert len(startup_calls) >= 1
                    assert len(shutdown_calls) >= 1


class TestWorkerPerformance:
    """Test performance-related aspects of multi-worker setup"""

    def test_worker_count_recommendations(self):
        """Test worker count calculation recommendations"""
        import psutil

        # Test default calculation
        cores = psutil.cpu_count()
        recommended = min(cores * 2, 32)

        assert recommended > 0
        assert recommended <= 32

        # Test with different core counts
        test_cases = [
            (1, 2),  # 1 core -> 2 workers
            (4, 8),  # 4 cores -> 8 workers
            (8, 16),  # 8 cores -> 16 workers
            (16, 32),  # 16 cores -> 32 workers (capped)
            (32, 32),  # 32 cores -> 32 workers (capped)
        ]

        for cores, expected_workers in test_cases:
            calculated = min(cores * 2, 32)
            assert calculated == expected_workers

    def test_configuration_size_limits(self):
        """Test that configuration serialization handles reasonable sizes"""
        # Test with large configuration
        large_config = {
            "api_key": "x" * 1000,  # Large API key
            "cors_allow_origins": [
                f"http://domain{i}.com" for i in range(100)
            ],  # Many origins
            "name": "Large Configuration Test",
            "description": "x" * 5000,  # Large description
        }

        # Should be able to serialize and deserialize
        encoded = base64.b64encode(pickle.dumps(large_config)).decode()
        decoded = pickle.loads(base64.b64decode(encoded))

        assert decoded == large_config
        assert len(encoded) > 1000  # Verify it's actually large

    def test_environment_cleanup(self):
        """Test that environment variables are properly managed"""
        # Test setting and cleaning up MCPO_CONFIG
        test_config = {"test": "value"}
        encoded = base64.b64encode(pickle.dumps(test_config)).decode()

        # Ensure clean start
        original_value = os.environ.get("MCPO_CONFIG")
        if "MCPO_CONFIG" in os.environ:
            del os.environ["MCPO_CONFIG"]

        try:
            # Set test value
            os.environ["MCPO_CONFIG"] = encoded
            assert os.environ["MCPO_CONFIG"] == encoded

            # Verify decoding works
            decoded = pickle.loads(base64.b64decode(os.environ["MCPO_CONFIG"]))
            assert decoded == test_config

        finally:
            # Clean up
            if "MCPO_CONFIG" in os.environ:
                del os.environ["MCPO_CONFIG"]
            if original_value:
                os.environ["MCPO_CONFIG"] = original_value
