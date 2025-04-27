"""
Tests for SSE transport implementation.
"""

import pytest
import logging
import asyncio
from unittest import mock
import sys
import json

from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.requests import Request
from starlette.exceptions import HTTPException

from mcp.server.fastmcp import FastMCP

# Import functions/classes to test from sse_transport.py
from cursor_notebook_mcp.sse_transport import (
    handle_sse, handle_root, http_exception_handler, 
    generic_exception_handler, create_starlette_app, run_sse_server,
)
from cursor_notebook_mcp.server import ServerConfig

# Create mock classes for tests
class MockScope:
    def __init__(self):
        self.type = "http"

class MockClient:
    def __init__(self, host="127.0.0.1", port=12345):
        self.host = host
        self.port = port

class MockRequest:
    def __init__(self):
        self.app = type('MockApp', (), {'state': type('MockState', (), {
            'mcp_server': mock.MagicMock(),
            'sse_transport': mock.MagicMock(),
            'config': mock.MagicMock()
        })})
        self.client = MockClient()
        self.url = "http://test/sse"
        self.scope = MockScope()
        
    async def _send(self, message):
        pass
        
    async def receive(self):
        return {"type": "http.disconnect"}

# Test basic route setup with TestClient
def test_root_route():
    """Test that the root route returns a 200 status."""
    with mock.patch('cursor_notebook_mcp.sse_transport.SseServerTransport') as mock_transport:
        # Create a Starlette app with the mock transport
        mcp_server = mock.MagicMock()
        config = mock.MagicMock()
        config.version = "0.2.3"  # Mock the version
        
        # Set a concrete value for log_level to avoid MagicMock comparison error
        config.log_level = logging.INFO  # Set to a concrete value, not a MagicMock
        
        # Create the app using the create_starlette_app function
        app = create_starlette_app(mcp_server, config)
        
        # Use the Starlette test client to make a request
        from starlette.testclient import TestClient
        client = TestClient(app)
        response = client.get("/")
        
        # Verify the response
        assert response.status_code == 200
        assert response.json()["status"] == "MCP SSE Server Running"
        assert response.json()["version"] == "0.2.3"

def test_sse_route_connection():
    """Test the SSE route calls the handle_sse function."""
    with mock.patch('cursor_notebook_mcp.sse_transport.handle_sse') as mock_handle_sse:
        # Set up the mock to return a valid response
        mock_handle_sse.return_value = PlainTextResponse("OK")
        
        # Create a Starlette app
        mcp_server = mock.MagicMock()
        config = mock.MagicMock()
        
        # Set a concrete value for log_level to avoid MagicMock comparison error
        config.log_level = logging.INFO  # Set to a concrete value, not a MagicMock
        
        app = create_starlette_app(mcp_server, config)
        
        # Use the test client
        from starlette.testclient import TestClient
        client = TestClient(app)
        
        # Note: We're testing for the function call, not the response
        # Older versions of Starlette might have warned about streaming responses,
        # but current behavior is different, so we skip that expectation
        try:
            client.get("/sse")
        except Exception as e:
            # The test client may raise an exception since SSE connections
            # don't complete normally - this is expected behavior
            pass
        
        # Verify handle_sse was called, which is the primary test objective
        assert mock_handle_sse.called

# --- SSE Endpoint Tests ---

@pytest.mark.asyncio
async def test_handle_sse_successful_connection():
    """Test successful SSE connection setup and MCP session."""
    # Create a mock request with all necessary components
    request = MockRequest()
    
    # Set up the transport context manager return value
    mock_streams = (mock.MagicMock(), mock.MagicMock())
    request.app.state.sse_transport.connect_sse.return_value.__aenter__.return_value = mock_streams
    
    # Set up the underlying MCP server
    mock_underlying_server = mock.MagicMock()
    mock_options = mock.MagicMock()
    mock_underlying_server.create_initialization_options.return_value = mock_options
    request.app.state.mcp_server._mcp_server = mock_underlying_server
    
    # Call the handler
    await handle_sse(request)
    
    # Verify the transport was connected properly
    request.app.state.sse_transport.connect_sse.assert_called_once_with(
        request.scope, request.receive, request._send
    )
    
    # Verify the MCP server was run with the correct parameters
    mock_underlying_server.run.assert_called_once_with(
        read_stream=mock_streams[0],
        write_stream=mock_streams[1],
        initialization_options=mock_options
    )

@pytest.mark.asyncio
async def test_handle_sse_exception_during_connection():
    """Test SSE connection that raises an exception."""
    # Create a mock request
    request = MockRequest()
    
    # Make the connect_sse raise an exception
    request.app.state.sse_transport.connect_sse.side_effect = Exception("Connection failed")
    
    # Call the handler (should not raise the exception)
    await handle_sse(request)
    
    # Verify connection was attempted
    request.app.state.sse_transport.connect_sse.assert_called_once()
    
    # No need to verify more as the function just logs and returns

@pytest.mark.asyncio
async def test_handle_root():
    """Test the root endpoint handler."""
    # Create a mock request
    request = MockRequest()
    request.app.state.config.version = "0.2.3"
    
    # Call the handler
    response = await handle_root(request)
    
    # Verify the response
    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    
    # For JSONResponse, we need to directly access the body and decode it
    response_body = json.loads(response.body.decode())
    assert response_body == {
        "status": "MCP SSE Server Running", 
        "version": "0.2.3"
    }

@pytest.mark.asyncio
async def test_http_exception_handler():
    """Test HTTP exception handler."""
    # Create a mock request
    request = MockRequest()
    
    # Create an HTTP exception
    exc = HTTPException(status_code=404, detail="Resource not found")
    
    # Call the handler
    response = await http_exception_handler(request, exc)
    
    # Verify the response
    assert isinstance(response, PlainTextResponse)
    assert response.status_code == 404
    assert "Resource not found" in response.body.decode()

@pytest.mark.asyncio
async def test_generic_exception_handler():
    """Test generic exception handler."""
    # Create a mock request
    request = MockRequest()
    
    # Create a generic exception
    exc = Exception("Something went wrong")
    
    # Call the handler
    response = await generic_exception_handler(request, exc)
    
    # Verify the response
    assert isinstance(response, PlainTextResponse)
    assert response.status_code == 500
    assert "Internal Server Error" in response.body.decode()

def test_create_starlette_app():
    """Test Starlette app creation."""
    # Create mock MCP server and config
    mcp_server = mock.MagicMock()
    config = mock.MagicMock()
    config.log_level = logging.DEBUG
    
    # Mock SseServerTransport to return a mock instead of actual transport
    with mock.patch('cursor_notebook_mcp.sse_transport.SseServerTransport') as mock_transport_class:
        mock_transport = mock.MagicMock()
        mock_transport_class.return_value = mock_transport
        
        # Call the function
        app = create_starlette_app(mcp_server, config)
        
        # Verify the app settings
        assert isinstance(app, Starlette)
        assert app.state.mcp_server == mcp_server
        assert app.state.config == config
        assert app.state.sse_transport == mock_transport
        assert app.debug is True  # Since config.log_level = DEBUG
        
        # Verify SseServerTransport was created correctly
        mock_transport_class.assert_called_once_with(endpoint="/messages/")

def test_create_starlette_app_with_high_log_level():
    """Test Starlette app creation with a higher log level."""
    # Create mock MCP server and config
    mcp_server = mock.MagicMock()
    config = mock.MagicMock()
    config.log_level = logging.WARNING  # Higher than DEBUG
    
    # Mock SseServerTransport
    with mock.patch('cursor_notebook_mcp.sse_transport.SseServerTransport') as mock_transport_class:
        # Call the function
        app = create_starlette_app(mcp_server, config)
        
        # Debug mode should be off since log level is higher than DEBUG
        assert app.debug is False

def test_run_sse_server_success():
    """Test successful SSE server startup."""
    # Mock dependencies
    mcp_server = mock.MagicMock()
    config = mock.MagicMock()
    config.host = "127.0.0.1"
    config.port = 8080
    config.log_level = logging.INFO
    
    # Mock the imported dependencies and functions
    with mock.patch('cursor_notebook_mcp.sse_transport.SseServerTransport', mock.MagicMock()), \
         mock.patch('cursor_notebook_mcp.sse_transport.uvicorn') as mock_uvicorn, \
         mock.patch('cursor_notebook_mcp.sse_transport.create_starlette_app') as mock_create_app:
        
        # Mock create_starlette_app to return a Starlette app
        mock_app = mock.MagicMock()
        mock_create_app.return_value = mock_app
        
        # Call the function
        run_sse_server(mcp_server, config)
        
        # Verify app creation
        mock_create_app.assert_called_once_with(mcp_server, config)
        
        # Verify uvicorn.run was called with the right parameters
        mock_uvicorn.run.assert_called_once_with(
            mock_app,
            host=config.host,
            port=config.port,
            log_level=config.log_level
        )

def test_run_sse_server_missing_deps():
    """Test SSE server startup with missing dependencies."""
    # Mock dependencies
    mcp_server = mock.MagicMock()
    config = mock.MagicMock()
    
    # Test with SseServerTransport = None to simulate missing dependency
    with mock.patch('cursor_notebook_mcp.sse_transport.SseServerTransport', None), \
         mock.patch('cursor_notebook_mcp.sse_transport.uvicorn', None), \
         mock.patch('cursor_notebook_mcp.sse_transport._sse_import_error', Exception("No uvicorn")):
        
        # Call should raise ImportError
        with pytest.raises(ImportError, match="SSE transport requires additional packages"):
            run_sse_server(mcp_server, config)

def test_run_sse_server_uvicorn_import_error():
    """Test SSE server startup with uvicorn import error."""
    # Mock dependencies
    mcp_server = mock.MagicMock()
    config = mock.MagicMock()
    
    # Mock create_starlette_app but raise ImportError for uvicorn.run
    with mock.patch('cursor_notebook_mcp.sse_transport.SseServerTransport', mock.MagicMock()), \
         mock.patch('cursor_notebook_mcp.sse_transport.uvicorn', mock.MagicMock()), \
         mock.patch('cursor_notebook_mcp.sse_transport.create_starlette_app') as mock_create_app, \
         mock.patch.object(sys.modules['cursor_notebook_mcp.sse_transport'].uvicorn, 'run', 
                          side_effect=ImportError("uvicorn not installed")):
        
        # Mock create_starlette_app to return a Starlette app
        mock_app = mock.MagicMock()
        mock_create_app.return_value = mock_app
        
        # Call should raise ImportError
        with pytest.raises(ImportError):
            run_sse_server(mcp_server, config)

def test_run_sse_server_generic_exception():
    """Test SSE server startup with a generic exception."""
    # Mock dependencies
    mcp_server = mock.MagicMock()
    config = mock.MagicMock()
    
    # Mock create_starlette_app but raise an exception for uvicorn.run
    with mock.patch('cursor_notebook_mcp.sse_transport.SseServerTransport', mock.MagicMock()), \
         mock.patch('cursor_notebook_mcp.sse_transport.uvicorn', mock.MagicMock()), \
         mock.patch('cursor_notebook_mcp.sse_transport.create_starlette_app') as mock_create_app, \
         mock.patch.object(sys.modules['cursor_notebook_mcp.sse_transport'].uvicorn, 'run', 
                          side_effect=Exception("Server startup failed")):
        
        # Mock create_starlette_app to return a Starlette app
        mock_app = mock.MagicMock()
        mock_create_app.return_value = mock_app
        
        # Call should raise the original exception
        with pytest.raises(Exception, match="Server startup failed"):
            run_sse_server(mcp_server, config) 