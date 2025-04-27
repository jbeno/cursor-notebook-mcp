"""
SSE Transport implementation for the Notebook MCP Server.

Uses Starlette and Uvicorn to serve the MCP server over SSE.
"""

import logging
import asyncio
from typing import Any

from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.errors import ServerErrorMiddleware

from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport

# Use try-except for optional SSE dependencies
try:
    import uvicorn
except ImportError as e:
    # Define dummy classes or raise a more informative error later if SSE is selected
    SseServerTransport = None
    uvicorn = None
    _sse_import_error = e
else:
    _sse_import_error = None

logger = logging.getLogger(__name__)

# Assuming ServerConfig type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .server import ServerConfig

# Define the SSE handler endpoint
async def handle_sse(request):
    """Handles incoming SSE connections and delegates to MCP transport."""
    mcp_server = request.app.state.mcp_server
    # Retrieve the shared transport instance
    transport = request.app.state.sse_transport 
    config = request.app.state.config
    client_host = request.client.host
    client_port = request.client.port
    log_prefix = f"[SSE {client_host}:{client_port}]"
    logger.info(f"{log_prefix} New SSE connection request from {client_host}:{client_port}")

    try:
        logger.debug(f"{log_prefix} Setting up SSE connection")
        # connect_sse handles the SSE handshake and provides streams
        async with transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            read_stream, write_stream = streams
            logger.info(f"{log_prefix} Connection established. Running MCP session.")
            
            # Get the underlying server instance that has the necessary methods
            underlying_server = mcp_server._mcp_server
            
            # Create options using the underlying server instance
            init_options = underlying_server.create_initialization_options()
            
            # Call run() on the underlying server instance, passing streams and options
            await underlying_server.run(
                read_stream=read_stream, 
                write_stream=write_stream,
                initialization_options=init_options
            )
            logger.info(f"{log_prefix} MCP session finished.")
    except Exception as e:
        # Log errors during the SSE connection or MCP run phase
        logger.error(f"{log_prefix} Error during SSE connection with {client_host}:{client_port}: {e}", exc_info=True)
        # Optionally re-raise or return an error response if possible (often too late)
        # Starlette might handle sending a 500 if the connection is still open
    finally:
        logger.info(f"{log_prefix} Closing SSE connection.")
        # Cleanup handled by context managers

# Define the root status endpoint
async def handle_root(request):
    """Simple status endpoint for the root path."""
    config = request.app.state.config
    return JSONResponse({"status": "MCP SSE Server Running", "version": config.version})

# Define exception handler
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP Exception: {exc.status_code} {exc.detail} for {request.url}")
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)

async def generic_exception_handler(request, exc):
    logger.exception(f"Unhandled exception for {request.url}: {exc}")
    return PlainTextResponse("Internal Server Error", status_code=500)

exception_handlers = {
    HTTPException: http_exception_handler,
    Exception: generic_exception_handler
}

# Function to create the Starlette app (refactored)
def create_starlette_app(mcp_server: FastMCP, config: 'ServerConfig') -> Starlette:
    """Creates the Starlette application instance."""
    # Create the SSE transport instance once, passing the **correct** endpoint path
    # This is the path the client will be instructed to POST messages back to.
    transport = SseServerTransport(endpoint="/messages/")
    
    routes = [
        Route("/", endpoint=handle_root, methods=["GET"]), # Root info
        Route("/sse", endpoint=handle_sse), # Initial SSE connection (GET)
        Mount("/messages", app=transport.handle_post_message) # Client message handler (POST)
    ]
    middleware = [
        Middleware(ServerErrorMiddleware, handler=generic_exception_handler)
    ]
    
    app = Starlette(
        routes=routes, 
        middleware=middleware,
        exception_handlers=exception_handlers,
        debug=config.log_level <= logging.DEBUG # Enable debug mode based on log level
    )
    # Store shared instances in app state
    app.state.mcp_server = mcp_server
    app.state.config = config
    app.state.sse_transport = transport # Store the transport instance
    return app


# Main function to run the SSE server
def run_sse_server(mcp_server: FastMCP, config: 'ServerConfig'):
    """Configures and runs the Uvicorn server for SSE transport."""
    if SseServerTransport is None or uvicorn is None:
        logger.error(f"SSE transport requires additional packages ('mcp-sdk[sse]', 'uvicorn'). Install error: {_sse_import_error}")
        raise ImportError(
            "SSE transport requires additional packages. "
            f"Please install with 'pip install \"cursor-notebook-mcp[sse]\"' or 'pip install uvicorn mcp-sdk[sse]'. Error: {_sse_import_error}"
        ) from _sse_import_error

    try:
        # Create the Starlette app
        app = create_starlette_app(mcp_server, config)
        
        logger.info(f"Starting Uvicorn server on {config.host}:{config.port}")
        uvicorn.run(
            app, 
            host=config.host, 
            port=config.port,
            log_level=config.log_level # Pass log level to uvicorn
        )
    except ImportError:
        logger.error("Failed to import uvicorn. Please install with '[sse]' extra: pip install cursor-notebook-mcp[sse]")
        raise # Re-raise for main server loop to catch
    except Exception as e:
        logger.exception(f"Failed to start or run Uvicorn server: {e}")
        raise # Re-raise for main server loop to catch 