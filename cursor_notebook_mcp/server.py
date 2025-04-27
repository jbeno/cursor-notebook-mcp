"""
Server entry point logic within the package.

Handles argument parsing, configuration, logging setup,
and launching the appropriate transport.
"""

import asyncio
import sys
import os
import argparse
from typing import Any, List, Dict
import logging
import re # Import re for the filter

# --- Package-Internal Imports ---
# Ensure these succeed when running as part of the package
try:
    from .tools import NotebookTools
    from .sse_transport import run_sse_server
except ImportError as e:
    print(f"Error importing package components: {e}. Ensure package structure is correct.", file=sys.stderr)
    sys.exit(1)

# --- External Dependencies ---
try:
    from mcp.server.fastmcp import FastMCP
    import nbformat
except ImportError as e:
    # This might occur if dependencies aren't installed correctly
    print(f"FATAL: Failed to import required libraries (mcp, nbformat, etc.). Error: {e}", file=sys.stderr)
    # Add hint about installing extras if it's an SSE component missing
    if "SseServerTransport" in str(e) or "uvicorn" in str(e) or "starlette" in str(e):
        print("Hint: SSE transport requires optional dependencies. Try installing with '[sse]' extra.", file=sys.stderr)
    sys.exit(1)

# --- Logging Setup ---
DEFAULT_LOG_DIR = os.path.expanduser("~/.cursor_notebook_mcp")
DEFAULT_LOG_LEVEL = logging.INFO

# Define the custom filter
class TraitletsValidationFilter(logging.Filter):
    """Filters out specific 'Additional properties are not allowed ('id' was unexpected)' errors from traitlets."""
    # Pre-compile regex for efficiency
    _pattern = re.compile(r"Notebook JSON is invalid: Additional properties are not allowed \('id' was unexpected\)")

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False to suppress the log record, True otherwise."""
        if record.name == 'traitlets' and record.levelno == logging.ERROR:
            # Check if the message matches the specific pattern we want to suppress
            if self._pattern.search(record.getMessage()):
                return False # Suppress this specific log message
        return True # Allow all other messages

def setup_logging(log_dir: str, log_level: int):
    """Configures the root logger based on provided parameters."""
    log_file = os.path.join(log_dir, "server.log")

    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Could not create log directory {log_dir}: {e}", file=sys.stderr)
        log_dir = None
        log_file = None

    logger = logging.getLogger() # Get root logger
    logger.setLevel(log_level)

    # Remove existing handlers to prevent duplicate logs on re-run
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # Use a more detailed formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')

    # File Handler
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8') # Specify encoding
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"WARNING: Could not set up file logging to {log_file}. Error: {e}", file=sys.stderr)
            log_file = None # Indicate failure

    # Stream Handler (stderr)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Apply the custom filter to the traitlets logger
    traitlets_logger = logging.getLogger('traitlets')
    # Ensure the logger's level isn't preventing ERRORs from reaching the filter
    # If the root logger level is INFO or DEBUG, this is fine.
    # If root is WARNING or higher, traitlets ERRORs might still be suppressed globally.
    # For robustness, ensure traitlets can process ERRORs for filtering:
    if traitlets_logger.level == 0 or traitlets_logger.level > logging.ERROR: # Check if level is NOTSET or higher than ERROR
         # If the logger's own level is restrictive, set it just enough to allow ERRORs
         # Note: This might slightly change behavior if it previously inherited a level > ERROR
         traitlets_logger.setLevel(logging.ERROR)

    traitlets_logger.addFilter(TraitletsValidationFilter())

    # Use root logger for initial messages
    initial_message = f"Logging initialized. Level: {logging.getLevelName(log_level)}."
    if log_file:
        logging.info(f"{initial_message} Log file: {log_file}")
    else:
        logging.info(f"{initial_message} Logging to stderr only.")

# --- Configuration Class ---
class ServerConfig:
    """Holds server configuration derived from arguments."""
    allowed_roots: List[str]
    max_cell_source_size: int
    max_cell_output_size: int
    log_dir: str
    log_level: int
    transport: str
    host: str
    port: int
    version: str = "0.2.3" # Dynamic version injected at build time or read from __init__

    def __init__(self, args: argparse.Namespace):
        self.log_dir = args.log_dir
        self.log_level = args.log_level_int
        self.transport = args.transport
        self.host = args.host
        self.port = args.port

        validated_roots = []
        if args.allow_root:
            for root in args.allow_root:
                if not os.path.isabs(root):
                    raise ValueError(f"--allow-root path must be absolute: {root}")
                if not os.path.isdir(root):
                    raise ValueError(f"--allow-root path must be an existing directory: {root}")
                validated_roots.append(os.path.realpath(root))
        self.allowed_roots = validated_roots

        if args.max_cell_source_size < 0:
            raise ValueError(f"--max-cell-source-size must be non-negative: {args.max_cell_source_size}")
        self.max_cell_source_size = args.max_cell_source_size

        if args.max_cell_output_size < 0:
             raise ValueError(f"--max-cell-output-size must be non-negative: {args.max_cell_output_size}")
        self.max_cell_output_size = args.max_cell_output_size

# --- Argument Parsing ---
def parse_arguments() -> argparse.Namespace:
    # Argument parser setup remains the same as before
    parser = argparse.ArgumentParser(
        description="Jupyter Notebook MCP Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--allow-root',
        action='append',
        required=True,
        metavar='DIR_PATH',
        help='Absolute path to a directory where notebooks are allowed. Can be used multiple times.'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default=DEFAULT_LOG_DIR,
        metavar='PATH',
        help='Directory to store log files.'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default=logging.getLevelName(DEFAULT_LOG_LEVEL),
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level.'
    )
    parser.add_argument(
        '--max-cell-source-size',
        type=int,
        default=10 * 1024 * 1024, # 10 MiB
        metavar='BYTES',
        help='Maximum allowed size (bytes) for a cell\'s source content.'
    )
    parser.add_argument(
        '--max-cell-output-size',
        type=int,
        default=10 * 1024 * 1024, # 10 MiB
        metavar='BYTES',
        help='Maximum allowed size (bytes) for a cell\'s serialized output.'
    )
    parser.add_argument(
        '--transport',
        type=str,
        default='stdio',
        choices=['stdio', 'sse'],
        help='Transport type to use.'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        metavar='IP_ADDR',
        help='Host to bind the SSE server to (only used with --transport=sse).'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        metavar='PORT_NUM',
        help='Port to bind the SSE server to (only used with --transport=sse).'
    )

    args = parser.parse_args()
    args.log_level_int = getattr(logging, args.log_level.upper())
    if os.path.exists(args.log_dir) and not os.path.isdir(args.log_dir):
         parser.error(f"--log-dir must be a directory path, not a file: {args.log_dir}")
    return args

# --- Main Execution Function (called by script entry point) ---
def main():
    """Parses arguments, sets up logging, initializes MCP, and runs the server."""
    args = None
    config = None
    logger = None

    try:
        args = parse_arguments()
        config = ServerConfig(args)
    except (SystemExit, ValueError) as e:
        print(f"ERROR: Configuration failed: {e}", file=sys.stderr)
        return sys.exit(e.code if isinstance(e, SystemExit) else 1)
    except Exception as e:
        print(f"CRITICAL: Failed during argument parsing or validation: {e}", file=sys.stderr)
        return sys.exit(1)

    try:
        setup_logging(config.log_dir, config.log_level)
        logger = logging.getLogger(__name__) # Get logger for this module (cursor_notebook_mcp.server)
    except Exception as e:
        print(f"CRITICAL: Failed during logging setup: {e}", file=sys.stderr)
        logging.basicConfig(level=logging.ERROR)
        logging.exception("Logging setup failed critically")
        return sys.exit(1)

    # Set a default logger if it's still None to prevent NoneType errors
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Notebook MCP Server starting (Version: {config.version}) - via {__name__}")
    logger.info(f"Allowed Roots: {config.allowed_roots}")
    logger.info(f"Transport Mode: {config.transport}")
    if config.transport == 'sse':
        logger.info(f"SSE Endpoint: http://{config.host}:{config.port}")
    logger.debug(f"Full configuration: {config.__dict__}")

    try:
        mcp_server = FastMCP("notebook_mcp")
        tool_provider = NotebookTools(config, mcp_server)
        logger.info("Notebook tools initialized and registered.")
    except Exception as e:
        logger.exception("Failed to initialize MCP server or tools.")
        return sys.exit(1)

    try:
        if config.transport == 'stdio':
            logger.info("Running server via stdio...")
            mcp_server.run(transport='stdio')
            logger.info("Server finished (stdio).")
        
        elif config.transport == 'sse':
            logger.info(f"Running server via SSE...")
            try:
                run_sse_server(mcp_server, config)
            except ImportError as e:
                logger.error(f"Failed to start SSE server due to missing packages: {e}")
                return sys.exit(1)
            except Exception as e:
                logger.exception("Failed to start or run SSE server.")
                return sys.exit(1)
            logger.info("Server finished (SSE).")
            
        else:
            logger.error(f"Internal Error: Invalid transport specified: {config.transport}")
            return sys.exit(1)
            
    except Exception as e:
        logger.exception("Server encountered a fatal error during execution.")
        return sys.exit(1)

# If this script is run directly (e.g., python -m cursor_notebook_mcp.server)
if __name__ == "__main__":
    print("Running server module directly...") 
    main() 