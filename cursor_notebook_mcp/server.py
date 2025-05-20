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
import tempfile
import getpass

# --- Package-Internal Imports ---
# Ensure these succeed when running as part of the package
try:
    from .tools import NotebookTools
    from .sftp_manager import SFTPManager
except ImportError as e:
    print(f"Error importing package components: {e}. Ensure package structure is correct.", file=sys.stderr)
    sys.exit(1)

# --- External Dependencies ---
try:
    from fastmcp import FastMCP
    import nbformat
    # from fastmcp.server import run_http_server # No longer needed, FastMCP.run() handles it
except ImportError as e:
    # This might occur if dependencies aren't installed correctly
    print(f"FATAL: Failed to import required libraries (mcp, nbformat, fastmcp, etc.). Error: {e}", file=sys.stderr)
    # FastMCP's http transport might have its own dependencies, typically uvicorn and starlette.
    # FastMCP bundles these if installed with 'http' extra: pip install fastmcp[http]
    if "uvicorn" in str(e) or "starlette" in str(e):
        print("Hint: HTTP transport requires optional dependencies. Try: pip install \"fastmcp[http]\"", file=sys.stderr)
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
    sftp_manager: Any = None
    raw_sftp_specs: List[str] = [] # Added to store original --sftp-root args
    version: str = "0.3.0" # Dynamic version injected at build time or read from __init__

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
                
        # Handle SFTP remote roots if specified
        self.sftp_manager = None
        if args.sftp_root:
            try:
                # Initialize SFTP manager
                self.sftp_manager = SFTPManager()
                self.raw_sftp_specs = args.sftp_root # Store the original spec list
                
                # Process each remote path
                for remote_spec in args.sftp_root:
                    # Create a virtual local path to represent remote path
                    local_virtual_path = os.path.join(
                        tempfile.gettempdir(), 
                        f"cursor_notebook_sftp_{abs(hash(remote_spec))}"
                    )
                    os.makedirs(local_virtual_path, exist_ok=True)
                    
                    # Add path mapping
                    username, host, _ = self.sftp_manager.add_path_mapping(remote_spec, local_virtual_path)
                    
                    # Get password if needed
                    password = None
                    if args.sftp_password:
                        password = args.sftp_password
                    elif args.sftp_key is None and not args.sftp_no_interactive:
                        # Only prompt if not using key file and interactive mode is enabled
                        if args.sftp_no_password_prompt:
                            # Skip the password prompt but still allow interactive auth
                            # This is useful when using SSH agent or when 2FA is the only interactive part
                            password = None
                        else:
                            # Prompt for password
                            password = getpass.getpass(f"SSH password for {username}@{host}: ")
                    
                    # Establish connection
                    if not self.sftp_manager.add_connection(
                        host, 
                        username, 
                        password=password, 
                        key_file=args.sftp_key,
                        port=args.sftp_port,
                        use_agent=not args.sftp_no_agent,
                        interactive=not args.sftp_no_interactive,
                        auth_mode=args.sftp_auth_mode
                    ):
                        raise ValueError(f"Failed to connect to {remote_spec}")
                    
                    # Add the virtual local path to allowed roots
                    validated_roots.append(os.path.realpath(local_virtual_path))
                    logging.info(f"Added virtual root mapping: {remote_spec} -> {local_virtual_path}")
            except ImportError as e:
                logging.error(f"Cannot use SFTP: {e}")
                raise ValueError(f"SFTP support requires paramiko: {e}")
                
        if not validated_roots:
            raise ValueError("No valid allowed roots (either local or remote) provided.")
            
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
        required=False,  # Not required if SFTP root is provided
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
        choices=['stdio', 'streamable-http', 'sse'],
        help='Transport type to use. "streamable-http" is recommended for web deployments; "sse" provides a deprecated two-endpoint SSE model.'
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
    
    # SFTP connection arguments
    sftp_group = parser.add_argument_group('SFTP Options')
    sftp_group.add_argument(
        '--sftp-root',
        action='append',
        metavar='USER@HOST:/PATH',
        help='Remote path to access via SFTP. Format: user@host:/path. Can be used multiple times.'
    )
    sftp_group.add_argument(
        '--sftp-password',
        type=str,
        help='SSH password for authentication (not recommended for security; prefer key-based auth).'
    )
    sftp_group.add_argument(
        '--sftp-key',
        type=str,
        metavar='KEY_FILE',
        help='SSH private key file for authentication.'
    )
    sftp_group.add_argument(
        '--sftp-port',
        type=int,
        default=22,
        help='SSH port for SFTP connections.'
    )
    sftp_group.add_argument(
        '--sftp-no-interactive',
        action='store_true',
        help='Disable interactive authentication (2FA will not work with this flag).'
    )
    sftp_group.add_argument(
        '--sftp-no-agent',
        action='store_true',
        help='Disable SSH agent authentication.'
    )
    sftp_group.add_argument(
        '--sftp-no-password-prompt',
        action='store_true',
        help='Skip password prompt but allow other interactive auth (useful with SSH agent or 2FA-only).'
    )
    sftp_group.add_argument(
        '--sftp-auth-mode',
        choices=['auto', 'key', 'password', 'key+interactive', 'password+interactive', 'interactive'],
        default='auto',
        help='Authentication mode (default: auto)'
    )

    args = parser.parse_args()
    args.log_level_int = getattr(logging, args.log_level.upper())
    if os.path.exists(args.log_dir) and not os.path.isdir(args.log_dir):
         parser.error(f"--log-dir must be a directory path, not a file: {args.log_dir}")
         
    # Ensure we have at least one root type
    if not args.allow_root and not args.sftp_root:
        parser.error("At least one of --allow-root or --sftp-root must be specified")
        
    return args

# --- Main Execution Function (called by script entry point) ---
def main():
    """Parses arguments, sets up logging, initializes MCP, and runs the server."""
    args = None
    config = None
    logger = None

    try:
        args = parse_arguments()
        setup_logging(args.log_dir, args.log_level_int) 
        logger = logging.getLogger(__name__) # Get logger *after* setup
        config = ServerConfig(args)
    except (SystemExit, ValueError) as e: # Catch config/argparse errors
        # Use basicConfig for logging if setup_logging hasn't run or failed early
        logging.basicConfig(level=logging.ERROR) 
        logging.exception("Configuration failed critically") # Log exception info
        print(f"ERROR: Configuration failed: {e}", file=sys.stderr)
        return sys.exit(e.code if isinstance(e, SystemExit) else 1)
    except Exception as e: # Catch unexpected errors during setup (like logging setup itself)
        # Use basicConfig as fallback if setup_logging failed
        logging.basicConfig(level=logging.ERROR) 
        logging.exception("Critical failure during initial setup") # Log the exception
        print(f"CRITICAL: Failed during argument parsing or validation: {e}", file=sys.stderr)
        return sys.exit(1)

    # Set a default logger if it's still None (shouldn't happen now, but defensive)
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.warning("Logger was not initialized correctly during setup.")

    logger.info(f"Notebook MCP Server starting (Version: {config.version}) - via {__name__}")
    logger.info(f"Allowed Roots: {config.allowed_roots}")
    logger.info(f"Transport Mode: {config.transport}")
    if config.transport == 'sse':
        logger.info(f"SSE Endpoint: http://{config.host}:{config.port}")
    if config.sftp_manager:
        logger.info(f"SFTP connections active")
    logger.debug(f"Full configuration: {config.__dict__}")

    try:
        mcp_server = FastMCP("notebook_mcp")
        tool_provider = NotebookTools(config, mcp_server)
        logger.info("Notebook tools initialized and registered.")
    except Exception as e:
        # *** Add basicConfig fallback here too if logger failed ***
        if logger is None: logging.basicConfig(level=logging.ERROR)
        logger.exception("Failed to initialize MCP server or tools.")
        return sys.exit(1)

    try:
        if config.transport == 'stdio':
            logger.info("Running server via stdio...")
            mcp_server.run(transport='stdio')
            logger.info("Server finished (stdio).")
        
        elif config.transport == 'streamable-http':
            logger.info(f"Running server using FastMCP's Streamable HTTP transport...")
            try:
                mcp_server.run(
                    transport="streamable-http",
                    host=config.host,
                    port=config.port,
                    log_level=logging.getLevelName(config.log_level).lower()
                )
            except ImportError as e:
                logger.error(f"Failed to start Streamable HTTP server due to missing packages: {e}")
                logger.error("Hint: HTTP transport requires optional dependencies. Try: pip install \"fastmcp[http]\"")
                return sys.exit(1)
            except Exception as e:
                logger.exception("Failed to start or run FastMCP Streamable HTTP server.")
                return sys.exit(1)
            logger.info("Server finished (Streamable HTTP).")

        elif config.transport == 'sse': 
            logger.info(f"Running server using FastMCP's (deprecated) SSE transport...")
            try:
                mcp_server.run(
                    transport="sse",
                    host=config.host,
                    port=config.port,
                    log_level=logging.getLevelName(config.log_level).lower()
                    # Default paths for 'sse' are typically /sse (GET) and /messages (POST)
                    # path="/custom_sse_path" # if you need to customize sse path
                    # message_path_prefix="/custom_message_path" # if you need to customize message path
                )
            except ImportError as e: 
                logger.error(f"Failed to start SSE server due to missing packages: {e}")
                logger.error("Hint: SSE transport requires optional dependencies. Try: pip install \"fastmcp[http]\" (provides SSE dependencies too)")
                return sys.exit(1)
            except Exception as e:
                logger.exception("Failed to start or run FastMCP SSE server.")
                return sys.exit(1)
            logger.info("Server finished (SSE).")
            
        else:
            logger.error(f"Internal Error: Invalid transport specified in validated config: {config.transport}")
            return sys.exit(1)
            
    except Exception as e:
        logger.exception("Server encountered a fatal error during execution.")
        return sys.exit(1)
    finally:
        # Clean up SFTP connections
        if config.sftp_manager:
            logger.info("Closing SFTP connections...")
            config.sftp_manager.close_all()

# If this script is run directly (e.g., python -m cursor_notebook_mcp.server)
if __name__ == "__main__":
    print("Running server module directly...") 
    main() 