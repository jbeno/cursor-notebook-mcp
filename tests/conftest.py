"""
Pytest configuration and fixtures for notebook MCP server tests.
"""

import pytest
import os
import sys
import asyncio
from pathlib import Path
from typing import List, Callable
import uuid
import shutil # Add import for shutil
import logging # Added for logger in cleanup
# from contextlib import suppress # No longer needed if cleanup_asyncio is removed/simplified
# import threading # No longer needed if SafeEventLoopPolicy is removed

# Add project root to sys.path to allow importing the package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import necessary components from the package and server script
from cursor_notebook_mcp.server import ServerConfig
from cursor_notebook_mcp.tools import NotebookTools
from fastmcp import FastMCP

logger = logging.getLogger(__name__) # Define logger at module level if not already present

# --- Fixtures ---

@pytest.fixture(scope="session")
def temp_notebook_dir(tmp_path_factory) -> Path:
    """Create a temporary directory to act as the allowed root for notebooks."""
    # Create a unique directory for the test session
    # tmp_path_factory is a built-in pytest fixture
    temp_dir = tmp_path_factory.mktemp("notebook_tests_")
    print(f"Created temporary test directory: {temp_dir}")
    return temp_dir

@pytest.fixture(scope="function") # Use function scope for config if tests might modify it indirectly
def server_config(temp_notebook_dir: Path) -> ServerConfig:
    """Provides a ServerConfig instance configured for testing."""
    # Create a dummy args object matching what parse_arguments would produce
    class MockArgs:
        allow_root = [str(temp_notebook_dir)]
        log_dir = str(temp_notebook_dir / "logs") # Keep logs within temp dir
        log_level_int = 10 # DEBUG
        max_cell_source_size = 10 * 1024 * 1024
        max_cell_output_size = 10 * 1024 * 1024
        transport = 'stdio' # Default, doesn't matter for direct tool testing
        host = '127.0.0.1'
        port = 8080
        # Add SFTP defaults expected by ServerConfig.__init__
        sftp_root = None
        sftp_password = None
        sftp_key = None
        sftp_port = 22
        sftp_no_interactive = False
        sftp_no_agent = False
        sftp_no_password_prompt = False
        sftp_auth_mode = 'auto'
        
    return ServerConfig(MockArgs())

@pytest.fixture(scope="function")
def mcp_server_inst() -> FastMCP:
    """Provides a clean FastMCP instance for each test function."""
    return FastMCP("test_notebook_mcp")

@pytest.fixture(scope="function")
def notebook_tools_inst(server_config: ServerConfig, mcp_server_inst: FastMCP) -> NotebookTools:
    """Provides an initialized NotebookTools instance with registered tools."""
    # Instantiating NotebookTools registers tools on mcp_server_inst
    return NotebookTools(server_config, mcp_server_inst)

@pytest.fixture
def notebook_path_factory(temp_notebook_dir: Path) -> Callable[[], str]:
    """Provides a function to generate unique notebook paths within the test dir."""
    def _create_path() -> str:
        # Generate a unique filename
        filename = f"test_nb_{uuid.uuid4()}.ipynb"
        return str(temp_notebook_dir / filename)
    
    return _create_path

# Fixture to provide an async event loop for tests marked with @pytest.mark.asyncio
# We're using pytest-asyncio's built-in event loop fixture
# Instead of redefining it, we'll use a custom event loop policy

# REMOVED SafeEventLoopPolicy and event_loop_policy fixture

# REMOVED or COMMENTED OUT cleanup_asyncio fixture
# @pytest.fixture(autouse=True)
# def cleanup_asyncio(request, event_loop_policy): # event_loop_policy would also be removed
#     ...

@pytest.fixture(scope="session")
def cli_command_path() -> str:
    """
    Returns the absolute path to the installed cursor-notebook-mcp script
    within the current environment's bin directory. Skips tests if not found.
    """
    # Find the bin/Scripts directory associated with the current Python executable
    python_executable = sys.executable
    venv_bin_path = Path(python_executable).parent

    # Construct the expected path to the script
    script_name = "cursor-notebook-mcp"
    # Handle potential .exe extension on Windows
    if sys.platform == "win32":
        script_name += ".exe"

    script_path = venv_bin_path / script_name

    if not script_path.exists():
        # Alternative check using shutil.which, which searches PATH
        found_path = shutil.which(script_name)
        if found_path:
             script_path = Path(found_path)
        else:
             pytest.skip(f"'{script_name}' command not found in venv bin or PATH.")

    return str(script_path) 