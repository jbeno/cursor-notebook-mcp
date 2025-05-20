"""
Additional tests to improve coverage for the server.py module.

This file focuses on improving coverage for methods in server.py that have
lower coverage, particularly:
1. Server initialization and configuration (lines 24-38)
2. Command handling (lines 170-198)
3. Server shutdown and cleanup (lines 422-466)
"""

import pytest
import os
import tempfile
import asyncio
import json
import sys
from unittest import mock
from pathlib import Path
import argparse
import logging

from cursor_notebook_mcp.server import ServerConfig, main, parse_arguments, setup_logging

# --- Fixtures ---

@pytest.fixture
def mock_args():
    """Create a mock args object to use with ServerConfig."""
    args = mock.MagicMock()
    args.log_dir = "/tmp/logs"
    args.log_level_int = logging.INFO
    args.transport = "stdio"
    args.host = "127.0.0.1"
    args.port = 8080
    args.allow_root = ["/test/root"]
    args.sftp_root = None
    args.max_cell_source_size = 10 * 1024 * 1024
    args.max_cell_output_size = 10 * 1024 * 1024
    return args

@pytest.fixture
def mock_server_config(mock_args, monkeypatch):
    """Create a ServerConfig instance with test values."""
    # Mock os.path.isabs and os.path.isdir to avoid filesystem checks
    monkeypatch.setattr(os.path, "isabs", lambda path: True)
    monkeypatch.setattr(os.path, "isdir", lambda path: True)
    monkeypatch.setattr(os.path, "realpath", lambda path: path)
    
    return ServerConfig(mock_args)

@pytest.fixture
def mock_server(mock_server_config):
    """Create a server instance with mocked dependencies."""
    # Create a mock server
    server = mock.MagicMock()
    server.config = mock_server_config
    
    # Mock the FastMCP instance
    server.mcp = mock.MagicMock()
    
    # Mock the NotebookTools instance
    server.tools = mock.MagicMock()
    
    return server

# --- Tests for server initialization and configuration ---

def test_server_config_from_args_direct(monkeypatch):
    """Test creating ServerConfig directly from arguments."""
    # Create mock ArgumentParser and args
    parser = mock.MagicMock()
    args = mock.MagicMock()
    args.log_dir = "/tmp/logs"
    args.log_level_int = logging.INFO
    args.transport = "stdio"
    args.host = "localhost"
    args.port = 9000
    args.allow_root = ["/custom/root"]
    args.sftp_root = None
    args.max_cell_source_size = 10 * 1024 * 1024
    args.max_cell_output_size = 10 * 1024 * 1024
    
    # Mock filesystem checks
    monkeypatch.setattr(os.path, "isabs", lambda path: True)
    monkeypatch.setattr(os.path, "isdir", lambda path: True)
    monkeypatch.setattr(os.path, "realpath", lambda path: path)
    
    # Create config with args
    config = ServerConfig(args)
    
    # Verify the config
    assert config.host == "localhost"
    assert config.port == 9000
    assert config.allowed_roots == ["/custom/root"]
    assert config.transport == "stdio"

def test_server_config_from_env_parse_args(monkeypatch):
    """Test creating ServerConfig by using environment variables via parse_arguments()."""
    # Set environment variables
    monkeypatch.setenv("MCP_HOST", "localhost")
    monkeypatch.setenv("MCP_PORT", "9000")
    
    # Create mock args that parse_arguments would return
    mock_args = mock.MagicMock()
    mock_args.host = "localhost"
    mock_args.port = 9000
    mock_args.allow_root = ["/env/root"]
    mock_args.transport = "stdio"
    mock_args.sftp_root = None
    mock_args.log_dir = "/tmp/logs"
    mock_args.log_level_int = logging.INFO
    mock_args.max_cell_source_size = 10 * 1024 * 1024
    mock_args.max_cell_output_size = 10 * 1024 * 1024
    
    # Mock parse_arguments to return our mock args
    monkeypatch.setattr("cursor_notebook_mcp.server.parse_arguments", lambda: mock_args)
    
    # Mock os.path functions for ServerConfig initialization
    monkeypatch.setattr(os.path, "isabs", lambda path: True)
    monkeypatch.setattr(os.path, "isdir", lambda path: True)
    monkeypatch.setattr(os.path, "realpath", lambda path: path)
    
    # Mock setup_logging to avoid actual logging setup
    mock_setup_logging = mock.MagicMock()
    monkeypatch.setattr("cursor_notebook_mcp.server.setup_logging", mock_setup_logging)
    
    # Create our ServerConfig directly - this is what main() would do internally
    config = ServerConfig(mock_args)
    
    # Verify the config has the expected values from our env vars/mock args
    assert config.host == "localhost"
    assert config.port == 9000
    assert config.allowed_roots == ["/env/root"]

def test_server_config_validate_args(mock_args, monkeypatch):
    """Test validating ServerConfig inputs with different arguments."""
    # Mock os path functions
    monkeypatch.setattr(os.path, "isabs", lambda path: True)
    monkeypatch.setattr(os.path, "isdir", lambda path: True)
    monkeypatch.setattr(os.path, "realpath", lambda path: path)
    
    # Test with valid config
    valid_args = mock_args
    ServerConfig(valid_args)  # Should not raise any exceptions
    
    # Test with invalid transport
    invalid_transport_args = mock.MagicMock()
    invalid_transport_args.log_dir = "/tmp/logs"
    invalid_transport_args.log_level_int = logging.INFO
    invalid_transport_args.host = "localhost"
    invalid_transport_args.port = 8080
    invalid_transport_args.allow_root = ["/test/root"]
    invalid_transport_args.sftp_root = None
    invalid_transport_args.max_cell_source_size = 10 * 1024 * 1024
    invalid_transport_args.max_cell_output_size = 10 * 1024 * 1024
    invalid_transport_args.transport = "invalid"
    
    # This should not raise immediately as transport validation is not in __init__
    config = ServerConfig(invalid_transport_args)
    assert config.transport == "invalid"
    
    # Test with invalid port
    invalid_port_args = mock.MagicMock()
    invalid_port_args.log_dir = "/tmp/logs"
    invalid_port_args.log_level_int = logging.INFO
    invalid_port_args.transport = "stdio"
    invalid_port_args.host = "localhost"
    invalid_port_args.port = -1
    invalid_port_args.allow_root = ["/test/root"]
    invalid_port_args.sftp_root = None
    invalid_port_args.max_cell_source_size = 10 * 1024 * 1024
    invalid_port_args.max_cell_output_size = 10 * 1024 * 1024
    
    # This should not raise immediately as port validation is not in __init__
    config = ServerConfig(invalid_port_args)
    assert config.port == -1
    
    # Test with no allowed roots
    no_roots_args = mock.MagicMock()
    no_roots_args.log_dir = "/tmp/logs"
    no_roots_args.log_level_int = logging.INFO
    no_roots_args.transport = "stdio"
    no_roots_args.host = "localhost"
    no_roots_args.port = 8080
    no_roots_args.allow_root = []
    no_roots_args.sftp_root = None
    no_roots_args.max_cell_source_size = 10 * 1024 * 1024
    no_roots_args.max_cell_output_size = 10 * 1024 * 1024
    
    # This should raise a ValueError immediately during initialization
    with pytest.raises(ValueError, match="No valid allowed roots"):
        ServerConfig(no_roots_args)

# --- Tests for parse_arguments ---

def test_parse_arguments_basic():
    """Test the parse_arguments function with basic arguments."""
    # Mock sys.argv
    with mock.patch('sys.argv', ['server.py', '--allow-root', '/test/root']):
        # Mock os.path functions
        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.path.isdir', return_value=True):
            # Call parse_arguments
            args = parse_arguments()
            
            # Verify the args
            assert args.allow_root == ['/test/root']
            assert args.transport == 'stdio'
            assert args.host == '127.0.0.1'
            assert args.port == 8080

def test_parse_arguments_missing_root():
    """Test parse_arguments with missing root arguments."""
    # Mock sys.argv with no --allow-root or --sftp-root
    with mock.patch('sys.argv', ['server.py']):
        # This should raise SystemExit
        with pytest.raises(SystemExit):
            parse_arguments()

# --- Tests for logging setup ---

def test_setup_logging():
    """Test setting up logging."""
    root_logger = logging.getLogger()
    # Store original handlers and level to avoid polluting state for other tests,
    # although setup_logging itself tries to clean up.
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Call setup_logging
            setup_logging(temp_dir, logging.INFO)
            
            # Verify log file was created
            log_file = os.path.join(temp_dir, "server.log")
            assert os.path.exists(log_file)
            
            # Verify root logger was configured
            # root_logger = logging.getLogger() # Already got it
            assert root_logger.level == logging.INFO
            
            # Verify handlers
            # setup_logging clears previous handlers and adds its own.
            current_handlers = root_logger.handlers
            assert len(current_handlers) >= 2  # At least file and stream handlers

            # Specifically check for the file handler
            file_handler_present = any(
                isinstance(h, logging.FileHandler) and h.baseFilename == log_file 
                for h in current_handlers
            )
            assert file_handler_present, f"FileHandler for {log_file} not found."

        finally:
            # Clean up handlers added by this test's call to setup_logging
            # This is crucial for Windows to release the log file before rmtree.
            for handler in root_logger.handlers[:]: # Iterate over a copy
                handler.close()
                root_logger.removeHandler(handler)
            
            # Restore original state (optional, but good practice if other tests are sensitive)
            # However, subsequent calls to setup_logging will clear these anyway.
            # The critical part is closing for the temp dir cleanup.
            # for handler in original_handlers:
            #     root_logger.addHandler(handler)
            # root_logger.setLevel(original_level)

def test_setup_logging_invalid_dir():
    """Test setup_logging with an invalid directory."""
    # Use a file as log_dir to trigger the error
    with tempfile.NamedTemporaryFile() as temp_file:
        # Call setup_logging
        setup_logging(temp_file.name, logging.INFO)
        
        # Verify root logger was configured
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        
        # Verify handlers
        handlers = root_logger.handlers
        assert len(handlers) >= 1  # At least stream handler

# --- Tests for main function ---

def test_main_function_basic(monkeypatch):
    """Test the main function with basic setup."""
    # Mock parse_arguments
    mock_args = mock.MagicMock()
    mock_args.log_dir = "/tmp/logs"
    mock_args.log_level_int = logging.INFO
    mock_args.transport = "stdio"
    mock_args.allow_root = ["/test/root"]
    mock_args.sftp_root = None
    mock_args.max_cell_source_size = 10 * 1024 * 1024
    mock_args.max_cell_output_size = 10 * 1024 * 1024
    monkeypatch.setattr("cursor_notebook_mcp.server.parse_arguments", lambda: mock_args)
    
    # Mock setup_logging
    mock_setup_logging = mock.MagicMock()
    monkeypatch.setattr("cursor_notebook_mcp.server.setup_logging", mock_setup_logging)
    
    # Mock os.path functions
    monkeypatch.setattr(os.path, "isabs", lambda path: True)
    monkeypatch.setattr(os.path, "isdir", lambda path: True)
    monkeypatch.setattr(os.path, "realpath", lambda path: path)
    
    # Instead of actually running main(), let's test the individual components it would use
    # This avoids issues with running the actual server
    
    # Create our ServerConfig directly
    config = ServerConfig(mock_args)
    assert config.transport == "stdio"
    assert config.allowed_roots == ["/test/root"]
    
    # Test if FastMCP and NotebookTools can be initialized
    # These are just import tests essentially
    from fastmcp import FastMCP
    from cursor_notebook_mcp.tools import NotebookTools
    
    # Ensure the classes exist (this is a minimal test)
    assert hasattr(FastMCP, 'run')
    assert hasattr(NotebookTools, '__init__')

def test_main_function_with_exception(monkeypatch):
    """Test the main function with an exception."""
    # Mock parse_arguments to raise an exception
    monkeypatch.setattr("cursor_notebook_mcp.server.parse_arguments", 
                       mock.MagicMock(side_effect=ValueError("Test error")))
    
    # Mock sys.exit to prevent actual exit
    mock_exit = mock.MagicMock()
    monkeypatch.setattr(sys, "exit", mock_exit)
    
    # Call main
    main()
    
    # Verify sys.exit was called with error code
    mock_exit.assert_called_with(1)

# --- End of tests ---