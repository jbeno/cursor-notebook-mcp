"""
Tests for SFTP initialization in ServerConfig and specific main() function paths from server.py.
"""

import pytest
import os
import sys
import argparse
import logging
from unittest import mock

# Assuming conftest.py provides necessary base fixtures if any, though these tests are more focused.
from cursor_notebook_mcp.server import ServerConfig, main, parse_arguments
# We need to be able to import SFTPManager to mock it, or ensure it's mockable if not directly imported by tests.
# If SFTPManager is in the same package, direct import like below should work if __init__.py is set up.
# from cursor_notebook_mcp.sftp_manager import SFTPManager 
# For now, we'll mock it via its path as used in server.py

# Use pytest-asyncio for async tests if any main paths become async, though main itself is sync.
# For now, these tests are synchronous as they test config and main's setup.

@pytest.fixture
def mock_base_args(tmp_path):
    """Provides a base argparse.Namespace with common arguments for ServerConfig tests."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    args = argparse.Namespace(
        log_dir=str(log_dir),
        log_level_int=logging.INFO,
        transport='stdio',
        host='127.0.0.1',
        port=8080,
        allow_root=[str(tmp_path)], # Provide a default local root
        sftp_root=None,
        sftp_password=None,
        sftp_key=None,
        sftp_port=22,
        sftp_no_interactive=False,
        sftp_no_agent=False,
        sftp_no_password_prompt=False,
        sftp_auth_mode='auto',
        max_cell_source_size=1024*1024,
        max_cell_output_size=1024*1024
    )
    return args

# --- ServerConfig SFTP Initialization Tests (lines 174-199, 201-202) --- 

@mock.patch('cursor_notebook_mcp.server.SFTPManager')
@mock.patch('os.path.realpath', side_effect=lambda p: p) # Mock realpath to return path as is
@mock.patch('os.makedirs') # Mock makedirs for virtual path creation
@mock.patch('tempfile.gettempdir', return_value='/tmp/test_sftp_temp')
def test_server_config_sftp_with_key(
    mock_gettempdir, mock_os_makedirs, mock_realpath, mock_sftp_manager_class, mock_base_args, tmp_path
):
    """Test ServerConfig SFTP init with a key file."""
    mock_sftp_instance = mock_sftp_manager_class.return_value
    mock_sftp_instance.add_path_mapping.return_value = ("testuser", "testhost", "/remote/path")
    mock_sftp_instance.add_connection.return_value = True

    mock_base_args.allow_root = None # Focus on SFTP root
    mock_base_args.sftp_root = ["testuser@testhost:/remote/path"]
    mock_base_args.sftp_key = "/path/to/key.pem"
    mock_base_args.sftp_password = None # Explicitly no password if key is used
    mock_base_args.sftp_no_interactive = True # Common with key-based auth

    config = ServerConfig(mock_base_args)

    mock_sftp_manager_class.assert_called_once()

    expected_local_virtual_path = os.path.join(
        '/tmp/test_sftp_temp', # This is the mocked gettempdir()
        f'cursor_notebook_sftp_{str(abs(hash("testuser@testhost:/remote/path")))}'
    )
    mock_sftp_instance.add_path_mapping.assert_called_once_with(
        "testuser@testhost:/remote/path",
        expected_local_virtual_path
    )
    mock_sftp_instance.add_connection.assert_called_once_with(
        'testhost', 'testuser', 
        password=None, 
        key_file="/path/to/key.pem", 
        port=22, 
        use_agent=True, # default because sftp_no_agent is False
        interactive=False, # because sftp_no_interactive is True
        auth_mode='auto'
    )
    assert len(config.allowed_roots) == 1
    assert config.sftp_manager is mock_sftp_instance
    assert config.raw_sftp_specs == ["testuser@testhost:/remote/path"]

@mock.patch('cursor_notebook_mcp.server.SFTPManager')
@mock.patch('getpass.getpass', return_value='sftp_password') # Mock password input
@mock.patch('os.path.realpath', side_effect=lambda p: p)
@mock.patch('os.makedirs')
@mock.patch('tempfile.gettempdir', return_value='/tmp/test_sftp_temp')
def test_server_config_sftp_with_password_prompt(
    mock_gettempdir, mock_os_makedirs, mock_realpath, mock_getpass, mock_sftp_manager_class, mock_base_args
):
    """Test ServerConfig SFTP init with password prompt."""
    mock_sftp_instance = mock_sftp_manager_class.return_value
    mock_sftp_instance.add_path_mapping.return_value = ("testuser", "testhost", "/remote/path")
    mock_sftp_instance.add_connection.return_value = True

    mock_base_args.allow_root = None
    mock_base_args.sftp_root = ["testuser@testhost:/promptpath"]
    mock_base_args.sftp_key = None # No key, to trigger password logic
    mock_base_args.sftp_password = None # No pre-set password
    mock_base_args.sftp_no_interactive = False # Allow interactive for prompt
    mock_base_args.sftp_no_password_prompt = False # Ensure prompt happens

    config = ServerConfig(mock_base_args)

    mock_getpass.assert_called_once_with("SSH password for testuser@testhost: ")
    mock_sftp_instance.add_connection.assert_called_once_with(
        'testhost', 'testuser', 
        password='sftp_password', 
        key_file=None, 
        port=22, 
        use_agent=True, 
        interactive=True, 
        auth_mode='auto'
    )
    assert config.sftp_manager is mock_sftp_instance

@mock.patch('cursor_notebook_mcp.server.SFTPManager')
@mock.patch('getpass.getpass') # Mock getpass to ensure it's NOT called
@mock.patch('os.path.realpath', side_effect=lambda p: p)
@mock.patch('os.makedirs')
@mock.patch('tempfile.gettempdir', return_value='/tmp/test_sftp_temp')
def test_server_config_sftp_no_password_prompt_flag(
    mock_gettempdir, mock_os_makedirs, mock_realpath, mock_getpass, mock_sftp_manager_class, mock_base_args
):
    """Test ServerConfig SFTP with sftp_no_password_prompt=True."""
    mock_sftp_instance = mock_sftp_manager_class.return_value
    mock_sftp_instance.add_path_mapping.return_value = ("testuser", "testhost", "/remote/path")
    mock_sftp_instance.add_connection.return_value = True

    mock_base_args.allow_root = None
    mock_base_args.sftp_root = ["testuser@testhost:/noprompt"]
    mock_base_args.sftp_key = None
    mock_base_args.sftp_password = None
    mock_base_args.sftp_no_interactive = False # Interactive enabled overall
    mock_base_args.sftp_no_password_prompt = True # But specific prompt disabled

    config = ServerConfig(mock_base_args)

    mock_getpass.assert_not_called()
    mock_sftp_instance.add_connection.assert_called_once_with(
        'testhost', 'testuser', 
        password=None, # Password should be None because of the flag
        key_file=None, 
        port=22, 
        use_agent=True, 
        interactive=True, 
        auth_mode='auto'
    )

@mock.patch('cursor_notebook_mcp.server.SFTPManager')
@mock.patch('os.path.realpath', side_effect=lambda p: p)
@mock.patch('os.makedirs')
@mock.patch('tempfile.gettempdir', return_value='/tmp/test_sftp_temp')
def test_server_config_sftp_connection_fails(
    mock_gettempdir, mock_os_makedirs, mock_realpath, mock_sftp_manager_class, mock_base_args
):
    """Test ServerConfig SFTP init when sftp_manager.add_connection fails."""
    mock_sftp_instance = mock_sftp_manager_class.return_value
    mock_sftp_instance.add_path_mapping.return_value = ("testuser", "testhost", "/remote/path")
    mock_sftp_instance.add_connection.return_value = False # Simulate connection failure

    mock_base_args.allow_root = None
    mock_base_args.sftp_root = ["testuser@testhost:/fails"]
    mock_base_args.sftp_key = "key.pem"

    with pytest.raises(ValueError, match="Failed to connect to testuser@testhost:/fails"):
        ServerConfig(mock_base_args)

@mock.patch('cursor_notebook_mcp.server.SFTPManager', side_effect=ImportError("paramiko not found"))
def test_server_config_sftp_paramiko_import_error(mock_sftp_manager_class_import_error, mock_base_args):
    """Test ServerConfig SFTP init when SFTPManager itself raises ImportError (simulating no paramiko)."""
    mock_base_args.allow_root = None
    mock_base_args.sftp_root = ["testuser@testhost:/importerr"]

    with pytest.raises(ValueError, match="SFTP support requires paramiko: paramiko not found"):
        ServerConfig(mock_base_args)

# --- main() function tests for SFTP log and streamable-http transport --- 

@mock.patch('sys.exit')
@mock.patch('cursor_notebook_mcp.server.NotebookTools')
@mock.patch('cursor_notebook_mcp.server.FastMCP')
@mock.patch('cursor_notebook_mcp.server.setup_logging')
@mock.patch('cursor_notebook_mcp.server.ServerConfig')
@mock.patch('cursor_notebook_mcp.server.parse_arguments')
@mock.patch('logging.getLogger')
def test_main_sftp_log_and_cleanup(
    mock_logging_get_logger, mock_parse_args, mock_server_config_class, 
    mock_setup_logging, mock_fast_mcp_class, mock_notebook_tools_class, mock_sys_exit, tmp_path
):
    """Test main logs SFTP status and calls sftp_manager.close_all() if SFTP was active."""
    # Setup args and config to simulate an SFTP setup
    args = argparse.Namespace(
        log_dir=str(tmp_path / "logs"), log_level_int=logging.INFO, transport='stdio',
        host='127.0.0.1', port=8080, allow_root=None, 
        sftp_root=["user@sftphost:/sftp_root_path"], sftp_password=None, sftp_key="key.pem",
        sftp_port=22, sftp_no_interactive=True, sftp_no_agent=False, 
        sftp_no_password_prompt=False, sftp_auth_mode='auto',
        max_cell_source_size=100, max_cell_output_size=100
    )
    mock_parse_args.return_value = args

    mock_sftp_mgr_instance = mock.MagicMock()
    # ServerConfig will create and assign sftp_manager if args.sftp_root is present
    # We need to mock ServerConfig instance's sftp_manager attribute
    config_instance = mock.MagicMock(
        version="test_ver", allowed_roots=[str(tmp_path / "sftp_virtual_root")], 
        transport='stdio', sftp_manager=mock_sftp_mgr_instance,
        host='127.0.0.1', port=8080, log_level=logging.INFO
    )
    mock_server_config_class.return_value = config_instance

    mock_logger = mock.MagicMock()
    mock_logging_get_logger.return_value = mock_logger

    mcp_server_instance = mock.MagicMock()
    mock_fast_mcp_class.return_value = mcp_server_instance
    # Simulate a normal completion of mcp_server.run to allow finally block to execute
    mcp_server_instance.run.return_value = None 

    main()

    mock_logger.info.assert_any_call("SFTP connections active")
    mock_sftp_mgr_instance.close_all.assert_called_once()
    # If mcp_server.run completes normally, main() itself might not call sys.exit.
    # The key is that close_all was called. 
    # If SystemExit was expected from main() due to an error, this assertion would be different.
    # For this test, we are primarily verifying the SFTP log and cleanup.
    # mock_sys_exit.assert_called() # This might not be true if main runs to completion without error

@mock.patch('sys.exit')
@mock.patch('cursor_notebook_mcp.server.NotebookTools')
@mock.patch('cursor_notebook_mcp.server.FastMCP')
@mock.patch('cursor_notebook_mcp.server.setup_logging')
@mock.patch('cursor_notebook_mcp.server.ServerConfig')
@mock.patch('cursor_notebook_mcp.server.parse_arguments')
@mock.patch('logging.getLogger')
def test_main_streamable_http_transport_success(
    mock_logging_get_logger, mock_parse_args, mock_server_config_class, 
    mock_setup_logging, mock_fast_mcp_class, mock_notebook_tools_class, mock_sys_exit, tmp_path
):
    """Test main() with streamable-http transport successfully."""
    args = argparse.Namespace(
        log_dir=str(tmp_path / "logs"), log_level_int=logging.INFO, transport='streamable-http',
        host='0.0.0.0', port=9090, allow_root=[str(tmp_path)], sftp_root=None,
        max_cell_source_size=100, max_cell_output_size=100
    )
    mock_parse_args.return_value = args

    config_instance = mock.MagicMock(
        version="test_ver", allowed_roots=[str(tmp_path)], transport='streamable-http', 
        sftp_manager=None, host='0.0.0.0', port=9090, log_level=logging.INFO
    )
    mock_server_config_class.return_value = config_instance

    mock_logger = mock.MagicMock()
    mock_logging_get_logger.return_value = mock_logger

    mcp_server_instance = mock.MagicMock()
    mock_fast_mcp_class.return_value = mcp_server_instance
    # mcp_server_instance.run will be called by main

    main()

    mcp_server_instance.run.assert_called_once_with(
        transport="streamable-http",
        host='0.0.0.0',
        port=9090,
        log_level=logging.getLevelName(logging.INFO).lower()
    )
    mock_logger.info.assert_any_call("Running server using FastMCP's Streamable HTTP transport...")
    mock_sys_exit.assert_not_called() # Should not exit if run is successful (unless run itself exits)


@mock.patch('sys.exit')
@mock.patch('cursor_notebook_mcp.server.NotebookTools')
@mock.patch('cursor_notebook_mcp.server.FastMCP')
@mock.patch('cursor_notebook_mcp.server.setup_logging')
@mock.patch('cursor_notebook_mcp.server.ServerConfig')
@mock.patch('cursor_notebook_mcp.server.parse_arguments')
@mock.patch('logging.getLogger')
def test_main_streamable_http_transport_importerror(
    mock_logging_get_logger, mock_parse_args, mock_server_config_class, 
    mock_setup_logging, mock_fast_mcp_class, mock_notebook_tools_class, mock_sys_exit, tmp_path
):
    """Test main() with streamable-http transport raising ImportError."""
    args = argparse.Namespace(
        log_dir=str(tmp_path / "logs"), log_level_int=logging.INFO, transport='streamable-http',
        host='0.0.0.0', port=9090, allow_root=[str(tmp_path)], sftp_root=None,
        max_cell_source_size=100, max_cell_output_size=100
    )
    mock_parse_args.return_value = args

    config_instance = mock.MagicMock(
        version="test_ver", allowed_roots=[str(tmp_path)], transport='streamable-http',
        sftp_manager=None, host='0.0.0.0', port=9090, log_level=logging.INFO
    )
    mock_server_config_class.return_value = config_instance

    mock_logger = mock.MagicMock()
    mock_logging_get_logger.return_value = mock_logger

    mcp_server_instance = mock.MagicMock()
    mock_fast_mcp_class.return_value = mcp_server_instance
    mcp_server_instance.run.side_effect = ImportError("uvicorn missing")

    main()

    mcp_server_instance.run.assert_called_once_with(
        transport="streamable-http",
        host='0.0.0.0',
        port=9090,
        log_level=logging.getLevelName(logging.INFO).lower()
    )
    mock_logger.error.assert_any_call("Failed to start Streamable HTTP server due to missing packages: uvicorn missing")
    mock_logger.error.assert_any_call('Hint: HTTP transport requires optional dependencies. Try: pip install "fastmcp[http]"')
    mock_sys_exit.assert_called_once_with(1) 