"""
Debugging test file to track down hanging issues between test modules.
"""

import pytest
import os
import sys
import time
import logging
import asyncio
from unittest import mock
import nbformat # Keep standard imports
from contextlib import suppress

# Configure logging to track execution
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('debug_hanging')

# Using the event_loop fixture from conftest.py

# --- Mock Fixtures --- 

@pytest.fixture(scope="module")
def mock_paramiko_module():
    """Provides a mock paramiko module for the duration of the module."""
    logger.debug("Creating mock paramiko module")
    _mock_paramiko = mock.MagicMock()
    _mock_paramiko.SSHException = type('MockSSHException', (Exception,), {})
    _mock_paramiko.AuthenticationException = type('MockAuthenticationException', (Exception,), {})
    _mock_paramiko.ssh_exception = mock.MagicMock()
    _mock_paramiko.ssh_exception.PasswordRequiredException = type('MockPasswordRequiredException', (Exception,), {})
    _mock_paramiko.RSAKey = mock.MagicMock()
    _mock_paramiko.RSAKey.from_private_key_file = mock.MagicMock(return_value=mock.MagicMock())
    
    # Keep track of original modules ONLY for this fixture's cleanup
    original_modules = {}
    modules_to_mock = {
        'paramiko': _mock_paramiko,
        'paramiko.ssh_exception': _mock_paramiko.ssh_exception
    }

    for mod_name in modules_to_mock:
        original_modules[mod_name] = sys.modules.get(mod_name, None)
        sys.modules[mod_name] = modules_to_mock[mod_name]
        logger.debug(f"Applied mock for {mod_name}")

    yield _mock_paramiko # Provide the mock if needed directly

    # Teardown: Restore original modules
    logger.debug("Restoring original modules after mock_paramiko_module fixture...")
    for mod_name, orig_mod in original_modules.items():
        if orig_mod is not None:
            sys.modules[mod_name] = orig_mod
            logger.debug(f"Restored original {mod_name}")
        elif mod_name in sys.modules:
            # Only delete if we actually added it
            if sys.modules[mod_name] is modules_to_mock.get(mod_name):
                 del sys.modules[mod_name]
                 logger.debug(f"Deleted mock {mod_name}")
            else:
                 logger.warning(f"Module {mod_name} was unexpectedly modified during test.")
        

@pytest.fixture
def mock_sftp_manager(monkeypatch): # Use monkeypatch for notebook_ops dependency
    """Create a mock SFTP manager for testing and mock it in notebook_ops"""
    logger.debug("Creating mock SFTP manager")
    mock_sftp_mgr = mock.MagicMock()
    mock_sftp_mgr.read_file.return_value = b"{\"cells\": [], \"metadata\": {}, \"nbformat\": 4, \"nbformat_minor\": 5}"
    mock_sftp_mgr.translate_path.return_value = (True, "/remote/path/notebook.ipynb", ("mock_client", "mock_sftp"))
    mock_client = mock.MagicMock()
    mock_sftp = mock.MagicMock()
    mock_sftp_mgr.connections = {"hostname": (mock_client, mock_sftp)}
    mock_sftp_mgr.path_mappings = {
        "user@hostname:/remote/path": ("hostname", "user", "/remote/path/", "/local/path/")
    }
    
    # Mock the SFTPManager class within notebook_ops
    # We import notebook_ops here, after mocks might be applied by other fixtures if needed
    from cursor_notebook_mcp import notebook_ops # Import locally scoped
    monkeypatch.setattr(notebook_ops, 'SFTPManager', lambda: mock_sftp_mgr, raising=False)
    logger.debug("Patched notebook_ops.SFTPManager")

    # Add proper implementation for add_path_mapping to the mock instance
    def mock_add_path_mapping(remote_spec, local_path):
        if not isinstance(remote_spec, str) or "@" not in remote_spec or ":" not in remote_spec:
            raise ValueError(f"Invalid remote spec: {remote_spec}")
            
        username, rest = remote_spec.split("@", 1)
        host, remote_path = rest.split(":", 1)
        
        if not remote_path.endswith('/'): remote_path += '/'
        if not local_path.endswith('/'): local_path += '/'
            
        mock_sftp_mgr.path_mappings[remote_spec] = (host, username, remote_path, local_path)
        return username, host, remote_path
    
    mock_sftp_mgr.add_path_mapping.side_effect = mock_add_path_mapping
    
    yield mock_sftp_mgr
    logger.debug("Mock SFTP manager fixture teardown")

# --- Test Functions --- 
# Import modules under test INSIDE the test functions or fixtures
# where they are needed, AFTER mocks have been applied.

@pytest.mark.asyncio
async def test_read_notebook_simple(tmp_path):
    """Simple test that reads a notebook file"""
    logger.debug("Starting test_read_notebook_simple")
    from cursor_notebook_mcp import notebook_ops # Import locally
    
    test_nb_path = tmp_path / "test.ipynb"
    nb = nbformat.v4.new_notebook()
    nbformat.write(nb, test_nb_path)
    
    allowed_roots = [str(tmp_path)]
    start_time = time.time()
    try:
        logger.debug("About to call read_notebook")
        result = await asyncio.wait_for(
            notebook_ops.read_notebook(str(test_nb_path), allowed_roots),
            timeout=30.0  # 30 second timeout
        )
        end_time = time.time()
        logger.debug(f"read_notebook completed successfully in {end_time - start_time:.2f} seconds.")
        assert isinstance(result, nbformat.NotebookNode)
    except asyncio.TimeoutError:
        logger.error("TIMEOUT in read_notebook call!")
        raise
    except Exception as e:
        logger.error(f"Error in read_notebook: {e}")
        raise
    finally:
        logger.debug("test_read_notebook_simple completed")

def test_sftp_manager_simple(mock_paramiko_module): # Depend on paramiko mock
    """Simple test of SFTP manager functionality"""
    logger.debug("Starting test_sftp_manager_simple")
    from cursor_notebook_mcp.sftp_manager import SFTPManager # Import locally
    
    # Now that paramiko is mocked (by fixture dependency), SFTPManager can be imported/used
    sftp_manager = SFTPManager()
    sftp_manager.connections = {}
    sftp_manager.path_mappings = {}
    
    remote_spec = "user@host:/remote/path"
    local_path = "/local/temp/path"
    username, host, remote_path = sftp_manager.add_path_mapping(remote_spec, local_path)
    
    assert username == "user"
    assert host == "host"
    assert remote_path == "/remote/path/"
    logger.debug("test_sftp_manager_simple completed")

@pytest.mark.asyncio
async def test_combined_operation(mock_sftp_manager, tmp_path): # Depends on SFTP mock
    """Run operations from both test files in a single test"""
    logger.debug("Starting test_combined_operation")
    from cursor_notebook_mcp import notebook_ops # Import locally
    
    test_nb_path = tmp_path / "test_combined.ipynb"
    nb = nbformat.v4.new_notebook()
    nbformat.write(nb, test_nb_path)
    
    # Part 1: Test SFTP-related code (using the provided mock_sftp_manager)
    try:
        logger.debug("Testing SFTP path mapping via mock")
        remote_spec = "user@host:/remote/path"
        local_path = "/local/temp/path"
        # Use the mock_sftp_manager fixture directly
        username, host, remote_path = mock_sftp_manager.add_path_mapping(remote_spec, local_path)
        assert username == "user"
    except Exception as e:
        logger.error(f"Error in SFTP part: {e}")
        raise
    
    # Part 2: Test notebook ops with SFTP (mocked via mock_sftp_manager fixture's monkeypatch)
    try:
        logger.debug("Testing notebook_ops with SFTP")
        allowed_roots = [str(tmp_path)]
        
        # Mock path resolution needed for this specific call
        with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                        return_value=(True, "/remote/path/notebook.ipynb")):
            result = await asyncio.wait_for(
                notebook_ops.read_notebook("ssh://user@hostname:/path/to/notebook.ipynb", 
                                           allowed_roots, 
                                           sftp_manager=mock_sftp_manager), # Pass the mock manager
                timeout=5.0 # Keep original timeout here
            )
        logger.debug("notebook_ops with SFTP completed successfully")
    except asyncio.TimeoutError:
        logger.error("TIMEOUT in notebook_ops with SFTP!")
        raise
    except Exception as e:
        logger.error(f"Error in notebook_ops with SFTP: {e}")
        raise
    
    logger.debug("test_combined_operation completed successfully") 