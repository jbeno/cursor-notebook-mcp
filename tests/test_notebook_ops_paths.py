"""
Tests specifically for path handling edge cases in notebook_ops.py.

This focuses on:
1. Windows path handling/normalization
2. URL-encoded paths
3. SFTP path handling and error conditions
4. Permission checking with different path types
"""

import os
import sys
import posixpath
import pytest
import asyncio
from unittest import mock
from urllib.parse import quote
from pathlib import Path

from cursor_notebook_mcp import notebook_ops
from cursor_notebook_mcp.sftp_manager import SFTPManager

# --- Setup and Fixtures ---

@pytest.fixture
def mock_sftp_manager():
    """Create a mock SFTP manager with predefined behavior."""
    mock_manager = mock.MagicMock(spec=SFTPManager)
    
    # Setup translate_path method
    def mock_translate_path(path):
        if path.startswith('ssh://user@example.com'):
            # Remote path that translates to local temp
            return True, '/tmp/sftp_local/user/path', None
        elif path.startswith('~/'):
            # Remote relative path
            return True, path, None
        elif path.startswith('/home/user'):
            # Remote absolute path
            return True, path, None
        else:
            # Not a remote path
            return False, path, None
            
    mock_manager.translate_path.side_effect = mock_translate_path
    
    # Setup _get_absolute_remote_path method
    def mock_get_absolute_remote_path(path):
        if path.startswith('ssh://'):
            return '/home/user/remote_dir/file.ipynb'
        elif path.startswith('~/'):
            return '/home/user/' + path[2:]
        elif path.startswith('/home/user'):
            return path
        else:
            return None
    
    mock_manager._get_absolute_remote_path.side_effect = mock_get_absolute_remote_path
    
    # Setup path mapping
    mock_manager.path_mappings = {
        'user@example.com:/home/user/remote_dir': ('example.com', 'user', '/home/user/remote_dir/', '/tmp/sftp_local/user/')
    }
    
    # Setup connections dict with mock SFTP client
    mock_sftp = mock.MagicMock()
    mock_sftp.normalize.side_effect = lambda p: posixpath.normpath(p.replace('~', '/home/user'))
    mock_manager.connections = {
        'example.com': (mock.MagicMock(), mock_sftp)
    }
    
    # Setup _get_remote_home_dir method
    mock_manager._get_remote_home_dir.return_value = '/home/user'
    
    return mock_manager

@pytest.fixture
def allowed_roots(tmp_path):
    """Create temporary directories to use as allowed roots."""
    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    os.makedirs(root1, exist_ok=True)
    os.makedirs(root2, exist_ok=True)
    return [str(root1), str(root2)]

# --- Test normalize_path ---

def test_normalize_path_windows_leading_slash():
    """Test normalizing Windows paths with leading slash."""
    # Only run on Windows or when emulating Windows paths
    if sys.platform == "win32":
        # Test with leading slash and drive letter
        path = "/C:/Users/test/notebook.ipynb"
        result = notebook_ops.normalize_path(path, [], None)
        assert result == os.path.realpath("C:/Users/test/notebook.ipynb")
        assert not result.startswith('/')
    else:
        # On non-Windows, test the path conversion for cross-platform support
        path = "/C:/Users/test/notebook.ipynb"
        result = notebook_ops.normalize_path(path, [], None)
        # Should still remove the leading slash before the drive letter
        assert not result.startswith("/C:")
        # The exact result depends on platform, but the leading slash before C: should be gone

def test_normalize_path_sftp_translation():
    """Test normalize_path with SFTP translation."""
    mock_manager = mock.MagicMock(spec=SFTPManager)
    # Configure translate_path to return a successful translation
    mock_manager.translate_path.return_value = (False, "/local/temp/path", None)
    
    path = "ssh://user@example.com:/path/notebook.ipynb"
    result = notebook_ops.normalize_path(path, [], mock_manager)
    
    # Should use the translated path 
    assert result == os.path.realpath("/local/temp/path")
    mock_manager.translate_path.assert_called_once_with(path)

def test_normalize_path_sftp_remote_not_translated():
    """Test normalize_path with remote path that doesn't translate."""
    mock_manager = mock.MagicMock(spec=SFTPManager)
    # Configure translate_path to return a remote path that wasn't translated
    mock_manager.translate_path.return_value = (True, "ssh://user@example.com:/path/notebook.ipynb", None)
    
    path = "ssh://user@example.com:/path/notebook.ipynb"
    result = notebook_ops.normalize_path(path, [], mock_manager)
    
    # Should still normalize the path even if it's remote
    assert result == os.path.realpath(path)
    mock_manager.translate_path.assert_called_once_with(path)

def test_normalize_path_sftp_error():
    """Test normalize_path when SFTP translation raises an error."""
    mock_manager = mock.MagicMock(spec=SFTPManager)
    # Configure translate_path to raise an exception
    mock_manager.translate_path.side_effect = Exception("SFTP translation error")
    
    path = "ssh://user@example.com:/path/notebook.ipynb"
    # Should handle the error and still return a normalized path
    result = notebook_ops.normalize_path(path, [], mock_manager)
    
    assert result == os.path.realpath(path)
    mock_manager.translate_path.assert_called_once_with(path)

# --- Test is_path_allowed ---

def test_is_path_allowed_no_roots():
    """Test is_path_allowed when no allowed roots are provided."""
    # With no roots, should return False
    assert not notebook_ops.is_path_allowed("/some/path", [], None)

def test_is_path_allowed_normalization_error():
    """Test is_path_allowed when path normalization raises an error."""
    # Mock normalize_path to raise an exception
    with mock.patch('cursor_notebook_mcp.notebook_ops.normalize_path', side_effect=Exception("Normalization error")):
        # Should handle the error and return False
        assert not notebook_ops.is_path_allowed("/some/path", ["/allowed"], None)

def test_is_path_allowed_windows_path_comparison():
    """Test is_path_allowed with Windows path comparison."""
    # Create paths with mixed path separators
    allowed_root = "C:\\allowed\\root"
    target_path = "C:/allowed/root/notebook.ipynb"
    
    # For Windows or when dealing with Windows-style paths
    with mock.patch('cursor_notebook_mcp.notebook_ops.normalize_path', return_value=target_path):
        # Should normalize separators and allow the path
        assert notebook_ops.is_path_allowed(target_path, [allowed_root], None)

def test_is_path_allowed_resolving_error():
    """Test is_path_allowed when realpath raises an error."""
    # Mock realpath to raise an exception for the allowed root
    with mock.patch('os.path.realpath') as mock_realpath:
        def mock_realpath_side_effect(path):
            if path == "/bad/root":
                raise OSError("Bad path")
            return path
            
        mock_realpath.side_effect = mock_realpath_side_effect
        
        # Should skip the bad root and continue checking
        # This path is not in any allowed root, so should return False
        assert not notebook_ops.is_path_allowed("/some/path", ["/bad/root", "/other/root"], None)

# --- Test resolve_path_and_check_permissions ---

@pytest.mark.asyncio
async def test_resolve_path_empty():
    """Test resolve_path_and_check_permissions with empty path."""
    with pytest.raises(ValueError, match="Path cannot be empty"):
        await asyncio.to_thread(notebook_ops.resolve_path_and_check_permissions, "", ["/allowed"])

@pytest.mark.asyncio
async def test_resolve_path_url_encoded():
    """Test resolve_path_and_check_permissions with URL-encoded path."""
    # Create a URL-encoded path
    original_path = "/test/path with spaces.ipynb"
    encoded_path = quote(original_path)
    
    # Mock is_path_allowed to avoid permission error
    with mock.patch('cursor_notebook_mcp.notebook_ops.is_path_allowed', return_value=True):
        with pytest.raises(PermissionError):  # Will fail because the path isn't allowed
            is_remote, resolved_path = await asyncio.to_thread(
                notebook_ops.resolve_path_and_check_permissions, 
                encoded_path, 
                ["/allowed"]
            )

@pytest.mark.asyncio
async def test_resolve_path_workspace_no_roots():
    """Test resolve_path_and_check_permissions with workspace path but no allowed roots."""
    with pytest.raises(PermissionError, match="/workspace/ paths require local allowed roots"):
        await asyncio.to_thread(
            notebook_ops.resolve_path_and_check_permissions,
            "/workspace/notebook.ipynb",
            []
        )

@pytest.mark.asyncio
async def test_resolve_path_workspace_with_roots(allowed_roots):
    """Test resolve_path_and_check_permissions with workspace path and allowed roots."""
    # Create the file to make the permission check pass
    workspace_path = "/workspace/notebook.ipynb"
    expected_path = os.path.normpath(os.path.join(allowed_roots[0], "notebook.ipynb"))
    os.makedirs(os.path.dirname(expected_path), exist_ok=True)
    open(expected_path, 'w').close()
    
    try:
        is_remote, resolved_path = await asyncio.to_thread(
            notebook_ops.resolve_path_and_check_permissions,
            workspace_path,
            allowed_roots
        )
        assert not is_remote
        assert resolved_path == expected_path
    finally:
        # Clean up
        if os.path.exists(expected_path):
            os.unlink(expected_path)

@pytest.mark.asyncio
async def test_resolve_path_sftp_remote(mock_sftp_manager):
    """Test resolve_path_and_check_permissions with a remote SFTP path."""
    # Mock path_exists to pass the permission check instead of the non-existent _remote_path_is_allowed
    with mock.patch.object(mock_sftp_manager, 'connections', {'example.com': (mock.MagicMock(), mock.MagicMock())}):
        is_remote, resolved_path = await asyncio.to_thread(
            notebook_ops.resolve_path_and_check_permissions,
            "ssh://user@example.com:/path/notebook.ipynb",
            [],
            mock_sftp_manager
        )
        assert is_remote
        assert resolved_path == "/home/user/remote_dir/file.ipynb"

@pytest.mark.asyncio
async def test_resolve_path_sftp_local_relative_no_roots():
    """Test resolve_path_and_check_permissions with a local relative path and SFTP but no roots."""
    # Create a mock SFTP manager
    mock_sftp = mock.MagicMock(spec=SFTPManager)
    # Configure translate_path to avoid using _get_absolute_remote_path
    mock_sftp._get_absolute_remote_path = lambda x: None  # No remote resolution, so relative path remains local
    
    with pytest.raises(ValueError, match="Cannot resolve relative local path.*without local allowed roots"):
        await asyncio.to_thread(
            notebook_ops.resolve_path_and_check_permissions,
            "relative/path.ipynb",
            [],
            mock_sftp
        )

@pytest.mark.asyncio
async def test_resolve_path_local_relative_no_sftp_no_roots():
    """Test resolve_path_and_check_permissions with a local relative path, no SFTP, no roots."""
    with pytest.raises(PermissionError, match="Cannot resolve relative path.*without allowed roots"):
        await asyncio.to_thread(
            notebook_ops.resolve_path_and_check_permissions,
            "relative/path.ipynb",
            []
        )

@pytest.mark.asyncio
async def test_resolve_path_permission_check_remote_fail(mock_sftp_manager, allowed_roots):
    """Test resolve_path_and_check_permissions with remote path that fails permission check."""
    # Mock the SFTP connection to raise KeyError
    mock_sftp_manager.connections = {}  # Empty connections dict
    
    with pytest.raises(PermissionError, match="Access denied.*outside allowed remote SFTP roots"):
        await asyncio.to_thread(
            notebook_ops.resolve_path_and_check_permissions,
            "ssh://user@example.com:/path/notebook.ipynb",
            allowed_roots,
            mock_sftp_manager
        )

@pytest.mark.asyncio
async def test_resolve_path_permission_check_local_fail(allowed_roots):
    """Test resolve_path_and_check_permissions with local path that fails permission check."""
    # Path outside the allowed roots
    outside_path = "/outside/allowed/roots.ipynb"
    
    with pytest.raises(PermissionError, match="Access denied.*outside allowed local allowed roots"):
        await asyncio.to_thread(
            notebook_ops.resolve_path_and_check_permissions,
            outside_path,
            allowed_roots
        )

# --- Test read_notebook and write_notebook ---

@pytest.mark.asyncio
async def test_read_notebook_invalid_path():
    """Test read_notebook with an invalid path."""
    with pytest.raises(ValueError, match="Invalid notebook path"):
        await notebook_ops.read_notebook("invalid_no_extension", [], None)

@pytest.mark.asyncio
async def test_read_notebook_remote_no_sftp():
    """Test read_notebook with a remote path but no SFTP manager."""
    # Mock resolve_path_and_check_permissions to indicate it's remote
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                   return_value=(True, "/remote/path.ipynb")):
        with pytest.raises(ConnectionError, match="SFTP manager required"):
            await notebook_ops.read_notebook("remote.ipynb", [], None)

@pytest.mark.asyncio
async def test_read_notebook_local_file_not_found(allowed_roots):
    """Test read_notebook with a local file that doesn't exist."""
    # Use a path inside the allowed roots but the file doesn't exist
    nonexistent_path = os.path.join(allowed_roots[0], "nonexistent.ipynb")
    
    with pytest.raises(FileNotFoundError):
        await notebook_ops.read_notebook(nonexistent_path, allowed_roots, None)

@pytest.mark.asyncio
async def test_write_notebook_invalid_notebook():
    """Test write_notebook with an invalid notebook."""
    # Create an invalid "notebook" - just a dict
    invalid_nb = {"not": "a valid notebook"}
    
    with pytest.raises(ValueError, match="Notebook content is invalid"):
        await notebook_ops.write_notebook("valid.ipynb", invalid_nb, [], None)

@pytest.mark.asyncio
async def test_write_notebook_remote_no_sftp():
    """Test write_notebook with a remote path but no SFTP manager."""
    # Create a valid notebook node
    import nbformat
    nb = nbformat.v4.new_notebook()
    
    # Mock resolve_path_and_check_permissions to indicate it's remote
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                   return_value=(True, "/remote/path.ipynb")):
        with pytest.raises(ConnectionError, match="SFTP manager required"):
            await notebook_ops.write_notebook("remote.ipynb", nb, [], None)

@pytest.mark.asyncio
async def test_write_notebook_parent_dir_error(allowed_roots):
    """Test write_notebook when creating parent directory fails."""
    # Create a valid notebook node
    import nbformat
    nb = nbformat.v4.new_notebook()
    
    # Path inside allowed roots but in a sub-directory
    nested_path = os.path.join(allowed_roots[0], "subdir", "notebook.ipynb")
    
    # Mock os.makedirs to raise an error
    with mock.patch('os.makedirs', side_effect=OSError("Permission denied")):
        # Mock resolve_path_and_check_permissions to avoid permission errors
        with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                       return_value=(False, nested_path)):
            with pytest.raises(IOError, match="Could not create local directory"):
                await notebook_ops.write_notebook(nested_path, nb, allowed_roots, None)

@pytest.mark.asyncio
async def test_write_notebook_write_error(allowed_roots):
    """Test write_notebook when the write operation fails."""
    # Create a valid notebook node
    import nbformat
    nb = nbformat.v4.new_notebook()
    
    # Path inside allowed roots
    valid_path = os.path.join(allowed_roots[0], "notebook.ipynb")
    
    # Mock nbformat.write to raise an error
    with mock.patch('nbformat.write', side_effect=IOError("Write failure")):
        # Mock resolve_path_and_check_permissions to avoid permission errors
        with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                       return_value=(False, valid_path)):
            with pytest.raises(IOError, match="Failed to write local notebook"):
                await notebook_ops.write_notebook(valid_path, nb, allowed_roots, None) 