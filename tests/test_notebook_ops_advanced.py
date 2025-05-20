"""
Tests targeting specific uncovered code paths in notebook_ops.py.

This file focuses on edge cases that aren't covered by the main tests.
"""

import pytest
import sys
import os
import posixpath
import asyncio
import nbformat
from unittest import mock
from urllib.parse import quote

from cursor_notebook_mcp import notebook_ops
from cursor_notebook_mcp.sftp_manager import SFTPManager

# --- URL Encoding and Windows Paths Tests ---

def test_normalize_path_url_encoded():
    """Test handling of URL-encoded paths."""
    # Create a path with spaces and special characters that needs URL encoding
    original_path = "/path/with spaces/and+special#chars.ipynb"
    encoded_path = quote(original_path)
    
    # Ensure the path is actually encoded
    assert '%20' in encoded_path  # Space becomes %20
    assert '%2B' in encoded_path  # + becomes %2B
    assert '%23' in encoded_path  # # becomes %23
    
    # Test normalize_path with the encoded path
    normalized = notebook_ops.normalize_path(encoded_path, ["/"], None)
    
    # Verify it was correctly decoded
    assert ' ' in normalized  # %20 becomes space
    assert '+' in normalized  # %2B becomes +
    assert '#' in normalized  # %23 becomes #
    assert normalized.endswith('and+special#chars.ipynb')

def test_normalize_path_windows_style_on_non_windows():
    """Test handling of Windows-style paths on non-Windows platforms."""
    # Skip test if running on Windows
    if sys.platform == "win32":
        pytest.skip("Test only relevant on non-Windows platforms")
    
    # Test Windows-style path with leading slash
    win_path = "/C:/Users/test/notebook.ipynb"
    normalized = notebook_ops.normalize_path(win_path, ["/"], None)
    
    # Verify the leading slash was removed
    assert not normalized.startswith("/C:")
    # The rest depends on platform, but at minimum we know the leading slash is gone

# --- SFTP Path Translation Error Handling ---

def test_normalize_path_sftp_translation_error():
    """Test normalize_path when SFTP translation raises an error."""
    # Create a mock SFTP manager
    mock_sftp = mock.MagicMock(spec=SFTPManager)
    # Make translate_path raise an exception
    mock_sftp.translate_path.side_effect = Exception("Translation failed")
    
    path = "/path/to/notebook.ipynb"
    # Should handle the error and return a normalized path
    result = notebook_ops.normalize_path(path, ["/"], mock_sftp)
    
    # Verify translate_path was called and the error was handled
    mock_sftp.translate_path.assert_called_once_with(path)
    assert result == os.path.realpath(path)  # Should still normalize the path

# --- is_path_allowed Edge Cases ---

def test_is_path_allowed_error_normalizing_path():
    """Test is_path_allowed when normalizing the path raises an error."""
    with mock.patch('cursor_notebook_mcp.notebook_ops.normalize_path', 
                  side_effect=Exception("Normalization error")):
        result = notebook_ops.is_path_allowed("/path", ["/allowed"], None)
        assert result is False  # Should return False if normalization fails

def test_is_path_allowed_windows_paths_with_different_separators():
    """Test is_path_allowed with Windows paths using different separators."""
    # Skip on non-Windows, but also set up the platform check to pass
    is_win32 = sys.platform == "win32"
    platform_check = is_win32 or True  # Force Windows path handling
    
    # We need to mock abs_target_path_local_repr
    with mock.patch('sys.platform', "win32" if platform_check else "linux"):
        # Test with Windows paths using forward slashes
        with mock.patch('cursor_notebook_mcp.notebook_ops.normalize_path', return_value='C:/Users/test/notebook.ipynb'), \
             mock.patch('os.path.realpath', side_effect=lambda p: p), \
             mock.patch('os.path.normpath', side_effect=lambda p: p):
            # Test with path outside root 
            result = notebook_ops.is_path_allowed('/path', ['D:/OtherRoot'], None)
            assert result is False  # Path outside of root
            
        # Test with Windows path that should match with forward slashes
        with mock.patch('cursor_notebook_mcp.notebook_ops.normalize_path', return_value='C:/Users/test/notebook.ipynb'), \
             mock.patch('os.path.realpath', side_effect=lambda p: p), \
             mock.patch('os.path.normpath', side_effect=lambda p: p):
            # Test with path inside root
            result = notebook_ops.is_path_allowed('/path', ['C:/Users'], None)
            assert result is True  # Path is inside root
            
        # Test with backslashes being converted to forward slashes
        with mock.patch('cursor_notebook_mcp.notebook_ops.normalize_path', return_value='C:/Users/test/notebook.ipynb'), \
             mock.patch('os.path.realpath', side_effect=lambda p: p), \
             mock.patch('os.path.normpath', side_effect=lambda p: p):
            # Test with backslashes in allowed root
            result = notebook_ops.is_path_allowed('/path', ['C:\\Users'], None)
            assert result is True  # Path is inside root after slash normalization

def test_is_path_allowed_error_resolving_root():
    """Test is_path_allowed when realpath for an allowed root raises an error."""
    # Create a custom function to simulate realpath behavior with error handling
    def mock_realpath(p):
        if p == '/invalid/root':
            raise OSError("Invalid root")
        elif p == '/valid/root':
            return '/valid/root'
        elif p == '/valid/path':
            return '/valid/root/valid/path'  # Make sure it's under valid root
        else:
            return p
            
    # Mock normalize_path to return a valid path that's under the valid root
    with mock.patch('cursor_notebook_mcp.notebook_ops.normalize_path', return_value='/valid/root/valid/path'), \
         mock.patch('os.path.normpath', side_effect=lambda p: p), \
         mock.patch('os.path.realpath', side_effect=mock_realpath):
        
        # Test with multiple roots, one valid and one invalid
        result = notebook_ops.is_path_allowed('/path', ['/invalid/root', '/valid/root'], None)
        # Even though invalid root causes an error, valid root should work and match
        assert result is True

# --- resolve_path_and_check_permissions Edge Cases ---

@pytest.mark.asyncio
async def test_resolve_path_and_check_permissions_workspace_path():
    """Test resolution of /workspace/ paths."""
    # Mock the permission check to return True
    with mock.patch('os.path.realpath', return_value='/allowed/path'), \
         mock.patch('os.path.isabs', return_value=True):
        
        is_remote, path = await asyncio.to_thread(
            notebook_ops.resolve_path_and_check_permissions,
            "/workspace/notebook.ipynb",
            ["/allowed"],
            None
        )
        
        assert is_remote is False
        assert path == os.path.normpath('/allowed/notebook.ipynb')

@pytest.mark.asyncio
async def test_resolve_path_null_path():
    """Test resolve_path_and_check_permissions with None or empty path."""
    with pytest.raises(ValueError, match="Path cannot be empty"):
        await asyncio.to_thread(notebook_ops.resolve_path_and_check_permissions, "", ["/allowed"], None)
    
    with pytest.raises(ValueError, match="Path cannot be empty"):
        await asyncio.to_thread(notebook_ops.resolve_path_and_check_permissions, None, ["/allowed"], None)

@pytest.mark.asyncio
async def test_resolve_path_permission_remote_exception():
    """Test handling of exceptions during remote permission checks."""
    # Create a minimal mock SFTP manager with key components for the test
    mock_sftp_manager = mock.MagicMock(spec=SFTPManager)
    mock_sftp_manager._get_absolute_remote_path.return_value = "/remote/path.ipynb"
    
    # Set up path mappings with a host connection that will raise KeyError
    mock_sftp_manager.path_mappings = {
        "spec": ("nonexistent_host", "user", "/remote/", "/local/")
    }
    mock_sftp_manager.connections = {}  # Empty connections dict will cause KeyError
    
    # Should raise PermissionError due to no valid connections
    with pytest.raises(PermissionError, match="Access denied:.*outside allowed remote SFTP roots"):
        await asyncio.to_thread(
            notebook_ops.resolve_path_and_check_permissions,
            "ssh://user@example.com/path.ipynb",
            [],
            mock_sftp_manager
        )

@pytest.mark.asyncio
async def test_resolve_path_permission_local_exception():
    """Test handling of exceptions during local permission checks."""
    # Mock os.path.realpath to raise an exception for a specific path
    with mock.patch('os.path.realpath', side_effect=lambda p: 
                  OSError("Invalid path") if p == '/allowed/bad' else p):
        
        # Should raise PermissionError due to exception in all permission checks
        with pytest.raises(PermissionError, match="Access denied:.*outside allowed local allowed roots"):
            await asyncio.to_thread(
                notebook_ops.resolve_path_and_check_permissions,
                "/path.ipynb",
                ["/allowed/bad"],  # Path that will cause realpath to fail
                None
            )

@pytest.mark.asyncio
async def test_read_notebook_local_file_not_found():
    """Test read_notebook with a missing local file."""
    # Mock os.path.exists to return False for the path
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(False, "/path/nonexistent.ipynb")), \
         mock.patch('builtins.open', side_effect=FileNotFoundError("No such file")):
        with pytest.raises(FileNotFoundError, match="Notebook file not found"):
            await notebook_ops.read_notebook("nonexistent.ipynb", ["/allowed"], None)

@pytest.mark.asyncio
async def test_read_notebook_unexpected_error():
    """Test read_notebook with an unexpected error during read."""
    # Mock open to raise an unexpected error
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(False, "/path/notebook.ipynb")), \
         mock.patch('builtins.open', side_effect=Exception("Unexpected error")):
        with pytest.raises(IOError, match="Failed to read local notebook file"):
            await notebook_ops.read_notebook("notebook.ipynb", ["/allowed"], None)

@pytest.mark.asyncio
async def test_write_notebook_local_makedirs_error():
    """Test write_notebook when creating parent directory fails."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": []
    }
    
    # Mock path resolution and makedirs to fail
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(False, "/path/dir/notebook.ipynb")), \
         mock.patch('nbformat.validate', return_value=None), \
         mock.patch('os.makedirs', side_effect=OSError("Permission denied")):
        with pytest.raises(IOError, match="Could not create local directory"):
            await notebook_ops.write_notebook("dir/notebook.ipynb", nb, ["/allowed"], None)

@pytest.mark.asyncio
async def test_write_notebook_unexpected_write_error():
    """Test write_notebook with an unexpected error during write."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": []
    }
    
    # Mock path resolution but make nbformat.write raise an error
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(False, "/path/notebook.ipynb")), \
         mock.patch('nbformat.validate', return_value=None), \
         mock.patch('os.makedirs', return_value=None), \
         mock.patch('asyncio.to_thread', side_effect=Exception("Unexpected error")):
        with pytest.raises(IOError, match="Failed to write local notebook"):
            await notebook_ops.write_notebook("notebook.ipynb", nb, ["/allowed"], None)

@pytest.mark.asyncio
async def test_resolve_path_remote_without_allowed_roots():
    """Test resolving a remote path without any allowed roots."""
    # Create a mock SFTP manager for remote path resolution
    mock_manager = mock.MagicMock(spec=SFTPManager)
    mock_manager._get_absolute_remote_path.return_value = "/remote/path.ipynb"
    
    # Set up path mappings and connections for permission check to succeed
    mock_sftp = mock.MagicMock()
    mock_sftp.normalize.return_value = "/remote"
    mock_manager.connections = {"host": (None, mock_sftp)}
    mock_manager.path_mappings = {
        "spec": ("host", "user", "/remote", "/local") 
    }
    mock_manager._get_remote_home_dir.return_value = "/home/user"
    
    # Mock the permission check to pass
    with mock.patch('posixpath.normpath', side_effect=lambda p: p), \
         mock.patch('posixpath.isabs', return_value=True):
        
        # This should work because remote paths with SFTP manager don't need allowed_roots
        is_remote, path = await asyncio.to_thread(
            notebook_ops.resolve_path_and_check_permissions,
            "ssh://user@host/path.ipynb",
            [],  # Empty allowed_roots
            mock_manager
        )
        
        assert is_remote is True
        assert path == "/remote/path.ipynb" 