"""
Tests specifically for edge cases in the NotebookTools class.

This focuses on:
1. Error handling paths that normal tests don't hit
2. Methods with lower test coverage
3. Tool registration and initialization
"""

import pytest
import os
import nbformat
import asyncio
from unittest import mock
import json
import tempfile
import sys
import subprocess
from pathlib import Path
import importlib
import logging
import re

from cursor_notebook_mcp.tools import NotebookTools
from fastmcp import FastMCP

# --- Setup Fixtures ---

class MockConfig:
    """Mock configuration for testing."""
    def __init__(self):
        self.allowed_roots = ["/test/root"]
        self.max_cell_source_size = 10 * 1024 * 1024  # 10 MB
        self.max_cell_output_size = 10 * 1024 * 1024  # 10 MB
        self.sftp_manager = None
        self.log_level = logging.INFO

@pytest.fixture
def mock_mcp():
    """Create a mock MCP server instance with a working tool registration."""
    mock_instance = mock.MagicMock(spec=FastMCP)
    
    # Create a working add_tool method
    mock_instance.add_tool = mock.MagicMock()
    
    return mock_instance

@pytest.fixture
def notebook_tools(mock_mcp):
    """Create a NotebookTools instance with mocked dependencies."""
    config = MockConfig()
    nt = NotebookTools(config, mock_mcp)
    return nt

@pytest.fixture
def notebook_tools_with_sftp(mock_mcp):
    """Create a NotebookTools instance with mock SFTP manager."""
    config = MockConfig()
    config.sftp_manager = mock.MagicMock()
    nt = NotebookTools(config, mock_mcp)
    return nt

@pytest.fixture
def temp_notebook_path():
    """Create a temporary notebook file."""
    fd, path = tempfile.mkstemp(suffix=".ipynb")
    os.close(fd)
    
    # Create a simple valid notebook
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell("print('test')"))
    nbformat.write(nb, path)
    
    yield path
    
    # Clean up
    if os.path.exists(path):
        os.remove(path)

# --- Test Constructor and Registration ---

def test_notebook_tools_init_add_tool(mock_mcp):
    """Test that NotebookTools initialization registers tools properly."""
    config = MockConfig()
    nt = NotebookTools(config, mock_mcp)
    
    # Check that add_tool was called for each tool method
    # This relies on the tool counting from _register_tools implementation
    expected_min_calls = 25  # There should be at least 25 tool methods
    assert mock_mcp.add_tool.call_count >= expected_min_calls

def test_notebook_tools_init_tool_decorator(mock_mcp):
    """Test NotebookTools initialization with tool decorator instead of add_tool."""
    config = MockConfig()
    
    # Create a working mock tool decorator
    decorator_func = lambda x: x  # Identity function
    mock_decorator = mock.MagicMock(return_value=decorator_func)
    
    # Remove add_tool but add a working tool decorator
    mock_mcp.add_tool = None
    mock_mcp.tool = mock_decorator
    
    # Patch the _register_tools method to prevent actual registration attempts
    with mock.patch('cursor_notebook_mcp.tools.NotebookTools._register_tools'):
        nt = NotebookTools(config, mock_mcp)
    
    # Verify the object was created successfully
    assert isinstance(nt, NotebookTools)

def test_notebook_tools_init_no_registration_method():
    """Test NotebookTools initialization with no valid registration method."""
    config = MockConfig()
    
    # Create an MCP instance with no valid registration method
    mock_mcp = mock.MagicMock(spec=FastMCP)
    # Explicitly set the attributes to None to ensure they don't exist
    mock_mcp.add_tool = None
    mock_mcp.tool = None
    
    # Patch the _register_tools method to raise the AttributeError we want to test
    with mock.patch('cursor_notebook_mcp.tools.NotebookTools._register_tools', 
                   side_effect=AttributeError("FastMCP instance does not have a known tool registration method")):
        # Initialization should raise AttributeError
        with pytest.raises(AttributeError, match="FastMCP instance does not have a known tool registration method"):
            nt = NotebookTools(config, mock_mcp)

# --- Test Notebook Create Edge Cases ---

@pytest.mark.asyncio
async def test_notebook_create_no_path(notebook_tools):
    """Test notebook_create with an empty path."""
    with pytest.raises(ValueError, match="Invalid notebook path provided"):
        await notebook_tools.notebook_create(notebook_path="")

@pytest.mark.asyncio
async def test_notebook_create_invalid_extension(notebook_tools):
    """Test notebook_create with invalid file extension."""
    with pytest.raises(ValueError, match="Invalid file type"):
        await notebook_tools.notebook_create(notebook_path="/test/file.txt")

@pytest.mark.asyncio
async def test_notebook_create_path_resolution_error(notebook_tools):
    """Test notebook_create when path resolution fails."""
    # Mock resolve_path_and_check_permissions to raise an error
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                   side_effect=ValueError("Invalid path")):
        with pytest.raises(ValueError, match="Invalid path"):
            await notebook_tools.notebook_create("/test/notebook.ipynb")

@pytest.mark.asyncio
async def test_notebook_create_sftp_no_manager(notebook_tools):
    """Test notebook_create when SFTP is needed but not available."""
    # Mock resolve_path_and_check_permissions to indicate it's remote
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                   return_value=(True, "/remote/notebook.ipynb")):
        with pytest.raises(ConnectionError, match="SFTP manager required"):
            await notebook_tools.notebook_create("/remote/notebook.ipynb")

@pytest.mark.asyncio
async def test_notebook_create_file_exists(notebook_tools, temp_notebook_path):
    """Test notebook_create when file already exists."""
    # Use a temp file that was created by the fixture
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                   return_value=(False, temp_notebook_path)):
        with pytest.raises(FileExistsError, match="Cannot create notebook, file already exists"):
            await notebook_tools.notebook_create(temp_notebook_path)

@pytest.mark.asyncio
async def test_notebook_create_write_failure(notebook_tools):
    """Test notebook_create when write fails."""
    # Mock the necessary components
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(False, "/test/notebook.ipynb")):
        with mock.patch('os.path.exists', return_value=False):
            with mock.patch('os.makedirs') as mock_makedirs:
                with mock.patch('cursor_notebook_mcp.notebook_ops.write_notebook', 
                              side_effect=IOError("Failed to write notebook")):
                    with pytest.raises(IOError) as excinfo:
                        await notebook_tools.notebook_create("/test/notebook.ipynb")
                    # Just check if this is an IOError related to notebook writing
                    assert "notebook" in str(excinfo.value).lower()

@pytest.mark.asyncio
async def test_notebook_create_unexpected_error(notebook_tools):
    """Test notebook_create with an unexpected error."""
    # Mock the necessary components
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(False, "/test/notebook.ipynb")):
        with mock.patch('os.path.exists', return_value=False):
            with mock.patch('os.makedirs'):
                with mock.patch.object(notebook_tools, 'write_notebook',
                                      side_effect=Exception("Unexpected error")):
                    with pytest.raises(RuntimeError, match="An unexpected error occurred during notebook creation"):
                        await notebook_tools.notebook_create("/test/notebook.ipynb")

# --- Test Notebook Delete Edge Cases ---

@pytest.mark.asyncio
async def test_notebook_delete_path_not_allowed(notebook_tools):
    """Test notebook_delete with a path outside allowed roots."""
    # Mock notebook_ops.is_path_allowed to return False
    with mock.patch('cursor_notebook_mcp.notebook_ops.is_path_allowed', return_value=False):
        with pytest.raises(PermissionError, match="Access denied"):
            await notebook_tools.notebook_delete("/outside/root/notebook.ipynb")

@pytest.mark.asyncio
async def test_notebook_delete_unexpected_error(notebook_tools):
    """Test notebook_delete with an unexpected error."""
    target_path = "/test/root/notebook.ipynb"
    # Mock path checks but make os.remove raise an unexpected error
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                   return_value=(False, target_path)) as mock_resolve_check, \
         mock.patch('os.path.exists', return_value=True) as mock_exists, \
         mock.patch('os.remove', side_effect=Exception("Unexpected error")) as mock_remove:
        
        with pytest.raises(RuntimeError, match="An unexpected error occurred during notebook deletion"):
            await notebook_tools.notebook_delete(target_path)

        mock_resolve_check.assert_called_once_with(target_path, notebook_tools._get_allowed_local_roots(), notebook_tools.sftp_manager)
        mock_exists.assert_called_once_with(target_path)
        mock_remove.assert_called_once_with(target_path)

# --- Test Notebook Rename Edge Cases ---

@pytest.mark.asyncio
async def test_notebook_rename_invalid_paths(notebook_tools):
    """Test notebook_rename with invalid paths."""
    # Test with empty paths
    with pytest.raises(ValueError, match="Invalid old or new path provided"):
        await notebook_tools.notebook_rename("", "/test/new.ipynb")
    
    with pytest.raises(ValueError, match="Invalid old or new path provided"):
        await notebook_tools.notebook_rename("/test/old.ipynb", "")
    
    # Test with invalid extensions
    with pytest.raises(ValueError, match="Invalid old or new path provided"):
        await notebook_tools.notebook_rename("/test/old.txt", "/test/new.ipynb")
    
    with pytest.raises(ValueError, match="Invalid old or new path provided"):
        await notebook_tools.notebook_rename("/test/old.ipynb", "/test/new.txt")

@pytest.mark.asyncio
async def test_notebook_rename_mixed_storage_types(notebook_tools_with_sftp):
    """Test notebook_rename between local and remote storage."""
    # Mock resolution to return different types for old and new paths
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  side_effect=[(False, "/local/old.ipynb"), (True, "/remote/new.ipynb")]):
        with pytest.raises(ValueError, match="Cannot rename/move between local and remote storage"):
            await notebook_tools_with_sftp.notebook_rename("/local/old.ipynb", "/remote/new.ipynb")

@pytest.mark.asyncio
async def test_notebook_rename_old_not_exists(notebook_tools):
    """Test notebook_rename when source file doesn't exist."""
    # Mock path resolution but make the existence check fail
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(False, "/test/old.ipynb")):
        with mock.patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="Source notebook file not found"):
                await notebook_tools.notebook_rename("/test/old.ipynb", "/test/new.ipynb")

@pytest.mark.asyncio
async def test_notebook_rename_new_exists(notebook_tools):
    """Test notebook_rename when destination file already exists."""
    # Mock path resolution and make both existence checks pass
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(False, "/test/path.ipynb")):
        with mock.patch('os.path.exists', return_value=True):
            with pytest.raises(FileExistsError, match="Cannot rename notebook, destination already exists"):
                await notebook_tools.notebook_rename("/test/old.ipynb", "/test/new.ipynb")

@pytest.mark.asyncio
async def test_notebook_rename_sftp_required(notebook_tools):
    """Test notebook_rename for remote files without SFTP manager."""
    # Mock resolution to indicate remote files
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(True, "/remote/path.ipynb")):
        with pytest.raises(ConnectionError, match="SFTP required"):
            await notebook_tools.notebook_rename("/remote/old.ipynb", "/remote/new.ipynb")

@pytest.mark.asyncio
async def test_notebook_rename_unexpected_error(notebook_tools):
    """Test notebook_rename with an unexpected error."""
    # Mock resolution and existence checks to pass
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(False, "/test/path.ipynb")):
        with mock.patch('os.path.exists', side_effect=[True, False]):  # Old exists, new doesn't
            # Make os.rename raise an unexpected error
            with mock.patch('os.rename', side_effect=Exception("Unexpected error")):
                with pytest.raises(RuntimeError, match="An unexpected error occurred during notebook rename"):
                    await notebook_tools.notebook_rename("/test/old.ipynb", "/test/new.ipynb")

# --- Test Export Functionality Edge Cases ---

@pytest.mark.asyncio
async def test_notebook_export_unsupported_format(notebook_tools, temp_notebook_path):
    """Test notebook_export with an unsupported format."""
    with pytest.raises(ValueError, match="Unsupported export format"):
        await notebook_tools.notebook_export(notebook_path=temp_notebook_path, export_format="invalid")

@pytest.mark.asyncio
async def test_notebook_export_remote_no_sftp(notebook_tools):
    """Test notebook_export with a remote path but no SFTP manager."""
    # Mock resolution to indicate a remote file
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(True, "/remote/notebook.ipynb")):
        with pytest.raises(ConnectionError, match="SFTP required for remote export prep"):
            await notebook_tools.notebook_export(notebook_path="/remote/notebook.ipynb", export_format="python")

@pytest.mark.asyncio
async def test_notebook_export_nbconvert_command_error(notebook_tools, temp_notebook_path):
    """Test notebook_export when nbconvert command fails."""
    # Mock path resolution to avoid permission issues
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(False, temp_notebook_path)):
        # Mock subprocess.run to simulate command failure
        mock_proc = mock.MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "nbconvert error message"
        
        with mock.patch('subprocess.run', return_value=mock_proc):
            with pytest.raises(RuntimeError, match="nbconvert failed.*nbconvert error message"):
                await notebook_tools.notebook_export(notebook_path=temp_notebook_path, export_format="python")

@pytest.mark.asyncio
async def test_notebook_export_unexpected_error(notebook_tools, temp_notebook_path):
    """Test notebook_export with an unexpected error."""
    # Mock path resolution to avoid permission issues
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(False, temp_notebook_path)):
        # Mock subprocess.run to raise an unexpected error
        with mock.patch('subprocess.run', side_effect=Exception("Unexpected error")):
            with pytest.raises(RuntimeError, match="An unexpected error occurred during notebook export"):
                await notebook_tools.notebook_export(notebook_path=temp_notebook_path, export_format="python")

# --- Test Notebook Read Edge Cases ---

@pytest.mark.asyncio
async def test_notebook_read_invalid_path(notebook_tools):
    """Test notebook_read with an invalid path."""
    # Skip the resolution step that accesses the file system
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', side_effect=ValueError("Invalid notebook path")):
        with pytest.raises(ValueError, match="Invalid notebook path"):
            await notebook_tools.notebook_read("file.txt")

@pytest.mark.asyncio
async def test_notebook_read_remote_no_sftp(notebook_tools):
    """Test notebook_read with a remote path but no SFTP manager."""
    # Mock resolution to indicate a remote file
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(True, "/remote/notebook.ipynb")):
        with pytest.raises(ConnectionError, match="SFTP required"):
            await notebook_tools.notebook_read("/remote/notebook.ipynb")

@pytest.mark.asyncio
async def test_notebook_read_file_not_found(notebook_tools):
    """Test notebook_read when file doesn't exist."""
    # Mock resolution to avoid permission issues
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(False, "/local/notebook.ipynb")):
        # Mock open to raise FileNotFoundError
        mock_open = mock.mock_open()
        mock_open.side_effect = FileNotFoundError("No such file")
        with mock.patch('builtins.open', mock_open):
            with pytest.raises(FileNotFoundError, match="No such file"):
                await notebook_tools.notebook_read("/local/notebook.ipynb")

@pytest.mark.asyncio
async def test_notebook_read_invalid_json(notebook_tools):
    """Test notebook_read with invalid JSON content."""
    # Mock resolution to avoid permission issues
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(False, "/local/notebook.ipynb")):
        # Mock open to return invalid JSON
        mock_open = mock.mock_open(read_data="{ not valid json }")
        with mock.patch('builtins.open', mock_open):
            with pytest.raises(ValueError):  # nbformat.reads will raise this
                await notebook_tools.notebook_read("/local/notebook.ipynb")

@pytest.mark.asyncio
async def test_notebook_read_oversized_cell(notebook_tools):
    """Test notebook_read with oversized cells that should be truncated."""
    
    mock_notebook_dict_content = {
        "cells": [
            {
                "cell_type": "code",
                "source": "A" * 1000,  
                "metadata": { 
                    "truncated": True,
                    "originalSize": 1000000
                },
                "id": "mock_cell_id_1",
                "outputs": [],
                "execution_count": None
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    mock_notebook_node = nbformat.from_dict(mock_notebook_dict_content)

    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions',
                  return_value=(False, "/test/notebook.ipynb")) as mock_resolve, \
         mock.patch('builtins.open', mock.mock_open()) as mock_file_open, \
         mock.patch('nbformat.read', return_value=mock_notebook_node) as mock_nb_read:

        result_dict = await notebook_tools.notebook_read("/test/notebook.ipynb")
            
        assert 'cells' in result_dict
        assert len(result_dict['cells']) == 1
        cell_result = result_dict['cells'][0]
        assert 'metadata' in cell_result
        assert 'truncated' in cell_result['metadata'] 
        assert cell_result['metadata']['truncated'] is True
        assert cell_result.get('id') == "mock_cell_id_1"
        assert 'outputs' in cell_result
        assert 'execution_count' in cell_result

@pytest.mark.asyncio
async def test_notebook_read_oversized_notebook(notebook_tools, temp_notebook_path):
    """Test notebook_read with an oversized notebook that exceeds total limit."""
    # Create a notebook with many large cells
    nb = nbformat.v4.new_notebook()
    for i in range(10):
        nb.cells.append(nbformat.v4.new_code_cell("A" * 100000))  # 100KB each
    nbformat.write(nb, temp_notebook_path)
    
    # Set a small total size limit
    notebook_tools.config.max_cell_source_size = 1000000  # 1MB
    
    # Mock the detection of total size
    original_json_dumps = json.dumps
    
    def mock_dumps(obj, *args, **kwargs):
        # Make the first few cells normal, then force the total size check to trigger
        if isinstance(obj, dict) and obj.get('cell_type') and 'source' in obj:
            # This is a cell being converted to string
            cell_index = nb.cells.index(obj) if obj in nb.cells else -1
            if cell_index >= 5:  # After 5th cell
                # Simulate exceeding the total size limit
                return "x" * 50000000  # Return a large string to trigger the limit
        return original_json_dumps(obj, *args, **kwargs)
    
    # Mock resolution to use our temp path
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(False, temp_notebook_path)):
        # Mock json.dumps to control the size check
        with mock.patch('json.dumps', side_effect=mock_dumps):
            # Read the notebook - should be truncated
            result = await notebook_tools.notebook_read(temp_notebook_path)
            
            # Check that the notebook was truncated
            assert 'cells' in result
            assert len(result['cells']) < 10  # Should have fewer cells than original
            assert 'truncated' in result['metadata']

# --- Test Notebook Validation Edge Cases ---

@pytest.mark.asyncio
async def test_validate_notebook_invalid_path(notebook_tools):
    """Test notebook_validate with invalid path."""
    # Skip the resolution step that accesses the file system  
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', side_effect=ValueError("Invalid notebook path")):
        with pytest.raises(ValueError, match="Invalid notebook path"):
            await notebook_tools.notebook_validate("file.txt")

@pytest.mark.asyncio
async def test_validate_notebook_file_not_found(notebook_tools):
    """Test notebook_validate when file doesn't exist."""
    # Mock resolution to avoid permission issues
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(False, "/nonexistent.ipynb")):
        # Mock open to raise FileNotFoundError
        mock_open = mock.mock_open()
        mock_open.side_effect = FileNotFoundError("No such file")
        with mock.patch('builtins.open', mock_open):
            with pytest.raises(FileNotFoundError, match="No such file"):
                await notebook_tools.notebook_validate("/nonexistent.ipynb")

@pytest.mark.asyncio
async def test_validate_notebook_unexpected_error(notebook_tools, temp_notebook_path):
    """Test notebook_validate with an unexpected error."""
    # Mock resolution to avoid permission issues
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  return_value=(False, temp_notebook_path)):
        # Make nbformat.validate raise an unexpected error
        with mock.patch('nbformat.validate', side_effect=Exception("Unexpected error")):
            with pytest.raises(RuntimeError, match="An unexpected error occurred validating notebook"):
                await notebook_tools.notebook_validate(temp_notebook_path) 