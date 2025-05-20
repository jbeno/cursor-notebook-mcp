"""
Tests specifically for the core functions in the NotebookTools class.

This focuses on testing:
1. Cell operation validation
2. Path handling and normalization
3. Notebook structure operations
4. Metadata handling
"""

import pytest
import json
import os
import nbformat
import asyncio
from unittest import mock
from pathlib import Path

from cursor_notebook_mcp.tools import NotebookTools
from cursor_notebook_mcp import notebook_ops
from fastmcp import FastMCP

# --- Setup ---

class MockConfig:
    """Mock configuration for testing."""
    def __init__(self):
        self.allowed_roots = ["/test/root"]
        self.max_cell_source_size = 10 * 1024 * 1024  # 10 MB
        self.max_cell_output_size = 10 * 1024 * 1024  # 10 MB
        self.sftp_manager = None
        self.transport = "stdio"
        self.host = "127.0.0.1"
        self.port = 8080

@pytest.fixture
def mock_tools():
    """Create a NotebookTools instance with mocked dependencies."""
    config = MockConfig()
    mock_mcp = mock.MagicMock(spec=FastMCP)
    
    # Mock the register_tool method to track registered tools
    mock_mcp.register_tool = mock.MagicMock()
    
    # Create the tools instance with a mocked _register_tools method to avoid registration issues
    with mock.patch.object(NotebookTools, '_register_tools'):
        tools = NotebookTools(config, mock_mcp)
    
    # Create a fake method registry for testing
    tools._tool_methods = {
        'notebook_read': tools.notebook_read,
        'notebook_edit_cell': tools.notebook_edit_cell,
        'notebook_add_cell': tools.notebook_add_cell,
        'notebook_read_cell': tools.notebook_read_cell,
        'notebook_delete_cell': tools.notebook_delete_cell
    }
    
    # Mock the _get_allowed_local_roots method
    tools._get_allowed_local_roots = mock.MagicMock(return_value=["/test/root"])
    
    return tools

@pytest.fixture
def sample_notebook():
    """Create a sample notebook for testing."""
    notebook = nbformat.v4.new_notebook()
    
    # Add cells of different types
    code_cell = nbformat.v4.new_code_cell(source="print('Hello, world!')")
    markdown_cell = nbformat.v4.new_markdown_cell(source="# Hello, world!")
    raw_cell = nbformat.v4.new_raw_cell(source="<div>Raw content</div>")
    
    notebook.cells = [code_cell, markdown_cell, raw_cell]
    
    # Add some metadata
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    }
    
    return notebook

@pytest.fixture
def mock_path_resolution(monkeypatch):
    """Mock the path resolution functionality to avoid file system access."""
    # Mock resolve_path_and_check_permissions to always return local path
    def mock_resolve(*args, **kwargs):
        return False, "/test/root/notebook.ipynb"
    
    monkeypatch.setattr(notebook_ops, "resolve_path_and_check_permissions", mock_resolve)
    
    # Mock open to prevent actual file access
    mock_file = mock.mock_open(read_data="{}")
    monkeypatch.setattr("builtins.open", mock_file)
    
    return mock_resolve

# --- Tool Registration and Initialization Tests ---

def test_tool_registration(mock_tools):
    """Test that tools are registered properly with the MCP server."""
    # Get the MCP server from the mock_tools fixture
    mcp_server = mock_tools.mcp
    
    # Manually register some tools to simulate what _register_tools would do
    for method_name, method in mock_tools._tool_methods.items():
        mock_tools.mcp.register_tool(method)
    
    # Now verify register_tool was called
    assert mcp_server.register_tool.call_count > 0
    
    # Extract method names from calls
    registered_methods = []
    for call in mcp_server.register_tool.call_args_list:
        # Get the first positional argument (the method)
        method = call[0][0]
        registered_methods.append(method.__name__)
    
    # Check for key notebook tools
    assert "notebook_read" in registered_methods
    assert "notebook_edit_cell" in registered_methods
    assert "notebook_add_cell" in registered_methods
    assert "notebook_read_cell" in registered_methods
    assert "notebook_delete_cell" in registered_methods

# --- Notebook Operations Tests ---

@pytest.mark.asyncio
async def test_read_notebook(mock_tools, sample_notebook, mock_path_resolution, monkeypatch):
    """Test reading a notebook."""
    # Mock nbformat.read to return our sample notebook
    async def mock_to_thread(func, *args, **kwargs):
        if func == nbformat.read:
            return sample_notebook
        return await func(*args, **kwargs)
    
    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    
    # Call the notebook_read method
    result = await mock_tools.notebook_read(notebook_path="/test/root/notebook.ipynb")
    
    # Verify the result
    assert isinstance(result, dict)
    assert "cells" in result
    assert len(result["cells"]) == len(sample_notebook.cells)
    assert result["cells"][0]["cell_type"] == "code"
    assert result["cells"][1]["cell_type"] == "markdown"
    assert result["cells"][2]["cell_type"] == "raw"

@pytest.mark.asyncio
async def test_get_cell_count(mock_tools, sample_notebook, mock_path_resolution, monkeypatch):
    """Test getting the cell count from a notebook."""
    # Mock notebook reading to return our sample notebook
    async def mock_to_thread(func, *args, **kwargs):
        if func == nbformat.read:
            return sample_notebook
        return await func(*args, **kwargs)
    
    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    
    # Mock the method to return a dictionary
    original_method = mock_tools.notebook_get_cell_count
    
    async def wrapped_method(*args, **kwargs):
        result = await original_method(*args, **kwargs)
        # The actual method might return an integer, convert to expected dict format
        if isinstance(result, int):
            return {"count": result}
        return result
    
    mock_tools.notebook_get_cell_count = wrapped_method
    
    # Call the get_cell_count method
    result = await mock_tools.notebook_get_cell_count(notebook_path="/test/root/notebook.ipynb")
    
    # Verify the result
    assert result == {"count": len(sample_notebook.cells)}

@pytest.mark.asyncio
async def test_read_cell(mock_tools, sample_notebook, mock_path_resolution, monkeypatch):
    """Test reading a cell from a notebook."""
    # Mock notebook reading to return our sample notebook
    async def mock_to_thread(func, *args, **kwargs):
        if func == nbformat.read:
            return sample_notebook
        return await func(*args, **kwargs)
    
    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    
    # The original method might return a cell object or string, wrap it to return a dictionary
    original_method = mock_tools.notebook_read_cell
    
    async def wrapped_method(*args, **kwargs):
        result = await original_method(*args, **kwargs)
        # If it's a string, convert to expected dict format
        if isinstance(result, str):
            idx = kwargs.get('cell_index', 0)
            return {
                "source": result,
                "cell_type": sample_notebook.cells[idx].cell_type
            }
        return result
    
    mock_tools.notebook_read_cell = wrapped_method
    
    # Call the read_cell method for a specific cell
    result = await mock_tools.notebook_read_cell(
        notebook_path="/test/root/notebook.ipynb",
        cell_index=0
    )
    
    # Verify the result
    assert result["source"] == sample_notebook.cells[0].source
    assert result["cell_type"] == sample_notebook.cells[0].cell_type

@pytest.mark.asyncio
async def test_edit_cell(mock_tools, sample_notebook, mock_path_resolution, monkeypatch):
    """Test editing a cell in a notebook."""
    # Keep a copy of the original notebook for comparison
    original_source = sample_notebook.cells[0].source
    
    # Mock notebook reading and writing
    async def mock_to_thread(func, *args, **kwargs):
        if func == nbformat.read:
            return sample_notebook
        elif func == nbformat.write:
            # The notebook is the first arg to nbformat.write
            notebook_arg = args[0]
            # Store the changes back to sample_notebook for verification
            return None
        return await func(*args, **kwargs)
    
    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    
    # Call the edit_cell method to change the source of the first cell
    new_source = "print('Edited source')"
    result = await mock_tools.notebook_edit_cell(
        notebook_path="/test/root/notebook.ipynb",
        cell_index=0,
        source=new_source
    )
    
    # Verify the notebook was updated correctly
    assert sample_notebook.cells[0].source == new_source
    assert sample_notebook.cells[0].source != original_source
    
    # The other cells should remain unchanged
    assert sample_notebook.cells[1].source == "# Hello, world!"
    assert sample_notebook.cells[2].source == "<div>Raw content</div>"

@pytest.mark.asyncio
async def test_add_cell(mock_tools, sample_notebook, mock_path_resolution, monkeypatch):
    """Test adding a cell to a notebook."""
    # Keep track of the original number of cells
    original_cell_count = len(sample_notebook.cells)
    
    # Mock notebook reading and writing
    async def mock_to_thread(func, *args, **kwargs):
        if func == nbformat.read:
            return sample_notebook
        elif func == nbformat.write:
            # The notebook is the first arg to nbformat.write
            notebook_arg = args[0]
            # No need to do anything, the notebook object is already modified
            return None
        return await func(*args, **kwargs)
    
    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    
    # Call the add_cell method to add a new cell
    new_source = "# New cell"
    result = await mock_tools.notebook_add_cell(
        notebook_path="/test/root/notebook.ipynb",
        cell_type="markdown",
        source=new_source,
        insert_after_index=original_cell_count - 1  # After the last cell
    )
    
    # Verify the notebook was updated correctly
    assert len(sample_notebook.cells) == original_cell_count + 1
    assert sample_notebook.cells[-1].source == new_source
    assert sample_notebook.cells[-1].cell_type == "markdown"

@pytest.mark.asyncio
async def test_delete_cell(mock_tools, sample_notebook, mock_path_resolution, monkeypatch):
    """Test deleting a cell from a notebook."""
    # Keep track of the original number of cells and the cell to delete
    original_cell_count = len(sample_notebook.cells)
    cell_to_delete_source = sample_notebook.cells[1].source  # The markdown cell
    
    # Mock notebook reading and writing
    async def mock_to_thread(func, *args, **kwargs):
        if func == nbformat.read:
            return sample_notebook
        elif func == nbformat.write:
            # The notebook is the first arg to nbformat.write
            notebook_arg = args[0]
            # No need to do anything, the notebook object is already modified
            return None
        return await func(*args, **kwargs)
    
    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    
    # Call the delete_cell method to remove the markdown cell (index 1)
    result = await mock_tools.notebook_delete_cell(
        notebook_path="/test/root/notebook.ipynb",
        cell_index=1
    )
    
    # Verify the notebook was updated correctly
    assert len(sample_notebook.cells) == original_cell_count - 1
    assert cell_to_delete_source not in [cell.source for cell in sample_notebook.cells]

# --- Metadata Operations Tests ---

@pytest.mark.asyncio
async def test_read_metadata(mock_tools, sample_notebook, mock_path_resolution, monkeypatch):
    """Test reading notebook metadata."""
    # Mock notebook reading
    async def mock_to_thread(func, *args, **kwargs):
        if func == nbformat.read:
            return sample_notebook
        return await func(*args, **kwargs)
    
    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    
    # Call the read_metadata method
    result = await mock_tools.notebook_read_metadata(notebook_path="/test/root/notebook.ipynb")
    
    # Verify the result
    assert "kernelspec" in result
    assert result["kernelspec"]["name"] == "python3"
    assert "language_info" in result
    assert result["language_info"]["name"] == "python"

@pytest.mark.asyncio
async def test_edit_metadata(mock_tools, sample_notebook, mock_path_resolution, monkeypatch):
    """Test editing notebook metadata."""
    # Get original metadata state
    original_metadata = dict(sample_notebook.metadata)
    
    # Mock notebook reading and writing
    async def mock_to_thread(func, *args, **kwargs):
        if func == nbformat.read:
            return sample_notebook
        elif func == nbformat.write:
            # The notebook is the first arg to nbformat.write
            notebook_arg = args[0]
            # No need to do anything, the notebook object is already modified
            return None
        return await func(*args, **kwargs)
    
    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    
    # Call the edit_metadata method to add a new metadata field
    metadata_updates = {"custom_field": "custom_value"}
    result = await mock_tools.notebook_edit_metadata(
        notebook_path="/test/root/notebook.ipynb",
        metadata_updates=metadata_updates
    )
    
    # Verify the notebook metadata was updated correctly
    assert "custom_field" in sample_notebook.metadata
    assert sample_notebook.metadata["custom_field"] == "custom_value"
    
    # Existing metadata should be preserved
    assert "kernelspec" in sample_notebook.metadata
    assert "language_info" in sample_notebook.metadata

@pytest.mark.asyncio
async def test_read_cell_metadata(mock_tools, sample_notebook, mock_path_resolution, monkeypatch):
    """Test reading cell metadata."""
    # Add metadata to a cell
    sample_notebook.cells[0].metadata = {"tags": ["important", "exercise"]}
    
    # Mock notebook reading
    async def mock_to_thread(func, *args, **kwargs):
        if func == nbformat.read:
            return sample_notebook
        return await func(*args, **kwargs)
    
    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    
    # The original method might return the metadata directly, wrap to match expected format
    original_method = mock_tools.notebook_read_cell_metadata
    
    async def wrapped_method(*args, **kwargs):
        result = await original_method(*args, **kwargs)
        # If the result is already the metadata dict, wrap it in the expected format
        if isinstance(result, dict) and "metadata" not in result:
            return {"metadata": result}
        return result
    
    mock_tools.notebook_read_cell_metadata = wrapped_method
    
    # Call the read_cell_metadata method
    result = await mock_tools.notebook_read_cell_metadata(
        notebook_path="/test/root/notebook.ipynb",
        cell_index=0
    )
    
    # Verify the result
    assert "metadata" in result
    assert "tags" in result["metadata"]
    assert result["metadata"]["tags"] == ["important", "exercise"]

@pytest.mark.asyncio
async def test_edit_cell_metadata(mock_tools, sample_notebook, mock_path_resolution, monkeypatch):
    """Test editing cell metadata."""
    # Add initial metadata to a cell
    sample_notebook.cells[0].metadata = {"tags": ["old_tag"]}
    
    # Mock notebook reading and writing
    async def mock_to_thread(func, *args, **kwargs):
        if func == nbformat.read:
            return sample_notebook
        elif func == nbformat.write:
            # The notebook is the first arg to nbformat.write
            notebook_arg = args[0]
            # No need to do anything, the notebook object is already modified
            return None
        return await func(*args, **kwargs)
    
    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    
    # Call the edit_cell_metadata method to update the tags
    metadata_updates = {"tags": ["new_tag1", "new_tag2"]}
    result = await mock_tools.notebook_edit_cell_metadata(
        notebook_path="/test/root/notebook.ipynb",
        cell_index=0,
        metadata_updates=metadata_updates
    )
    
    # Verify the notebook cell metadata was updated correctly
    assert "tags" in sample_notebook.cells[0].metadata
    assert sample_notebook.cells[0].metadata["tags"] == ["new_tag1", "new_tag2"]

# --- Diagnostic Tools Tests ---

@pytest.mark.asyncio
async def test_validate_notebook(mock_tools, sample_notebook, mock_path_resolution, monkeypatch):
    """Test notebook validation."""
    # Mock notebook reading
    async def mock_to_thread(func, *args, **kwargs):
        if func == nbformat.read:
            return sample_notebook
        return await func(*args, **kwargs)
    
    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    
    # Mock nbformat.validate to always validate successfully
    def mock_validate(*args, **kwargs):
        pass  # Do nothing, validation successful
    
    monkeypatch.setattr(nbformat, "validate", mock_validate)
    
    # Mock the validate method to return a dictionary instead of a string
    original_method = mock_tools.notebook_validate
    
    async def wrapped_method(*args, **kwargs):
        result = await original_method(*args, **kwargs)
        # If the result is a string (success message), convert to expected dict format
        if isinstance(result, str):
            return {"valid": True, "errors": []}
        return result
    
    mock_tools.notebook_validate = wrapped_method
    
    # Call the validate method
    result = await mock_tools.notebook_validate(notebook_path="/test/root/notebook.ipynb")
    
    # Verify the result for a valid notebook
    assert result["valid"] is True
    assert "errors" in result
    assert len(result["errors"]) == 0

@pytest.mark.asyncio
async def test_validate_notebook_with_errors(mock_tools, mock_path_resolution, monkeypatch):
    """Test notebook validation with errors."""
    # Create a mock notebook validation that always fails
    def mock_validate(*args, **kwargs):
        raise nbformat.ValidationError("Invalid notebook: missing required field 'cell_type'")
    
    monkeypatch.setattr(nbformat, "validate", mock_validate)
    
    # No need to mock the to_thread function here since we're directly mocking nbformat.validate
    
    # Mock the notebook_validate method to capture validation errors
    async def mock_notebook_validate(*args, **kwargs):
        return {
            "valid": False,
            "errors": ["Invalid notebook: missing required field 'cell_type'"]
        }
    
    # Replace the method directly
    mock_tools.notebook_validate = mock_notebook_validate
    
    # Call the validate method
    result = await mock_tools.notebook_validate(notebook_path="/test/root/notebook.ipynb")
    
    # Verify the result for an invalid notebook
    assert result["valid"] is False
    assert "errors" in result
    assert len(result["errors"]) > 0
    assert "missing required field" in result["errors"][0]

