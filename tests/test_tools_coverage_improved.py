"""
Additional tests to improve coverage for the NotebookTools class.

This file focuses on improving coverage for methods in tools.py that have
lower coverage, particularly:
1. diagnose_imports
2. notebook_export
3. Error handling paths in various methods
"""

import pytest
import tempfile
import os
import nbformat
import json
import sys
import subprocess
import importlib
from unittest import mock
from pathlib import Path

from cursor_notebook_mcp.tools import NotebookTools

@pytest.fixture
def notebook_tools():
    """Create a NotebookTools instance for testing."""
    # Create a mock config object with the required attributes
    config = mock.MagicMock()
    config.allowed_roots = [str(Path.cwd())]
    config.max_cell_source_size = 1024 * 1024  # 1MB
    config.max_cell_output_size = 1024 * 1024  # 1MB
    config.sftp_manager = None
    
    # Create a mock MCP instance
    mcp_instance = mock.MagicMock()
    
    # Create tools with mock objects
    tools = NotebookTools(config, mcp_instance)
    
    return tools

@pytest.fixture
def sample_notebook_path():
    """Create a temporary notebook file for testing."""
    # Create a simple notebook
    nb = nbformat.v4.new_notebook()
    # Add a code cell
    nb.cells.append(nbformat.v4.new_code_cell(source="print('Hello, world!')"))
    # Add a markdown cell
    nb.cells.append(nbformat.v4.new_markdown_cell(source="# Markdown Cell"))
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False, mode='w') as f:
        notebook_path = f.name
        nbformat.write(nb, f)
        
    yield notebook_path
    
    # Clean up
    if os.path.exists(notebook_path):
        os.unlink(notebook_path)

# --- Tests for notebook_export method ---

@pytest.mark.asyncio
async def test_notebook_validate_with_custom_notebook(notebook_tools, sample_notebook_path):
    """Test notebook_validate with a custom notebook."""
    # Add the temporary directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Create a valid notebook
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(source="print('Hello')"))
    
    # Write the notebook to the temporary file
    with open(sample_notebook_path, 'w') as f:
        nbformat.write(nb, f)
    
    # Call the method to validate the notebook
    result = await notebook_tools.notebook_validate(
        notebook_path=sample_notebook_path
    )
    
    # Verify the result indicates the notebook is valid
    assert "valid" in result.lower()

# --- Tests for error handling in other methods ---

@pytest.mark.asyncio
async def test_notebook_get_info_with_metadata(notebook_tools, sample_notebook_path):
    """Test notebook_get_info with custom metadata."""
    # Add the temporary directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Create a notebook with custom metadata
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "custom_field": "custom_value"
    }
    
    # Write the notebook to the temporary file
    with open(sample_notebook_path, 'w') as f:
        nbformat.write(nb, f)
    
    # Call the method to get notebook info
    result = await notebook_tools.notebook_get_info(
        notebook_path=sample_notebook_path
    )
    
    # Verify the result contains the metadata
    assert "metadata" in result
    assert "kernelspec" in result["metadata"]
    assert "custom_field" in result["metadata"]
    assert result["metadata"]["custom_field"] == "custom_value"

@pytest.mark.asyncio
async def test_notebook_read_with_error_handling(notebook_tools):
    """Test notebook_read with error handling."""
    # Mock resolve_path_and_check_permissions to raise an error
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  side_effect=ValueError("Invalid path")):
        # Call notebook_read - should raise the same error
        with pytest.raises(ValueError, match="Invalid path"):
            await notebook_tools.notebook_read("/invalid/path.ipynb")

@pytest.mark.asyncio
async def test_notebook_read_metadata_with_error_handling(notebook_tools):
    """Test notebook_read_metadata with error handling."""
    # Mock resolve_path_and_check_permissions to raise an error
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', 
                  side_effect=PermissionError("Access denied")):
        # Call notebook_read_metadata - should raise the same error
        with pytest.raises(PermissionError, match="Access denied"):
            await notebook_tools.notebook_read_metadata("/invalid/path.ipynb")

@pytest.mark.asyncio
async def test_notebook_merge_cells_incompatible_types(notebook_tools, sample_notebook_path):
    """Test notebook_merge_cells with incompatible cell types."""
    # Add the temporary directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # The sample notebook has a code cell followed by a markdown cell
    # Trying to merge them should raise an error
    with pytest.raises(ValueError, match="Cannot merge cells of different types"):
        await notebook_tools.notebook_merge_cells(
            notebook_path=sample_notebook_path,
            first_cell_index=0
        )