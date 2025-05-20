"""
Tests to improve coverage for the NotebookTools class.
"""

import pytest
import tempfile
import os
import nbformat
import json
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

@pytest.mark.asyncio
async def test_log_prefix(notebook_tools):
    """Test the _log_prefix method."""
    # Call the method with various parameters
    prefix1 = notebook_tools._log_prefix("test_tool")
    prefix2 = notebook_tools._log_prefix("test_tool", path="test.ipynb")
    prefix3 = notebook_tools._log_prefix("test_tool", path="test.ipynb", index=1)
    
    # Verify the results
    assert "test_tool" in prefix1
    assert "test.ipynb" in prefix2
    assert "index" in prefix3
    assert "1" in prefix3

@pytest.mark.asyncio
async def test_get_allowed_local_roots(notebook_tools):
    """Test the _get_allowed_local_roots method."""
    # Call the method
    roots = notebook_tools._get_allowed_local_roots()
    
    # Verify the result
    assert isinstance(roots, list)
    assert str(Path.cwd()) in roots

# Removed test_notebook_split_cell as its functionality is more comprehensively covered by tests in tests/test_notebook_tools.py
# @pytest.mark.asyncio
# async def test_notebook_split_cell(notebook_tools, sample_notebook_path):
#     """Test splitting a cell."""
#     # Add the temporary directory to allowed roots
#     notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
#     # First edit a cell to have multiple lines
#     await notebook_tools.notebook_edit_cell(
#         notebook_path=sample_notebook_path,
#         cell_index=0,
#         source="print('First line')\nprint('Second line')"
#     )
    
#     # Call the method to split the cell
#     result = await notebook_tools.notebook_split_cell(
#         notebook_path=sample_notebook_path,
#         cell_index=0,
#         split_at_line=1  # Split after the first line
#     )
    
#     # Verify the result
#     assert "Successfully split" in result
    
#     # Verify the cell was split
#     cell_count = await notebook_tools.notebook_get_cell_count(notebook_path=sample_notebook_path)
#     assert cell_count == 3
    
#     # Check the content of the split cells
#     with open(sample_notebook_path, 'r') as f:
#         nb = nbformat.read(f, as_version=4)
    
#     # Check that the cells were split correctly
#     # The split might not be exactly as expected, so let's just check that we have the right number of cells
#     assert len(nb.cells) == 3
    
#     # Check that at least one cell contains each line
#     all_sources = [str(cell.source) for cell in nb.cells]
#     all_text = ''.join(all_sources)
#     assert "First line" in all_text
#     assert "Second line" in all_text

# Removed test_notebook_search as its functionality is far more comprehensively covered by the version in tests/test_notebook_tools.py
# @pytest.mark.asyncio
# async def test_notebook_search(notebook_tools, sample_notebook_path):
#     """Test searching in a notebook."""
#     # Add the temporary directory to allowed roots
#     notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
#     # Call the method to search
#     result = await notebook_tools.notebook_search(
#         notebook_path=sample_notebook_path,
#         query="Hello"
#     )
    
#     # Verify the result
#     assert isinstance(result, list)
#     assert len(result) > 0
#     assert "Hello" in str(result)
# No function follows this one, end of file.