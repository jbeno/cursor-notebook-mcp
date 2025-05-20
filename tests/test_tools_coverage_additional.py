"""
Additional tests to improve coverage for the NotebookTools class.
"""

import pytest
import tempfile
import os
import nbformat
import json
import re
import subprocess
import sys
import posixpath
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
async def test_notebook_edit_cell_metadata(notebook_tools, sample_notebook_path):
    """Test editing cell metadata."""
    # Add the temporary directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Call the method to edit cell metadata
    result = await notebook_tools.notebook_edit_cell_metadata(
        notebook_path=sample_notebook_path,
        cell_index=0,
        metadata_updates={"tags": ["test-tag"]}
    )
    
    # Verify the result
    assert "Successfully updated" in result
    
    # Verify the metadata was updated
    cell_metadata = await notebook_tools.notebook_read_cell_metadata(
        notebook_path=sample_notebook_path,
        cell_index=0
    )
    
    assert "tags" in cell_metadata
    assert "test-tag" in cell_metadata["tags"]

@pytest.mark.asyncio
async def test_notebook_edit_metadata(notebook_tools, sample_notebook_path):
    """Test editing notebook metadata."""
    # Add the temporary directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Call the method to edit notebook metadata
    result = await notebook_tools.notebook_edit_metadata(
        notebook_path=sample_notebook_path,
        metadata_updates={"kernelspec": {"name": "python3", "display_name": "Python 3"}}
    )
    
    # Verify the result
    assert "Successfully updated" in result
    
    # Verify the metadata was updated
    notebook_metadata = await notebook_tools.notebook_read_metadata(
        notebook_path=sample_notebook_path
    )
    
    assert "kernelspec" in notebook_metadata
    assert notebook_metadata["kernelspec"]["name"] == "python3"

# Removed test_notebook_validate as its functionality is more comprehensively covered by test_validate in tests/test_notebook_tools.py
# @pytest.mark.asyncio
# async def test_notebook_validate(notebook_tools, sample_notebook_path):
#     """Test validating a notebook."""
#     # Add the temporary directory to allowed roots
#     notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
#     
#     # Call the method to validate the notebook
#     result = await notebook_tools.notebook_validate(notebook_path=sample_notebook_path)
#     
#     # Verify the result
#     assert "valid" in result.lower()

@pytest.mark.asyncio
async def test_notebook_export(notebook_tools, sample_notebook_path):
    """Test exporting a notebook."""
    # Add the temporary directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Call the method to export the notebook
    result = await notebook_tools.notebook_export(
        notebook_path=sample_notebook_path,
        export_format="html"
    )
    
    # Verify the result
    assert "Successfully exported" in result
    assert "html" in result.lower()
    
    # Extract the output path from the result
    output_path = result.split("to ")[-1].strip()
    
    # Verify the file exists
    assert os.path.exists(output_path)
    
    # Clean up
    if os.path.exists(output_path):
        os.unlink(output_path)

@pytest.mark.asyncio
async def test_notebook_move_cell(notebook_tools, sample_notebook_path):
    """Test moving a cell."""
    # Add the temporary directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Call the method to move a cell
    result = await notebook_tools.notebook_move_cell(
        notebook_path=sample_notebook_path,
        cell_index=0,
        new_index=1
    )
    
    # Verify the result
    assert "Successfully moved" in result
    
    # Verify the cell was moved
    with open(sample_notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)
    
    assert "Markdown Cell" in nb.cells[0].source
    assert "Hello, world!" in nb.cells[1].source

# --- Tests focusing on lines 1337-1353 in notebook_export method ---

@pytest.mark.asyncio
async def test_notebook_export_remote_path(notebook_tools, sample_notebook_path):
    """Test notebook_export with a remote path, targeting lines 1337-1353."""
    remote_path = "/home/user/notebook.ipynb"
    # This is the path of the temporary .ipynb file created from downloaded remote content
    local_temp_notebook_input_path = "/tmp/temp_notebook_for_nbconvert.ipynb" 
    # This is where nbconvert will write its output, derived from the temp input dir and remote basename
    expected_local_exported_html_path = os.path.join(os.path.dirname(local_temp_notebook_input_path), os.path.basename(remote_path).replace(".ipynb", ".html"))

    mock_sftp_manager = mock.MagicMock()
    with open(sample_notebook_path, 'rb') as f_sample:
        sample_content_bytes = f_sample.read()
    mock_sftp_manager.read_file.return_value = sample_content_bytes
    mock_sftp_manager.write_file = mock.MagicMock(return_value=None)
    notebook_tools.sftp_manager = mock_sftp_manager
    notebook_tools.config.sftp_manager = mock_sftp_manager

    mock_temp_file_obj = mock.MagicMock()
    mock_temp_file_obj.name = local_temp_notebook_input_path
    mock_temp_file_context = mock.MagicMock()
    mock_temp_file_context.__enter__.return_value = mock_temp_file_obj
    mock_temp_file_context.__exit__.return_value = None

    mock_exported_file_content = b"<html><body>Mocked HTML</body></html>"
    # Mock open to correctly simulate reading the *expected* local exported html path
    mock_open_for_html = mock.mock_open(read_data=mock_exported_file_content)
    
    # When os.path.isfile is called for expected_local_exported_html_path, it should be True.
    # For other paths, it can be True as well if needed, or a more specific mock.
    def isfile_side_effect(path):
        if path == expected_local_exported_html_path:
            return True
        return True # Default to True for other relevant paths if any
    mock_isfile_conditional = mock.MagicMock(side_effect=isfile_side_effect)

    # Mock for os.path.exists, specifically for the cleanup part
    def pathexists_side_effect(path):
        if path == local_temp_notebook_input_path:
            return True # Ensure cleanup logic thinks the temp file exists
        return False # Default for other paths if necessary
    mock_pathexists_conditional = mock.MagicMock(side_effect=pathexists_side_effect)

    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', return_value=(True, remote_path)) as mock_resolve, \
         mock.patch('tempfile.NamedTemporaryFile', return_value=mock_temp_file_context) as mock_tempfile_creator, \
         mock.patch('subprocess.run', return_value=mock.MagicMock(returncode=0, stdout=f"Writing 123 bytes to {expected_local_exported_html_path}")) as mock_subprocess, \
         mock.patch('os.path.isfile', mock_isfile_conditional) as mock_isfile, \
         mock.patch('os.path.exists', mock_pathexists_conditional) as mock_exists, \
         mock.patch('os.makedirs') as mock_makedirs, \
         mock.patch('builtins.open', mock_open_for_html) as mock_builtin_open, \
         mock.patch('os.remove') as mock_os_remove:

        result = await notebook_tools.notebook_export(
            notebook_path=remote_path,
            export_format="html"
        )

    assert "Successfully exported notebook" in result
    assert remote_path in result
    assert expected_local_exported_html_path in result # This should now match the success message
    
    expected_remote_target_html_path = posixpath.join(posixpath.dirname(remote_path), os.path.basename(expected_local_exported_html_path))
    assert expected_remote_target_html_path in result

    mock_resolve.assert_called_once_with(remote_path, notebook_tools._get_allowed_local_roots(), mock_sftp_manager)
    mock_sftp_manager.read_file.assert_called_once_with(remote_path)
    mock_tempfile_creator.assert_called_once_with(delete=False, suffix=".ipynb")
    mock_temp_file_obj.write.assert_called_once_with(sample_content_bytes)
    
    expected_nbconvert_cmd_part_input = local_temp_notebook_input_path
    # The output base for nbconvert is derived from expected_local_exported_html_path
    expected_nbconvert_cmd_part_output_base = os.path.splitext(expected_local_exported_html_path)[0]
    assert mock_subprocess.call_args[0][0][5] == expected_nbconvert_cmd_part_input # Index 5 for input path
    assert mock_subprocess.call_args[0][0][7] == expected_nbconvert_cmd_part_output_base # Index 7 for output base

    mock_builtin_open.assert_called_with(expected_local_exported_html_path, 'rb')
    mock_sftp_manager.write_file.assert_called_once_with(expected_remote_target_html_path, mock_exported_file_content)
    mock_os_remove.assert_any_call(local_temp_notebook_input_path)

@pytest.mark.asyncio
async def test_notebook_export_remote_translation_error(notebook_tools, sample_notebook_path):
    """Test notebook_export when creating the temp file for downloaded remote content fails."""
    remote_path = "/home/user/notebook.ipynb"

    mock_sftp_manager = mock.MagicMock()
    with open(sample_notebook_path, 'rb') as f_sample:
        sample_content_bytes = f_sample.read()
    mock_sftp_manager.read_file.return_value = sample_content_bytes
    notebook_tools.sftp_manager = mock_sftp_manager
    notebook_tools.config.sftp_manager = mock_sftp_manager

    # Mock tempfile.NamedTemporaryFile to raise IOError during __enter__ or write
    # to simulate failure in creating/writing the temp file for downloaded content.
    mock_temp_file_creator = mock.MagicMock()
    # Ensure it's specifically IOError for the test's assertion on __cause__
    mock_temp_file_creator.side_effect = IOError("Simulated error creating/writing temporary file for download")

    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions', return_value=(True, remote_path)), \
         mock.patch('tempfile.NamedTemporaryFile', mock_temp_file_creator):
        
        with pytest.raises(IOError) as exc_info: # Expect IOError (or OSError) directly
            await notebook_tools.notebook_export(
                notebook_path=remote_path,
                export_format="html"
            )
        # Check the error message of the raised IOError
        assert "Simulated error creating/writing temporary file for download" in str(exc_info.value)
        # No need to check __cause__ as IOError is raised directly

@pytest.mark.asyncio
async def test_notebook_export_output_file_parsing(notebook_tools, sample_notebook_path):
    """Test notebook_export when output file is found via stdout parsing, targeting lines 1419-1431."""
    # Add the directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Mock subprocess.run to pretend nbconvert succeeded but with a different output file
    mock_process = mock.MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = f"Writing 1234 bytes to {sample_notebook_path}.html"
    
    # Mock os.path.isfile to first return False, then True for the parsed path
    def mock_isfile_side_effect(path):
        if ".html" in path:
            return True if "parsed_output" in path else False
        return True  # Original notebook exists
    
    mock_isfile = mock.MagicMock(side_effect=mock_isfile_side_effect)
    
    # Create a mock search match
    mock_match = mock.MagicMock()
    mock_match.group = lambda x: f"{os.path.dirname(sample_notebook_path)}/parsed_output.html"
    
    # Apply all mocks
    with mock.patch('subprocess.run', return_value=mock_process):
        with mock.patch('os.path.isfile', mock_isfile):
            with mock.patch('os.makedirs'):
                with mock.patch('re.search', return_value=mock_match):
                    # Call notebook_export
                    result = await notebook_tools.notebook_export(
                        notebook_path=sample_notebook_path,
                        export_format="html"
                    )
    
    # Verify the result contains the parsed path
    assert "Successfully exported" in result
    assert "parsed_output.html" in result

@pytest.mark.asyncio
async def test_notebook_export_output_file_not_found(notebook_tools, sample_notebook_path):
    """Test notebook_export when output file is not found, targeting lines 1419-1431."""
    # Add the directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Mock subprocess.run to pretend nbconvert succeeded
    mock_process = mock.MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "Some output without file path"
    
    # Mock os.path.isfile to return False (file not found)
    mock_isfile = mock.MagicMock(return_value=False)
    
    # Apply all mocks
    with mock.patch('subprocess.run', return_value=mock_process):
        with mock.patch('os.path.isfile', mock_isfile):
            with mock.patch('os.makedirs'):
                # Call notebook_export
                with pytest.raises(FileNotFoundError, match="Output file expected at"):
                    await notebook_tools.notebook_export(
                        notebook_path=sample_notebook_path,
                        export_format="html"
                    )

@pytest.mark.asyncio
async def test_notebook_export_subprocess_timeout(notebook_tools, sample_notebook_path):
    """Test notebook_export when subprocess times out, targeting lines 1469-1471."""
    # Add the directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Create a TimeoutExpired exception to be raised
    timeout_exc = subprocess.TimeoutExpired(cmd=["nbconvert", "--to", "html"], timeout=60)
    
    # Mock subprocess.run to raise TimeoutExpired
    with mock.patch('subprocess.run', side_effect=timeout_exc):
        with mock.patch('os.makedirs'):
            # Call notebook_export
            with pytest.raises(RuntimeError, match="nbconvert command timed out"):
                await notebook_tools.notebook_export(
                    notebook_path=sample_notebook_path,
                    export_format="html"
                )

@pytest.mark.asyncio
async def test_notebook_export_cleanup_error(notebook_tools, sample_notebook_path):
    """Test notebook_export when temp file cleanup fails, targeting lines 1476-1478."""
    # Setup path resolution mock to pass the permission check
    remote_path = "/home/user/notebook.ipynb"
    
    # Set up our mocked SFTP manager
    mock_sftp_manager = mock.MagicMock()
    mock_sftp_manager.read_file.return_value = open(sample_notebook_path, 'rb').read()
    mock_sftp_manager.translate_path.return_value = (
        remote_path,  # Remote path
        "/tmp/local/equivalent/notebook.ipynb",  # Local equivalent
        True  # Is remote
    )
    notebook_tools.sftp_manager = mock_sftp_manager
    
    # Setup mock for tempfile.NamedTemporaryFile
    mock_temp_file = mock.MagicMock()
    mock_temp_file.name = "/tmp/temp_notebook.ipynb"
    mock_temp_file.__enter__.return_value = mock_temp_file
    
    # Apply all mocks
    with mock.patch('cursor_notebook_mcp.notebook_ops.resolve_path_and_check_permissions',
                   return_value=(True, remote_path)):
        with mock.patch('tempfile.NamedTemporaryFile', return_value=mock_temp_file):
            with mock.patch('subprocess.run', side_effect=Exception("nbconvert error")):
                with mock.patch('os.path.exists', return_value=True):
                    with mock.patch('os.makedirs') as mock_makedirs:
                        with mock.patch('os.remove', side_effect=Exception("Cleanup error")):
                            # Call notebook_export
                            with pytest.raises(RuntimeError, match="An unexpected error occurred during notebook export"):
                                await notebook_tools.notebook_export(
                                    notebook_path=remote_path,
                                    export_format="html"
                                )

# --- Tests for other uncovered lines in tools.py ---

@pytest.mark.asyncio
async def test_notebook_read_size_error(notebook_tools, sample_notebook_path):
    """Test notebook_read when cell serialization fails, targeting lines 1574-1577."""
    # Add the directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Create a notebook with a problematic cell
    with open(sample_notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Add a cell with content that will cause json.dumps to fail
    with mock.patch('json.dumps', side_effect=TypeError("Cannot serialize cell")):
        # Call notebook_read
        result = await notebook_tools.notebook_read(notebook_path=sample_notebook_path)
        
        # Check that the error is handled correctly
        assert any("processing_error" in cell.get("metadata", {}) for cell in result["cells"])

@pytest.mark.asyncio
async def test_notebook_search_attribute_error(notebook_tools, sample_notebook_path):
    """Test notebook_search with AttributeError, targeting lines 1716-1718."""
    # Add the directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Create a simple notebook with normal cells for writing
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(source="print('Hello')"))
    
    with open(sample_notebook_path, 'w') as f:
        nbformat.write(nb, f)
    
    # Now create a replacement notebook for our mocked read
    # We'll mock nbformat.reads to return a notebook with a problematic cell
    mock_nb = mock.MagicMock()
    mock_nb.cells = []
    
    # Create a cell that will raise AttributeError when 'source' is accessed
    problematic_cell = mock.MagicMock()
    problematic_cell.cell_type = 'code'
    type(problematic_cell).source = mock.PropertyMock(side_effect=AttributeError("No source attribute"))
    mock_nb.cells.append(problematic_cell)
    
    # Mock nbformat.reads to return our problematic notebook
    with mock.patch('nbformat.reads', return_value=mock_nb):
        with mock.patch('nbformat.read', return_value=mock_nb):
            # Call notebook_search
            result = await notebook_tools.notebook_search(
                notebook_path=sample_notebook_path,
                query="Hello"
            )
    
    # The method should handle the error and continue
    assert isinstance(result, list)
    assert len(result) == 1
    assert "message" in result[0]
    assert "No matches found" in result[0]["message"]

@pytest.mark.asyncio
async def test_notebook_search_cell_error(notebook_tools, sample_notebook_path):
    """Test notebook_search with cell-specific error, targeting lines 1731-1733."""
    # Add the directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # We need to create a notebook with cells that will be processed differently
    nb = nbformat.v4.new_notebook()
    
    # First cell works normally
    nb.cells.append(nbformat.v4.new_code_cell(source="print('Hello')"))
    
    # Second cell will cause an exception during processing
    # We'll mock this behavior when reading the notebook
    nb.cells.append(nbformat.v4.new_code_cell(source="print('Another cell')"))
    
    # Write the notebook
    with open(sample_notebook_path, 'w') as f:
        nbformat.write(nb, f)
    
    # Mock the splitlines method for the second cell to raise an exception
    original_notebook = None
    with open(sample_notebook_path, 'r', encoding='utf-8') as f:
        original_notebook = nbformat.read(f, as_version=4)
    
    # Create a modified notebook where the second cell's source will raise an exception
    modified_notebook = nbformat.v4.new_notebook()
    modified_notebook.cells.append(original_notebook.cells[0])  # First cell is normal
    
    # Create a mock cell for the second one that raises an exception on splitlines()
    mock_cell = original_notebook.cells[1]
    modified_source = mock.MagicMock()
    modified_source.splitlines.side_effect = Exception("Error processing cell")
    mock_cell.source = modified_source
    modified_notebook.cells.append(mock_cell)
    
    # Mock nbformat.read to return our modified notebook
    with mock.patch('nbformat.read', return_value=modified_notebook):
        # Call notebook_search
        result = await notebook_tools.notebook_search(
            notebook_path=sample_notebook_path,
            query="Hello"
        )
    
    # Should find match in first cell even though second cell causes error
    assert isinstance(result, list)
    assert len(result) > 0
    assert result[0]["cell_index"] == 0

@pytest.mark.asyncio
async def test_extract_code_outline_error(notebook_tools):
    """Test _extract_code_outline with exceptions, targeting lines 1771-1773."""
    # Test with a syntax error
    invalid_code = "def function(:"  # Syntax error
    result = notebook_tools._extract_code_outline(invalid_code)
    assert "<Syntax Error>" in result
    
    # Test with a general exception during AST parsing
    with mock.patch('ast.parse', side_effect=Exception("AST error")):
        result = notebook_tools._extract_code_outline("def function(): pass")
        assert any("<AST Parsing Error" in item for item in result)

@pytest.mark.asyncio
async def test_extract_markdown_outline_error(notebook_tools):
    """Test _extract_markdown_outline with exceptions, targeting lines 1817-1819."""
    # We can't directly mock str.split, so use a different approach that doesn't serialize
    
    # Test with AttributeError
    with mock.patch('re.match', side_effect=AttributeError("No match method")):
        result = notebook_tools._extract_markdown_outline("# Heading")
        assert "<Missing Source>" in result[0]
    
    # Test with general exception in markdown processing
    with mock.patch('re.match', side_effect=Exception("Parsing error")):
        result = notebook_tools._extract_markdown_outline("# Heading")
        assert "Parsing error" in result[0]  # Look for the error message text

@pytest.mark.asyncio
async def test_get_first_line_context_error(notebook_tools):
    """Test _get_first_line_context with exception, targeting lines 1836-1837."""
    # Create a custom string-like object that will raise an exception on splitlines()
    class ErrorString:
        def __str__(self):
            return "Test string"
            
        def splitlines(self):
            raise Exception("Splitlines error")
    
    # Call the method with our custom object
    result = notebook_tools._get_first_line_context(ErrorString())
    assert "<Error getting context>" in result

@pytest.mark.asyncio
async def test_notebook_validate_validation_error(notebook_tools, sample_notebook_path):
    """Test notebook_validate with validation error, targeting lines 1894-1895."""
    # Add the directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Create an invalid notebook structure but valid JSON
    with open(sample_notebook_path, 'w') as f:
        json.dump({
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            # Missing "cells" key
        }, f)
    
    # Call notebook_validate
    result = await notebook_tools.notebook_validate(notebook_path=sample_notebook_path)
    
    # Check that validation failed but didn't raise an exception
    assert "validation failed" in result.lower()

@pytest.mark.asyncio
async def test_notebook_read_cell_output_too_large(notebook_tools, sample_notebook_path):
    """Test notebook_read_cell_output with too large output, targeting lines 1979-1982."""
    # Add the directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Create a notebook with a large output
    nb = nbformat.v4.new_notebook()
    code_cell = nbformat.v4.new_code_cell(source="print('Large output')")
    code_cell['outputs'] = [
        nbformat.v4.new_output(
            output_type='stream',
            name='stdout',
            text='A' * (notebook_tools.config.max_cell_output_size + 1)  # Exceeds limit
        )
    ]
    nb.cells.append(code_cell)
    
    with open(sample_notebook_path, 'w') as f:
        nbformat.write(nb, f)
    
    # Call notebook_read_cell_output
    result = await notebook_tools.notebook_read_cell_output(
        notebook_path=sample_notebook_path,
        cell_index=0
    )
    
    # Check for error output structure
    assert len(result) == 1
    assert result[0]["output_type"] == "error"
    assert "OutputSizeError" in result[0]["ename"]
    assert "truncated" in result[0]["evalue"]

@pytest.mark.asyncio
async def test_notebook_read_cell_output_serialization_error(notebook_tools, sample_notebook_path):
    """Test notebook_read_cell_output with serialization error, targeting lines 1990-1992."""
    # Add the directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Create a notebook with a code cell
    nb = nbformat.v4.new_notebook()
    code_cell = nbformat.v4.new_code_cell(source="print('Hello')")
    code_cell['outputs'] = [
        nbformat.v4.new_output(
            output_type='stream',
            name='stdout',
            text='Hello'
        )
    ]
    nb.cells.append(code_cell)
    
    with open(sample_notebook_path, 'w') as f:
        nbformat.write(nb, f)
    
    # Mock json.dumps to raise an error
    with mock.patch('json.dumps', side_effect=TypeError("Cannot serialize")):
        # Call notebook_read_cell_output
        with pytest.raises(ValueError, match="Could not determine size of cell output"):
            await notebook_tools.notebook_read_cell_output(
                notebook_path=sample_notebook_path,
                cell_index=0
            )

@pytest.mark.asyncio
async def test_notebook_split_cell_empty_source(notebook_tools, sample_notebook_path):
    """Test notebook_split_cell with empty source, targeting lines 2036-2038."""
    # Add the directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Create a notebook with an empty code cell
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(source=""))
    
    with open(sample_notebook_path, 'w') as f:
        nbformat.write(nb, f)
    
    # Call notebook_split_cell with line 1 (should work with empty source)
    result = await notebook_tools.notebook_split_cell(
        notebook_path=sample_notebook_path,
        cell_index=0,
        split_at_line=1
    )
    
    # Verify success
    assert "Successfully split cell" in result
    
    # Check that we now have 2 empty cells
    cell_count = await notebook_tools.notebook_get_cell_count(
        notebook_path=sample_notebook_path
    )
    assert cell_count == 2

@pytest.mark.asyncio
async def test_notebook_change_cell_type_already_same_type(notebook_tools, sample_notebook_path):
    """Test notebook_change_cell_type when cell is already the target type, targeting line 2248-2250."""
    # Add the directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Create a notebook with a code cell
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(source="print('Hello')"))
    
    with open(sample_notebook_path, 'w') as f:
        nbformat.write(nb, f)
    
    # Call notebook_change_cell_type with the same type
    result = await notebook_tools.notebook_change_cell_type(
        notebook_path=sample_notebook_path,
        cell_index=0,
        new_type="code"  # Already code
    )
    
    # Verify no change was needed
    assert "already of type 'code'" in result
    assert "No change needed" in result

@pytest.mark.asyncio
async def test_notebook_duplicate_cell_with_id(notebook_tools, sample_notebook_path):
    """Test notebook_duplicate_cell with cell IDs, targeting lines 2299-2301."""
    # Add the directory to allowed roots
    notebook_tools.config.allowed_roots.append(os.path.dirname(sample_notebook_path))
    
    # Create a notebook with a cell that has an ID
    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell(source="print('Hello')")
    cell.id = "original-id"
    nb.cells.append(cell)
    
    with open(sample_notebook_path, 'w') as f:
        nbformat.write(nb, f)
    
    # Mock the random ID generation to check it's called
    with mock.patch('random.choices', return_value=list("newcellid")):
        # Call notebook_duplicate_cell
        result = await notebook_tools.notebook_duplicate_cell(
            notebook_path=sample_notebook_path,
            cell_index=0
        )
    
    # Verify success
    assert "Successfully duplicated cell" in result
    
    # Check that we now have 2 cells
    cell_count = await notebook_tools.notebook_get_cell_count(
        notebook_path=sample_notebook_path
    )
    assert cell_count == 2
    
    # We can't easily verify the new ID without reading raw JSON,
    # but the coverage for that code path should be improved

if __name__ == "__main__":
    pytest.main()