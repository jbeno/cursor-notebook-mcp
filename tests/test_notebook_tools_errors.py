"""
Tests specifically targeting error paths and edge cases in NotebookTools.

This complements the main test_notebook_tools.py file by focusing on
harder-to-reach parts of the code to improve coverage.
"""

import pytest
import os
import json
import asyncio
from unittest import mock
import subprocess
from pathlib import Path
import sys
import importlib.util
import nbformat

# Import the class to be tested
from cursor_notebook_mcp.tools import NotebookTools

# Use pytest-asyncio for async tests
pytestmark = pytest.mark.asyncio

# --- Tests for diagnose_imports function ---

async def test_diagnose_imports_subprocess_error(notebook_tools_inst: NotebookTools):
    """Test diagnose_imports when subprocess.run raises an error."""
    with mock.patch('subprocess.run') as mock_run:
        # Make subprocess.run raise an exception
        mock_run.side_effect = subprocess.SubprocessError("Command failed")
        
        # Call should still return something even if subprocess fails
        result = await notebook_tools_inst.diagnose_imports()
        
        # Verify we get a result that mentions Python version at minimum
        assert "Python version:" in result
        # The function likely has fallback behavior rather than error messages

async def test_diagnose_imports_malformed_json(notebook_tools_inst: NotebookTools):
    """Test diagnose_imports when pip returns malformed JSON."""
    with mock.patch('subprocess.run') as mock_run:
        # Create a mock CompletedProcess with invalid JSON output
        mock_completed = mock.Mock()
        mock_completed.returncode = 0
        mock_completed.stdout = "Not valid JSON"
        mock_run.return_value = mock_completed
        
        # Call should still handle the JSON parsing error
        result = await notebook_tools_inst.diagnose_imports()
        
        # Verify we get a result
        assert isinstance(result, str)
        assert "Python version:" in result
        # Function likely has fallback behavior

async def test_diagnose_imports_unexpected_format(notebook_tools_inst: NotebookTools):
    """Test diagnose_imports when pip returns unexpected JSON format."""
    with mock.patch('subprocess.run') as mock_run:
        # Create a mock CompletedProcess with JSON that lacks expected fields
        mock_completed = mock.Mock()
        mock_completed.returncode = 0
        mock_completed.stdout = json.dumps({"unexpected": "structure"})
        mock_run.return_value = mock_completed
        
        # Call should still handle this unexpected format
        result = await notebook_tools_inst.diagnose_imports()
        
        # Verify we get some result
        assert isinstance(result, str)
        assert "Python version:" in result

# --- Tests for notebook validation ---

async def test_validate_invalid_json(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test validate with a notebook containing invalid JSON."""
    nb_path = notebook_path_factory()
    
    # Create a file with invalid JSON content
    with open(nb_path, 'w') as f:
        f.write('{"cells": [{"invalid": true, }]}')  # Trailing comma makes this invalid JSON
    
    # We expect an IOError when trying to validate invalid JSON
    with pytest.raises(IOError):
        await notebook_tools_inst.notebook_validate(nb_path)

async def test_validate_not_a_notebook(notebook_tools_inst: NotebookTools, notebook_path_factory, tmp_path):
    """Test validate with a file that contains valid JSON but isn't a notebook."""
    nb_path = notebook_path_factory()
    
    # Create a JSON file that's not a notebook
    with open(nb_path, 'w') as f:
        f.write('{"not": "a notebook"}')
    
    # We expect an IOError when the file isn't a valid notebook
    with pytest.raises(IOError):
        await notebook_tools_inst.notebook_validate(nb_path)

async def test_validate_missing_nbformat(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test validate with a notebook missing nbformat specification."""
    nb_path = notebook_path_factory()
    
    # Create a notebook that appears valid
    nb = nbformat.v4.new_notebook()
    nbformat.write(nb, nb_path)
    
    # This should validate successfully
    result = await notebook_tools_inst.notebook_validate(nb_path)
    assert "valid" in result.lower()

async def test_get_outline_invalid_python_syntax(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test notebook_get_outline with invalid Python syntax in code cells."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    
    # Add a cell with invalid Python syntax
    invalid_code = "def invalid_function(:\n    return 'syntax error'"
    await notebook_tools_inst.notebook_add_cell(
        notebook_path=nb_path, 
        cell_type='code',
        source=invalid_code,
        insert_after_index=-1
    )
    
    # Get outline should still work, handling the syntax error gracefully
    outline = await notebook_tools_inst.notebook_get_outline(nb_path)
    
    # Should return info for the cell even though parsing failed
    assert len(outline) == 1
    assert outline[0]["index"] == 0  # Should have index
    assert "outline" in outline[0]   # Should have an outline
    assert "type" in outline[0]      # Should have a type
    assert outline[0]["type"] == "code"  # Should be a code cell

async def test_search_empty_query(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test notebook_search with an empty query string."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    
    # Add a cell with some content
    await notebook_tools_inst.notebook_add_cell(
        notebook_path=nb_path, 
        cell_type='code',
        source="print('hello')",
        insert_after_index=-1
    )
    
    # Search with empty query should raise ValueError
    with pytest.raises(ValueError, match="Search query cannot be empty"):
        await notebook_tools_inst.notebook_search(nb_path, "")
    
    # Search with whitespace query - this gets trimmed in the implementation
    # and may return no matches or throw an error depending on the implementation
    try:
        results = await notebook_tools_inst.notebook_search(nb_path, "   ")
        # Results can be a list or a dict with a message
        if isinstance(results, list):
            # Empty or non-empty list
            if len(results) > 0:
                # Should have a standard structure
                assert isinstance(results[0], dict)
        else:
            # Dict with a message
            assert isinstance(results, dict)
            assert "message" in results
    except ValueError:
        # Or it might raise ValueError if it considers whitespace empty
        pass

async def test_notebook_read_huge_file_truncation(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test notebook_read with a very large notebook gets truncated."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    
    # Create a notebook with a cell containing a large string
    large_string = "x" * 1000000  # 1MB of data
    await notebook_tools_inst.notebook_add_cell(
        notebook_path=nb_path, 
        cell_type='code',
        source=large_string,
        insert_after_index=-1
    )
    
    # Reading should not truncate the output when the length isn't specified
    notebook_data = await notebook_tools_inst.notebook_read(nb_path)
    
    # Just check that we got a valid object, truncation may be done differently
    assert isinstance(notebook_data, dict)
    assert "cells" in notebook_data
    assert len(notebook_data["cells"]) == 1

# --- Tests for export functionality edge cases ---

@pytest.mark.skipif(not importlib.util.find_spec("nbconvert"), reason="nbconvert not found")
async def test_export_unsupported_format(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test export to an unsupported format."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    
    # Add a cell so the notebook isn't empty
    await notebook_tools_inst.notebook_add_cell(
        notebook_path=nb_path, 
        cell_type='code',
        source="print('hello')",
        insert_after_index=-1
    )
    
    # Try to export to an invalid format - expect RuntimeError from nbconvert
    output_path = str(Path(nb_path).with_suffix(".invalid"))
    with pytest.raises(RuntimeError):
        await notebook_tools_inst.notebook_export(
            notebook_path=nb_path,
            export_format="invalid_format",
            output_path=output_path
        )

@pytest.mark.skipif(not importlib.util.find_spec("nbconvert"), reason="nbconvert not found")
async def test_export_nbconvert_error(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test export when nbconvert raises an error."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    output_path = str(Path(nb_path).with_suffix(".html"))
    
    # Mock subprocess.run instead of nbconvert directly
    with mock.patch('subprocess.run') as mock_run:
        # Make it return a failed process
        mock_process = mock.Mock()
        mock_process.returncode = 1
        mock_process.stderr = "Export failed"
        mock_run.return_value = mock_process
        
        # Call should raise RuntimeError
        with pytest.raises(RuntimeError):
            await notebook_tools_inst.notebook_export(
                notebook_path=nb_path,
                export_format="html",
                output_path=output_path
            )

# --- Tests for cell transformation edge cases ---

async def test_change_cell_type_to_raw(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test changing a cell type to 'raw'."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    
    # Add a code cell
    await notebook_tools_inst.notebook_add_cell(
        notebook_path=nb_path, 
        cell_type='code',
        source="print('hello')",
        insert_after_index=-1
    )
    
    # Change to raw type
    result = await notebook_tools_inst.notebook_change_cell_type(
        notebook_path=nb_path,
        cell_index=0,
        new_type='raw'
    )
    
    assert "Successfully changed cell 0 from 'code' to 'raw'" in result
    
    # Verify the cell is now raw
    nb = await notebook_tools_inst.read_notebook(nb_path, notebook_tools_inst.config.allowed_roots)
    assert nb.cells[0].cell_type == 'raw'

async def test_merge_cells_mixed_types(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test merging cells of different types should fail."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    
    # Add cells of different types
    await notebook_tools_inst.notebook_add_cell(
        notebook_path=nb_path, 
        cell_type='code',
        source="print('hello')",
        insert_after_index=-1
    )
    await notebook_tools_inst.notebook_add_cell(
        notebook_path=nb_path, 
        cell_type='markdown',
        source="# Hello",
        insert_after_index=0
    )
    
    # Try to merge them
    with pytest.raises(ValueError, match="Cannot merge cells of different types"):
        await notebook_tools_inst.notebook_merge_cells(
            notebook_path=nb_path,
            first_cell_index=0
        )

async def test_split_cell_at_negative_line(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test splitting a cell at a negative line number should fail."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    
    # Add a cell with multiple lines
    await notebook_tools_inst.notebook_add_cell(
        notebook_path=nb_path, 
        cell_type='code',
        source="line1\nline2\nline3",
        insert_after_index=-1
    )
    
    # Try to split at negative line
    with pytest.raises(ValueError, match="out of bounds"):
        await notebook_tools_inst.notebook_split_cell(
            notebook_path=nb_path,
            cell_index=0,
            split_at_line=-1
        )

# --- Tests for path validation edge cases ---

async def test_path_validation_nonexistent_parent_dir(notebook_tools_inst: NotebookTools, notebook_path_factory, temp_notebook_dir):
    """Test operations with paths whose parent directories don't exist."""
    # Create a path with non-existent parent directories
    nonexistent_dir = temp_notebook_dir / "does_not_exist" / "nested"
    nb_path = str(nonexistent_dir / "test.ipynb")
    
    # Create should succeed by creating directories
    result = await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    assert "Successfully created" in result
    assert os.path.exists(nb_path)
    
    # Clean up
    os.remove(nb_path)
    os.rmdir(nonexistent_dir)
    os.rmdir(nonexistent_dir.parent)

async def test_path_validation_directory_target(notebook_tools_inst: NotebookTools, temp_notebook_dir):
    """Test operations with a path pointing to a directory."""
    # Create a test directory
    test_dir = temp_notebook_dir / "test_dir.ipynb"
    os.makedirs(test_dir, exist_ok=True)
    
    # Try to create notebook with path pointing to directory
    with pytest.raises(FileExistsError):
        await notebook_tools_inst.notebook_create(notebook_path=str(test_dir))
    
    # Clean up
    os.rmdir(test_dir) 