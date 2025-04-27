"""
Tests for core notebook operations in notebook_ops.py.
"""

import pytest
import os
import nbformat
from unittest import mock
from pathlib import Path
import logging
from io import StringIO # Import StringIO from io

# Import functions to test
from cursor_notebook_mcp import notebook_ops
from cursor_notebook_mcp.server import setup_logging


@pytest.mark.asyncio
async def test_read_notebook_io_error(tmp_path):
    """Test read_notebook handles IOError from nbformat.read."""
    # Create a dummy file path within the temp directory
    dummy_path = tmp_path / "dummy_read.ipynb"
    dummy_path.touch() # Create the file so path checks pass
    allowed_roots = [str(tmp_path)]
    
    # Mock nbformat.read to raise IOError
    with mock.patch('nbformat.read', side_effect=IOError("Cannot read file")):
        with pytest.raises(IOError, match="Cannot read file"):
            # Call the async function from notebook_ops with await
            await notebook_ops.read_notebook(str(dummy_path), allowed_roots)

@pytest.mark.asyncio
async def test_read_notebook_validation_error(tmp_path):
    """Test read_notebook handles ValidationError from nbformat.read."""
    dummy_path = tmp_path / "dummy_validation.ipynb"
    dummy_path.touch()
    allowed_roots = [str(tmp_path)]
    validation_error_instance = nbformat.ValidationError("Invalid notebook format")
    
    with mock.patch('nbformat.read', side_effect=validation_error_instance):
        # Expect the function to catch ValidationError and re-raise it as IOError
        with pytest.raises(IOError, match=r"Failed to read notebook file.*?Invalid notebook format"):
            await notebook_ops.read_notebook(str(dummy_path), allowed_roots)

@pytest.mark.asyncio
async def test_write_notebook_io_error(tmp_path):
    """Test write_notebook handles IOError from nbformat.write."""
    dummy_path = tmp_path / "dummy_write.ipynb"
    # Do not create the file beforehand for write test
    allowed_roots = [str(tmp_path)]
    nb = nbformat.v4.new_notebook() # Create an empty notebook object
    
    # Mock nbformat.write to raise IOError
    with mock.patch('nbformat.write', side_effect=IOError("Cannot write file")):
        with pytest.raises(IOError, match="Cannot write file"):
            await notebook_ops.write_notebook(str(dummy_path), nb, allowed_roots)

@pytest.mark.asyncio
async def test_read_notebook_file_not_found(tmp_path):
    """Test read_notebook handles FileNotFoundError."""
    non_existent_path = tmp_path / "non_existent.ipynb"
    allowed_roots = [str(tmp_path)]
    
    # Ensure the file does not exist
    assert not non_existent_path.exists()
    
    with pytest.raises(FileNotFoundError):
        await notebook_ops.read_notebook(str(non_existent_path), allowed_roots)

@pytest.mark.asyncio
async def test_read_notebook_generic_exception(tmp_path):
    """Test read_notebook handles generic Exception from nbformat.read."""
    dummy_path = tmp_path / "dummy_generic_read.ipynb"
    dummy_path.touch()
    allowed_roots = [str(tmp_path)]
    generic_error = Exception("Some generic read error")

    with mock.patch('nbformat.read', side_effect=generic_error):
        # Expect the function to catch Exception and re-raise it as IOError
        with pytest.raises(IOError, match=r"Failed to read notebook file.*?Some generic read error"):
            await notebook_ops.read_notebook(str(dummy_path), allowed_roots)

@pytest.mark.asyncio
async def test_write_notebook_generic_exception(tmp_path):
    """Test write_notebook handles generic Exception from nbformat.write."""
    dummy_path = tmp_path / "dummy_generic_write.ipynb"
    allowed_roots = [str(tmp_path)]
    nb = nbformat.v4.new_notebook()
    generic_error = Exception("Some generic write error")

    with mock.patch('nbformat.write', side_effect=generic_error):
        with pytest.raises(IOError, match=r"Failed to write notebook file.*?Some generic write error"):
            await notebook_ops.write_notebook(str(dummy_path), nb, allowed_roots)


# --- setup_logging Tests (Synchronous) ---

@mock.patch('os.makedirs', side_effect=OSError("Permission denied to create dir"))
@mock.patch('logging.FileHandler') # Mock FileHandler to prevent actual file creation
@mock.patch('sys.stderr', new_callable=StringIO) # Use imported StringIO
def test_setup_logging_makedirs_error(mock_stderr, mock_filehandler, mock_makedirs, tmp_path):
    """Test setup_logging handles OSError when creating log directory."""
    log_dir = str(tmp_path / "unwritable_logs")
    setup_logging(log_dir, logging.INFO)

    mock_makedirs.assert_called_once_with(log_dir, exist_ok=True)
    # Check that the error was printed to stderr
    assert "Could not create log directory" in mock_stderr.getvalue()
    assert "Permission denied to create dir" in mock_stderr.getvalue()
    # Check that FileHandler was NOT called because log_dir creation failed
    mock_filehandler.assert_not_called()

@mock.patch('os.makedirs') # Mock makedirs to succeed
@mock.patch('logging.FileHandler', side_effect=IOError("Cannot open log file for writing"))
@mock.patch('sys.stderr', new_callable=StringIO) # Use imported StringIO
def test_setup_logging_filehandler_error(mock_stderr, mock_filehandler, mock_makedirs, tmp_path):
    """Test setup_logging handles error when creating FileHandler."""
    log_dir = str(tmp_path / "logs")
    log_file_path = os.path.join(log_dir, "server.log")

    setup_logging(log_dir, logging.INFO)

    mock_makedirs.assert_called_once_with(log_dir, exist_ok=True)
    mock_filehandler.assert_called_once_with(log_file_path, encoding='utf-8')
    # Check that the warning was printed to stderr
    assert "Could not set up file logging" in mock_stderr.getvalue()
    assert "Cannot open log file for writing" in mock_stderr.getvalue()

# --- Additional Tests for notebook_ops.py Coverage ---

def test_is_path_allowed_empty_roots():
    """Test is_path_allowed behavior with empty allowed_roots list."""
    test_path = "/some/path"
    allowed_roots = []
    
    # Function should return False if no roots are configured
    assert not notebook_ops.is_path_allowed(test_path, allowed_roots)

def test_is_path_allowed_path_resolve_error():
    """Test is_path_allowed behavior when path resolution fails."""
    with mock.patch('os.path.realpath', side_effect=Exception("Path resolution error")):
        test_path = "/some/path"
        allowed_roots = ["/allowed/root"]
        
        # Function should return False if path resolution fails
        assert not notebook_ops.is_path_allowed(test_path, allowed_roots)

def test_is_path_allowed_root_resolve_error():
    """Test is_path_allowed behavior when allowed root resolution fails."""
    # First call returns successfully, second call raises exception
    def mock_realpath(path):
        if path == "/some/path":
            return "/some/path"
        raise Exception("Root resolution error")
    
    with mock.patch('os.path.realpath', side_effect=mock_realpath):
        test_path = "/some/path"
        allowed_roots = ["/allowed/root"]
        
        # Function should continue to the next root and return False
        assert not notebook_ops.is_path_allowed(test_path, allowed_roots)

@pytest.mark.asyncio
async def test_read_notebook_non_absolute_path():
    """Test read_notebook rejects non-absolute paths."""
    non_abs_path = "relative/path/notebook.ipynb"
    allowed_roots = ["/valid/root"]
    
    with pytest.raises(ValueError, match="Invalid notebook path: Only absolute paths are allowed"):
        await notebook_ops.read_notebook(non_abs_path, allowed_roots)

@pytest.mark.asyncio
async def test_read_notebook_outside_allowed_roots(tmp_path):
    """Test read_notebook rejects paths outside allowed roots."""
    dummy_path = "/some/path/outside/notebook.ipynb"
    allowed_roots = [str(tmp_path)]
    
    with pytest.raises(PermissionError, match="Access denied: Path .* is outside the allowed workspace roots"):
        await notebook_ops.read_notebook(dummy_path, allowed_roots)

@pytest.mark.asyncio
async def test_read_notebook_invalid_extension(tmp_path):
    """Test read_notebook rejects non-notebook files."""
    dummy_path = tmp_path / "not_a_notebook.txt"
    dummy_path.touch()
    allowed_roots = [str(tmp_path)]
    
    with pytest.raises(ValueError, match="Invalid file type: .* must point to a .ipynb file"):
        await notebook_ops.read_notebook(str(dummy_path), allowed_roots)

@pytest.mark.asyncio
async def test_write_notebook_non_absolute_path():
    """Test write_notebook rejects non-absolute paths."""
    non_abs_path = "relative/path/notebook.ipynb"
    allowed_roots = ["/valid/root"]
    nb = nbformat.v4.new_notebook()
    
    with pytest.raises(ValueError, match="Invalid notebook path: Only absolute paths are allowed for writing"):
        await notebook_ops.write_notebook(non_abs_path, nb, allowed_roots)

@pytest.mark.asyncio
async def test_write_notebook_outside_allowed_roots(tmp_path):
    """Test write_notebook rejects paths outside allowed roots."""
    dummy_path = "/some/path/outside/notebook.ipynb"
    allowed_roots = [str(tmp_path)]
    nb = nbformat.v4.new_notebook()
    
    with pytest.raises(PermissionError, match="Access denied: Path .* is outside the allowed workspace roots"):
        await notebook_ops.write_notebook(dummy_path, nb, allowed_roots)

@pytest.mark.asyncio
async def test_write_notebook_invalid_extension(tmp_path):
    """Test write_notebook rejects non-notebook files."""
    dummy_path = str(tmp_path / "not_a_notebook.txt")
    allowed_roots = [str(tmp_path)]
    nb = nbformat.v4.new_notebook()
    
    with pytest.raises(ValueError, match="Invalid file type for writing: .* must point to a .ipynb file"):
        await notebook_ops.write_notebook(dummy_path, nb, allowed_roots)

@pytest.mark.asyncio
async def test_write_notebook_create_parent_dir(tmp_path):
    """Test write_notebook creates parent directory."""
    parent_dir = tmp_path / "nested" / "dir"
    dummy_path = parent_dir / "notebook.ipynb"
    allowed_roots = [str(tmp_path)]
    nb = nbformat.v4.new_notebook()
    
    # Ensure the parent directory doesn't exist yet
    assert not parent_dir.exists()
    
    # Mock the directory creation and nbformat.write to verify they're called
    with mock.patch('os.path.isdir', return_value=False), \
         mock.patch('os.makedirs') as mock_makedirs, \
         mock.patch('nbformat.write') as mock_write:
        
        await notebook_ops.write_notebook(str(dummy_path), nb, allowed_roots)
        
        mock_makedirs.assert_called_once_with(str(parent_dir), exist_ok=True)
        mock_write.assert_called_once()

@pytest.mark.asyncio
async def test_write_notebook_parent_dir_creation_fails(tmp_path):
    """Test write_notebook handles error during parent directory creation."""
    parent_dir = tmp_path / "nested" / "dir"
    dummy_path = parent_dir / "notebook.ipynb"
    allowed_roots = [str(tmp_path)]
    nb = nbformat.v4.new_notebook()
    
    # Mock directory checks and creation to simulate failure
    with mock.patch('os.path.isdir', return_value=False), \
         mock.patch('os.makedirs', side_effect=OSError("Failed to create directory")):
        
        with pytest.raises(IOError, match="Could not create directory for notebook .* Failed to create directory"):
            await notebook_ops.write_notebook(str(dummy_path), nb, allowed_roots)