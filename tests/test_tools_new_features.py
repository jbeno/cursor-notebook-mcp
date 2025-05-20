"""
Tests for newer features and specific uncovered areas of NotebookTools.
"""

import pytest
import os
import nbformat
from pathlib import Path
import json
import asyncio
from unittest import mock
import sys # For os.path.sep checks if needed by test logic

# Fixtures like notebook_tools_inst, temp_notebook_dir, notebook_path_factory, tmp_path_factory
# are expected to be available from conftest.py
from cursor_notebook_mcp.tools import NotebookTools # Ensure NotebookTools is imported for type hinting if used

pytestmark = pytest.mark.asyncio

# --- Tests for notebook_get_server_path_context ---

async def test_get_server_path_context_no_project_dir(notebook_tools_inst: NotebookTools):
    """Test get_server_path_context when no project_directory is provided."""
    context = await notebook_tools_inst.notebook_get_server_path_context()
    assert context["project_directory_status"] == "not_provided"
    assert context["effective_notebook_base_path_for_project"] is None
    assert "Project directory not provided" in context["path_construction_guidance"]
    assert isinstance(context["allowed_roots"], list)
    # The notebook_tools_inst fixture uses server_config, which uses temp_notebook_dir as an allowed root
    # Let's ensure it's correctly reported.
    # server_config fixture in conftest.py defines allowed_roots = [str(temp_notebook_dir)]
    # So, notebook_tools_inst.config.allowed_roots[0] should be that path.
    expected_root = notebook_tools_inst.config.allowed_roots[0]
    assert any(os.path.normpath(expected_root) == os.path.normpath(r) for r in context["allowed_roots"])
    assert context["sftp_enabled"] is False # By default, sftp_manager is None in the fixture
    assert context["original_sftp_specs"] == []

async def test_get_server_path_context_project_is_root(notebook_tools_inst: NotebookTools, temp_notebook_dir: Path):
    """Test get_server_path_context when project_directory is an allowed root."""
    project_dir = str(temp_notebook_dir)
    # Ensure the config for this instance uses this path as an allowed root
    original_roots = notebook_tools_inst.config.allowed_roots
    notebook_tools_inst.config.allowed_roots = [project_dir]

    context = await notebook_tools_inst.notebook_get_server_path_context(project_directory=project_dir)
    assert context["project_directory_status"] == "is_an_allowed_root"
    assert context["effective_notebook_base_path_for_project"] == "" # Empty string for root
    assert "is directly one of the server's allowed roots" in context["path_construction_guidance"]
    assert context["provided_project_directory"] == project_dir

    # Restore original config if it matters for other tests using the same instance, though fixtures are function-scoped
    notebook_tools_inst.config.allowed_roots = original_roots

async def test_get_server_path_context_project_within_root(notebook_tools_inst: NotebookTools, temp_notebook_dir: Path):
    """Test get_server_path_context when project_directory is within an allowed root."""
    allowed_root = str(temp_notebook_dir)
    notebook_tools_inst.config.allowed_roots = [allowed_root] # Ensure this is the only root for clarity

    sub_project_dir_name = "my_sub_project"
    project_dir_path = temp_notebook_dir / sub_project_dir_name
    project_dir_path.mkdir(exist_ok=True) # Create the subdirectory
    project_dir_str = str(project_dir_path)
    
    context = await notebook_tools_inst.notebook_get_server_path_context(project_directory=project_dir_str)
    
    assert context["project_directory_status"] == "is_within_allowed_root"
    
    expected_base = sub_project_dir_name
    # On Windows, os.path.relpath will use backslashes.
    # The tool's effective_notebook_base_path_for_project might be normalized to posix if server_path_style is Posix.
    # Let's check server_path_style and adjust expectation.
    
    current_os_sep_base = expected_base # os.path.relpath default
    if context["server_path_style"] == 'Posix' and os.path.sep != '/':
        # If server is Posix and we are on Windows, relpath gives Windows, but tool might convert
        assert context["effective_notebook_base_path_for_project"] == current_os_sep_base.replace(os.path.sep, '/')
    else: # Either server is Windows, or both are Posix, or server is Windows and we are Posix (os.sep already /)
        assert context["effective_notebook_base_path_for_project"] == current_os_sep_base

    assert f"within server allowed root: '{allowed_root}'" in context["path_construction_guidance"]
    
    # Check the example path in guidance is correct for the OS
    # Construct the expected example path using os.path.join for OS-agnostic check
    expected_example_path = os.path.join(expected_base, "my_notebook.ipynb")
    assert f"'{expected_example_path}'" in context["path_construction_guidance"]


async def test_get_server_path_context_project_outside_root(notebook_tools_inst: NotebookTools, tmp_path_factory: pytest.TempPathFactory):
    """Test get_server_path_context when project_directory is outside allowed roots."""
    # notebook_tools_inst.config.allowed_roots is set by its fixture (e.g., to temp_notebook_dir)
    # Create another temp dir which will be outside the one in config.allowed_roots
    outside_project_dir_path = tmp_path_factory.mktemp("outside_project_for_path_context")
    
    context = await notebook_tools_inst.notebook_get_server_path_context(project_directory=str(outside_project_dir_path))
    
    assert context["project_directory_status"] == "outside_allowed_roots"
    assert context["effective_notebook_base_path_for_project"] is None
    assert "is NOT an allowed root nor within any allowed root" in context["path_construction_guidance"]

async def test_get_server_path_context_sftp_enabled(notebook_tools_inst: NotebookTools, temp_notebook_dir: Path):
    """Test get_server_path_context when SFTP is enabled."""
    mock_sftp_manager = mock.MagicMock()
    # raw_sftp_specs comes from ServerConfig, which is part of notebook_tools_inst.config
    # ServerConfig initializes raw_sftp_specs = []
    # The tool accesses self.config.raw_sftp_specs
    
    original_sftp_manager = notebook_tools_inst.config.sftp_manager
    original_raw_specs = notebook_tools_inst.config.raw_sftp_specs
    
    notebook_tools_inst.config.sftp_manager = mock_sftp_manager
    notebook_tools_inst.sftp_manager = mock_sftp_manager # Also update on tools instance directly
    notebook_tools_inst.config.raw_sftp_specs = ["user@host:/remote_projects/projectA"]
    # The tool uses getattr(self.sftp_manager, 'root_path', None)
    # Set it on the mock_sftp_manager directly
    mock_sftp_manager.root_path = "/sftp_configured_root"


    local_project_dir = str(temp_notebook_dir / "my_local_project_with_sftp")
    os.makedirs(local_project_dir, exist_ok=True)

    context = await notebook_tools_inst.notebook_get_server_path_context(project_directory=local_project_dir)
    
    assert context["sftp_enabled"] is True
    assert context["sftp_root"] == "/sftp_configured_root"
    assert context["original_sftp_specs"] == ["user@host:/remote_projects/projectA"]
    
    # When project_directory is resolved locally within allowed_roots (as in this test setup),
    # the primary guidance will be about local path construction.
    # The explicit "SFTP is active" string might not be the leading part of the guidance or present in the same way
    # as when project_directory is outside local roots.
    # The key indicators context["sftp_enabled"] and context["original_sftp_specs"] are more reliable here.
    # We already assert that the detailed SFTP pathing advice for outside_allowed_roots is NOT present.
    # If SFTP being generally mentioned is important, we could make this check more flexible, e.g., case-insensitive or partial.
    # For now, we focus on the structured data.

    # This assertion should hold true because local_project_dir is within temp_notebook_dir (an allowed root)
    assert context["project_directory_status"] == "is_within_allowed_root" 
    assert "If your project directory corresponds to one of these SFTP locations" not in context["path_construction_guidance"]

    # Restore original config
    notebook_tools_inst.config.sftp_manager = original_sftp_manager
    notebook_tools_inst.sftp_manager = original_sftp_manager
    notebook_tools_inst.config.raw_sftp_specs = original_raw_specs

async def test_get_server_path_context_project_resolution_error(notebook_tools_inst: NotebookTools):
    """Test get_server_path_context with a project_directory that causes a resolution error (e.g., null byte)."""
    invalid_project_dir = "/some/path\0with/nullbyte"
    
    # Mock os.path.abspath to raise ValueError for the specific invalid path
    original_abspath = os.path.abspath
    def mock_abspath(path):
        if path == os.path.normpath(invalid_project_dir): # normpath might process it first
            raise ValueError("Simulated error due to null byte in path")
        return original_abspath(path)

    with mock.patch('os.path.abspath', side_effect=mock_abspath):
        context = await notebook_tools_inst.notebook_get_server_path_context(project_directory=invalid_project_dir)
    
    assert context["project_directory_status"] == "resolution_error"
    assert "Error processing project_directory" in context["path_construction_guidance"]
    assert invalid_project_dir in context["path_construction_guidance"]
    assert "Simulated error due to null byte in path" in context["path_construction_guidance"]

# --- Tests for notebook_bulk_add_cells ---

async def test_bulk_add_cells_simple(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test adding multiple cells in bulk."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)

    cells_to_add = [
        {"cell_type": "markdown", "source": "# Title"},
        {"cell_type": "code", "source": "print('hello')"},
        {"cell_type": "markdown", "source": "Some text."}
    ]

    result = await notebook_tools_inst.notebook_bulk_add_cells(
        notebook_path=nb_path, 
        cells_to_add=cells_to_add, 
        insert_after_index=-1 # Add at the beginning
    )
    assert "Successfully added 3 cells" in result

    nb_content = await notebook_tools_inst.notebook_read(notebook_path=nb_path)
    assert len(nb_content['cells']) == 3
    assert nb_content['cells'][0]['source'] == "# Title"
    assert nb_content['cells'][0]['cell_type'] == "markdown"
    assert nb_content['cells'][1]['source'] == "print('hello')"
    assert nb_content['cells'][1]['cell_type'] == "code"
    assert nb_content['cells'][2]['source'] == "Some text."
    assert nb_content['cells'][2]['cell_type'] == "markdown"

async def test_bulk_add_cells_empty_list(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test bulk add with an empty list of cells."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    await notebook_tools_inst.notebook_add_cell(nb_path, "code", "print(1)", -1) # Add one initial cell

    result = await notebook_tools_inst.notebook_bulk_add_cells(
        notebook_path=nb_path, 
        cells_to_add=[], 
        insert_after_index=0
    )
    assert "No cells provided to add" in result
    count = await notebook_tools_inst.notebook_get_cell_count(notebook_path=nb_path)
    assert count == 1 # No change in cell count

async def test_bulk_add_cells_in_middle(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test bulk adding cells in the middle of existing cells."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    await notebook_tools_inst.notebook_add_cell(nb_path, "markdown", "Cell 0", -1)
    await notebook_tools_inst.notebook_add_cell(nb_path, "markdown", "Cell 1 (will be Cell 3)", 0)

    cells_to_insert = [
        {"cell_type": "code", "source": "Inserted Code 1"},
        {"cell_type": "code", "source": "Inserted Code 2"}
    ]
    await notebook_tools_inst.notebook_bulk_add_cells(
        notebook_path=nb_path, 
        cells_to_add=cells_to_insert, 
        insert_after_index=0 # Insert after "Cell 0"
    )

    nb_content = await notebook_tools_inst.notebook_read(notebook_path=nb_path)
    assert len(nb_content['cells']) == 4
    assert nb_content['cells'][0]['source'] == "Cell 0"
    assert nb_content['cells'][1]['source'] == "Inserted Code 1"
    assert nb_content['cells'][1]['cell_type'] == "code"
    assert nb_content['cells'][2]['source'] == "Inserted Code 2"
    assert nb_content['cells'][2]['cell_type'] == "code"
    assert nb_content['cells'][3]['source'] == "Cell 1 (will be Cell 3)"

async def test_bulk_add_cells_invalid_cell_data_format(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test bulk add with malformed cell data."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    
    malformed_cells = [
        {"type": "markdown", "content": "Missing source/cell_type keys"} # Incorrect keys
    ]
    with pytest.raises(ValueError, match="must be a dictionary with 'cell_type' and 'source' keys"):
        await notebook_tools_inst.notebook_bulk_add_cells(nb_path, malformed_cells, -1)

    malformed_cells_2 = [
        {"cell_type": "markdown"} # Missing source
    ]
    with pytest.raises(ValueError, match="must be a dictionary with 'cell_type' and 'source' keys"):
        await notebook_tools_inst.notebook_bulk_add_cells(nb_path, malformed_cells_2, -1)

async def test_bulk_add_cells_invalid_cell_type(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test bulk add with an invalid cell_type."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    cells_with_invalid_type = [
        {"cell_type": "picture", "source": "some_image.png"}
    ]
    with pytest.raises(ValueError, match="Invalid cell_type 'picture'"):
        await notebook_tools_inst.notebook_bulk_add_cells(nb_path, cells_with_invalid_type, -1)

async def test_bulk_add_cells_source_too_large(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test bulk add where one cell's source exceeds the size limit."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    limit = notebook_tools_inst.config.max_cell_source_size
    cells = [
        {"cell_type": "code", "source": "print('ok')"},
        {"cell_type": "markdown", "source": "A" * (limit + 1)}
    ]
    with pytest.raises(ValueError, match="Source content for cell at index 1 .* exceeds maximum allowed size"):
        await notebook_tools_inst.notebook_bulk_add_cells(nb_path, cells, -1)

async def test_bulk_add_cells_invalid_insert_index(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test bulk add with an insert_after_index that is out of bounds."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    # Notebook is empty, so 0 cells. insert_after_index=0 means initial_insertion_point=1, which is out of bounds (0 to 0).
    cells = [{"cell_type": "code", "source": "print(1)"}]
    with pytest.raises(IndexError, match="Initial insertion point 1 .* is out of bounds for notebook with 0 cells"):
        await notebook_tools_inst.notebook_bulk_add_cells(nb_path, cells, 0)
    
    await notebook_tools_inst.notebook_add_cell(nb_path, "code", "cell0", -1) # Now 1 cell
    # insert_after_index=1 means initial_insertion_point=2, out of bounds (0 to 1)
    with pytest.raises(IndexError, match="Initial insertion point 2 .* is out of bounds for notebook with 1 cells"):
        await notebook_tools_inst.notebook_bulk_add_cells(nb_path, cells, 1)

# --- Tests for notebook_edit_cell_output ---

async def test_edit_cell_output_add_new(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test adding output to a code cell with no prior output."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    await notebook_tools_inst.notebook_add_cell(nb_path, "code", "print('Hello')", -1)

    new_outputs = [
        {"output_type": "stream", "name": "stdout", "text": "Hello\n"}
    ]
    result = await notebook_tools_inst.notebook_edit_cell_output(nb_path, 0, new_outputs)
    assert "Successfully edited/added outputs" in result

    read_outputs = await notebook_tools_inst.notebook_read_cell_output(nb_path, 0)
    assert len(read_outputs) == 1
    assert read_outputs[0]["output_type"] == "stream"
    assert read_outputs[0]["text"] == "Hello\n"

async def test_edit_cell_output_replace_existing(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test replacing existing output of a code cell."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    await notebook_tools_inst.notebook_add_cell(nb_path, "code", "print('Old'); 1/0", -1)
    
    # Manually set an initial output
    # Use the public tool method notebook_read, which handles allowed_roots internally
    nb_dict_content = await notebook_tools_inst.notebook_read(notebook_path=nb_path) 
    nb_obj = nbformat.from_dict(nb_dict_content) # Convert dict to NotebookNode for modification

    initial_error_output = {
        "output_type": "error", 
        "ename": "ZeroDivisionError", 
        "evalue": "division by zero", 
        "traceback": ["Traceback line 1"]
    }
    nb_obj['cells'][0]['outputs'] = [nbformat.from_dict(initial_error_output)] # Convert to NotebookNode
    await notebook_tools_inst.write_notebook(nb_path, nb_obj, notebook_tools_inst.config.allowed_roots)

    new_stream_output = [
        {"output_type": "stream", "name": "stdout", "text": "New output\n"}
    ]
    await notebook_tools_inst.notebook_edit_cell_output(nb_path, 0, new_stream_output)
    
    read_outputs = await notebook_tools_inst.notebook_read_cell_output(nb_path, 0)
    assert len(read_outputs) == 1
    assert read_outputs[0]["output_type"] == "stream"
    assert read_outputs[0]["text"] == "New output\n"

async def test_edit_cell_output_clear_by_empty_list(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test clearing outputs by providing an empty list."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    await notebook_tools_inst.notebook_add_cell(nb_path, "code", "print('stuff')", -1)
    # Use the public tool method notebook_read
    nb_dict_content = await notebook_tools_inst.notebook_read(notebook_path=nb_path)
    nb_obj = nbformat.from_dict(nb_dict_content)
    nb_obj['cells'][0]['outputs'] = [nbformat.from_dict({"output_type": "stream", "name": "stdout", "text": "stuff\n"})] # Convert to NotebookNode
    await notebook_tools_inst.write_notebook(nb_path, nb_obj, notebook_tools_inst.config.allowed_roots)

    await notebook_tools_inst.notebook_edit_cell_output(nb_path, 0, []) # Empty list
    read_outputs = await notebook_tools_inst.notebook_read_cell_output(nb_path, 0)
    assert read_outputs == []

async def test_edit_cell_output_not_code_cell(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test attempting to edit output of a non-code cell."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    await notebook_tools_inst.notebook_add_cell(nb_path, "markdown", "# Text", -1)
    outputs = [{"output_type": "stream", "name": "stdout", "text": "Hello\n"}]
    with pytest.raises(ValueError, match="Outputs can only be edited for 'code' cells"):
        await notebook_tools_inst.notebook_edit_cell_output(nb_path, 0, outputs)

async def test_edit_cell_output_invalid_outputs_param_type(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test providing a non-list type for the outputs parameter."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    await notebook_tools_inst.notebook_add_cell(nb_path, "code", "a=1", -1)
    with pytest.raises(ValueError, match="The 'outputs' parameter must be a list"):
        await notebook_tools_inst.notebook_edit_cell_output(nb_path, 0, {"output_type": "stream"}) # A dict, not list

async def test_edit_cell_output_invalid_item_type_in_list(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test an item in the outputs list is not a dictionary."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    await notebook_tools_inst.notebook_add_cell(nb_path, "code", "a=1", -1)
    with pytest.raises(ValueError, match="Each item in 'outputs' must be a dictionary"):
        await notebook_tools_inst.notebook_edit_cell_output(nb_path, 0, ["not_a_dict"])

async def test_edit_cell_output_missing_output_type(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test an output item is missing the 'output_type' key."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    await notebook_tools_inst.notebook_add_cell(nb_path, "code", "a=1", -1)
    with pytest.raises(ValueError, match="missing required key 'output_type'"):
        await notebook_tools_inst.notebook_edit_cell_output(nb_path, 0, [{"name": "stdout"}])

async def test_edit_cell_output_invalid_stream_output(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test stream output item is missing required keys."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    await notebook_tools_inst.notebook_add_cell(nb_path, "code", "a=1", -1)
    with pytest.raises(ValueError, match="Stream output .* missing 'name' string key"):
        await notebook_tools_inst.notebook_edit_cell_output(nb_path, 0, [{"output_type": "stream", "text": "hi"}])
    with pytest.raises(ValueError, match="Stream output .* missing 'text' key"):
        await notebook_tools_inst.notebook_edit_cell_output(nb_path, 0, [{"output_type": "stream", "name": "stdout"}])

async def test_edit_cell_output_invalid_display_data_output(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test display_data output item is missing required keys."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    await notebook_tools_inst.notebook_add_cell(nb_path, "code", "a=1", -1)
    with pytest.raises(ValueError, match="Output type 'display_data'.* missing 'data' dictionary key"):
        await notebook_tools_inst.notebook_edit_cell_output(nb_path, 0, [{"output_type": "display_data", "metadata": {}}])
    with pytest.raises(ValueError, match="Output type 'display_data'.* missing 'metadata' dictionary key"):
        await notebook_tools_inst.notebook_edit_cell_output(nb_path, 0, [{"output_type": "display_data", "data": {}}])

async def test_edit_cell_output_invalid_error_output(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test error output item is missing required keys."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    await notebook_tools_inst.notebook_add_cell(nb_path, "code", "a=1", -1)
    with pytest.raises(ValueError, match="Error output .* missing 'ename' string key"):
        await notebook_tools_inst.notebook_edit_cell_output(nb_path, 0, [{"output_type": "error", "evalue": "val", "traceback": []}])
    with pytest.raises(ValueError, match="Error output .* missing 'evalue' string key"):
        await notebook_tools_inst.notebook_edit_cell_output(nb_path, 0, [{"output_type": "error", "ename": "nm", "traceback": []}])
    with pytest.raises(ValueError, match="Error output .* missing 'traceback' list key"):
        await notebook_tools_inst.notebook_edit_cell_output(nb_path, 0, [{"output_type": "error", "ename": "nm", "evalue": "val"}])

async def test_edit_cell_output_data_too_large(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test provided output data exceeds max_cell_output_size."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    await notebook_tools_inst.notebook_add_cell(nb_path, "code", "a=1", -1)
    limit = notebook_tools_inst.config.max_cell_output_size
    large_text = "B" * (limit + 1)
    outputs = [
        {"output_type": "stream", "name": "stdout", "text": large_text}
    ]
    with pytest.raises(ValueError, match="Provided output data size .* exceeds maximum allowed size"):
        await notebook_tools_inst.notebook_edit_cell_output(nb_path, 0, outputs)

async def test_edit_cell_output_cell_index_out_of_bounds(notebook_tools_inst: NotebookTools, notebook_path_factory):
    """Test edit_cell_output with an out-of-bounds cell_index."""
    nb_path = notebook_path_factory()
    await notebook_tools_inst.notebook_create(notebook_path=nb_path)
    await notebook_tools_inst.notebook_add_cell(nb_path, "code", "a=1", -1) # One cell at index 0
    outputs = [{"output_type": "stream", "name": "stdout", "text": "Hello\n"}]
    with pytest.raises(IndexError, match="Cell index 1 is out of bounds"):
        await notebook_tools_inst.notebook_edit_cell_output(nb_path, 1, outputs) 