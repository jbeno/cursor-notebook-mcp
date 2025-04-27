"""
Defines the MCP tools for Jupyter Notebook operations.

Uses a class-based approach to manage dependencies on configuration and core operations.
"""

import os
import sys
import subprocess
import importlib.util
import logging
import json
from typing import Any, List, Dict, Callable, Coroutine, Union
import ast
import re

import nbformat
from nbformat import NotebookNode

from . import notebook_ops

logger = logging.getLogger(__name__)

class NotebookTools:
    """Encapsulates notebook manipulation tools for MCP."""

    def __init__(self, config: Any, mcp_instance: Any):
        """
        Initializes the NotebookTools provider.

        Args:
            config: A configuration object (like ServerConfig) containing attributes like
                    allowed_roots, max_cell_source_size, max_cell_output_size.
            mcp_instance: The FastMCP server instance to register tools against.
        """
        self.config = config
        self.mcp = mcp_instance
        # Make notebook operations available to tool methods
        self.read_notebook = notebook_ops.read_notebook
        self.write_notebook = notebook_ops.write_notebook
        self.is_path_allowed = notebook_ops.is_path_allowed

        # Register tools upon instantiation
        self._register_tools()

    def _log_prefix(self, tool_name: str, **kwargs) -> str:
        """Helper to create a consistent log prefix."""
        args_str = ", ".join(f"{k}='{v}'" for k, v in kwargs.items())
        return f"[Tool: {tool_name}({args_str})]"

    def _register_tools(self):
        """Registers all tool methods with the MCP instance."""
        tools_to_register = [
            self.notebook_create,
            self.notebook_delete,
            self.notebook_rename,
            self.notebook_edit_cell,
            self.notebook_add_cell,
            self.notebook_delete_cell,
            self.notebook_read_cell,
            self.notebook_get_cell_count,
            self.notebook_read_metadata,
            self.notebook_edit_metadata,
            self.notebook_read_cell_metadata,
            self.notebook_edit_cell_metadata,
            self.notebook_clear_cell_outputs,
            self.notebook_clear_all_outputs,
            self.notebook_move_cell,
            self.diagnose_imports,
            self.notebook_validate,
            self.notebook_get_info,
            self.notebook_read_cell_output,
            self.notebook_split_cell,
            self.notebook_merge_cells,
            self.notebook_export,
            self.notebook_read,
            self.notebook_change_cell_type,
            self.notebook_duplicate_cell,
            self.notebook_get_outline,
            self.notebook_search,
        ]
        for tool_method in tools_to_register:
            # Use the method's name and docstring for registration
            if hasattr(self.mcp, 'add_tool'):
                self.mcp.add_tool(tool_method)
            elif hasattr(self.mcp, 'tool') and callable(self.mcp.tool):
                # If add_tool doesn't exist, try applying the .tool() decorator programmatically
                # This assumes tool_method already has the correct signature and docstring
                decorated_tool = self.mcp.tool()(tool_method)
                # Need to ensure the decorated tool replaces the original method if necessary,
                # but FastMCP might handle registration internally when called like this.
                # Let's assume the decorator call handles registration.
                pass # Decorator call executed, registration might have happened.
            else:
                # If neither works, log an error
                logger.error(f"Could not find a method to register tool '{tool_method.__name__}' on FastMCP instance.")
                # Optionally raise an error here
                raise AttributeError("FastMCP instance does not have a known tool registration method (tried add_tool, tool decorator)")
            logger.debug(f"Registered tool: {tool_method.__name__}")

    # --- Tool Definitions --- 
    # These methods will be registered automatically by _register_tools
    
    async def notebook_create(self, notebook_path: str) -> str:
        """Creates a new, empty Jupyter Notebook (.ipynb) file at the specified path.

        Args:
            notebook_path: The absolute path where the new .ipynb file should be created.
                           Must be within an allowed root directory.
        """
        log_prefix = self._log_prefix('notebook_create', path=notebook_path)
        logger.info(f"{log_prefix} Called.")

        try:
            # --- Security and Existence Check (Before Write) ---
            if not os.path.isabs(notebook_path):
                 raise ValueError("Invalid notebook path: Only absolute paths are allowed.")
            if not self.is_path_allowed(notebook_path, self.config.allowed_roots):
                 raise PermissionError(f"Access denied: Path '{notebook_path}' is outside the allowed workspace roots.")

            resolved_path = os.path.realpath(notebook_path)
            if not resolved_path.endswith(".ipynb"):
                 raise ValueError(f"Invalid file type: '{resolved_path}' must point to a .ipynb file.")
            if os.path.exists(resolved_path):
                 raise FileExistsError(f"Cannot create notebook, file already exists: {resolved_path}")

            # --- Create and Write --- 
            nb = nbformat.v4.new_notebook()
            await self.write_notebook(notebook_path, nb, self.config.allowed_roots)
            
            logger.info(f"{log_prefix} SUCCESS - Created new notebook at {resolved_path}")
            return f"Successfully created new notebook: {notebook_path}"

        except (PermissionError, FileExistsError, ValueError, IOError) as e:
            logger.error(f"{log_prefix} FAILED - {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred during notebook creation: {e}") from e

    async def notebook_delete(self, notebook_path: str) -> str:
        """Deletes a Jupyter Notebook (.ipynb) file at the specified path.

        Args:
            notebook_path: The absolute path to the notebook file to delete.
                           Must be within an allowed root directory.
        """
        log_prefix = self._log_prefix('notebook_delete', path=notebook_path)
        logger.info(f"{log_prefix} Called.")

        try:
            # Security Checks
            if not os.path.isabs(notebook_path):
                raise ValueError("Invalid notebook path: Only absolute paths are allowed.")
            if not self.is_path_allowed(notebook_path, self.config.allowed_roots):
                raise PermissionError(f"Access denied: Path is outside the allowed workspace roots.")
            if not notebook_path.endswith(".ipynb"):
                raise ValueError("Invalid file type: Path must point to a .ipynb file.")

            resolved_path = os.path.realpath(notebook_path)
            if not os.path.isfile(resolved_path):
                raise FileNotFoundError(f"Notebook file not found at: {resolved_path}")

            # Delete the file
            os.remove(resolved_path)
            logger.info(f"{log_prefix} SUCCESS - Deleted notebook at {resolved_path}")
            return f"Successfully deleted notebook: {notebook_path}"

        except (ValueError, PermissionError, FileNotFoundError, OSError) as e:
            # Let specific, expected errors propagate for tests
            if isinstance(e, (ValueError, PermissionError, FileNotFoundError)):
                logger.error(f"{log_prefix} FAILED - {e}") # Log the specific error
                raise # Re-raise the original exception
            # Handle unexpected OSErrors more generically
            logger.error(f"{log_prefix} FAILED - {e}")
            raise IOError(f"Failed to delete notebook file due to OS error: {e}") from e
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred during notebook deletion: {e}") from e

    async def notebook_rename(self, old_path: str, new_path: str) -> str:
        """Renames/Moves a Jupyter Notebook (.ipynb) file from one path to another.

        Args:
            old_path: The absolute path to the existing notebook file.
            new_path: The absolute path where the notebook file should be moved/renamed to.
                      Both paths must be within an allowed root directory.
        """
        log_prefix = self._log_prefix('notebook_rename', old=old_path, new=new_path)
        logger.info(f"{log_prefix} Called.")

        try:
            # Security Checks
            if not os.path.isabs(old_path) or not os.path.isabs(new_path):
                raise ValueError("Invalid notebook path(s): Only absolute paths are allowed.")
            if not self.is_path_allowed(old_path, self.config.allowed_roots) or \
               not self.is_path_allowed(new_path, self.config.allowed_roots):
                raise PermissionError(f"Access denied: One or both paths are outside the allowed workspace roots.")
            if not old_path.endswith(".ipynb") or not new_path.endswith(".ipynb"):
                raise ValueError("Invalid file type: Both paths must point to .ipynb files.")

            resolved_old_path = os.path.realpath(old_path)
            resolved_new_path = os.path.realpath(new_path)

            if not os.path.isfile(resolved_old_path):
                raise FileNotFoundError(f"Source notebook file not found at: {resolved_old_path}")
            if os.path.exists(resolved_new_path):
                raise FileExistsError(f"Cannot rename notebook, destination already exists: {resolved_new_path}")

            # Create parent directory of destination if it doesn't exist
            os.makedirs(os.path.dirname(resolved_new_path), exist_ok=True)

            # Rename/move the file
            os.rename(resolved_old_path, resolved_new_path)
            logger.info(f"{log_prefix} SUCCESS - Renamed notebook from {resolved_old_path} to {resolved_new_path}")
            return f"Successfully renamed notebook from {old_path} to {new_path}"

        except (ValueError, PermissionError, FileNotFoundError, FileExistsError, OSError) as e:
            # Let specific, expected errors propagate for tests
            if isinstance(e, (ValueError, PermissionError, FileNotFoundError, FileExistsError)):
                logger.error(f"{log_prefix} FAILED - {e}") # Log the specific error
                raise # Re-raise the original exception
            # Handle unexpected OSErrors more generically
            logger.error(f"{log_prefix} FAILED - {e}")
            raise IOError(f"Failed to rename notebook file due to OS error: {e}") from e
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred during notebook rename: {e}") from e

    async def notebook_edit_cell(self, notebook_path: str, cell_index: int, source: str) -> str:
        """Replaces the source content of a specific cell in a Jupyter Notebook.

        Args:
            notebook_path: Absolute path to the .ipynb file within an allowed root.
            cell_index: The 0-based index of the cell to edit.
            source: The new source code or markdown content for the cell.
        """
        log_prefix = self._log_prefix('notebook_edit_cell', path=notebook_path, index=cell_index)
        logger.info(f"{log_prefix} Called.")
        try:
            # Validate source size
            if len(source.encode('utf-8')) > self.config.max_cell_source_size:
                raise ValueError(f"Source content exceeds maximum allowed size ({self.config.max_cell_source_size} bytes).")

            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")
            
            nb.cells[cell_index].source = source
            await self.write_notebook(notebook_path, nb, self.config.allowed_roots)
            logger.info(f"{log_prefix} SUCCESS - Edited cell.")
            return f"Successfully edited cell {cell_index} in {notebook_path}"
        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_add_cell(self, notebook_path: str, cell_type: str, source: str, insert_after_index: int) -> str:
        """Adds a new cell to a Jupyter Notebook after the specified index.

        Args:
            notebook_path: Absolute path to the .ipynb file.
            cell_type: Type of cell ('code' or 'markdown'). Must be lowercase.
            source: The source code or markdown content for the new cell.
            insert_after_index: The 0-based index after which to insert the new cell (-1 to insert at the beginning).
        """
        log_prefix = self._log_prefix('notebook_add_cell', path=notebook_path, type=cell_type, after_index=insert_after_index)
        logger.info(f"{log_prefix} Called.")
        try:
            # Validate source size
            if len(source.encode('utf-8')) > self.config.max_cell_source_size:
                raise ValueError(f"Source content exceeds maximum allowed size ({self.config.max_cell_source_size} bytes).")

            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            
            if cell_type == 'code':
                new_cell = nbformat.v4.new_code_cell(source)
            elif cell_type == 'markdown':
                new_cell = nbformat.v4.new_markdown_cell(source)
            else:
                raise ValueError("Invalid cell_type: Must be 'code' or 'markdown'.")

            insertion_index = insert_after_index + 1
            if not 0 <= insertion_index <= len(nb.cells):
                 raise IndexError(f"Insertion index {insertion_index} (based on insert_after_index {insert_after_index}) is out of bounds (0-{len(nb.cells)}).")

            nb.cells.insert(insertion_index, new_cell)
            await self.write_notebook(notebook_path, nb, self.config.allowed_roots)
            logger.info(f"{log_prefix} SUCCESS - Added cell at index {insertion_index}.")
            return f"Successfully added {cell_type} cell at index {insertion_index} in {notebook_path}"
        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_delete_cell(self, notebook_path: str, cell_index: int) -> str:
        """Deletes a specific cell from a Jupyter Notebook.

        Args:
            notebook_path: Absolute path to the .ipynb file.
            cell_index: The 0-based index of the cell to delete.
        """
        log_prefix = self._log_prefix('notebook_delete_cell', path=notebook_path, index=cell_index)
        logger.info(f"{log_prefix} Called.")
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")

            del nb.cells[cell_index]
            await self.write_notebook(notebook_path, nb, self.config.allowed_roots)
            logger.info(f"{log_prefix} SUCCESS - Deleted cell.")
            return f"Successfully deleted cell {cell_index} from {notebook_path}"
        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_read_cell(self, notebook_path: str, cell_index: int) -> str:
        """Reads the source content of a specific cell from a Jupyter Notebook.

        Args:
            notebook_path: Absolute path to the .ipynb file.
            cell_index: The 0-based index of the cell to read.
        """
        log_prefix = self._log_prefix('notebook_read_cell', path=notebook_path, index=cell_index)
        logger.info(f"{log_prefix} Called.")
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")

            source = nb.cells[cell_index].source
            logger.info(f"{log_prefix} SUCCESS - Read cell.")
            # Apply size limit for safety, even though separate tool reads outputs
            MAX_LEN_BYTES = self.config.max_cell_source_size # Reuse config
            if len(source.encode('utf-8')) > MAX_LEN_BYTES:
                 logger.warning(f"{log_prefix} WARNING - Source content truncated ({MAX_LEN_BYTES} byte limit).")
                 # Truncate based on bytes, find closest character boundary
                 encoded_source = source.encode('utf-8')
                 truncated_bytes = encoded_source[:MAX_LEN_BYTES]
                 try:
                     source = truncated_bytes.decode('utf-8', errors='ignore') + "... (truncated)"
                 except UnicodeDecodeError:
                     source = "[Source truncated - unable to decode cleanly]"
            return source
        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_get_cell_count(self, notebook_path: str) -> int:
        """Returns the total number of cells in the notebook."""
        log_prefix = self._log_prefix('notebook_get_cell_count', path=notebook_path)
        logger.info(f"{log_prefix} Called.")
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            count = len(nb.cells)
            logger.info(f"{log_prefix} SUCCESS - Count: {count}")
            return count
        except (ValueError, FileNotFoundError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_read_metadata(self, notebook_path: str) -> dict:
        """Reads the top-level metadata of the notebook."""
        log_prefix = self._log_prefix('notebook_read_metadata', path=notebook_path)
        logger.info(f"{log_prefix} Called.")
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            metadata = dict(nb.metadata)
            logger.info(f"{log_prefix} SUCCESS - Read metadata.")
            return metadata
        except (ValueError, FileNotFoundError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_edit_metadata(self, notebook_path: str, metadata_updates: dict) -> str:
        """Updates the top-level metadata of the notebook. Merges provided updates."""
        log_prefix = self._log_prefix('notebook_edit_metadata', path=notebook_path)
        logger.info(f"{log_prefix} Called with updates: {metadata_updates}")
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            nb.metadata.update(metadata_updates)
            await self.write_notebook(notebook_path, nb, self.config.allowed_roots)
            logger.info(f"{log_prefix} SUCCESS - Updated metadata.")
            return f"Successfully updated metadata for {notebook_path}"
        except (ValueError, FileNotFoundError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_read_cell_metadata(self, notebook_path: str, cell_index: int) -> dict:
        """Reads the metadata of a specific cell."""
        log_prefix = self._log_prefix('notebook_read_cell_metadata', path=notebook_path, index=cell_index)
        logger.info(f"{log_prefix} Called.")
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")
            metadata = dict(nb.cells[cell_index].metadata)
            logger.info(f"{log_prefix} SUCCESS - Read cell metadata.")
            return metadata
        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_edit_cell_metadata(self, notebook_path: str, cell_index: int, metadata_updates: dict) -> str:
        """Updates the metadata of a specific cell. Merges provided updates."""
        log_prefix = self._log_prefix('notebook_edit_cell_metadata', path=notebook_path, index=cell_index)
        logger.info(f"{log_prefix} Called with updates: {metadata_updates}")
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")
            nb.cells[cell_index].metadata.update(metadata_updates)
            await self.write_notebook(notebook_path, nb, self.config.allowed_roots)
            logger.info(f"{log_prefix} SUCCESS - Updated cell metadata.")
            return f"Successfully updated metadata for cell {cell_index} in {notebook_path}"
        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_clear_cell_outputs(self, notebook_path: str, cell_index: int) -> str:
        """Clears the output(s) of a specific cell."""
        log_prefix = self._log_prefix('notebook_clear_cell_outputs', path=notebook_path, index=cell_index)
        logger.info(f"{log_prefix} Called.")
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")
            
            cell = nb.cells[cell_index]
            changed = False
            if hasattr(cell, 'outputs') and cell.outputs:
                cell.outputs = []
                changed = True
            if hasattr(cell, 'execution_count') and cell.execution_count is not None:
                 cell.execution_count = None
                 changed = True

            if changed:
                await self.write_notebook(notebook_path, nb, self.config.allowed_roots)
                logger.info(f"{log_prefix} SUCCESS - Cleared outputs.")
                return f"Successfully cleared outputs for cell {cell_index} in {notebook_path}"
            else:
                logger.info(f"{log_prefix} SUCCESS - No outputs/count to clear.")
                return f"No outputs or execution count found to clear for cell {cell_index} in {notebook_path}"

        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_clear_all_outputs(self, notebook_path: str) -> str:
        """Clears all outputs from all code cells in the notebook."""
        log_prefix = self._log_prefix('notebook_clear_all_outputs', path=notebook_path)
        logger.info(f"{log_prefix} Called.")
        cleared_count = 0
        changed = False
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == 'code':
                    cell_changed = False
                    if hasattr(cell, 'outputs') and cell.outputs:
                        cell.outputs = []
                        cell_changed = True
                    if hasattr(cell, 'execution_count') and cell.execution_count is not None:
                        cell.execution_count = None
                        cell_changed = True
                    if cell_changed:
                        cleared_count += 1
                        changed = True # Mark that the notebook object was modified
            
            if changed:
                 await self.write_notebook(notebook_path, nb, self.config.allowed_roots)
                 logger.info(f"{log_prefix} SUCCESS - Cleared outputs for {cleared_count} cells.")
                 return f"Successfully cleared outputs for {cleared_count} code cells in {notebook_path}"
            else:
                 logger.info(f"{log_prefix} SUCCESS - No outputs needed clearing.")
                 return f"No code cell outputs found to clear in {notebook_path}"

        except (ValueError, FileNotFoundError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_move_cell(self, notebook_path: str, from_index: int, to_index: int) -> str:
        """Moves a cell from one position to another."""
        log_prefix = self._log_prefix('notebook_move_cell', path=notebook_path, from_idx=from_index, to_idx=to_index)
        logger.info(f"{log_prefix} Called.")
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            num_cells = len(nb.cells)
            
            if not 0 <= from_index < num_cells:
                raise IndexError(f"Source index {from_index} is out of bounds (0-{num_cells-1}).")
            if not 0 <= to_index <= num_cells: # Allow moving to the very end
                raise IndexError(f"Destination index {to_index} is out of bounds (0-{num_cells}).")
                
            if from_index == to_index or from_index == to_index - 1:
                 logger.info(f"{log_prefix} SKIPPED - Cell move results in no change.")
                 return f"Cell at index {from_index} was not moved (source and destination are effectively the same)."

            cell_to_move = nb.cells.pop(from_index)
            # Adjust insertion index if pop happened before target
            actual_to_index = to_index
            if from_index < to_index:
                actual_to_index -= 1 # The target index shifted down by one

            nb.cells.insert(actual_to_index, cell_to_move)

            await self.write_notebook(notebook_path, nb, self.config.allowed_roots)
            logger.info(f"{log_prefix} SUCCESS - Moved cell from {from_index} to {to_index} (inserted at {actual_to_index}).")
            return f"Successfully moved cell from index {from_index} to {to_index} in {notebook_path}"
        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def diagnose_imports(self) -> str:
        """Diagnostic tool to troubleshoot import issues, particularly with nbconvert."""
        # This tool is largely self-contained and doesn't need config/ops
        log_prefix = self._log_prefix('diagnose_imports')
        logger.info(f"{log_prefix} Called.")
        
        result = []
        result.append("=== Python Environment Diagnostics ===")
        result.append(f"Python version: {sys.version}")
        result.append(f"Python executable: {sys.executable}")
        result.append(f"Current working directory: {os.getcwd()}")
        
        result.append("\n=== Module Search Paths (sys.path) ===")
        for i, path in enumerate(sys.path):
            result.append(f"{i}: {path}")
        
        result.append("\n=== nbconvert Detection Tests ===")
        try:
            import nbconvert
            result.append(f"Direct import SUCCESS! Version: {getattr(nbconvert, '__version__', 'N/A')}, Path: {getattr(nbconvert, '__file__', 'N/A')}")
        except ImportError as e:
            result.append(f"Direct import FAILED: {e}")
        
        result.append("\nAttempting importlib.util.find_spec('nbconvert')...")
        try:
            spec = importlib.util.find_spec("nbconvert")
            if spec and spec.origin:
                result.append(f"SUCCESS! Spec found at: {spec.origin}")
            else:
                result.append("FAILED: No spec found for nbconvert")
        except Exception as e:
             result.append(f"Error using find_spec: {e}")

        result.append("\n=== Checking pip list output ===")
        try:
            cmd = [sys.executable, "-m", "pip", "list"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            output = proc.stdout
            nbconvert_line = next((line for line in output.split('\n') if 'nbconvert' in line.split()), None)
            if nbconvert_line:
                result.append(f"Found in pip list: {nbconvert_line.strip()}")
            else:
                result.append("nbconvert not found in pip list output")
        except FileNotFoundError:
             result.append("FAILED: 'pip' command not found.")
        except subprocess.TimeoutExpired:
             result.append("FAILED: 'pip list' command timed out.")
        except Exception as e:
            result.append(f"Error running pip list: {e}")
        
        return "\n".join(result)

    async def notebook_validate(self, notebook_path: str) -> str:
        """Validates the notebook against the nbformat schema."""
        log_prefix = self._log_prefix('notebook_validate', path=notebook_path)
        logger.info(f"{log_prefix} Called.")
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            nbformat.validate(nb)
            logger.info(f"{log_prefix} SUCCESS - Notebook is valid.")
            return "Notebook is valid according to the nbformat schema."
        except nbformat.ValidationError as e:
            logger.warning(f"{log_prefix} VALIDATION FAILED: {e}")
            # Provide a more structured error message if possible
            return f"Notebook validation failed: {e}"
        except (ValueError, FileNotFoundError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_get_info(self, notebook_path: str) -> dict:
        """Gets general information about the notebook (cell count, kernel, language)."""
        log_prefix = self._log_prefix('notebook_get_info', path=notebook_path)
        logger.info(f"{log_prefix} Called.")
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            info = {
                "path": notebook_path, # Add path for context
                "cell_count": len(nb.cells),
                "metadata": dict(nb.metadata), 
                "kernelspec": nb.metadata.get("kernelspec", None),
                "language_info": nb.metadata.get("language_info", None)
            }
            logger.info(f"{log_prefix} SUCCESS - Gathered notebook info.")
            return info
        except (ValueError, FileNotFoundError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_read_cell_output(self, notebook_path: str, cell_index: int) -> List[dict]:
        """Reads the output(s) of a specific cell from a Jupyter Notebook.

        Args:
            notebook_path: Absolute path to the .ipynb file.
            cell_index: The 0-based index of the cell to read outputs from.

        Returns:
            A list containing the cell's output objects (as dictionaries).
        """
        log_prefix = self._log_prefix('notebook_read_cell_output', path=notebook_path, index=cell_index)
        logger.info(f"{log_prefix} Called.")
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")

            cell = nb.cells[cell_index]
            if cell.cell_type != 'code':
                logger.info(f"{log_prefix} SUCCESS - Cell {cell_index} is not a code cell, returning empty list.")
                return []

            outputs = cell.get('outputs', [])
            if not outputs:
                logger.info(f"{log_prefix} SUCCESS - Cell {cell_index} has no outputs.")
                return []

            # Check output size limit
            try:
                output_bytes = json.dumps(outputs).encode('utf-8')
                output_size = len(output_bytes)
                if output_size > self.config.max_cell_output_size:
                    logger.error(f"{log_prefix} FAILED - Output size ({output_size} bytes) exceeds limit ({self.config.max_cell_output_size} bytes). Returning truncated representation.")
                    # Return a placeholder indicating truncation
                    return [{
                        'output_type': 'error',
                        'ename': 'OutputSizeError',
                        'evalue': f'Output truncated - size ({output_size} bytes) exceeds limit ({self.config.max_cell_output_size} bytes)',
                        'traceback': []
                    }]
            except (TypeError, OverflowError) as json_err:
                logger.error(f"{log_prefix} FAILED - Could not serialize outputs for size check: {json_err}")
                raise ValueError(f"Could not determine size of cell output due to serialization error: {json_err}")

            logger.info(f"{log_prefix} SUCCESS - Read outputs for cell {cell_index} (Size: {output_size} bytes).")
            return outputs # Already JSON serializable list[dict]

        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_split_cell(self, notebook_path: str, cell_index: int, split_at_line: int) -> str:
        """Splits a cell into two cells at a specified line number.

        Args:
            notebook_path: Absolute path to the .ipynb file.
            cell_index: The 0-based index of the cell to split.
            split_at_line: The 1-based line number where the split occurs (this line starts the new cell).
        """
        log_prefix = self._log_prefix('notebook_split_cell', path=notebook_path, index=cell_index, line=split_at_line)
        logger.info(f"{log_prefix} Called.")
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            num_cells = len(nb.cells)
            if not 0 <= cell_index < num_cells:
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{num_cells-1}).")

            cell_to_split = nb.cells[cell_index]
            source_lines = cell_to_split.source.splitlines(True)
            num_lines = len(source_lines)
            split_index = split_at_line - 1 # Convert to 0-based index

            if not 0 <= split_index <= num_lines:
                 raise ValueError(f"Split line {split_at_line} is out of bounds (1-{num_lines + 1}).")

            source_part1 = "".join(source_lines[:split_index])
            source_part2 = "".join(source_lines[split_index:])

            # Update original cell
            cell_to_split.source = source_part1

            # Create new cell with the same type and metadata
            cell_type = cell_to_split.cell_type
            if cell_type == 'code':
                new_cell = nbformat.v4.new_code_cell(source=source_part2)
            elif cell_type == 'markdown':
                new_cell = nbformat.v4.new_markdown_cell(source=source_part2)
            elif cell_type == 'raw':
                 new_cell = nbformat.v4.new_raw_cell(source=source_part2)
            else:
                 logger.warning(f"{log_prefix} - Unknown cell type '{cell_type}'. Creating raw cell.")
                 new_cell = nbformat.v4.new_raw_cell(source=source_part2)
            new_cell.metadata.update(dict(cell_to_split.metadata)) # Copy metadata

            # Insert the new cell
            nb.cells.insert(cell_index + 1, new_cell)

            await self.write_notebook(notebook_path, nb, self.config.allowed_roots)
            logger.info(f"{log_prefix} SUCCESS - Split cell {cell_index} at line {split_at_line}. New cell inserted at index {cell_index + 1}.")
            return f"Successfully split cell {cell_index} at line {split_at_line}."

        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_merge_cells(self, notebook_path: str, first_cell_index: int) -> str:
        """Merges a cell with the cell immediately following it.

        Args:
            notebook_path: Absolute path to the .ipynb file.
            first_cell_index: The 0-based index of the first cell in the pair to merge.
        """
        log_prefix = self._log_prefix('notebook_merge_cells', path=notebook_path, index=first_cell_index)
        logger.info(f"{log_prefix} Called.")
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            num_cells = len(nb.cells)

            if not 0 <= first_cell_index < num_cells - 1:
                raise IndexError(f"First cell index {first_cell_index} is invalid or it's the last cell (cannot merge). Valid range: 0-{num_cells-2}.")

            cell1 = nb.cells[first_cell_index]
            cell2 = nb.cells[first_cell_index + 1]

            if cell1.cell_type != cell2.cell_type:
                raise ValueError(f"Cannot merge cells of different types ({cell1.cell_type} and {cell2.cell_type}).")

            source1 = cell1.source
            source2 = cell2.source
            separator = '\n' if source1 and not source1.endswith('\n') else ''
            combined_source = source1 + separator + source2

            # Check combined source size
            if len(combined_source.encode('utf-8')) > self.config.max_cell_source_size:
                raise ValueError(f"Merged source content exceeds maximum allowed size ({self.config.max_cell_source_size} bytes).")

            cell1.source = combined_source
            # Merge metadata? Decide on strategy (e.g., keep cell1's, merge dicts?)
            # For now, keep cell1's metadata.

            del nb.cells[first_cell_index + 1]

            await self.write_notebook(notebook_path, nb, self.config.allowed_roots)
            logger.info(f"{log_prefix} SUCCESS - Merged cell {first_cell_index + 1} into cell {first_cell_index}.")
            return f"Successfully merged cell {first_cell_index + 1} into cell {first_cell_index}."

        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_export(self, notebook_path: str, export_format: str, output_path: str) -> str:
        """Exports a notebook to a specified format using nbconvert.

        Args:
            notebook_path: Absolute path to the .ipynb file to export.
            export_format: Desired output format (e.g., 'python', 'html').
            output_path: Absolute path for the exported file (must be within allowed roots).
        """
        log_prefix = self._log_prefix('notebook_export', nb=notebook_path, format=export_format, out=output_path)
        logger.info(f"{log_prefix} Called.")

        try:
            # Use helper for path validation
            if not os.path.isabs(notebook_path) or not os.path.isabs(output_path):
                raise ValueError("Input and output paths must be absolute.")
            if not self.is_path_allowed(notebook_path, self.config.allowed_roots) or \
               not self.is_path_allowed(output_path, self.config.allowed_roots):
                raise PermissionError("Input or output path is outside allowed roots.")
                
            resolved_notebook_path = os.path.realpath(notebook_path)
            resolved_output_path = os.path.realpath(output_path)

            if not os.path.isfile(resolved_notebook_path):
                raise FileNotFoundError(f"Notebook file not found: {resolved_notebook_path}")
            if resolved_notebook_path == resolved_output_path:
                 raise ValueError("Output path cannot be the same as the input notebook path.")
            if not resolved_notebook_path.endswith('.ipynb'):
                 raise ValueError("Input path must be a .ipynb file.")

            output_dir = os.path.dirname(resolved_output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            export_format_lower = export_format.lower()
            if export_format_lower == 'script': export_format_lower = 'python'
            
            # --- Run nbconvert --- 
            logger.info(f"{log_prefix} - Running nbconvert via subprocess...")
            output_filename_base = os.path.splitext(os.path.basename(resolved_output_path))[0]
            cmd = [
                sys.executable, "-m", "nbconvert",
                f"--to={export_format_lower}",
                f"--output={output_filename_base}", # nbconvert adds extension
                resolved_notebook_path
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(sys.path)
            
            logger.debug(f"{log_prefix} Command: {' '.join(cmd)}")
            logger.debug(f"{log_prefix} CWD: {output_dir}")
            
            process = subprocess.run(
                cmd, cwd=output_dir, env=env,
                capture_output=True, text=True, timeout=60
            )
            
            if process.returncode != 0:
                logger.error(f"{log_prefix} FAILED - nbconvert error (code {process.returncode}): {process.stderr}")
                raise RuntimeError(f"nbconvert failed: {process.stderr}")
                
            logger.debug(f"{log_prefix} nbconvert STDOUT: {process.stdout}")
            if process.stderr:
                 logger.debug(f"{log_prefix} nbconvert STDERR: {process.stderr}")

            # --- Verify output file --- 
            # nbconvert determines the exact output filename and extension
            # Try to find the output file based on the base name
            actual_output_file = None
            expected_extension = f".{export_format_lower}" # Guess common extension
            potential_output = os.path.join(output_dir, output_filename_base + expected_extension)
            if os.path.exists(potential_output):
                actual_output_file = potential_output
            else:
                # Search for files starting with the base name in the output dir
                for filename in os.listdir(output_dir):
                    if filename.startswith(output_filename_base):
                        actual_output_file = os.path.join(output_dir, filename)
                        logger.info(f"{log_prefix} Found probable output file: {actual_output_file}")
                        break
            
            if not actual_output_file or not os.path.exists(actual_output_file):
                 raise FileNotFoundError(f"nbconvert ran but output file '{output_filename_base + expected_extension}' (or similar) not found in {output_dir}. nbconvert output: {process.stdout} {process.stderr}")

            logger.info(f"{log_prefix} SUCCESS - Exported notebook to {actual_output_file}")
            return f"Successfully exported notebook to {export_format} format at {actual_output_file}"

        except (ValueError, FileNotFoundError, PermissionError, IOError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except subprocess.TimeoutExpired:
            logger.error(f"{log_prefix} FAILED - nbconvert process timed out")
            raise RuntimeError("nbconvert process timed out after 60 seconds")
        except Exception as e:
            # Check if it's likely nbconvert is not installed
            if isinstance(e, RuntimeError) and "No module named nbconvert" in str(e):
                 logger.error(f"{log_prefix} FAILED - nbconvert module not found. Is it installed?")
                 raise ImportError("nbconvert does not seem to be installed in the server environment.") from e
            logger.exception(f"{log_prefix} FAILED - Unexpected error during export: {e}")
            raise RuntimeError(f"An unexpected error occurred during export: {e}") from e

    async def notebook_read(self, notebook_path: str) -> dict:
        """Reads an entire notebook and returns its structure as a dictionary.

        Args:
            notebook_path: Absolute path to the .ipynb file within an allowed root.
        """
        log_prefix = self._log_prefix('notebook_read', path=notebook_path)
        logger.info(f"{log_prefix} Called.")
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            nb_dict = dict(nb)
            
            # Optionally truncate large cell outputs/sources here if needed
            # This is similar to notebook_read_cell_output logic but applied globally
            total_size = 0
            MAX_TOTAL_SIZE = 50 * 1024 * 1024 # Example: Limit total read size to 50MB
            
            for i, cell in enumerate(nb_dict.get('cells', [])):
                try:
                    cell_bytes = json.dumps(cell).encode('utf-8')
                    cell_size = len(cell_bytes)
                    total_size += cell_size
                    # Check individual cell limits (redundant with specific tools, but safe)
                    if cell_size > max(self.config.max_cell_source_size, self.config.max_cell_output_size) * 1.1: # Allow some buffer
                         logger.warning(f"{log_prefix} Cell {i} size ({cell_size} bytes) exceeds limits. Truncating representation.")
                         cell['source'] = "[Content truncated due to size]"
                         cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': '[Output truncated due to size]'}]
                         cell['metadata'] = {'truncated': True}
                except Exception as size_err:
                     logger.warning(f"{log_prefix} Could not estimate size for a cell: {size_err}")
                     cell['source'] = "[Content could not be processed]"
                     cell['outputs'] = []
                     cell['metadata'] = {'processing_error': True}

                if total_size > MAX_TOTAL_SIZE:
                     logger.error(f"{log_prefix} FAILED - Total notebook size ({total_size}) exceeds limit ({MAX_TOTAL_SIZE} bytes). Returning partial structure.")
                     # Remove remaining cells to prevent excessive data transfer
                     nb_dict['cells'] = nb_dict['cells'][:i + 1]
                     nb_dict['metadata']['truncated'] = 'total_size_limit_exceeded'
                     break # Stop processing cells

            logger.info(f"{log_prefix} SUCCESS - Read entire notebook (Estimated size: {total_size} bytes).")
            return nb_dict
        except (ValueError, FileNotFoundError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_change_cell_type(self, notebook_path: str, cell_index: int, new_type: str) -> str:
        """Changes the type of a specific cell in a Jupyter Notebook.
        
        Args:
            notebook_path: Absolute path to the .ipynb file.
            cell_index: The 0-based index of the cell to change.
            new_type: Target type: 'code', 'markdown', or 'raw'.
        """
        log_prefix = self._log_prefix('notebook_change_cell_type', path=notebook_path, index=cell_index, type=new_type)
        logger.info(f"{log_prefix} Called.")
        
        valid_types = ['code', 'markdown', 'raw']
        if new_type not in valid_types:
            raise ValueError(f"Invalid cell type: {new_type}. Must be one of: {', '.join(valid_types)}")
        
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")
            
            cell = nb.cells[cell_index]
            current_type = cell.cell_type
            if current_type == new_type:
                return f"Cell {cell_index} is already of type '{new_type}'. No change needed."
            
            source = cell.source
            metadata = dict(cell.metadata)
            
            if new_type == 'code': new_cell = nbformat.v4.new_code_cell(source=source)
            elif new_type == 'markdown': new_cell = nbformat.v4.new_markdown_cell(source=source)
            else: new_cell = nbformat.v4.new_raw_cell(source=source)
            
            new_cell.metadata.update(metadata)
            nb.cells[cell_index] = new_cell
            
            await self.write_notebook(notebook_path, nb, self.config.allowed_roots)
            logger.info(f"{log_prefix} SUCCESS - Changed cell type from '{current_type}' to '{new_type}'")
            return f"Successfully changed cell {cell_index} from '{current_type}' to '{new_type}'"
        
        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_duplicate_cell(self, notebook_path: str, cell_index: int, count: int = 1) -> str:
        """Duplicates a specific cell in a Jupyter Notebook multiple times.
        
        Args:
            notebook_path: Absolute path to the .ipynb file.
            cell_index: The 0-based index of the cell to duplicate.
            count: Number of copies to create (default: 1).
        """
        log_prefix = self._log_prefix('notebook_duplicate_cell', path=notebook_path, index=cell_index, count=count)
        logger.info(f"{log_prefix} Called.")
        
        if count < 1: raise ValueError(f"Count must be a positive integer: {count}")
        
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")
            
            cell_to_duplicate = nb.cells[cell_index]
            insertion_index = cell_index + 1
            
            for i in range(count):
                cell_type = cell_to_duplicate.cell_type
                source = cell_to_duplicate.source
                metadata = dict(cell_to_duplicate.metadata)

                if cell_type == 'code': new_cell = nbformat.v4.new_code_cell(source=source)
                elif cell_type == 'markdown': new_cell = nbformat.v4.new_markdown_cell(source=source)
                elif cell_type == 'raw': new_cell = nbformat.v4.new_raw_cell(source=source)
                else: 
                    logger.warning(f"{log_prefix} - Unknown cell type: {cell_type}. Duplicating as raw cell.")
                    new_cell = nbformat.v4.new_raw_cell(source=source)
                
                new_cell.metadata.update(metadata)
                nb.cells.insert(insertion_index + i, new_cell)
            
            await self.write_notebook(notebook_path, nb, self.config.allowed_roots)
            new_cells_text = "cell" if count == 1 else f"{count} cells"
            logger.info(f"{log_prefix} SUCCESS - Duplicated cell, created {new_cells_text}")
            return f"Successfully duplicated cell {cell_index}, creating {new_cells_text} after it."
        
        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    async def notebook_get_outline(self, notebook_path: str) -> List[Dict[str, Union[int, str, List[str]]]]:
        """Analyzes a Jupyter notebook file to extract its structure.

        Reads the notebook, iterates through cells, identifies their type,
        extracts definitions (func, class), headings (H1, H2), significant comments (comment:),
        and calculates line counts. Returns a default context line if no other outline is found.

        Args:
            notebook_path: The absolute path to the .ipynb notebook file.

        Returns:
            A list of dictionaries, where each dictionary represents a cell
            and contains:
            - 'index': The 0-based index of the cell.
            - 'type': The cell type ('code' or 'markdown').
            - 'line_count': The total number of lines in the cell.
            - 'outline': A list of strings representing the outline items found.
                         Guaranteed to contain at least one item.
        """
        log_prefix = self._log_prefix('notebook_get_outline', path=notebook_path)
        logger.info(f"{log_prefix} Called.")
        try:
            # read_notebook already performs path validation and read
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)

            # <<< Add check for empty notebook here >>>
            if not nb.cells:
                logger.info(f"{log_prefix} SUCCESS - Notebook is empty.")
                return [{"message": "Notebook is empty or has no cells"}]

            structure_map: List[Dict[str, Union[int, str, List[str]]]] = []

            for index, cell in enumerate(nb.cells):
                outline_items = []
                line_count = len(cell.source.splitlines())

                if cell.cell_type == 'code':
                    outline_items = self._extract_code_outline(cell.source)
                elif cell.cell_type == 'markdown':
                    outline_items = self._extract_markdown_outline(cell.source)

                # Ensure there's at least one outline item (context if needed)
                if not outline_items:
                    outline_items = self._get_first_line_context(cell.source)

                cell_info: Dict[str, Union[int, str, List[str]]] = {
                    "index": index,
                    "type": cell.cell_type,
                    "line_count": line_count,
                    "outline": outline_items
                }
                structure_map.append(cell_info)

            logger.info(f"{log_prefix} SUCCESS - Generated outline ({len(structure_map)} cells analyzed).")
            return structure_map # Return the structured list

        except (ValueError, FileNotFoundError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}") from e

    # --- Helper methods for outline generation --- 

    def _extract_code_outline(self, source: str) -> List[str]:
        """Extracts functions, classes, and comment headings from code."""
        outline = []
        # First pass for comment headings
        try:
            lines = source.splitlines()
            for line in lines:
                match = re.match(r'^\s*#\s+(.*)', line) # Match comments like '# Heading'
                if match and match.group(1):
                     outline.append(f"comment: {match.group(1).strip()}")
        except Exception as e:
            logger.warning(f"Error parsing comments for outline: {e}")
            # Continue to AST parsing even if comment parsing fails

        # Second pass for AST elements (functions, classes)
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    outline.append(f"func: {node.name}")
                elif isinstance(node, ast.ClassDef):
                    outline.append(f"class: {node.name}")
        except SyntaxError:
            # If syntax is invalid, AST parsing fails. Add indicator.
            # Avoid adding if already found comment headings.
            if not any(item.startswith("comment:") for item in outline):
                 outline.append("<Syntax Error>")
            # Keep any comment headings found before the error
        except Exception as e:
            # Catch other potential AST parsing errors
            if not outline: # Avoid adding if we already have items
                outline.append(f"<AST Parsing Error: {e}>")
        return outline

    def _extract_markdown_outline(self, source: str) -> List[str]:
        """Extracts markdown headings (H1, H2, etc.) and HTML headings (h1-h6) from markdown."""
        headings = []
        # Regex to find HTML headings like <h1...>...</h1>, capturing level and content
        # Handles attributes in the opening tag and ignores case for tag names
        html_heading_re = re.compile(r'<h([1-6])[^>]*>(.*?)</h\1>', re.IGNORECASE | re.DOTALL)
        try:
            lines = source.split('\n')
            for line in lines:
                stripped_line = line.strip()
                # Check for Markdown heading first
                md_match = re.match(r'^(#+)\s*(.*)', stripped_line) # Allow zero spaces after #
                if md_match:
                    level = len(md_match.group(1))
                    heading_text = md_match.group(2).strip()
                    if heading_text:
                        headings.append(f"H{level}: {heading_text}")
                else:
                    # If not Markdown, check for HTML headings using finditer for multiple matches per line
                    for html_match in html_heading_re.finditer(stripped_line):
                        level = int(html_match.group(1))
                        # Basic cleanup: remove potential inner tags for outline brevity
                        heading_text = re.sub(r'<.*?>', '', html_match.group(2)).strip()
                        if heading_text:
                            headings.append(f"H{level}: {heading_text}")

        except AttributeError:
             headings.append("<Missing Source>") # Should be rare
        except Exception as e:
             headings.append(f"<Markdown Parsing Error: {e}>")
        return headings

    def _get_first_line_context(self, source: str) -> List[str]:
        """Gets the first non-empty line as context if no other outline found."""
        try:
            for line in source.splitlines():
                stripped_line = line.strip()
                if stripped_line:
                    # Truncate long lines for brevity
                    context = stripped_line[:100] + ('...' if len(stripped_line) > 100 else '')
                    return [f"context: {context}"]
            # If loop finishes without finding a non-empty line
            return ["<Empty Cell>"]
        except Exception as e:
            logger.warning(f"Error getting first line context: {e}")
            return ["<Error getting context>"] 

    async def notebook_search(
        self,
        notebook_path: str,
        query: str
    ) -> List[Dict[str, Union[int, str]]]:
        """Searches within a notebook's code and markdown cells for a query string.

        Performs a case-insensitive search within the source of each cell.

        Args:
            notebook_path: Absolute path to the .ipynb notebook file.
            query: The string to search for.

        Returns:
            A list of dictionaries, where each dictionary represents a match:
            - 'cell_index': The 0-based index of the matching cell.
            - 'cell_type': The type of the matching cell ('code' or 'markdown').
            - 'match_line_number': The 1-based line number within the cell where the match occurred.
            - 'snippet': A truncated snippet of the line containing the match.
        """
        log_prefix = self._log_prefix('notebook_search', path=notebook_path, query=query)
        logger.info(f"{log_prefix} Called.")
        
        if not query:
            raise ValueError("Search query cannot be empty.")
            
        results: List[Dict[str, Union[int, str]]] = []
        try:
            nb = await self.read_notebook(notebook_path, self.config.allowed_roots)
            query_lower = query.lower()
            MAX_SNIPPET_LEN = 150 # Max length for the snippet

            for index, cell in enumerate(nb.cells):
                try:
                    source = cell.source
                    cell_type = cell.cell_type
                    lines = source.splitlines()
                    for line_num_0based, line in enumerate(lines):
                        if query_lower in line.lower():
                            line_num_1based = line_num_0based + 1
                            # Create snippet, truncating if necessary
                            snippet = line.strip()
                            if len(snippet) > MAX_SNIPPET_LEN:
                                # Try to find query position for better truncation
                                try:
                                    match_start = snippet.lower().index(query_lower)
                                    start = max(0, match_start - MAX_SNIPPET_LEN // 3)
                                    end = min(len(snippet), match_start + len(query) + (MAX_SNIPPET_LEN * 2 // 3))
                                    prefix = "..." if start > 0 else ""
                                    suffix = "..." if end < len(snippet) else ""
                                    snippet = prefix + snippet[start:end] + suffix
                                except ValueError:
                                    # Fallback if query not found after lowercasing (shouldn't happen often)
                                    snippet = snippet[:MAX_SNIPPET_LEN] + "..."
                            
                            results.append({
                                "cell_index": index,
                                "cell_type": cell_type,
                                "match_line_number": line_num_1based,
                                "snippet": snippet
                            })
                except AttributeError:
                    # Skip cells that might unexpectedly lack a source attribute
                    logger.warning(f"{log_prefix} Skipping cell {index} due to missing source.")
                    continue
                except Exception as cell_err:
                    # Log error processing a specific cell but continue searching others
                    logger.error(f"{log_prefix} Error processing cell {index}: {cell_err}")
                    continue # Continue to the next cell

            # <<< Add the check for empty results here >>>
            if not results:
                logger.info(f"{log_prefix} SUCCESS - No matches found.")
                # Return a specific message instead of an empty list
                return [{"message": "No matches found"}] 
            else:
                logger.info(f"{log_prefix} SUCCESS - Found {len(results)} match(es).")
                return results

        except (ValueError, FileNotFoundError, IOError, PermissionError) as e:
            logger.error(f"{log_prefix} FAILED - Specific error: {e}")
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred during notebook search: {e}") from e 