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
import asyncio # Ensure asyncio is imported
import tempfile
import posixpath

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
        # Get SFTP manager from config if available
        self.sftp_manager = getattr(config, 'sftp_manager', None)

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
            self.notebook_edit_cell_output,
            self.notebook_bulk_add_cells,
            self.notebook_get_server_path_context,
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
    
    # Helper to access config with type check
    def _get_allowed_local_roots(self) -> List[str]:
        # ServerConfig stores combined roots in allowed_roots.
        # We need to differentiate based on sftp_manager presence?
        # For simplicity now, assume config.allowed_roots contains *all* roots
        # (local ones from --allow-root and local temp paths for --sftp-root).
        # notebook_ops.resolve_path_and_check_permissions handles the logic.
        if hasattr(self.config, 'allowed_roots') and isinstance(self.config.allowed_roots, list):
            return self.config.allowed_roots
        return [] # Return empty list if not configured

    async def notebook_create(self, notebook_path: str) -> str:
        """Creates a new, empty Jupyter Notebook (.ipynb) file at the specified path.

        Parameters
        ----------
        notebook_path : str
            Path to create the notebook. Can be:
            - Absolute path on local filesystem (e.g., '/path/to/notebook.ipynb')
            - Relative path from allowed root (e.g., 'notebook.ipynb')
            - Remote path if SFTP is enabled (e.g., '~/notebook.ipynb' or '/home/user/notebook.ipynb')
            
            For SFTP sessions, both absolute paths on the remote server and relative 
            paths from the SFTP root directory are supported. Tilde ('~') expansion is supported.

        Returns
        -------
        str
            Success message with the path of the created notebook.

        Raises
        ------
        ValueError
            If the path doesn't end with '.ipynb' or cannot be resolved.
        PermissionError
            If the path is outside allowed workspace roots.
        FileExistsError
            If the notebook already exists.
        IOError
            If writing the file fails.
        ConnectionError
            If SFTP manager is required but not available for remote existence check.

        Notes
        -----
        When working with remote notebooks via SFTP:
        - Parent directories will be created automatically if they don't exist.
        - Use relative paths, absolute paths, or paths starting with '~'.
        """
        log_prefix = self._log_prefix('notebook_create', path=notebook_path)
        logger.info(f"{log_prefix} Called.")

        try:
            # Basic validation
            if not notebook_path or not isinstance(notebook_path, str):
                 raise ValueError("Invalid notebook path provided.")
            if not notebook_path.endswith(".ipynb"):
                raise ValueError(f"Invalid file type: '{notebook_path}' must point to a .ipynb file.")

            # Resolve path and check permissions using the new central function
            # Pass the original user path here.
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )
            logger.debug(f"Permissions checked for '{notebook_path}', resolved to '{absolute_op_path}', is_remote={is_remote}")

            # Check if file exists using the absolute operation path
            file_exists = False
            if is_remote:
                if not self.sftp_manager:
                     raise ConnectionError("SFTP manager required but not available for remote existence check.")
                file_exists = await asyncio.to_thread(self.sftp_manager.path_exists, absolute_op_path)
            else:
                file_exists = os.path.exists(absolute_op_path)

            if file_exists:
                # Use the resolved absolute path in the error for clarity
                raise FileExistsError(f"Cannot create notebook, file already exists: {absolute_op_path}")

            # --- Create and Write --- 
            nb = nbformat.v4.new_notebook()
            # Call write_notebook with the ORIGINAL user path.
            # write_notebook now uses the same central resolution logic.
            await self.write_notebook(notebook_path, nb, self._get_allowed_local_roots(), self.sftp_manager)

            # Log success with original path
            logger.info(f"{log_prefix} SUCCESS - Created new notebook: {notebook_path} (at {absolute_op_path})")
            return f"Successfully created new notebook: {notebook_path}"

        except (PermissionError, FileExistsError, ValueError, IOError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise # Re-raise specific, expected errors
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error during creation: {e}")
            raise RuntimeError(f"An unexpected error occurred during notebook creation for '{notebook_path}': {e}") from e

    async def notebook_delete(self, notebook_path: str) -> str:
        """Deletes a Jupyter Notebook (.ipynb) file at the specified path.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file to delete. Can be:
            - Absolute path on local filesystem (e.g., '/path/to/notebook.ipynb')
            - Relative path from allowed root (e.g., 'notebook.ipynb')
            - Remote path if SFTP is enabled (e.g., '/home/user/notebook.ipynb', '~/notebook.ipynb')

        Returns
        -------
        str
            Success message with the path of the deleted notebook.

        Raises
        ------
        ValueError
            If the path doesn't end with '.ipynb' or cannot be resolved.
        PermissionError
            If the path is outside allowed workspace roots.
        FileNotFoundError
            If the notebook doesn't exist at the resolved path.
        IOError
            If the file deletion fails.
        ConnectionError
            If SFTP is required but unavailable.

        Notes
        -----
        - The deletion is permanent and cannot be undone.
        - For remote notebooks via SFTP, the file is deleted on the remote server.
        """
        log_prefix = self._log_prefix('notebook_delete', path=notebook_path)
        logger.info(f"{log_prefix} Called.")

        try:
            # Basic validation
            if not notebook_path or not isinstance(notebook_path, str):
                 raise ValueError("Invalid notebook path provided.")
            if not notebook_path.endswith(".ipynb"):
                raise ValueError(f"Invalid file type: '{notebook_path}' must point to a .ipynb file.")

            # Resolve path and check permissions using the new central function
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )
            logger.debug(f"{log_prefix} Resolved to '{absolute_op_path}', is_remote={is_remote}")

            # Check if file exists before attempting deletion (optional, remove is idempotent)
            exists = False
            if is_remote:
                if not self.sftp_manager:
                     raise ConnectionError("SFTP manager required but not available for remote operation.")
                logger.debug(f"{log_prefix} Checking existence of remote absolute_op_path: '{absolute_op_path}' via sftp_manager.path_exists (bypassing cache)")
                exists = await asyncio.to_thread(self.sftp_manager.path_exists, absolute_op_path, bypass_cache=True)
                logger.debug(f"{log_prefix} Remote absolute_op_path '{absolute_op_path}' exists: {exists}")
            else:
                logger.debug(f"{log_prefix} Checking existence of local absolute_op_path: '{absolute_op_path}'")
                exists = os.path.exists(absolute_op_path)
                logger.debug(f"{log_prefix} Local absolute_op_path '{absolute_op_path}' exists: {exists}")

            if not exists:
                # Standard practice is for delete to be idempotent, so often no error if not found.
                # However, to match original stricter behavior / user expectation from logs:
                logger.warning(f"{log_prefix} File not found at resolved path: {absolute_op_path}")
                raise FileNotFoundError(f"Notebook file not found at: {absolute_op_path}")

            # Perform the delete operation
            if is_remote:
                await asyncio.to_thread(self.sftp_manager.remove_file, absolute_op_path)
            else:
                await asyncio.to_thread(os.remove, absolute_op_path)
            
            logger.info(f"{log_prefix} SUCCESS - Deleted notebook: {notebook_path} (from {absolute_op_path})")
            return f"Successfully deleted notebook: {notebook_path}"

        except (ValueError, PermissionError, FileNotFoundError, IOError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise # Re-raise specific, expected errors
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred during notebook deletion for '{notebook_path}': {e}") from e

    async def notebook_rename(self, old_path: str, new_path: str) -> str:
        """Renames/Moves a Jupyter Notebook (.ipynb) file from one path to another.

        Parameters
        ----------
        old_path : str
            Path to the existing notebook file. Can be relative, absolute, or use '~'.
        new_path : str
            Path where the notebook file should be moved/renamed to. Can be relative, absolute, or use '~'.

        Returns
        -------
        str
            Success message with both original old and new paths.

        Raises
        ------
        ValueError
            If either path is invalid (e.g., doesn't end with .ipynb) or paths resolve to different storage types (local vs remote).
        PermissionError
            If either path resolves outside allowed workspace roots.
        FileNotFoundError
            If the source notebook doesn't exist at the resolved path.
        FileExistsError
            If a file already exists at the resolved destination path.
        IOError
            If the file rename/move operation fails.
        ConnectionError
            If SFTP is required but unavailable or paths are on different hosts.

        Notes
        -----
        - The parent directory of the destination path will be created if it doesn't exist.
        - Both paths must resolve to the same storage type (both local or both remote on the same host).
        """
        log_prefix = self._log_prefix('notebook_rename', old=old_path, new=new_path)
        logger.info(f"{log_prefix} Called.")

        try:
            # Basic validation
            if not old_path or not old_path.endswith(".ipynb") or \
               not new_path or not new_path.endswith(".ipynb"):
                raise ValueError("Invalid old or new path provided. Both must exist and end with .ipynb")

            # Resolve and check permissions for BOTH paths
            is_remote_old, abs_old_path = notebook_ops.resolve_path_and_check_permissions(
                old_path, self._get_allowed_local_roots(), self.sftp_manager
            )
            logger.debug(f"{log_prefix} Resolved old_path '{old_path}' to abs_old_path: '{abs_old_path}', is_remote: {is_remote_old}")

            is_remote_new, abs_new_path = notebook_ops.resolve_path_and_check_permissions(
                new_path, self._get_allowed_local_roots(), self.sftp_manager
            )
            logger.debug(f"{log_prefix} Resolved new_path '{new_path}' to abs_new_path: '{abs_new_path}', is_remote: {is_remote_new}")

            if is_remote_old != is_remote_new:
                 raise ValueError("Cannot rename/move between local and remote storage.")

            # Check existence of old path and non-existence of new path
            old_exists = False
            new_exists = False
            if is_remote_old:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                logger.debug(f"{log_prefix} Checking existence of remote abs_old_path: '{abs_old_path}' via sftp_manager.path_exists (bypassing cache)")
                old_exists = await asyncio.to_thread(self.sftp_manager.path_exists, abs_old_path, bypass_cache=True)
                logger.debug(f"{log_prefix} Remote abs_old_path '{abs_old_path}' exists: {old_exists}")
                
                logger.debug(f"{log_prefix} Checking existence of remote abs_new_path: '{abs_new_path}' via sftp_manager.path_exists (bypassing cache)")
                new_exists = await asyncio.to_thread(self.sftp_manager.path_exists, abs_new_path, bypass_cache=True)
                logger.debug(f"{log_prefix} Remote abs_new_path '{abs_new_path}' exists: {new_exists}")
            else:
                logger.debug(f"{log_prefix} Checking existence of local abs_old_path: '{abs_old_path}'")
                old_exists = os.path.exists(abs_old_path)
                logger.debug(f"{log_prefix} Local abs_old_path '{abs_old_path}' exists: {old_exists}")

                logger.debug(f"{log_prefix} Checking existence of local abs_new_path: '{abs_new_path}'")
                new_exists = os.path.exists(abs_new_path)
                logger.debug(f"{log_prefix} Local abs_new_path '{abs_new_path}' exists: {new_exists}")

            if not old_exists:
                raise FileNotFoundError(f"Source notebook file not found at: {abs_old_path}")
            if new_exists:
                raise FileExistsError(f"Cannot rename notebook, destination already exists: {abs_new_path}")

            # Perform the rename using the absolute paths
            if is_remote_old:
                await asyncio.to_thread(self.sftp_manager.rename_file, abs_old_path, abs_new_path)
            else:
                await asyncio.to_thread(os.rename, abs_old_path, abs_new_path)

            logger.info(f"{log_prefix} SUCCESS - Renamed {abs_old_path} to {abs_new_path}")
            # Return original user paths in the success message
            return f"Successfully renamed notebook from {old_path} to {new_path}"

        except (ValueError, PermissionError, FileNotFoundError, FileExistsError, IOError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error during rename: {e}")
            raise RuntimeError(f"An unexpected error occurred during notebook rename: {e}") from e

    async def notebook_edit_cell(self, notebook_path: str, cell_index: int, source: str) -> str:
        """Replaces the source content of a specific cell in a Jupyter Notebook.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').
        cell_index : int
            The 0-based index of the cell to edit.
        source : str
            The new source content for the cell.

        Returns
        -------
        str
            Success message with cell index and original notebook path.

        Raises
        ------
        ValueError
            If source content exceeds maximum allowed size or path is invalid.
        IndexError
            If cell_index is out of bounds.
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading or writing the notebook fails.
        ConnectionError
            If SFTP is required but unavailable.

        Notes
        -----
        - For markdown cells containing LaTeX, use '$ ... $' for inline math and '$$ ... $$' for display math.
        """
        log_prefix = self._log_prefix('notebook_edit_cell', path=notebook_path, index=cell_index)
        logger.info(f"{log_prefix} Called.")
        try:
            # Validate source size
            if len(source.encode('utf-8')) > self.config.max_cell_source_size:
                raise ValueError(f"Source content exceeds maximum allowed size ({self.config.max_cell_source_size} bytes).")

            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Edit cell
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")

            nb.cells[cell_index].source = source

            # 4. Write notebook
            if is_remote:
                json_content = nbformat.writes(nb)
                await asyncio.to_thread(self.sftp_manager.write_file, absolute_op_path, json_content)
            else:
                await asyncio.to_thread(nbformat.write, nb, absolute_op_path)

            logger.info(f"{log_prefix} SUCCESS - Edited cell {cell_index} in {absolute_op_path}.")
            return f"Successfully edited cell {cell_index} in {notebook_path}"

        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred editing cell in '{notebook_path}': {e}") from e

    async def notebook_add_cell(self, notebook_path: str, cell_type: str, source: str, insert_after_index: int) -> str:
        """Adds a new cell to a Jupyter Notebook after the specified index.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').
        cell_type : str
            Type of cell ('code' or 'markdown').
        source : str
            The source content for the new cell.
        insert_after_index : int
            The 0-based index after which to insert the new cell (-1 for beginning).

        Returns
        -------
        str
            Success message with the new cell's insertion index and original path.

        Raises
        ------
        ValueError
            If cell_type is invalid or source content exceeds maximum allowed size.
        IndexError
            If insert_after_index results in an invalid insertion index.
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading or writing the notebook fails.
        ConnectionError
            If SFTP is required but unavailable.

        Notes
        -----
        - For markdown cells containing LaTeX, use '$ ... $' for inline math and '$$ ... $$' for display math.
        """
        log_prefix = self._log_prefix('notebook_add_cell', path=notebook_path, type=cell_type, after_index=insert_after_index)
        logger.info(f"{log_prefix} Called.")
        try:
            # Validate source size before path resolution
            if len(source.encode('utf-8')) > self.config.max_cell_source_size:
                raise ValueError(f"Source content exceeds maximum allowed size ({self.config.max_cell_source_size} bytes).")

            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read the notebook using the correct method
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Perform the cell addition logic
            if cell_type == 'code':
                new_cell = nbformat.v4.new_code_cell(source)
            elif cell_type == 'markdown':
                new_cell = nbformat.v4.new_markdown_cell(source)
            else:
                raise ValueError("Invalid cell_type: Must be 'code' or 'markdown'.")

            insertion_index = insert_after_index + 1
            if not 0 <= insertion_index <= len(nb.cells):
                 # Use the actual number of cells read for the error message
                 raise IndexError(f"Insertion index {insertion_index} (based on insert_after_index {insert_after_index}) is out of bounds (0-{len(nb.cells)}).")

            nb.cells.insert(insertion_index, new_cell)

            # 4. Write the modified notebook back
            if is_remote:
                json_content = nbformat.writes(nb)
                await asyncio.to_thread(self.sftp_manager.write_file, absolute_op_path, json_content)
            else:
                await asyncio.to_thread(nbformat.write, nb, absolute_op_path)

            logger.info(f"{log_prefix} SUCCESS - Added {cell_type} cell at index {insertion_index} to {absolute_op_path}")
            # Return message using original path for user consistency
            return f"Successfully added {cell_type} cell at index {insertion_index} in {notebook_path}"

        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred adding cell to '{notebook_path}': {e}") from e

    async def notebook_delete_cell(self, notebook_path: str, cell_index: int) -> str:
        """Deletes a specific cell from a Jupyter Notebook.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').
        cell_index : int
            The 0-based index of the cell to delete.

        Returns
        -------
        str
            Success message with the deleted cell index and original path.

        Raises
        ------
        IndexError
            If cell_index is out of bounds.
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading or writing the notebook fails.
        ValueError
            If the file is not a valid notebook (e.g., during read).
        ConnectionError
            If SFTP is required but unavailable.

        Notes
        -----
        - Cell deletion is permanent.
        """
        log_prefix = self._log_prefix('notebook_delete_cell', path=notebook_path, index=cell_index)
        logger.info(f"{log_prefix} Called.")
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Delete cell
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")

            del nb.cells[cell_index]

            # 4. Write notebook
            if is_remote:
                json_content = nbformat.writes(nb)
                await asyncio.to_thread(self.sftp_manager.write_file, absolute_op_path, json_content)
            else:
                await asyncio.to_thread(nbformat.write, nb, absolute_op_path)

            logger.info(f"{log_prefix} SUCCESS - Deleted cell {cell_index} from {absolute_op_path}.")
            return f"Successfully deleted cell {cell_index} from {notebook_path}"

        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred deleting cell from '{notebook_path}': {e}") from e

    async def notebook_read_cell(self, notebook_path: str, cell_index: int) -> str:
        """Reads the source content of a specific cell from a Jupyter Notebook.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').
        cell_index : int
            The 0-based index of the cell to read.

        Returns
        -------
        str
            The source content of the specified cell (potentially truncated).

        Raises
        ------
        IndexError
            If cell_index is out of bounds.
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading the notebook fails.
        ValueError
             If the file is not a valid notebook (e.g., during read).
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_read_cell', path=notebook_path, index=cell_index)
        logger.info(f"{log_prefix} Called.")
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Extract cell source
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")

            source = nb.cells[cell_index].source
            logger.info(f"{log_prefix} SUCCESS - Read cell {cell_index} from {absolute_op_path}.")

            # Apply size limit for safety
            MAX_LEN_BYTES = self.config.max_cell_source_size
            if len(source.encode('utf-8')) > MAX_LEN_BYTES:
                 logger.warning(f"{log_prefix} WARNING - Source content truncated ({MAX_LEN_BYTES} byte limit).")
                 encoded_source = source.encode('utf-8')
                 truncated_bytes = encoded_source[:MAX_LEN_BYTES]
                 try:
                     source = truncated_bytes.decode('utf-8', errors='ignore') + "... (truncated)"
                 except UnicodeDecodeError:
                     source = "[Source truncated - unable to decode cleanly]"
            return source

        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred reading cell from '{notebook_path}': {e}") from e

    async def notebook_get_cell_count(self, notebook_path: str) -> int:
        """Returns the total number of cells in the notebook.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').

        Returns
        -------
        int
            The total number of cells in the notebook.

        Raises
        ------
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading the notebook fails.
        ValueError
             If the file is not a valid notebook (e.g., during read).
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_get_cell_count', path=notebook_path)
        logger.info(f"{log_prefix} Called.")
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Get cell count
            count = len(nb.cells)
            logger.info(f"{log_prefix} SUCCESS - Count: {count} for {absolute_op_path}")
            return count

        except (ValueError, FileNotFoundError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred getting cell count for '{notebook_path}': {e}") from e

    async def notebook_read_metadata(self, notebook_path: str) -> dict:
        """Reads the top-level metadata of the notebook.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').

        Returns
        -------
        dict
            Dictionary containing the notebook's metadata.

        Raises
        ------
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading the notebook fails.
        ValueError
             If the file is not a valid notebook (e.g., during read).
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_read_metadata', path=notebook_path)
        logger.info(f"{log_prefix} Called.")
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Extract metadata
            metadata = dict(nb.metadata)
            logger.info(f"{log_prefix} SUCCESS - Read metadata from {absolute_op_path}.")
            return metadata

        except (ValueError, FileNotFoundError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred reading metadata from '{notebook_path}': {e}") from e

    async def notebook_edit_metadata(self, notebook_path: str, metadata_updates: dict) -> str:
        """Updates the top-level metadata of the notebook.

        If `metadata_updates` is an empty dictionary `{}`, the notebook's metadata
        will be reset to the default metadata of a new notebook (which includes
        default `kernelspec` and `language_info`). Otherwise, the provided updates
        are merged into the existing metadata.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').
        metadata_updates : dict
            Dictionary containing metadata fields to update or add.

        Returns
        -------
        str
            Success message confirming metadata was updated.

        Raises
        ------
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading or writing the notebook fails.
        ValueError
             If the file is not a valid notebook (e.g., during read).
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_edit_metadata', path=notebook_path)
        logger.info(f"{log_prefix} Called with updates: {metadata_updates}")
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Update metadata
            if not metadata_updates: # Handles {} or None, though type hint is dict
                default_nb = nbformat.v4.new_notebook()
                nb.metadata = default_nb.metadata
                logger.info(f"{log_prefix} Cleared notebook metadata to defaults.")
            else:
                nb.metadata.update(metadata_updates)
                logger.info(f"{log_prefix} Updated notebook metadata.")

            # 4. Write notebook
            if is_remote:
                json_content = nbformat.writes(nb)
                await asyncio.to_thread(self.sftp_manager.write_file, absolute_op_path, json_content)
            else:
                await asyncio.to_thread(nbformat.write, nb, absolute_op_path)

            logger.info(f"{log_prefix} SUCCESS - Updated metadata for {absolute_op_path}.")
            return f"Successfully updated metadata for {notebook_path}"

        except (ValueError, FileNotFoundError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred updating metadata for '{notebook_path}': {e}") from e

    async def notebook_read_cell_metadata(self, notebook_path: str, cell_index: int) -> dict:
        """Reads the metadata of a specific cell.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').
        cell_index : int
            The 0-based index of the cell to read metadata from.

        Returns
        -------
        dict
            Dictionary containing the cell's metadata.

        Raises
        ------
        IndexError
            If cell_index is out of bounds.
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading the notebook fails.
        ValueError
             If the file is not a valid notebook (e.g., during read).
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_read_cell_metadata', path=notebook_path, index=cell_index)
        logger.info(f"{log_prefix} Called.")
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Extract cell metadata
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")
            metadata = dict(nb.cells[cell_index].metadata)
            logger.info(f"{log_prefix} SUCCESS - Read metadata for cell {cell_index} from {absolute_op_path}.")
            return metadata

        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred reading cell metadata from '{notebook_path}': {e}") from e

    async def notebook_edit_cell_metadata(self, notebook_path: str, cell_index: int, metadata_updates: dict) -> str:
        """Updates the metadata of a specific cell.

        If `metadata_updates` is an empty dictionary `{}`, the cell's metadata
        will be cleared to an empty dictionary. Otherwise, the provided updates
        are merged into the existing cell metadata.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').
        cell_index : int
            The 0-based index of the cell to update metadata for.
        metadata_updates : dict
            Dictionary containing metadata fields to update or add.

        Returns
        -------
        str
            Success message confirming cell metadata was updated.

        Raises
        ------
        IndexError
            If cell_index is out of bounds.
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading or writing the notebook fails.
        ValueError
             If the file is not a valid notebook (e.g., during read).
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_edit_cell_metadata', path=notebook_path, index=cell_index)
        logger.info(f"{log_prefix} Called with updates: {metadata_updates}")
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Update cell metadata
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")
            
            cell = nb.cells[cell_index]
            if not metadata_updates: # Handles {} or None
                cell.metadata = {}
                logger.info(f"{log_prefix} Cleared metadata for cell {cell_index}.")
            else:
                cell.metadata.update(metadata_updates)
                logger.info(f"{log_prefix} Updated metadata for cell {cell_index}.")

            # 4. Write notebook
            if is_remote:
                json_content = nbformat.writes(nb)
                await asyncio.to_thread(self.sftp_manager.write_file, absolute_op_path, json_content)
            else:
                await asyncio.to_thread(nbformat.write, nb, absolute_op_path)

            logger.info(f"{log_prefix} SUCCESS - Updated metadata for cell {cell_index} in {absolute_op_path}.")
            return f"Successfully updated metadata for cell {cell_index} in {notebook_path}"

        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred updating cell metadata for '{notebook_path}': {e}") from e

    async def notebook_clear_cell_outputs(self, notebook_path: str, cell_index: int) -> str:
        """Clears the output(s) of a specific code cell in a Jupyter Notebook.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').
        cell_index : int
            0-based index of the cell to clear outputs from.
            
        Returns
        -------
        str
            Success message confirming the cell's outputs were cleared or stating no action was needed.
            
        Raises
        ------
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading or writing the notebook fails.
        ValueError
            If the file is not a valid notebook (e.g., during read).
        IndexError
            If the cell_index is out of range.
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_clear_cell_outputs', path=notebook_path, index=cell_index)
        logger.info(f"{log_prefix} Called.")
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Clear outputs
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")

            cell = nb.cells[cell_index]
            changed = False
            if cell.cell_type == 'code': # Only clear for code cells
                if hasattr(cell, 'outputs') and cell.outputs:
                    cell.outputs = []
                    changed = True
                if hasattr(cell, 'execution_count') and cell.execution_count is not None:
                    cell.execution_count = None
                    changed = True

            # 4. Write notebook only if changed
            if changed:
                if is_remote:
                    json_content = nbformat.writes(nb)
                    await asyncio.to_thread(self.sftp_manager.write_file, absolute_op_path, json_content)
                else:
                    await asyncio.to_thread(nbformat.write, nb, absolute_op_path)
                logger.info(f"{log_prefix} SUCCESS - Cleared outputs for cell {cell_index} in {absolute_op_path}.")
                return f"Successfully cleared outputs for cell {cell_index} in {notebook_path}"
            else:
                logger.info(f"{log_prefix} SUCCESS - No outputs/count to clear for cell {cell_index} (type: {cell.cell_type}) in {absolute_op_path}.")
                return f"No outputs or execution count found to clear for cell {cell_index} in {notebook_path}"

        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred clearing cell outputs for '{notebook_path}': {e}") from e

    async def notebook_clear_all_outputs(self, notebook_path: str) -> str:
        """Clears all outputs from all code cells in a Jupyter Notebook.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').
            
        Returns
        -------
        str
            Success message indicating how many cells were cleared or stating no action was needed.
            
        Raises
        ------
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading or writing the notebook fails.
        ValueError
            If the file is not a valid notebook (e.g., during read).
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_clear_all_outputs', path=notebook_path)
        logger.info(f"{log_prefix} Called.")
        cleared_count = 0
        changed = False
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Clear outputs for all code cells
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

            # 4. Write notebook only if changed
            if changed:
                 if is_remote:
                     json_content = nbformat.writes(nb)
                     await asyncio.to_thread(self.sftp_manager.write_file, absolute_op_path, json_content)
                 else:
                     await asyncio.to_thread(nbformat.write, nb, absolute_op_path)
                 logger.info(f"{log_prefix} SUCCESS - Cleared outputs for {cleared_count} cells in {absolute_op_path}.")
                 return f"Successfully cleared outputs for {cleared_count} code cells in {notebook_path}"
            else:
                 logger.info(f"{log_prefix} SUCCESS - No outputs needed clearing in {absolute_op_path}.")
                 return f"No code cell outputs found to clear in {notebook_path}"

        except (ValueError, FileNotFoundError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred clearing all outputs for '{notebook_path}': {e}") from e

    async def notebook_move_cell(self, notebook_path: str, cell_index: int, new_index: int) -> str:
        """Moves a specific cell from one index to another in a Jupyter Notebook.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').
        cell_index : int
            The 0-based index of the cell to move.
        new_index : int
            The 0-based index to which the cell should be moved.

        Returns
        -------
        str
            Success message with the moved cell's original and new indices.

        Raises
        ------
        IndexError
            If cell_index or new_index is out of bounds.
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading or writing the notebook fails.
        ValueError
            If the file is not a valid notebook (e.g., during read).
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_move_cell', path=notebook_path, from_index=cell_index, to_index=new_index)
        logger.info(f"{log_prefix} Called.")
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Move cell
            num_cells = len(nb.cells)
            if not 0 <= cell_index < num_cells or not 0 <= new_index < num_cells:
                raise IndexError(f"Cell index {cell_index} or new index {new_index} is out of bounds (0-{num_cells-1}).")

            if cell_index == new_index:
                 logger.info(f"{log_prefix} SUCCESS - Cell index {cell_index} and new index {new_index} are the same. No move needed.")
                 return f"Cell {cell_index} is already at index {new_index} in {notebook_path}."

            cell = nb.cells.pop(cell_index)
            nb.cells.insert(new_index, cell)

            # 4. Write notebook
            if is_remote:
                json_content = nbformat.writes(nb)
                await asyncio.to_thread(self.sftp_manager.write_file, absolute_op_path, json_content)
            else:
                await asyncio.to_thread(nbformat.write, nb, absolute_op_path)

            logger.info(f"{log_prefix} SUCCESS - Moved cell from {cell_index} to {new_index} in {absolute_op_path}.")
            return f"Successfully moved cell from {cell_index} to {new_index} in {notebook_path}"

        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred moving cell in '{notebook_path}': {e}") from e

    async def notebook_export(self, notebook_path: str, export_format: str) -> str:
        """Exports a notebook to a specified format using nbconvert.

        The output file is placed in the same directory as the input notebook.
        For remote notebooks (SFTP), the output is placed in the corresponding 
        local temporary directory structure, and the absolute local path is returned.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file to export (relative, absolute, or '~').
        export_format : str
            Desired output format (e.g., 'python', 'html', 'pdf').

        Returns
        -------
        str
            Success message with the absolute path to the exported file.

        Raises
        ------
        ValueError
            If the export format is unsupported or path resolution fails.
        FileNotFoundError
            If the input notebook doesn't exist or the output file isn't created.
        PermissionError
            If the resolved input or output path is disallowed.
        IOError
            If reading/writing fails during remote download/local save.
        ConnectionError
            If SFTP is required but unavailable.
        RuntimeError
            If the nbconvert subprocess fails or times out.
        """
        log_prefix = self._log_prefix('notebook_export', path=notebook_path, format=export_format)
        logger.info(f"{log_prefix} Called.")
        
        supported_formats = ['python', 'script', 'html', 'markdown', 'pdf', 'latex', 'rst']
        if export_format not in supported_formats:
            raise ValueError(f"Unsupported export format: '{export_format}'. Supported: {supported_formats}")
        
        temp_input_file = None # Keep track of temporary file for cleanup
        try:
            # 1. Resolve input notebook path and check permissions
            # absolute_op_path here is the absolute path for the operation (remote or local)
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Determine input path for nbconvert & output directory related components
            nbconvert_input_path = absolute_op_path # This is the initial path for nbconvert (local or remote string)
            # output_dir_for_nbconvert will be where nbconvert actually writes.
            # output_basename will be used to construct the output filename.
            output_dir_for_nbconvert = os.path.dirname(absolute_op_path) # Default for local path
            output_basename = os.path.basename(absolute_op_path)       # Default for local path

            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required for remote export prep.")
                logger.debug(f"Downloading remote file {absolute_op_path} to temporary location for export.")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb") as tmp_f:
                    tmp_f.write(content_bytes)
                    temp_input_file = tmp_f.name
                nbconvert_input_path = temp_input_file # nbconvert reads from this local temp file
                logger.debug(f"Using temporary file {nbconvert_input_path} for nbconvert input.")

                # For remote files, nbconvert writes to the directory of the local temp input file.
                output_dir_for_nbconvert = os.path.dirname(temp_input_file)
                # The basename for the output file still comes from the original remote path.
                output_basename = os.path.basename(absolute_op_path) 
                logger.debug(f"SFTP: nbconvert output directory set to: {output_dir_for_nbconvert}, basename from: {output_basename}")
            
            # 3. Construct the absolute output path for nbconvert
            output_filename_base = os.path.splitext(output_basename)[0]
            ext_map = { 'python': '.py', 'script': '.py', 'html': '.html', 'markdown': '.md', 'pdf': '.pdf', 'latex': '.tex', 'rst': '.rst' }
            output_ext = ext_map.get(export_format)
            if not output_ext:
                 # Should not happen due to check above, but defensive
                 raise ValueError(f"Internal error: Could not map format '{export_format}' to extension.") 
            
            # Output path is in the resolved directory (local original or local temp)
            absolute_output_path = os.path.join(output_dir_for_nbconvert, output_filename_base + output_ext)
            logger.debug(f"Determined absolute output path: {absolute_output_path}")

            # Ensure output directory exists (important for local temp dirs too)
            if output_dir_for_nbconvert:
                 os.makedirs(output_dir_for_nbconvert, exist_ok=True)

            # 4. Construct and run nbconvert command
            cmd = [
                 sys.executable,
                 "-m", "nbconvert", 
                 "--to", export_format, 
                 nbconvert_input_path, # Use local original or local temp path
                 "--output", os.path.splitext(absolute_output_path)[0] # nbconvert adds extension, provide base
                 # Alternatively, use --output-dir and let nbconvert name the file?
                 # "--output-dir", output_dir
            ]
            
            logger.info(f"Running nbconvert: {' '.join(cmd)}")
            proc = await asyncio.to_thread(
                 subprocess.run,
                 cmd, capture_output=True, text=True, timeout=60
            )
            
            stdout = proc.stdout.strip() if proc.stdout else ""
            stderr = proc.stderr.strip() if proc.stderr else ""

            if proc.returncode != 0:
                logger.error(f"{log_prefix} FAILED - nbconvert returned {proc.returncode}. stdout: {stdout}, stderr: {stderr}")
                error_details = stderr if stderr else stdout
                raise RuntimeError(f"nbconvert failed (code {proc.returncode}): {error_details}")
            
            # Determine the actual output file produced (nbconvert might slightly alter name)
            # Let's trust the constructed absolute_output_path if nbconvert succeeded
            actual_output_file = absolute_output_path
            if not os.path.isfile(actual_output_file):
                 # Attempt to parse from stdout as fallback
                 output_file_match = re.search(r"Writing \d+ bytes to (.*)", stdout)
                 parsed_output_file = output_file_match.group(1).strip() if output_file_match else None
                 if parsed_output_file and os.path.isfile(parsed_output_file):
                      actual_output_file = parsed_output_file
                      logger.warning(f"{log_prefix} nbconvert output path mismatch? Used parsed path: {actual_output_file}")
                 else:
                      logger.error(f"{log_prefix} FAILED - Expected output file not found after nbconvert: {actual_output_file} (or parsed: {parsed_output_file})")
                      raise FileNotFoundError(f"Output file expected at '{actual_output_file}' was not created by nbconvert.")
            
            # If remote, upload the exported file back to the SFTP server
            if is_remote:
                if not self.sftp_manager: 
                    # This should ideally not be reached if download worked, but defensive
                    raise ConnectionError("SFTP manager required to upload exported file but is not available.")
                
                target_remote_dir = os.path.dirname(absolute_op_path) # absolute_op_path is the original remote .ipynb path
                target_remote_filename = os.path.basename(actual_output_file) # Use basename from the local export
                target_remote_path = posixpath.join(target_remote_dir, target_remote_filename) # Use posixpath for remote paths

                logger.info(f"{log_prefix} Uploading exported file from local {actual_output_file} to remote {target_remote_path}")
                with open(actual_output_file, 'rb') as f_export_content:
                    exported_content_bytes = f_export_content.read()
                
                await asyncio.to_thread(self.sftp_manager.write_file, target_remote_path, exported_content_bytes)
                logger.info(f"{log_prefix} SUCCESS - Exported '{notebook_path}' to local {actual_output_file} and uploaded to remote {target_remote_path}")
                return f"Successfully exported notebook '{notebook_path}' to {target_remote_path} (via local temp: {actual_output_file})"
            else:
                logger.info(f"{log_prefix} SUCCESS - Exported '{notebook_path}' to {actual_output_file}")
                # Return the absolute path where the file was actually created
                return f"Successfully exported notebook '{notebook_path}' to {actual_output_file}"
        
        except (ValueError, PermissionError, FileNotFoundError, ConnectionError, IOError, NotImplementedError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}")
            raise
        except subprocess.TimeoutExpired:
            logger.error(f"{log_prefix} FAILED - nbconvert command timed out.")
            raise RuntimeError("nbconvert command timed out after 60 seconds.")
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error during export: {e}")
            raise RuntimeError(f"An unexpected error occurred during notebook export: {e}") from e 
        finally:
            # Clean up temporary input file if created
            if temp_input_file and os.path.exists(temp_input_file):
                 try:
                      os.remove(temp_input_file)
                      logger.debug(f"Cleaned up temporary file: {temp_input_file}")
                 except Exception as e_clean:
                      logger.warning(f"Failed to clean up temporary file {temp_input_file}: {e_clean}")

    async def notebook_read(self, notebook_path: str) -> dict:
        """Reads an entire notebook and returns its structure as a dictionary.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').

        Returns
        -------
        dict
            Dictionary representation of the notebook, potentially with truncated cells/outputs.

        Raises
        ------
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading the notebook fails.
        ValueError
             If the file is not a valid notebook or cannot be serialized.
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_read', path=notebook_path)
        logger.info(f"{log_prefix} Called.")
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Convert to dict and handle truncation (existing logic)
            nb_dict = dict(nb)
            
            total_size = 0
            MAX_TOTAL_SIZE = 50 * 1024 * 1024 # 50MB limit
            MAX_CELL_SIZE = max(self.config.max_cell_source_size, self.config.max_cell_output_size)
            
            cells = nb_dict.get('cells', [])
            truncated_cells = []
            for i, cell in enumerate(cells):
                try:
                    cell_bytes = json.dumps(cell).encode('utf-8')
                    cell_size = len(cell_bytes)
                    total_size += cell_size

                    if cell_size > MAX_CELL_SIZE * 1.1: # Allow buffer
                         logger.warning(f"{log_prefix} Cell {i} size ({cell_size} bytes) exceeds limit. Truncating representation.")
                         truncated_cell = {
                             'cell_type': cell.get('cell_type', 'unknown'),
                             'metadata': { 'truncated': 'cell_size_limit_exceeded' },
                             'source': "[Content truncated due to size]"
                         }
                         # Add truncated output placeholder if code cell
                         if truncated_cell['cell_type'] == 'code':
                             truncated_cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': '[Output truncated due to size]'}]
                         truncated_cells.append(truncated_cell)
                    else:
                         truncated_cells.append(cell) # Keep original cell

                except Exception as size_err:
                     logger.warning(f"{log_prefix} Could not estimate size for cell {i}: {size_err}")
                     truncated_cells.append({
                         'cell_type': cell.get('cell_type', 'unknown'),
                         'metadata': { 'processing_error': True },
                         'source': "[Content could not be processed]"
                     })

                if total_size > MAX_TOTAL_SIZE:
                     logger.error(f"{log_prefix} FAILED - Total notebook size ({total_size}) exceeds limit ({MAX_TOTAL_SIZE} bytes) reading {absolute_op_path}. Returning partial structure.")
                     # Replace remaining cells with a single truncated message
                     nb_dict['cells'] = truncated_cells[:i] + [{
                         'cell_type': 'markdown', 
                         'metadata': {'truncated': 'total_size_limit_exceeded'},
                         'source': f"... (Remaining cells truncated - total size exceeded {MAX_TOTAL_SIZE} bytes) ..."
                     }]
                     # Ensure top-level metadata indicates truncation
                     if 'metadata' not in nb_dict: nb_dict['metadata'] = {}
                     nb_dict['metadata']['truncated'] = 'total_size_limit_exceeded'
                     break # Stop processing cells
            else:
                 # If loop completed without exceeding total size, update cells list
                 nb_dict['cells'] = truncated_cells

            logger.info(f"{log_prefix} SUCCESS - Read entire notebook from {absolute_op_path} (Estimated size: {total_size} bytes).")
            return nb_dict

        except (ValueError, FileNotFoundError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error reading '{notebook_path}': {e}")
            raise RuntimeError(f"An unexpected error occurred reading notebook: {e}") from e

    async def notebook_get_outline(self, notebook_path: str) -> List[Dict[str, Union[int, str, List[str]]]]:
        """Analyzes a Jupyter notebook file to extract its structure.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').

        Returns
        -------
        list of dict
            List representing the notebook structure, or a message if empty. See implementation for details.

        Raises
        ------
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading the notebook fails.
        ValueError
             If the file is not a valid notebook (e.g., during read).
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_get_outline', path=notebook_path)
        logger.info(f"{log_prefix} Called.")
        try:
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            if not nb.cells:
                logger.info(f"{log_prefix} SUCCESS - Notebook is empty ({absolute_op_path}).")
                return [{ "message": "Notebook is empty or has no cells" }]

            structure_map: List[Dict[str, Union[int, str, List[str]]]] = []
            for index, cell in enumerate(nb.cells):
                outline_items = []
                line_count = len(cell.source.splitlines())
                cell_type = cell.cell_type
                if cell_type == 'code':
                    outline_items = self._extract_code_outline(cell.source)
                elif cell_type == 'markdown':
                    outline_items = self._extract_markdown_outline(cell.source)
                if not outline_items:
                    outline_items = self._get_first_line_context(cell.source)
                cell_info: Dict[str, Union[int, str, List[str]]] = {
                    "index": index, "type": cell_type,
                    "line_count": line_count, "outline": outline_items
                }
                structure_map.append(cell_info)

            logger.info(f"{log_prefix} SUCCESS - Generated outline ({len(structure_map)} cells analyzed) from {absolute_op_path}.")
            return structure_map
        except (ValueError, FileNotFoundError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error getting outline for '{notebook_path}': {e}")
            raise RuntimeError(f"An unexpected error occurred getting notebook outline: {e}") from e

    async def notebook_search(
        self,
        notebook_path: str,
        query: str
    ) -> List[Dict[str, Union[int, str]]]:
        """Searches within a notebook's code and markdown cells for a query string.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').
        query : str
            The search query string.

        Returns
        -------
        list of dict
            List of dictionaries containing cell index, cell type, and match line number.

        Raises
        ------
        ValueError
            If query is empty or path is invalid.
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading the notebook fails.
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_search', path=notebook_path, query=query)
        logger.info(f"{log_prefix} Called.")
        if not query:
            logger.error(f"{log_prefix} Query is empty.")
            raise ValueError("Search query cannot be empty.")
        results: List[Dict[str, Union[int, str]]] = []
        try:
            logger.debug(f"{log_prefix} Resolving path and checking permissions...")
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )
            logger.debug(f"{log_prefix} Path resolved: is_remote={is_remote}, absolute_op_path={absolute_op_path}")

            logger.debug(f"{log_prefix} Reading notebook content...")
            nb = None
            if is_remote:
                if not self.sftp_manager: 
                    logger.error(f"{log_prefix} SFTP manager not available for remote read.")
                    raise ConnectionError("SFTP manager required for remote notebook search.")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                logger.debug(f"{log_prefix} SFTP read {len(content_bytes)} bytes. Parsing notebook...")
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
                logger.debug(f"{log_prefix} Parsed remote notebook.")
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    logger.debug(f"{log_prefix} Local read from {absolute_op_path}. Parsing notebook...")
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)
                    logger.debug(f"{log_prefix} Parsed local notebook.")
            
            if nb is None:
                logger.error(f"{log_prefix} Notebook object is None after read attempt.")
                raise IOError(f"Failed to load notebook content from {notebook_path}")

            logger.info(f"{log_prefix} Notebook loaded. Cell count: {len(nb.cells)}. Starting search...")
            query_lower = query.lower()
            MAX_SNIPPET_LEN = 150
            for index, cell in enumerate(nb.cells):
                logger.debug(f"{log_prefix} Processing cell {index} of type {cell.cell_type}")
                try:
                    source = cell.source
                    cell_type = cell.cell_type
                    lines = source.splitlines()
                    for line_num_0based, line in enumerate(lines):
                        if query_lower in line.lower():
                            line_num_1based = line_num_0based + 1
                            snippet = line.strip()
                            if len(snippet) > MAX_SNIPPET_LEN:
                                try:
                                    match_start = snippet.lower().index(query_lower)
                                    start = max(0, match_start - MAX_SNIPPET_LEN // 3)
                                    end = min(len(snippet), match_start + len(query) + (MAX_SNIPPET_LEN * 2 // 3))
                                    prefix = "..." if start > 0 else ""
                                    suffix = "..." if end < len(snippet) else ""
                                    snippet = prefix + snippet[start:end] + suffix
                                except ValueError: # query_lower not in snippet.lower() - should not happen if outer if was true
                                    snippet = snippet[:MAX_SNIPPET_LEN] + "..."
                            results.append({
                                "cell_index": index,
                                "cell_type": cell_type,
                                "match_line_number": line_num_1based,
                                "snippet": snippet
                            })
                            logger.debug(f"{log_prefix} Found match in cell {index}, line {line_num_1based}")
                except AttributeError as attr_err:
                    logger.warning(f"{log_prefix} Skipping cell {index} due to AttributeError (likely missing source): {attr_err} in {absolute_op_path}.")
                    continue
                except Exception as cell_err:
                    logger.error(f"{log_prefix} Error processing cell {index} in {absolute_op_path}. Type: {type(cell_err).__name__}, Error: {cell_err}", exc_info=(self.config.log_level == logging.DEBUG))
                    continue
            logger.info(f"{log_prefix} Search loop completed.")

            if not results:
                logger.info(f"{log_prefix} SUCCESS - No matches found in {absolute_op_path}.")
                return [{"message": f"No matches found for query '{query}' in notebook '{notebook_path}'."}]
            else:
                logger.info(f"{log_prefix} SUCCESS - Found {len(results)} match(es) in {absolute_op_path}.")
                return results
        except (ValueError, FileNotFoundError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - Known error: {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e: # Generic catch-all
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {type(e).__name__}: {e}") # exc_info=True is implicit with logger.exception
            raise RuntimeError(f"An unexpected error occurred during notebook search for '{notebook_path}': {e}") from e

    # --- Helper methods for outline generation (Restored AGAIN - FULL Version) --- 

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

        # Second pass for AST elements (functions, classes)
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    outline.append(f"func: {node.name}")
                elif isinstance(node, ast.ClassDef):
                    outline.append(f"class: {node.name}")
        except SyntaxError:
            if not any(item.startswith("comment:") for item in outline):
                 outline.append("<Syntax Error>")
        except Exception as e:
            if not outline:
                outline.append(f"<AST Parsing Error: {e}>")
        return outline

    def _extract_markdown_outline(self, source: str) -> List[str]:
        """Extracts markdown headings (H1, H2, etc.) and HTML headings (h1-h6)."""
        headings = []
        html_heading_re = re.compile(r'<h([1-6])[^>]*>(.*?)</h\1>', re.IGNORECASE | re.DOTALL)
        try:
            lines = source.split('\n')
            for line in lines:
                stripped_line = line.strip()
                md_match = re.match(r'^(#+)\s*(.*)', stripped_line)
                if md_match:
                    level = len(md_match.group(1))
                    heading_text = md_match.group(2).strip()
                    if heading_text:
                        headings.append(f"H{level}: {heading_text}")
                else:
                    for html_match in html_heading_re.finditer(stripped_line):
                        level = int(html_match.group(1))
                        heading_text = re.sub(r'<.*?>', '', html_match.group(2)).strip()
                        if heading_text:
                            headings.append(f"H{level}: {heading_text}")
        except AttributeError:
             headings.append("<Missing Source>")
        except Exception as e:
             headings.append(f"<Markdown Parsing Error: {e}>")
        return headings

    def _get_first_line_context(self, source: str) -> List[str]:
        """Gets the first non-empty line as context if no other outline found."""
        try:
            for line in source.splitlines():
                stripped_line = line.strip()
                if stripped_line:
                    context = stripped_line[:100] + ('... ' if len(stripped_line) > 100 else '')
                    return [f"context: {context}"]
            return ["<Empty Cell>"]
        except Exception as e:
            logger.warning(f"Error getting first line context: {e}")
            return ["<Error getting context>"]
            
    # --- End of Outline Helpers ---

    async def notebook_validate(self, notebook_path: str) -> str:
        """Validates the notebook against the nbformat schema.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').

        Returns
        -------
        str
            Success message or validation error details.

        Raises
        ------
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading the notebook fails.
        ValueError
             If the file is not a valid notebook (e.g., during read).
        ConnectionError
            If SFTP is required but unavailable.
        nbformat.ValidationError
             Propagated if validation fails structurally.
        """
        log_prefix = self._log_prefix('notebook_validate', path=notebook_path)
        logger.info(f"{log_prefix} Called.")
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP manager required for remote validation.")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Validate notebook
            nbformat.validate(nb)
            logger.info(f"{log_prefix} SUCCESS - Notebook at {absolute_op_path} is valid.")
            return "Notebook is valid."

        # Catch ValidationError specifically FIRST
        except nbformat.ValidationError as e:
            logger.warning(f"{log_prefix} VALIDATION FAILED ({absolute_op_path}): {e}")
            return f"Notebook validation failed: {e}"
        # Then catch other expected errors
        except (ValueError, FileNotFoundError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        # Finally catch generic exceptions
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error validating notebook: {e}")
            raise RuntimeError(f"An unexpected error occurred validating notebook: {e}") from e

    async def notebook_get_info(self, notebook_path: str) -> dict:
        """Gets general information about the notebook (cell count, metadata, etc.).

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').

        Returns
        -------
        dict
            Dictionary containing notebook information. See implementation for keys.

        Raises
        ------
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading the notebook fails.
        ValueError
             If the file is not a valid notebook (e.g., during read).
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_get_info', path=notebook_path)
        logger.info(f"{log_prefix} Called.")
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Extract information
            info = {
                'cell_count': len(nb.cells),
                'metadata': dict(nb.metadata),
                'nbformat': nb.nbformat,
                'nbformat_minor': nb.nbformat_minor
            }

            logger.info(f"{log_prefix} SUCCESS - Retrieved notebook information for {absolute_op_path}.")
            return info

        except (ValueError, FileNotFoundError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error retrieving notebook info: {e}")
            raise RuntimeError(f"An unexpected error occurred retrieving notebook info: {e}") from e

    async def notebook_read_cell_output(self, notebook_path: str, cell_index: int) -> List[dict]:
        """Reads the output(s) of a specific cell from a Jupyter Notebook.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').
        cell_index : int
            The 0-based index of the cell to read outputs from.

        Returns
        -------
        list of dict
            A list containing the cell's output objects, or an empty list if none/not applicable.
            Can include a placeholder error object if output size limit is exceeded.

        Raises
        ------
        IndexError
            If cell_index is out of bounds.
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading the notebook fails.
        ValueError
            If the file is not valid or output cannot be serialized/checked.
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_read_cell_output', path=notebook_path, index=cell_index)
        logger.info(f"{log_prefix} Called.")
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Extract output
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")

            cell = nb.cells[cell_index]
            if cell.cell_type != 'code':
                logger.info(f"{log_prefix} SUCCESS - Cell {cell_index} is not a code cell ({absolute_op_path}), returning empty list.")
                return []

            outputs = cell.get('outputs', [])
            if not outputs:
                logger.info(f"{log_prefix} SUCCESS - Cell {cell_index} has no outputs ({absolute_op_path}).")
                return []

            # *** Check output size limit and return error structure if exceeded ***
            try:
                output_bytes = json.dumps(outputs).encode('utf-8')
                output_size = len(output_bytes)
                if output_size > self.config.max_cell_output_size:
                    logger.error(f"{log_prefix} FAILED - Output size ({output_size} bytes) exceeds limit ({self.config.max_cell_output_size} bytes) for cell {cell_index} in {absolute_op_path}.")
                    # *** Return the error structure ***
                    return [{
                        'output_type': 'error',
                        'ename': 'OutputSizeError',
                        'evalue': f'Output truncated - size ({output_size} bytes) exceeds limit ({self.config.max_cell_output_size} bytes)',
                        'traceback': []
                    }]
            except (TypeError, OverflowError) as json_err:
                logger.error(f"{log_prefix} FAILED - Could not serialize outputs for size check in {absolute_op_path}: {json_err}")
                # Raise ValueError if serialization itself fails
                raise ValueError(f"Could not determine size of cell output: {json_err}")

            logger.info(f"{log_prefix} SUCCESS - Read outputs for cell {cell_index} from {absolute_op_path} (Size: {output_size} bytes).")
            return outputs

        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error reading cell outputs: {e}")
            raise RuntimeError(f"An unexpected error occurred reading cell outputs: {e}") from e

    async def notebook_split_cell(self, notebook_path: str, cell_index: int, split_at_line: int) -> str:
        """Splits a cell into two cells at a specified line number.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').
        cell_index : int
            The 0-based index of the cell to split.
        split_at_line : int
            The 1-based line number where the split occurs (this line starts the new cell).

        Returns
        -------
        str
            Success message.

        Raises
        ------
        IndexError
            If cell_index is out of bounds.
        ValueError
            If split_at_line is out of bounds for the cell or path is invalid.
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading or writing the notebook fails.
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_split_cell', path=notebook_path, index=cell_index, line=split_at_line)
        logger.info(f"{log_prefix} Called.")
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Perform split logic (Corrected)
            num_cells = len(nb.cells)
            if not 0 <= cell_index < num_cells:
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{num_cells-1}).")

            cell_to_split = nb.cells[cell_index]
            # Use splitlines(True) to preserve endings for accurate joining
            source_lines = cell_to_split.source.splitlines(True) 
            num_lines = len(source_lines) 
            # Handle edge case of empty source string
            if not source_lines and split_at_line == 1:
                 split_index_0based = 0 # Splitting an empty cell at line 1 results in two empty cells
            elif not 1 <= split_at_line <= num_lines + 1:
                 # Validate 1-based line number against actual number of lines
                 raise ValueError(f"Split line {split_at_line} is out of bounds for cell with {num_lines} lines (valid range: 1-{num_lines + 1}).")
            else:
                 split_index_0based = split_at_line - 1 # Convert valid 1-based line number to 0-based index

            source_part1 = "".join(source_lines[:split_index_0based])
            source_part2 = "".join(source_lines[split_index_0based:])

            # Update original cell
            cell_to_split.source = source_part1

            # Create new cell (Logic remains the same)
            cell_type = cell_to_split.cell_type
            if cell_type == 'code':
                new_cell = nbformat.v4.new_code_cell(source=source_part2)
            elif cell_type == 'markdown':
                new_cell = nbformat.v4.new_markdown_cell(source=source_part2)
            elif cell_type == 'raw':
                 new_cell = nbformat.v4.new_raw_cell(source=source_part2)
            else:
                 logger.warning(f"{log_prefix} - Unknown cell type '{cell_type}' being split. Creating raw cell for second part.")
                 new_cell = nbformat.v4.new_raw_cell(source=source_part2)
            # Ensure metadata is copied correctly (use deepcopy for safety)
            from copy import deepcopy
            new_cell.metadata.update(deepcopy(cell_to_split.metadata)) 
            # Regenerate ID for the new cell
            if hasattr(new_cell, 'id'):
                 import random
                 import string
                 new_cell.id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

            # Insert the new cell
            insertion_point = cell_index + 1
            nb.cells.insert(insertion_point, new_cell)

            # 4. Write notebook (Logic remains the same)
            if is_remote:
                json_content = nbformat.writes(nb)
                await asyncio.to_thread(self.sftp_manager.write_file, absolute_op_path, json_content)
            else:
                await asyncio.to_thread(nbformat.write, nb, absolute_op_path)

            logger.info(f"{log_prefix} SUCCESS - Split cell {cell_index} at line {split_at_line}. New cell inserted at index {insertion_point} in {absolute_op_path}.")
            return f"Successfully split cell {cell_index} at line {split_at_line} in {notebook_path}."
        
        # Restore the exception handling block
        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error splitting cell in '{notebook_path}': {e}")
            raise RuntimeError(f"An unexpected error occurred splitting cell: {e}") from e

    async def notebook_merge_cells(self, notebook_path: str, first_cell_index: int) -> str:
        """Merges a cell with the cell immediately following it.

        Parameters
        ----------
        notebook_path : str
             Path to the notebook file (relative, absolute, or '~').
        first_cell_index : int
            The 0-based index of the first cell in the pair to merge.

        Returns
        -------
        str
            Success message.

        Raises
        ------
        IndexError
            If first_cell_index is invalid or the last cell.
        ValueError
            If cells have different types, merged content exceeds size limit, or path is invalid.
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading or writing the notebook fails.
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_merge_cells', path=notebook_path, index=first_cell_index)
        logger.info(f"{log_prefix} Called.")
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Perform merge logic
            num_cells = len(nb.cells)
            if not 0 <= first_cell_index < num_cells - 1:
                raise IndexError(f"First cell index {first_cell_index} is invalid or it's the last cell (0-{num_cells-2}).")

            cell1 = nb.cells[first_cell_index]
            cell2 = nb.cells[first_cell_index + 1]

            if cell1.cell_type != cell2.cell_type:
                raise ValueError(f"Cannot merge cells of different types ({cell1.cell_type} and {cell2.cell_type}).")

            source1 = cell1.source
            source2 = cell2.source

            # Add separator only if source1 is non-empty and doesn't end with newline
            separator = '\n' if source1 and not source1.endswith('\n') else ''
            combined_source = source1 + separator + source2

            # Check combined source size
            if len(combined_source.encode('utf-8')) > self.config.max_cell_source_size:
                raise ValueError(f"Merged source content exceeds maximum allowed size ({self.config.max_cell_source_size} bytes).")

            cell1.source = combined_source
            # Merge metadata? Keep cell1's for now.
            # cell1.metadata.update(cell2.metadata) # Optional: if merging metadata is desired

            del nb.cells[first_cell_index + 1]

            # 4. Write notebook
            if is_remote:
                json_content = nbformat.writes(nb)
                await asyncio.to_thread(self.sftp_manager.write_file, absolute_op_path, json_content)
            else:
                await asyncio.to_thread(nbformat.write, nb, absolute_op_path)

            logger.info(f"{log_prefix} SUCCESS - Merged cell {first_cell_index + 1} into cell {first_cell_index} in {absolute_op_path}.")
            return f"Successfully merged cell {first_cell_index + 1} into cell {first_cell_index} in {notebook_path}."

        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error merging cells in '{notebook_path}': {e}")
            raise RuntimeError(f"An unexpected error occurred merging cells: {e}") from e

    async def notebook_change_cell_type(self, notebook_path: str, cell_index: int, new_type: str) -> str:
        """Changes the type of a specific cell in a Jupyter Notebook.
        
        Parameters
        ----------
        notebook_path : str
             Path to the notebook file (relative, absolute, or '~').
        cell_index : int
            The 0-based index of the cell to change.
        new_type : str
            Target cell type ('code', 'markdown', or 'raw').

        Returns
        -------
        str
            Success message or message indicating no change was needed.

        Raises
        ------
        IndexError
            If cell_index is out of bounds.
        ValueError
            If new_type is invalid or path is invalid.
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading or writing the notebook fails.
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_change_cell_type', path=notebook_path, index=cell_index, type=new_type)
        logger.info(f"{log_prefix} Called.")
        
        valid_types = ['code', 'markdown', 'raw']
        if new_type not in valid_types:
            raise ValueError(f"Invalid cell type: {new_type}. Must be one of: {', '.join(valid_types)}")
        
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Change cell type
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")
            
            cell = nb.cells[cell_index]
            current_type = cell.cell_type
            if current_type == new_type:
                logger.info(f"{log_prefix} Cell {cell_index} is already type '{new_type}'. No change needed.")
                return f"Cell {cell_index} is already of type '{new_type}'. No change needed."
            
            # *** Create a new cell object instead of just changing type ***
            source = cell.source
            metadata = dict(cell.metadata)
            
            if new_type == 'code': 
                new_cell = nbformat.v4.new_code_cell(source=source)
                # Initialize standard code cell fields if they weren't there
                new_cell.outputs = cell.get('outputs', []) 
                new_cell.execution_count = cell.get('execution_count', None)
            elif new_type == 'markdown': 
                new_cell = nbformat.v4.new_markdown_cell(source=source)
            else: # new_type == 'raw'
                new_cell = nbformat.v4.new_raw_cell(source=source)
            
            # Copy metadata and replace the cell in the list
            new_cell.metadata.update(metadata)
            # Preserve original ID if it exists?
            if hasattr(cell, 'id'): new_cell.id = cell.id 
            nb.cells[cell_index] = new_cell
            
            # 4. Write notebook
            if is_remote:
                json_content = nbformat.writes(nb)
                await asyncio.to_thread(self.sftp_manager.write_file, absolute_op_path, json_content)
            else:
                await asyncio.to_thread(nbformat.write, nb, absolute_op_path)

            logger.info(f"{log_prefix} SUCCESS - Changed cell type from '{current_type}' to '{new_type}' in {absolute_op_path}")
            return f"Successfully changed cell type from '{current_type}' to '{new_type}' in {notebook_path}"
        
        # Restore exception handling block
        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error changing cell type: {e}")
            raise RuntimeError(f"An unexpected error occurred changing cell type: {e}") from e

    async def notebook_duplicate_cell(self, notebook_path: str, cell_index: int, count: int = 1) -> str:
        """Duplicates a specific cell in a Jupyter Notebook multiple times.
        
        Parameters
        ----------
        notebook_path : str
             Path to the notebook file (relative, absolute, or '~').
        cell_index : int
            The 0-based index of the cell to duplicate.
        count : int, optional
            Number of copies to create (default: 1).
            
        Returns
        -------
        str
            Success message.
            
        Raises
        ------
        IndexError
            If cell_index is out of range.
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading or writing the notebook fails.
        ValueError
            If count is less than 1 or path is invalid.
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_duplicate_cell', path=notebook_path, index=cell_index, count=count)
        logger.info(f"{log_prefix} Called.")
        
        if count < 1: raise ValueError(f"Count must be a positive integer: {count}")
        
        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP required")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Duplicate cell
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")
            
            cell_to_duplicate = nb.cells[cell_index]
            # Use deepcopy to avoid shared references issues, especially with metadata/outputs
            from copy import deepcopy
            
            insertion_index = cell_index + 1
            for i in range(count):
                 # Create a true copy of the cell node
                 new_cell = deepcopy(cell_to_duplicate)
                 # Regenerate ID for the new cell
                 if hasattr(new_cell, 'id'):
                     # Simple way to generate a plausible ID, though not guaranteed unique
                     # across sessions. Consider using uuid module if strict uniqueness needed.
                     import random
                     import string
                     new_cell.id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
                 
                 nb.cells.insert(insertion_index + i, new_cell)
            
            # 4. Write notebook
            if is_remote:
                json_content = nbformat.writes(nb)
                await asyncio.to_thread(self.sftp_manager.write_file, absolute_op_path, json_content)
            else:
                await asyncio.to_thread(nbformat.write, nb, absolute_op_path)

            new_cells_text = "cell" if count == 1 else f"{count} cells"
            logger.info(f"{log_prefix} SUCCESS - Duplicated cell {cell_index}, created {new_cells_text} after it in {absolute_op_path}.")
            return f"Successfully duplicated cell {cell_index}, creating {new_cells_text} after it in {notebook_path}."
        
        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error duplicating cell: {e}")
            raise RuntimeError(f"An unexpected error occurred duplicating cell: {e}") from e

    # Placeholder for write_notebook called by notebook_create
    # In reality, this might be notebook_ops.write_notebook
    async def write_notebook(self, path, nb, roots, sftp_mgr):
        await notebook_ops.write_notebook(path, nb, roots, sftp_mgr)

    # NOTE: Need to add sftp_manager.remove_file(path) - DONE

    async def notebook_edit_cell_output(self, notebook_path: str, cell_index: int, outputs: List[Dict[str, Any]]) -> str:
        """Edits or adds the output(s) of a specific code cell in a Jupyter Notebook.

        If the cell already has outputs, they will be replaced.
        If the cell is a code cell and has no outputs, the provided outputs will be added.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').
        cell_index : int
            The 0-based index of the code cell whose outputs are to be edited/added.
        outputs : List[Dict[str, Any]]
            A list of dictionaries representing the new outputs for the cell.
            Each dictionary must conform to the Jupyter Notebook cell output format.
            For example, a simple text output:
            `[{"output_type": "stream", "name": "stdout", "text": "Hello World\\n"}]`
            An error output:
            `[{"output_type": "error", "ename": "ValueError", "evalue": "A sample error", "traceback": ["frame1", "frame2"]}]`
            A display_data output (e.g., for an image):
            `[{"output_type": "display_data", "data": {"image/png": "base64encodeddatacomeshere..."}, "metadata": {}}]`

        Returns
        -------
        str
            Success message confirming the cell's outputs were edited/added.

        Raises
        ------
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading or writing the notebook fails.
        ValueError
            If the file is not a valid notebook, if the cell is not a code cell,
            if the provided outputs are not a list, or if the output data is too large.
        IndexError
            If the cell_index is out of range.
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix('notebook_edit_cell_output', path=notebook_path, index=cell_index)
        logger.info(f"{log_prefix} Called.")

        if not isinstance(outputs, list):
            raise ValueError("The 'outputs' parameter must be a list of dictionaries.")

        # Validate individual output items before further processing
        for i, output_item in enumerate(outputs):
            if not isinstance(output_item, dict):
                raise ValueError(f"Each item in 'outputs' must be a dictionary. Item at index {i} is not a dict: {type(output_item)}")
            if "output_type" not in output_item:
                raise ValueError(f"Item at index {i} in 'outputs' is missing required key 'output_type'. Got: {output_item}")
            
            output_type = output_item["output_type"]
            if not isinstance(output_type, str):
                raise ValueError(f"'output_type' for item at index {i} in 'outputs' must be a string. Got: {type(output_type)}")

            # Basic validation for common output types
            if output_type == "stream":
                if "name" not in output_item or not isinstance(output_item["name"], str):
                    raise ValueError(f"Stream output at index {i} is missing 'name' string key. Got: {output_item}")
                if "text" not in output_item: # nbformat allows string or list of strings, we'll be flexible on read but enforce presence on write
                    raise ValueError(f"Stream output at index {i} is missing 'text' key. Got: {output_item}")
            elif output_type in ["display_data", "execute_result"]:
                if "data" not in output_item or not isinstance(output_item["data"], dict):
                    raise ValueError(f"Output type '{output_type}' at index {i} is missing 'data' dictionary key. Got: {output_item}")
                if "metadata" not in output_item or not isinstance(output_item["metadata"], dict):
                    raise ValueError(f"Output type '{output_type}' at index {i} is missing 'metadata' dictionary key. Got: {output_item}")
            elif output_type == "error":
                if "ename" not in output_item or not isinstance(output_item["ename"], str):
                    raise ValueError(f"Error output at index {i} is missing 'ename' string key. Got: {output_item}")
                if "evalue" not in output_item or not isinstance(output_item["evalue"], str):
                    raise ValueError(f"Error output at index {i} is missing 'evalue' string key. Got: {output_item}")
                if "traceback" not in output_item or not isinstance(output_item["traceback"], list):
                    raise ValueError(f"Error output at index {i} is missing 'traceback' list key. Got: {output_item}")
            # Not raising error for unknown output_types, nbformat will handle that if it's truly invalid

        try:
            # Validate output size before potentially large data processing
            try:
                output_bytes = json.dumps(outputs).encode('utf-8')
                output_size = len(output_bytes)
                if output_size > self.config.max_cell_output_size:
                    logger.error(f"{log_prefix} FAILED - Provided output size ({output_size} bytes) exceeds limit ({self.config.max_cell_output_size} bytes).")
                    raise ValueError(f"Provided output data size ({output_size} bytes) exceeds maximum allowed size ({self.config.max_cell_output_size} bytes).")
            except (TypeError, OverflowError) as json_err:
                logger.error(f"{log_prefix} FAILED - Could not serialize provided outputs for size check: {json_err}")
                raise ValueError(f"Could not process provided outputs for size validation: {json_err}")

            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read notebook
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP manager required for remote notebook operations.")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Validate cell index and type, then edit outputs
            if not 0 <= cell_index < len(nb.cells):
                raise IndexError(f"Cell index {cell_index} is out of bounds (0-{len(nb.cells)-1}).")

            cell = nb.cells[cell_index]
            if cell.cell_type != 'code':
                raise ValueError(f"Outputs can only be edited for 'code' cells. Cell {cell_index} is of type '{cell.cell_type}'.")

            # Assign the new outputs
            cell.outputs = [NotebookNode(output) for output in outputs] # Ensure outputs are NotebookNodes

            # Optionally, clear execution_count if outputs are manually set,
            # or require it to be part of the 'outputs' if relevant for some output types.
            # For simplicity, we'll just set the outputs. User can clear separately if needed.
            # if hasattr(cell, 'execution_count'):
            #    cell.execution_count = None

            # 4. Write notebook
            if is_remote:
                json_content = nbformat.writes(nb)
                await asyncio.to_thread(self.sftp_manager.write_file, absolute_op_path, json_content)
            else:
                await asyncio.to_thread(nbformat.write, nb, absolute_op_path)

            logger.info(f"{log_prefix} SUCCESS - Edited/Added outputs for cell {cell_index} in {absolute_op_path}.")
            return f"Successfully edited/added outputs for cell {cell_index} in {notebook_path}"

        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error editing cell outputs: {e}")
            raise RuntimeError(f"An unexpected error occurred editing cell outputs for '{notebook_path}': {e}") from e

    async def notebook_bulk_add_cells(
        self,
        notebook_path: str,
        cells_to_add: List[Dict[str, str]],
        insert_after_index: int
    ) -> str:
        """Adds multiple new cells sequentially to a Jupyter Notebook after a specified index.

        All cells provided in `cells_to_add` will be inserted as a contiguous block.
        The first cell in `cells_to_add` is inserted at `insert_after_index + 1`,
        the second at `insert_after_index + 2`, and so on.

        Parameters
        ----------
        notebook_path : str
            Path to the notebook file (relative, absolute, or '~').
        cells_to_add : List[Dict[str, str]]
            A list of dictionaries, where each dictionary defines a cell to add.
            Each dictionary must have two keys:
            - "cell_type": str, type of cell ('code' or 'markdown').
            - "source": str, the source content for the new cell.
            Example: `[{"cell_type": "code", "source": "print('Hello')"}, {"cell_type": "markdown", "source": "# Section"}]`
        insert_after_index : int
            The 0-based index after which to insert the new block of cells.
            -1 inserts the block at the beginning of the notebook.

        Returns
        -------
        str
            Success message with the number of cells added and original path.

        Raises
        ------
        ValueError
            If `cells_to_add` is malformed, `cell_type` is invalid,
            or source content for any cell exceeds maximum allowed size.
        IndexError
            If `insert_after_index` results in an invalid initial insertion index.
        FileNotFoundError
            If the notebook file doesn't exist at the resolved path.
        PermissionError
            If the notebook path resolves outside allowed workspace roots.
        IOError
            If reading or writing the notebook fails.
        ConnectionError
            If SFTP is required but unavailable.
        """
        log_prefix = self._log_prefix(
            'notebook_bulk_add_cells',
            path=notebook_path,
            num_cells=len(cells_to_add) if cells_to_add else 0,
            after_index=insert_after_index
        )
        logger.info(f"{log_prefix} Called.")

        if not cells_to_add:
            logger.info(f"{log_prefix} No cells provided to add. Operation skipped.")
            return f"No cells provided to add to {notebook_path}. No changes made."

        # Validate all cell data before reading the notebook
        for i, cell_data in enumerate(cells_to_add):
            if not isinstance(cell_data, dict) or "cell_type" not in cell_data or "source" not in cell_data:
                raise ValueError(
                    f"Each item in 'cells_to_add' must be a dictionary with 'cell_type' and 'source' keys. "
                    f"Problem found at index {i}: {cell_data}"
                )
            source = cell_data["source"]
            cell_type = cell_data["cell_type"]
            if not isinstance(source, str):
                 raise ValueError(f"Source for cell at index {i} must be a string. Got: {type(source)}")
            if not isinstance(cell_type, str):
                 raise ValueError(f"Cell type for cell at index {i} must be a string. Got: {type(cell_type)}")

            if len(source.encode('utf-8')) > self.config.max_cell_source_size:
                raise ValueError(
                    f"Source content for cell at index {i} (type: {cell_type}) "
                    f"exceeds maximum allowed size ({self.config.max_cell_source_size} bytes)."
                )
            if cell_type not in ['code', 'markdown']:
                raise ValueError(f"Invalid cell_type '{cell_type}' for cell at index {i}. Must be 'code' or 'markdown'.")

        try:
            # 1. Resolve path and check permissions
            is_remote, absolute_op_path = notebook_ops.resolve_path_and_check_permissions(
                notebook_path, self._get_allowed_local_roots(), self.sftp_manager
            )

            # 2. Read the notebook
            nb: NotebookNode
            if is_remote:
                if not self.sftp_manager: raise ConnectionError("SFTP manager required for remote notebook operations.")
                content_bytes = await asyncio.to_thread(self.sftp_manager.read_file, absolute_op_path)
                nb = nbformat.reads(content_bytes.decode('utf-8'), as_version=4)
            else:
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                    nb = await asyncio.to_thread(nbformat.read, f, as_version=4)

            # 3. Determine insertion point and validate
            current_num_cells = len(nb.cells)
            # The first new cell will be inserted at this index
            initial_insertion_point = insert_after_index + 1

            if not (0 <= initial_insertion_point <= current_num_cells):
                raise IndexError(
                    f"Initial insertion point {initial_insertion_point} (based on insert_after_index {insert_after_index}) "
                    f"is out of bounds for notebook with {current_num_cells} cells. "
                    f"Valid range for initial insertion point: 0 to {current_num_cells}."
                )

            # 4. Add cells
            for i, cell_data in enumerate(cells_to_add):
                cell_type = cell_data['cell_type']
                source = cell_data['source']
                
                new_cell: NotebookNode
                if cell_type == 'code':
                    new_cell = nbformat.v4.new_code_cell(source)
                elif cell_type == 'markdown': # Already validated to be 'code' or 'markdown'
                    new_cell = nbformat.v4.new_markdown_cell(source)
                # else: # Should not be reached due to prior validation, but as a safeguard:
                #    logger.error(f"{log_prefix} Internal error: Unexpected cell_type '{cell_type}' encountered during cell creation.")
                #    raise ValueError(f"Internal error: Unexpected cell_type '{cell_type}'")

                nb.cells.insert(initial_insertion_point + i, new_cell)
            
            # 5. Write the modified notebook back
            if is_remote:
                json_content = nbformat.writes(nb)
                await asyncio.to_thread(self.sftp_manager.write_file, absolute_op_path, json_content)
            else:
                await asyncio.to_thread(nbformat.write, nb, absolute_op_path)

            logger.info(
                f"{log_prefix} SUCCESS - Added {len(cells_to_add)} cells starting after index {insert_after_index} "
                f"(actual initial index {initial_insertion_point}) to {absolute_op_path}"
            )
            return (
                f"Successfully added {len(cells_to_add)} cells to {notebook_path} "
                f"starting after index {insert_after_index}."
            )

        except (ValueError, FileNotFoundError, IndexError, IOError, PermissionError, ConnectionError) as e:
            logger.error(f"{log_prefix} FAILED - {type(e).__name__}: {e}", exc_info=(self.config.log_level == logging.DEBUG))
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error: {e}")
            raise RuntimeError(f"An unexpected error occurred bulk adding cells to '{notebook_path}': {e}") from e

    async def notebook_get_server_path_context(self, project_directory: str = None) -> Dict[str, Any]:
        """
        Provides information about the server's path configuration and, if a project
        directory is given, validates it against the server's allowed roots and offers
        path construction guidance.

        Parameters
        ----------
        project_directory : str, optional
            The absolute path to the user's current project directory on the client side.
            If provided, the tool will analyze its relationship with the server's allowed roots.

        Returns
        -------\n\
        Dict[str, Any]
            A dictionary containing:
            - 'allowed_roots': List[str] - Absolute paths of server's allowed root directories.
            - 'server_os_path_separator': str - The OS-specific path separator used by the server.
            - 'server_path_style': str - 'Posix' or 'Windows'.
            - 'sftp_enabled': bool - True if SFTP is configured and active.
            - 'sftp_root': Optional[str] - The SFTP root path if SFTP is enabled and configured.
            - 'original_sftp_specs': Optional[List[str]] - The original --sftp-root user@host:/path specifications if SFTP is enabled.
            - 'provided_project_directory': Optional[str] - The project_directory that was passed in.
            - 'project_directory_status': Optional[str] - Status of the project directory:
                - 'not_provided': If project_directory was not given.
                - 'is_an_allowed_root': If project_directory is one of the allowed_roots.
                - 'is_within_allowed_root': If project_directory is a subdirectory of an allowed_root.
                - 'outside_allowed_roots': If project_directory is not under any allowed_root.
                - 'resolution_error': If there was an error processing the project directory.
            - 'effective_notebook_base_path_for_project': Optional[str] - The relative path prefix
              to use for notebook operations to target them within the project_directory,
              relative to an allowed_root. Empty if project_directory is an allowed_root.
              None if project_directory is outside allowed_roots or not provided.
            - 'path_construction_guidance': str - Guidance messages for path construction.
        """
        log_prefix = self._log_prefix('notebook_get_server_path_context', project_dir=project_directory)
        logger.info(f"{log_prefix} Called.")

        sftp_enabled = False
        sftp_root = None
        if self.sftp_manager:
            sftp_enabled = True
            sftp_root = getattr(self.sftp_manager, 'root_path', None) # This might still be useful for a specific connection's root if SFTPManager exposes it
            if sftp_root is None:
                sftp_root = getattr(self.config, 'sftp_root_path', None) # Fallback, though raw_sftp_specs is more comprehensive

        server_path_style = 'Windows' if os.path.sep == '\\\\' else 'Posix'
        allowed_roots_abs = [os.path.abspath(p) for p in self._get_allowed_local_roots()]

        # Initialize project-specific fields
        project_dir_status = "not_provided"
        effective_base_path = None
        guidance = "Project directory not provided. Paths will be resolved relative to server's allowed_roots."
        original_sftp_specs_from_config = getattr(self.config, 'raw_sftp_specs', [])

        if project_directory:
            try:
                # Normalize project directory path (ensure it's absolute and clean)
                # This assumes project_directory is a local path from the client's perspective
                # If server is remote (SFTP), this comparison logic needs more advanced handling.
                # For now, assuming local server or SFTP where local mapping is relevant.
                norm_project_dir = os.path.abspath(os.path.normpath(project_directory))
                guidance = f"Project directory: '{norm_project_dir}'.\\n"

                project_is_allowed_root = False
                project_within_allowed_root = False
                relevant_allowed_root_for_project = None

                for root_abs in allowed_roots_abs:
                    if norm_project_dir == root_abs:
                        project_is_allowed_root = True
                        relevant_allowed_root_for_project = root_abs
                        break
                    # Check if norm_project_dir starts with root_abs + path_separator
                    # This implies norm_project_dir is a subdirectory of root_abs
                    if norm_project_dir.startswith(root_abs + os.path.sep) or norm_project_dir.startswith(root_abs + '/'): # Check both separators for robustness
                        # Additional check to ensure it's a true subdirectory and not just a shared prefix
                        # e.g. /foo/bar should not match /foo/ba
                        if len(norm_project_dir) > len(root_abs) and norm_project_dir[len(root_abs)] in (os.path.sep, '/'):
                            project_within_allowed_root = True
                            relevant_allowed_root_for_project = root_abs
                            break
                
                if project_is_allowed_root:
                    project_dir_status = "is_an_allowed_root"
                    effective_base_path = "" # Operations can be relative to project dir itself
                    guidance += "Project directory is directly one of the server's allowed roots.\\n"
                    guidance += "For unqualified notebook paths (e.g., 'my_notebook.ipynb'), use them as is."
                elif project_within_allowed_root and relevant_allowed_root_for_project:
                    project_dir_status = "is_within_allowed_root"
                    # Calculate the part of the project_directory path relative to the found allowed_root
                    effective_base_path = os.path.relpath(norm_project_dir, relevant_allowed_root_for_project)
                    # Ensure Posix-style separators for cross-platform consistency if needed by server/nbformat
                    if server_path_style == 'Posix' and os.path.sep != '/':
                        effective_base_path = effective_base_path.replace(os.path.sep, '/')

                    guidance += f"Project directory is within server allowed root: '{relevant_allowed_root_for_project}'.\\n"
                    guidance += f"To target files within your project, prepend unqualified paths with: '{effective_base_path}{os.path.sep}'.\\n"
                    guidance += f"Example: 'my_notebook.ipynb' should be passed as '{effective_base_path}{os.path.sep}my_notebook.ipynb'."
                else:
                    project_dir_status = "outside_allowed_roots"
                    guidance += "Project directory is NOT an allowed root nor within any allowed root.\n"
                    if sftp_enabled and original_sftp_specs_from_config:
                        guidance += f"SFTP is active. Original SFTP remote roots configured: {original_sftp_specs_from_config}.\\n"
                        guidance += "If your project directory corresponds to one of these SFTP locations, paths for tools should be relative to that SFTP root. E.g., if SFTP root is 'user@host:/remote/project' and your project is '/remote/project', then 'my_notebook.ipynb' targets '/remote/project/my_notebook.ipynb'.\\n"
                    guidance += "Notebook operations using unqualified paths might not target your project directory directly unless it aligns with an SFTP root as described above.\\n"
                    guidance += "Alternatively, provide absolute paths (if allowed by server) or paths relative to one of the server's direct allowed_roots: "
                    guidance += f"{', '.join(allowed_roots_abs)}."

            except Exception as e:
                logger.error(f"{log_prefix} Error processing project_directory '{project_directory}': {e}")
                project_dir_status = "resolution_error"
                guidance = f"Error processing project_directory '{project_directory}': {e}. Path guidance may be unreliable."
        
        try:
            context = {
                "allowed_roots": allowed_roots_abs,
                "server_os_path_separator": os.path.sep,
                "server_path_style": server_path_style,
                "sftp_enabled": sftp_enabled,
                "sftp_root": sftp_root, # This specific sftp_root might be less relevant now compared to original_sftp_specs
                "original_sftp_specs": original_sftp_specs_from_config if sftp_enabled else [],
                "provided_project_directory": project_directory,
                "project_directory_status": project_dir_status,
                "effective_notebook_base_path_for_project": effective_base_path,
                "path_construction_guidance": guidance,
            }
            logger.info(f"{log_prefix} SUCCESS - Server path context: {context}")
            return context
        except Exception as e:
            logger.exception(f"{log_prefix} FAILED - Unexpected error getting server path context: {e}")
            raise RuntimeError(f"An unexpected error occurred while fetching server path context: {e}") from e
