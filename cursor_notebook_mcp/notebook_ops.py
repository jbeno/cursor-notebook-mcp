"""
Core notebook file operations (read, write, validate path).

These functions are designed to be independent of global state and 
receive necessary configuration (like allowed roots) explicitly.
"""

import os
import logging
from typing import List, Optional, Tuple
import asyncio
import nbformat
from .sftp_manager import SFTPManager
import posixpath # For remote paths
from urllib.parse import unquote
import sys

logger = logging.getLogger(__name__)

# Uncomment normalize_path and is_path_allowed
def normalize_path(path: str, allowed_roots: List[str], sftp_manager=None) -> str:
    """
    Normalize path for security checks, translating remote paths when needed.
    
    Args:
        path: The path to normalize
        allowed_roots: List of allowed root directories
        sftp_manager: Optional SFTP manager for remote paths
        
    Returns:
        Normalized path (local representation)
    """
    # Handle URL-encoded paths (especially important for Windows paths)
    if '%' in path:
        # URL decode the path first (e.g., /c%3A/path -> /c:/path)
        path = unquote(path)
        logger.debug(f"URL-decoded path: {path}")
    
    # Handle leading slash + drive letter pattern on Windows (like /C:/path)
    if sys.platform == "win32" and path.startswith('/') and len(path) > 3 and path[1].isalpha() and path[2] == ':':
        # Remove leading slash to get a proper Windows path (e.g., /C:/path -> C:/path)
        path = path[1:]
        logger.debug(f"Removed leading slash from Windows path: {path}")
    # Handle Linux-style path with Windows drive letter (/c:/path)
    elif path.startswith('/') and len(path) > 3 and path[1].isalpha() and path[2] == ':':
        # This might be a Windows path on a non-Windows system or just a URL-encoded path
        # We'll remove the leading slash if there's a drive letter pattern
        path = path[1:]
        logger.debug(f"Detected and normalized Windows-style path: {path}")
        
    # Handle SFTP paths
    if sftp_manager:
        try:
            # See if this is a remote path that needs translation
            is_remote, local_path, _ = sftp_manager.translate_path(path)
            
            # If we got a local translation, use it
            if not is_remote and local_path != path:
                logger.debug(f"Translated remote path '{path}' to local path '{local_path}'")
                return os.path.realpath(local_path)
                
            # Handle remote paths that didn't translate
            if is_remote:
                logger.debug(f"Path '{path}' is remote but was not translated")
        except Exception as e:
            logger.error(f"Error translating path '{path}': {e}")
    
    # Return absolute, normalized path
    return os.path.realpath(path)

def is_path_allowed(target_path: str, allowed_roots: List[str], sftp_manager=None) -> bool:
    """Checks if the target path is within one of the allowed roots."""
    if not allowed_roots:
        logger.warning("Security check skipped: No allowed roots configured.")
        return False # Or True? Defaulting to False for safety.

    try:
        # Normalize the path (handles SFTP translation)
        # Use the local representation returned by normalize_path for checking against local roots
        abs_target_path_local_repr = normalize_path(target_path, allowed_roots, sftp_manager)
        
    except Exception as e:
        logger.error(f"Error normalizing path '{target_path}' for permission check: {e}")
        return False # Cannot validate unresolved path

    # Standard check against allowed roots using the local representation
    for allowed_root in allowed_roots:
        try:
            # Ensure allowed_root is also absolute and resolved
            abs_allowed_root = os.path.normpath(os.path.realpath(allowed_root))
            # Use normpath and realpath on the target path as well for robust comparison
            abs_target_real = os.path.normpath(os.path.realpath(abs_target_path_local_repr))
            
            # For Windows paths, ensure consistent path separators
            if sys.platform == "win32" or (
                # Also normalize if it looks like we're comparing Windows-style paths
                # (has drive letter or contains backslashes)
                (abs_allowed_root.find('\\') >= 0 or (len(abs_allowed_root) > 1 and abs_allowed_root[1] == ':')) or
                (abs_target_real.find('\\') >= 0 or (len(abs_target_real) > 1 and abs_target_real[1] == ':'))
            ):
                # Replace backslashes with forward slashes for consistent comparison
                abs_allowed_root = abs_allowed_root.replace('\\', '/')
                abs_target_real = abs_target_real.replace('\\', '/')
                # Compare with forward slashes
                if abs_target_real == abs_allowed_root or abs_target_real.startswith(abs_allowed_root + '/'):
                    logger.debug(f"Windows path '{target_path}' is allowed within root '{allowed_root}'")
                    return True
            else:
                # Standard Unix path comparison with OS-specific separator
                if abs_target_real == abs_allowed_root or abs_target_real.startswith(abs_allowed_root + os.sep):
                    return True
        except Exception as e:
            logger.error(f"Error resolving allowed root '{allowed_root}' or checking path '{abs_target_path_local_repr}': {e}")
            continue # Try the next root

    # Log if disallowed, using the original path for user clarity
    logger.warning(f"Permission check failed: Path '{target_path}' (resolved to '{abs_target_path_local_repr}') is outside allowed roots: {allowed_roots}")
    return False

# --- New Simplified Resolution Logic --- 

def resolve_path_and_check_permissions(
    user_path: str,
    allowed_local_roots: List[str], # Local roots from --allow-root
    sftp_manager: Optional[SFTPManager] = None, # SFTP config
) -> Tuple[bool, str]:
    """
    Resolves user path, determines if remote/local, checks permissions.

    Returns:
        Tuple (is_remote, absolute_op_path)
        is_remote: Boolean indicating if the path is SFTP managed.
        absolute_op_path: The absolute path (remote or local) for file operations.

    Raises:
        ValueError: If path is invalid or cannot be resolved.
        PermissionError: If the resolved path is outside allowed roots.
    """
    if not user_path:
        raise ValueError("Path cannot be empty.")

    # Handle URL-encoded paths (especially for Windows paths)
    if '%' in user_path:
        user_path = unquote(user_path)
        logger.debug(f"URL-decoded path: {user_path}")
    
    # Handle Windows paths that start with a leading slash and drive letter
    if sys.platform == "win32" and user_path.startswith('/') and len(user_path) > 3 and user_path[1].isalpha() and user_path[2] == ':':
        user_path = user_path[1:]  # Remove leading slash for proper Windows path
        logger.debug(f"Removed leading slash from Windows path: {user_path}")
    # Also handle on non-Windows systems for consistency
    elif user_path.startswith('/') and len(user_path) > 3 and user_path[1].isalpha() and user_path[2] == ':':
        user_path = user_path[1:]
        logger.debug(f"Detected and normalized Windows-style path: {user_path}")

    is_remote = False
    absolute_op_path = None

    # Handle /workspace/ paths locally first (Cursor specific)
    if user_path.startswith("/workspace/"):
        if not allowed_local_roots:
             raise PermissionError("Access denied: /workspace/ paths require local allowed roots.")
        relative_path = user_path[len("/workspace/"):]
        absolute_op_path = os.path.normpath(os.path.join(allowed_local_roots[0], relative_path))
        is_remote = False
        logger.debug(f"Resolved /workspace path '{user_path}' to local '{absolute_op_path}'")
    # Check if SFTP is active
    elif sftp_manager:
        resolved_remote = sftp_manager._get_absolute_remote_path(user_path)
        if resolved_remote:
            is_remote = True
            absolute_op_path = resolved_remote
            logger.debug(f"Resolved path '{user_path}' to remote '{absolute_op_path}'")
        else:
            is_remote = False
            if os.path.isabs(user_path):
                 absolute_op_path = os.path.normpath(user_path)
            elif allowed_local_roots:
                 absolute_op_path = os.path.normpath(os.path.join(allowed_local_roots[0], user_path))
            else:
                 raise ValueError(f"Cannot resolve relative local path '{user_path}' without local allowed roots.")
            logger.debug(f"Path '{user_path}' did not resolve via SFTP mappings, treating as local '{absolute_op_path}'.")
    # No SFTP manager, purely local resolution
    else:
        is_remote = False
        if os.path.isabs(user_path):
            absolute_op_path = os.path.normpath(user_path)
        elif allowed_local_roots:
            absolute_op_path = os.path.normpath(os.path.join(allowed_local_roots[0], user_path))
        else:
            # If relative path and no allowed roots, it's inherently disallowed/unresolvable
            raise PermissionError(f"Access denied: Cannot resolve relative path '{user_path}' without allowed roots.")
        logger.debug(f"Resolved path '{user_path}' to local '{absolute_op_path}'")

    # Ensure we have a resolved path at this point
    if absolute_op_path is None:
         # This case should ideally be prevented by the logic above
         raise ValueError(f"Internal error: Path resolution failed for '{user_path}'")

    # --- Permission Check (Refined Logic) --- 
    is_allowed = False
    if is_remote and sftp_manager:
        # For remote paths, check if absolute_op_path falls under any *normalized* remote prefix
        for spec, (host, username, remote_prefix, _) in sftp_manager.path_mappings.items():
            try:
                sftp = sftp_manager.connections[host][1]
                user_home = sftp_manager._get_remote_home_dir(sftp, host, username)
                temp_remote_prefix = remote_prefix
                if temp_remote_prefix.startswith('~') and user_home:
                     temp_remote_prefix = posixpath.join(user_home, temp_remote_prefix[1:].lstrip('/'))
                
                normalized_remote_prefix = temp_remote_prefix # Start with potentially tilde-expanded
                # Attempt normalization robustly
                try:
                    normalized_remote_prefix = posixpath.normpath(normalized_remote_prefix)
                    if not posixpath.isabs(normalized_remote_prefix):
                        normalized_remote_prefix = sftp.normalize(normalized_remote_prefix)
                except Exception as norm_err:
                    logger.warning(f"Could not fully normalize remote prefix '{remote_prefix}' for mapping '{spec}', using '{normalized_remote_prefix}': {norm_err}")
                    # Continue comparison with the best normalized path we have

                if absolute_op_path == normalized_remote_prefix or \
                   absolute_op_path.startswith(normalized_remote_prefix + '/'):
                    is_allowed = True
                    logger.debug(f"Permission granted: Remote path '{absolute_op_path}' is within allowed prefix '{normalized_remote_prefix}'.")
                    break
            except KeyError:
                 logger.warning(f"No active connection for host {host} while checking permissions for '{spec}', skipping.")
                 continue
            except Exception as e:
                 logger.warning(f"Error normalizing/checking remote prefix '{remote_prefix}' for permissions: {e}")
                 continue
    else:
        # --- LOCAL PATH CHECK (Refined) ---
        if not allowed_local_roots:
            logger.error("Security Violation: Cannot check local path permission without allowed_local_roots.")
            # is_allowed remains False
        else:
            for root in allowed_local_roots:
                try:
                    # Ensure comparison uses resolved absolute paths for both
                    abs_root = os.path.realpath(root)
                    # Resolve potential symlinks etc. in the operation path too
                    abs_op_path_real = os.path.realpath(absolute_op_path)

                    # Check if the real operation path is exactly the root or starts with root + separator
                    if abs_op_path_real == abs_root or abs_op_path_real.startswith(abs_root + os.sep):
                        is_allowed = True
                        logger.debug(f"Permission granted: Local path '{abs_op_path_real}' is within allowed root '{abs_root}'.")
                        break # Found an allowed root
                except Exception as e:
                     # Log error if realpath fails for a root or the op path
                     logger.error(f"Error resolving real paths during permission check: root='{root}', op_path='{absolute_op_path}': {e}")
                     continue # Try next root

    if not is_allowed:
        scope = "remote SFTP roots defined by --sftp-root" if is_remote else "local allowed roots defined by --allow-root"
        logger.error(f"Security Violation: Resolved path '{absolute_op_path}' is outside allowed {scope}.")
        # Ensure the error message clearly states which roots were checked
        raise PermissionError(f"Access denied: Path '{user_path}' resolves to '{absolute_op_path}' which is outside allowed {scope}.")

    return is_remote, absolute_op_path

# --- Read/Write Operations --- 

async def read_notebook(
    notebook_path: str, # Original user path
    allowed_roots: List[str],
    sftp_manager: Optional[SFTPManager] = None,
) -> nbformat.NotebookNode:
    """Reads a notebook file safely after resolving path and checking permissions."""
    if not notebook_path or not notebook_path.endswith(".ipynb"):
         raise ValueError(f"Invalid notebook path: '{notebook_path}'")

    try:
        is_remote, absolute_op_path = resolve_path_and_check_permissions(
            notebook_path, allowed_roots, sftp_manager
        )

        logger.debug(f"Reading notebook '{notebook_path}' resolved to '{absolute_op_path}' (is_remote={is_remote})")

        if is_remote:
            if not sftp_manager:
                 raise ConnectionError("SFTP manager required but not available for remote read.")
            # SFTP read using the absolute remote path
            content_bytes = await asyncio.to_thread(sftp_manager.read_file, absolute_op_path)
            content_str = content_bytes.decode('utf-8')
        else:
            # Local read using the absolute local path
            try:
                # Standard sync read in thread
                with open(absolute_op_path, "r", encoding='utf-8') as f:
                     content_str = await asyncio.to_thread(f.read)
            except FileNotFoundError:
                logger.error(f"Local notebook file not found at: {absolute_op_path}")
                raise FileNotFoundError(f"Notebook file not found at: {absolute_op_path}")
            except Exception as e:
                logger.error(f"Error reading local notebook {absolute_op_path}: {e}")
                raise IOError(f"Failed to read local notebook file '{absolute_op_path}': {e}") from e

        # Parse content
        nb = nbformat.reads(content_str, as_version=4)
        logger.info(f"Successfully read and parsed notebook: {notebook_path} (from {absolute_op_path})")
        return nb

    except (ValueError, PermissionError, FileNotFoundError, ConnectionError, IOError) as e:
         logger.error(f"Failed to read notebook '{notebook_path}': {e}")
         raise # Re-raise specific expected errors
    except Exception as e:
         logger.exception(f"Unexpected error reading notebook '{notebook_path}': {e}")
         raise IOError(f"An unexpected error occurred while reading '{notebook_path}': {e}") from e

async def write_notebook(
    notebook_path: str, # Original user path
    nb: nbformat.NotebookNode,
    allowed_roots: List[str],
    sftp_manager: Optional[SFTPManager] = None
):
    """Writes a notebook object safely after resolving path and checking permissions."""
    if not notebook_path:
         raise ValueError("Notebook path cannot be empty.")
    if not notebook_path.endswith('.ipynb'):
         # Add check for write as well
         raise ValueError(f"Invalid notebook path for writing: '{notebook_path}'")

    # Validate notebook content first
    try:
        nbformat.validate(nb)
    except nbformat.ValidationError as e:
        logger.error(f"Invalid notebook format before writing '{notebook_path}': {e}")
        raise ValueError(f"Notebook content is invalid: {e}") from e

    try:
        is_remote, absolute_op_path = resolve_path_and_check_permissions(
            notebook_path, allowed_roots, sftp_manager
        )

        logger.debug(f"Writing notebook '{notebook_path}' resolved to '{absolute_op_path}' (is_remote={is_remote})")

        if is_remote:
            if not sftp_manager:
                 raise ConnectionError("SFTP manager required but not available for remote write.")
            # SFTP write using the absolute remote path
            json_content = nbformat.writes(nb)
            await asyncio.to_thread(sftp_manager.write_file, absolute_op_path, json_content)
            logger.info(f"Successfully wrote remote notebook: {absolute_op_path}")
        else:
            # Local write using the absolute local path
            parent_dir = os.path.dirname(absolute_op_path)
            if parent_dir:
                try:
                    os.makedirs(parent_dir, exist_ok=True)
                except OSError as e:
                    logger.error(f"Local: Failed to create parent directory '{parent_dir}': {e}")
                    raise IOError(f"Could not create local directory '{parent_dir}': {e}") from e
            try:
                # nbformat.write needs the node, not the JSON string
                await asyncio.to_thread(nbformat.write, nb, absolute_op_path)
                logger.info(f"Successfully wrote local notebook: {absolute_op_path}")
            except Exception as e:
                 logger.error(f"Failed to write local notebook file '{absolute_op_path}' using nbformat: {e}")
                 raise IOError(f"Failed to write local notebook '{absolute_op_path}': {e}") from e
            
    except (ValueError, PermissionError, FileNotFoundError, ConnectionError, IOError) as e:
        logger.error(f"Failed to write notebook '{notebook_path}': {e}")
        raise # Re-raise specific expected errors
    except Exception as e:
        logger.exception(f"Unexpected error writing notebook '{notebook_path}': {e}")
        raise IOError(f"An unexpected error occurred while writing '{notebook_path}': {e}") from e
# End of file - removed commented-out old functions 