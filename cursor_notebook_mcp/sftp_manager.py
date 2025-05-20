"""
SFTP connection manager for remote notebook operations.

Handles SSH/SFTP connectivity using paramiko to provide transparent
access to remote notebook files without relying on platform-specific tools.
"""

import os
import re
import logging
import tempfile
import getpass
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import posixpath # Make sure this is imported at the top
import errno # Import errno for specific error codes

try:
    import paramiko
except ImportError:
    raise ImportError("Paramiko is required for SFTP operations. Install with 'pip install paramiko'.")

logger = logging.getLogger(__name__)

class InteractiveHandler:
    """Interactive authentication handler for SSH connections."""
    
    def __init__(self, password: Optional[str] = None):
        self.password = password
        
    def handler(self, title, instructions, prompt_list):
        """Handle interactive authentication prompts from SSH server."""
        logger.debug(f"Interactive Auth: {title}, {instructions}, {prompt_list}")
        responses = []
        
        for prompt in prompt_list:
            prompt_str = prompt[0]
            echo = prompt[1]  # Whether to echo the input (False for passwords)
            
            # If password is already provided and this looks like a password prompt, use it
            if not echo and self.password and ("password" in prompt_str.lower() or "passphrase" in prompt_str.lower()):
                logger.debug("Using provided password for prompt")
                responses.append(self.password)
                self.password = None  # Use password only once
                continue
                
            # Otherwise, prompt the user interactively
            if echo:
                response = input(f"{prompt_str}: ")
            else:
                response = getpass.getpass(f"{prompt_str}: ")
            responses.append(response)
            
        return responses

class SFTPManager:
    """Manages SFTP connections and path translations."""
    
    def __init__(self):
        # Store connections by host
        self.connections = {}  # host -> (client, sftp)
        # Path mapping: remote_path_prefix -> (host, username, remote_path, local_prefix)
        self.path_mappings = {}
        # Cache of remote file existence checks
        self._path_exists_cache = {}
        
    def add_connection(self, host: str, username: str, password: Optional[str]=None, 
                      key_file: Optional[str]=None, port: int=22, 
                      use_agent: bool=True, interactive: bool=True,
                      auth_mode: str="auto") -> bool:
        """
        Add and authenticate an SFTP connection with support for interactive auth.
        
        Args:
            host: SSH server hostname
            username: SSH username
            password: SSH password (optional)
            key_file: Path to SSH private key file (optional)
            port: SSH port
            use_agent: Whether to attempt using SSH agent for authentication
            interactive: Whether to allow interactive authentication for 2FA
            auth_mode: Authentication mode: "auto", "key", "password", "key+interactive", 
                       "password+interactive", "interactive"
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            logger.info(f"Connecting to {host}:{port} as {username} using auth mode: {auth_mode}")
            
            # For servers requiring 2FA, we need to handle custom authentication
            # First try to connect with just public key auth
            connect_kwargs = {
                "hostname": host,
                "port": port,
                "username": username,
                "allow_agent": use_agent,
                "look_for_keys": True,  # Allow looking for keys
            }
            
            # Check if the key file is encrypted and needs a passphrase
            key_needs_passphrase = False
            if key_file and os.path.exists(key_file):
                try:
                    # Try to load the key without a password first
                    paramiko.RSAKey.from_private_key_file(key_file)
                except paramiko.ssh_exception.PasswordRequiredException:
                    # The key is encrypted and needs a passphrase
                    key_needs_passphrase = True
                    if not password and interactive:
                        password = getpass.getpass(f"Enter passphrase for key '{key_file}': ")
                except Exception as e:
                    logger.warning(f"Error checking private key: {e}")
            
            # Add key file if provided
            if key_file:
                connect_kwargs["key_filename"] = key_file
                logger.info(f"Using key file for authentication: {key_file}")
                
            # Add password if provided (could be key password or server password)
            if password:
                connect_kwargs["password"] = password
                if key_needs_passphrase:
                    logger.info("Using password for key authentication")
                else:
                    logger.info("Using password for server authentication")
            
            # Handle different authentication modes
            if auth_mode == "auto":
                try:
                    # Try standard connection first
                    logger.debug("Attempting standard client.connect...")
                    client.connect(**connect_kwargs)
                    logger.debug("Standard client.connect successful.")
                except (paramiko.AuthenticationException, paramiko.ssh_exception.PasswordRequiredException) as auth_err:
                    # Handle specific auth errors, potentially falling back to interactive
                    logger.debug(f"Standard auth failed ({type(auth_err).__name__}), checking interactive fallback...")
                    if interactive:
                        if isinstance(auth_err, paramiko.ssh_exception.PasswordRequiredException):
                             logger.info("Key requires a passphrase")
                             password = getpass.getpass(f"Enter passphrase for key '{key_file}': ")
                             connect_kwargs["password"] = password
                             # Retry standard connect with passphrase
                             try:
                                 client.connect(**connect_kwargs)
                             except Exception as retry_err:
                                 logger.error(f"Retry connect with passphrase failed: {retry_err}")
                                 # If retry fails, attempt interactive transport as last resort
                                 self._connect_with_interactive(client, host, port, username, password, key_file, use_agent)
                        else: # Original error was AuthenticationException
                            # Try manual Transport approach for 2FA support
                            logger.debug("Trying interactive mode via Transport...")
                            self._connect_with_interactive(client, host, port, username, password, key_file, use_agent)
                    else:
                         logger.error(f"Standard auth failed and interactive mode disabled: {auth_err}")
                         raise auth_err # Re-raise original auth error
                except Exception as e:
                    # If initial connect fails with *non-auth* error, don't retry in generic handler.
                    # Log the error and let it fall through to the failure return.
                    logger.warning(f"Initial connection attempt failed with unexpected error: {e}. NOT retrying connection.")
                    raise # Re-raise the unexpected connection error
            elif auth_mode == "key":
                # Pure key-based auth without any interactive prompts
                connect_kwargs["allow_agent"] = use_agent
                connect_kwargs["look_for_keys"] = True
                client.connect(**connect_kwargs)
            elif auth_mode == "password":
                # Pure password auth
                connect_kwargs["allow_agent"] = False
                connect_kwargs["look_for_keys"] = False
                client.connect(**connect_kwargs)
            elif auth_mode == "key+interactive":
                # Direct approach for 2FA servers using a single authentication handshake
                logger.info("Using key + interactive authentication mode")
                
                # For servers requiring both key and 2FA, we need a special Transport approach
                # that follows the required sequence: publickey, then keyboard-interactive
                transport = paramiko.Transport((host, port))
                transport.start_client()
                
                # First, try public key authentication
                logger.info("Attempting public key authentication first...")
                
                if key_file:
                    try:
                        # Try to load the private key
                        if password:
                            try:
                                key = paramiko.RSAKey.from_private_key_file(key_file, password)
                            except:
                                # Try other key types
                                try:
                                    key = paramiko.DSSKey.from_private_key_file(key_file, password)
                                except:
                                    key = paramiko.Ed25519Key.from_private_key_file(key_file, password)
                        else:
                            try:
                                key = paramiko.RSAKey.from_private_key_file(key_file)
                            except:
                                # Try other key types
                                try:
                                    key = paramiko.DSSKey.from_private_key_file(key_file)
                                except:
                                    key = paramiko.Ed25519Key.from_private_key_file(key_file)
                        
                        # Attempt public key authentication first
                        transport.auth_publickey(username, key)
                        logger.info("Public key authentication successful")
                    except Exception as key_err:
                        logger.error(f"Public key authentication failed: {key_err}")
                        raise  # If public key auth fails in this mode, don't proceed
                else:
                    # Try agent keys if no key file specified
                    agent_auth_success = False
                    if use_agent:
                        try:
                            agent = paramiko.Agent()
                            for key in agent.get_keys():
                                try:
                                    transport.auth_publickey(username, key)
                                    if transport.is_authenticated():
                                        logger.info("Public key authentication via agent successful")
                                        agent_auth_success = True
                                        break
                                except:
                                    continue
                        except Exception as agent_err:
                            logger.warning(f"Agent auth failed: {agent_err}")
                    
                    if not agent_auth_success:
                        logger.error("No key file provided and agent authentication failed")
                        raise paramiko.AuthenticationException("Public key authentication failed - required for key+interactive mode")
                
                # Now handle the keyboard-interactive part (only if public key was successful)
                if not transport.is_authenticated():
                    auth_handler = InteractiveHandler(password)
                    logger.info("Public key authentication succeeded, now proceeding with interactive authentication")
                    transport.auth_interactive(username, auth_handler.handler)
                
                # Create client from authenticated transport
                if transport.is_authenticated():
                    client._transport = transport
                else:
                    raise paramiko.AuthenticationException("Authentication failed with key+interactive mode")
            elif auth_mode == "password+interactive":
                # Direct approach for 2FA servers requiring password then keyboard-interactive
                logger.info("Using password + interactive authentication mode")
                
                # Ensure we have a password for the initial authentication
                if not password and interactive:
                    password = getpass.getpass(f"SSH password for {username}@{host}: ")
                elif not password:
                    raise paramiko.AuthenticationException("Password required for password+interactive mode")
                
                # For servers requiring both password and 2FA, we need a special Transport approach
                # that follows the required sequence: password, then keyboard-interactive
                transport = paramiko.Transport((host, port))
                transport.start_client()
                
                # First try password authentication
                logger.info("Attempting password authentication first...")
                try:
                    transport.auth_password(username, password)
                    logger.info("Password authentication successful")
                except Exception as pwd_err:
                    logger.error(f"Password authentication failed: {pwd_err}")
                    raise  # If password auth fails in this mode, don't proceed
                
                # Now handle the keyboard-interactive part for 2FA (only if password was successful)
                if not transport.is_authenticated():
                    # For the 2FA step, don't reuse the password
                    auth_handler = InteractiveHandler(None)
                    logger.info("Password authentication succeeded, now proceeding with interactive authentication")
                    transport.auth_interactive(username, auth_handler.handler)
                
                # Create client from authenticated transport
                if transport.is_authenticated():
                    client._transport = transport
                else:
                    raise paramiko.AuthenticationException("Authentication failed with password+interactive mode")
            elif auth_mode == "interactive":
                # Pure interactive auth without key attempt first
                logger.info("Using pure interactive authentication mode")
                transport = paramiko.Transport((host, port))
                transport.start_client()
                
                auth_handler = InteractiveHandler(password)
                transport.auth_interactive(username, auth_handler.handler)
                
                # Create client from authenticated transport
                if transport.is_authenticated():
                    client._transport = transport
                else:
                    raise paramiko.AuthenticationException("Interactive authentication failed")
            else:
                raise ValueError(f"Unknown auth_mode: {auth_mode}")
            
            # Open SFTP session only if authentication seems successful
            # Use client.get_transport().is_authenticated() for reliability
            transport = client.get_transport()
            if transport and transport.is_authenticated():
                sftp = client.open_sftp()
                self.connections[host] = (client, sftp)
                logger.info(f"Successfully connected and opened SFTP to {host}")
                return True
            else:
                 # This path might be reached if using Transport modes directly
                 logger.error(f"Authentication failed or transport not active for {host}")
                 if client: client.close()
                 return False
                 
        except Exception as e:
            # Catch all other exceptions during the process
            # Includes the re-raised non-auth connection error from above
            logger.error(f"Failed to connect to {host}: {e}")
            if client: client.close() # Ensure client is closed on any error
            return False
    
    def _connect_with_interactive(self, client, host, port, username, password, key_file, use_agent):
        """
        Helper method to connect with key + interactive authentication.
        Used for the key+interactive auth mode and as a fallback in auto mode.
        """
        # Set up authentication using Transport for more control
        transport = paramiko.Transport((host, port))
        transport.start_client()
        
        # Try key-based auth first
        if key_file:
            try:
                # Try to load the private key
                if password:
                    try:
                        key = paramiko.RSAKey.from_private_key_file(key_file, password)
                    except:
                        # Try other key types
                        try:
                            key = paramiko.DSSKey.from_private_key_file(key_file, password)
                        except:
                            key = paramiko.Ed25519Key.from_private_key_file(key_file, password)
                else:
                    try:
                        key = paramiko.RSAKey.from_private_key_file(key_file)
                    except:
                        # Try other key types
                        try:
                            key = paramiko.DSSKey.from_private_key_file(key_file)
                        except:
                            key = paramiko.Ed25519Key.from_private_key_file(key_file)
                            
                transport.auth_publickey(username, key)
            except Exception as key_err:
                logger.warning(f"Key auth failed: {key_err}")
        
        # If we get here and still need auth, try interactive with agent
        if not transport.is_authenticated() and use_agent:
            try:
                agent = paramiko.Agent()
                for key in agent.get_keys():
                    try:
                        transport.auth_publickey(username, key)
                        if transport.is_authenticated():
                            break
                    except:
                        pass
            except Exception as agent_err:
                logger.warning(f"Agent auth failed: {agent_err}")
                    
        # If we're still not authenticated, try keyboard-interactive
        if not transport.is_authenticated():
            try:
                auth_handler = InteractiveHandler(password)
                transport.auth_interactive(username, auth_handler.handler)
            except Exception as interactive_err:
                logger.warning(f"Interactive auth failed: {interactive_err}")
                
        # Create client from authenticated transport
        if transport.is_authenticated():
            client._transport = transport
        else:
            raise paramiko.AuthenticationException("All authentication methods failed")
    
    def add_path_mapping(self, remote_spec: str, local_path: str) -> Tuple[str, str, str]:
        """
        Add a path mapping for translating between remote and local paths.
        
        Args:
            remote_spec: Connection string in format user@host:/path
            local_path: Local path to map to
            
        Returns:
            Tuple of (username, host, remote_path)
        """
        # Parse remote spec: user@host:/remote/path
        match = re.match(r"(.+)@(.+):(.+)", remote_spec)
        if not match:
            raise ValueError(f"Invalid remote spec: {remote_spec}")
            
        username, host, remote_path = match.groups()
        
        # Ensure trailing slash on paths
        if not remote_path.endswith('/'):
            remote_path += '/'
        if not local_path.endswith('/'):
            local_path += '/'
            
        # Store the mapping
        self.path_mappings[remote_spec] = (host, username, remote_path, local_path)
        logger.info(f"Added path mapping: {remote_spec} -> {local_path}")
        return username, host, remote_path
        
    def translate_path(self, path: str) -> Tuple[bool, str, Optional[Tuple]]:
        """
        Translate a path between remote and local representations.
        Also attempts to identify if an absolute path matches a remote mapping.
        """
        path_norm = os.path.normpath(path) # OS-specific normalization for input `path`

        # Check all mappings
        for remote_spec, (host, username, remote_prefix, local_prefix) in self.path_mappings.items():
            # remote_prefix is from user@host:/remote/path, so it's a remote path string
            # local_prefix is the local temporary path for this mapping
            remote_prefix_norm = posixpath.normpath(remote_prefix) # Use posixpath for remote prefix
            local_prefix_norm = os.path.normpath(local_prefix)    # Use os.path for local prefix

            # 1. Check if local path maps to remote
            # path_norm here is an OS-specific local path
            if path_norm.startswith(local_prefix_norm):
                # Ensure it's a proper subpath or exact match
                if path_norm == local_prefix_norm or path_norm.startswith(local_prefix_norm + os.sep):
                    if host in self.connections:
                         # Construct remote path relative to remote_prefix
                         relative_part = os.path.relpath(path_norm, local_prefix_norm)
                         # Convert relative_part to use Posix separators before joining
                         posix_relative_part = relative_part.replace(os.sep, '/')
                         remote_path_translated = posixpath.join(remote_prefix_norm, posix_relative_part) if posix_relative_part != '.' else remote_prefix_norm
                         logger.debug(f"Translated local '{path}' to remote '{remote_path_translated}' using spec '{remote_spec}'")
                         return True, posixpath.normpath(remote_path_translated), self.connections[host]
                    else:
                         logger.error(f"No active SFTP connection for host {host} (mapping: {remote_spec}) when translating local path {path}")
                         return False, path_norm, None

            # 2. Check if the input path *is* an absolute remote path matching this mapping prefix
            # This is crucial for handling calls with pre-resolved absolute remote paths
            # Convert path_norm to a Posix-style string for comparison with remote_prefix_norm
            path_as_posix = path.replace(os.sep, '/') # Use original path for safer replacement
            # If original path was already posix, path_norm might have converted some things on windows
            # If path was like "C:/foo/bar", path_norm is "C:\foo\bar", path_as_posix is "C:/foo/bar"
            # If path was like "/foo/bar", path_norm on win is "\foo\bar", path_as_posix is "/foo/bar"
            # Ensure the path_as_posix is also normalized using posixpath for a clean comparison
            normalized_path_as_posix = posixpath.normpath(path_as_posix)

            if normalized_path_as_posix.startswith(remote_prefix_norm):
                # Ensure it's a proper subpath or exact match
                if normalized_path_as_posix == remote_prefix_norm or normalized_path_as_posix.startswith(remote_prefix_norm + '/'): 
                    if host in self.connections:
                         logger.debug(f"Input path '{path}' (as Posix '{normalized_path_as_posix}') matches remote prefix '{remote_prefix_norm}' for host '{host}'")
                         return True, normalized_path_as_posix, self.connections[host]
                    else:
                         logger.error(f"No active SFTP connection for host {host} (mapping: {remote_spec}) when checking remote path {path}")
                         return False, path_norm, None # Return OS-specific local if no connection

        # 3. If no remote mapping matched, assume it's a local path
        logger.debug(f"Path '{path}' did not match any SFTP mapping prefixes, treating as local.")
        return False, path_norm, None # path_norm is os.path.normpath(path)

    def path_exists(self, path: str, bypass_cache: bool = False) -> bool:
        """
        Check if a path exists (works for remote or local paths).
        Handles tilde expansion before checking.

        Args:
            path: The path string to check.
            bypass_cache: If True, ignore cached result and do not update cache.
        """
        try:
            # Determine if local or remote AND get the correct absolute path for checking
            is_remote, check_path, conn_info = self._resolve_path_for_operation(path)

            if not bypass_cache:
                if check_path in self._path_exists_cache:
                    logger.debug(f"SFTP Cache: Returning cached existence for {check_path}: {self._path_exists_cache[check_path]}")
                    return self._path_exists_cache[check_path]
            else:
                logger.debug(f"SFTP Cache: Bypassing cache for existence check of {check_path}")

            exists = False
            if is_remote:
                _, sftp = conn_info
                try:
                    logger.debug(f"SFTP: Checking existence of {check_path}")
                    sftp.stat(check_path)
                    exists = True
                except FileNotFoundError:
                    exists = False
                except Exception as sftp_e:
                    logger.error(f"SFTP Error checking existence of {check_path}: {sftp_e}")
                    exists = False # Assume false on error
            else:
                logger.debug(f"Local: Checking existence of {check_path}")
                exists = os.path.exists(check_path)

            if not bypass_cache:
                self._path_exists_cache[check_path] = exists
                logger.debug(f"SFTP Cache: Updated cache for {check_path}: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error determining path type or checking existence for '{path}': {e}")
            return False # Default to False on resolution errors

    def list_dir(self, path: str) -> List[str]:
        """
        List directory contents (local or remote).
        Handles tilde expansion.
        """
        is_remote, op_path, conn_info = self._resolve_path_for_operation(path)
        try:
            if is_remote:
                _, sftp = conn_info
                logger.debug(f"SFTP: Listing directory {op_path}")
                return sftp.listdir(op_path)
            else:
                logger.debug(f"Local: Listing directory {op_path}")
                return os.listdir(op_path)
        except IOError as e:
            if is_remote:
                if getattr(e, 'errno', None) == errno.ENOENT or "no such file" in str(e).lower():
                    raise FileNotFoundError(f"SFTP: Directory not found at {op_path}: {e}") from e
                elif getattr(e, 'errno', None) == errno.EACCES or "permission denied" in str(e).lower():
                    raise PermissionError(f"SFTP: Permission denied for {op_path}: {e}") from e
            raise # Re-raise original or already translated error for local, or untranslated SFTP IOError
        except Exception as e:
            logger.error(f"Error listing directory '{path}' (resolved to '{op_path}'): {e}")
            raise # Re-raise other exceptions

    def read_file(self, path: str) -> bytes:
        """
        Read file contents (local or remote).
        Handles tilde expansion.
        """
        is_remote, op_path, conn_info = self._resolve_path_for_operation(path)
        try:
            if is_remote:
                _, sftp = conn_info
                logger.debug(f"SFTP: Reading file {op_path}")
                with sftp.open(op_path, 'rb') as f:
                    return f.read()
            else:
                logger.debug(f"Local: Reading file {op_path}")
                with open(op_path, 'rb') as f:
                    return f.read()
        except IOError as e:
            if is_remote:
                if getattr(e, 'errno', None) == errno.ENOENT or "no such file" in str(e).lower():
                    raise FileNotFoundError(f"SFTP: File not found at {op_path}: {e}") from e
                elif getattr(e, 'errno', None) == errno.EACCES or "permission denied" in str(e).lower():
                    raise PermissionError(f"SFTP: Permission denied for {op_path}: {e}") from e
            # For local, IOError could be FileNotFoundError or PermissionError, let it propagate or be more specific if needed
            # If os.open raises, it's usually specific enough (e.g. FileNotFoundError, PermissionError)
            raise # Re-raise original or already translated error for local, or untranslated SFTP IOError
        except Exception as e:
            logger.error(f"Error reading file '{path}' (resolved to '{op_path}'): {e}")
            raise # Re-raise other exceptions

    def write_file(self, path: str, content: Union[str, bytes]) -> None:
        """
        Write file contents (local or remote).
        Handles tilde expansion and parent directory creation.
        """
        is_remote, op_path, conn_info = self._resolve_path_for_operation(path)
        try:
            content_bytes = content.encode('utf-8') if isinstance(content, str) else content

            if is_remote:
                _, sftp = conn_info
                logger.debug(f"SFTP: Writing file {op_path}")
                parent_dir = posixpath.dirname(op_path) # Use posixpath for remote
                if parent_dir and parent_dir != '/':
                    # _sftp_makedirs will handle its own specific errors
                    self._sftp_makedirs(sftp, parent_dir)

                with sftp.open(op_path, 'wb') as f:
                    f.set_pipelined(True)
                    f.write(content_bytes)
                logger.info(f"SFTP: Successfully wrote file: {op_path}")
            else:
                # Local write
                logger.debug(f"Local: Writing file {op_path}")
                parent_dir = os.path.dirname(op_path)
                if parent_dir:
                    try:
                        os.makedirs(parent_dir, exist_ok=True)
                    except OSError as e:
                        logger.error(f"Local: Failed to create parent directory '{parent_dir}': {e}")
                        # Let this be specific, e.g. PermissionError from os.makedirs
                        raise IOError(f"Could not create local directory '{parent_dir}': {e}") from e 
                with open(op_path, 'wb') as f:
                    f.write(content_bytes)
                logger.info(f"Local: Successfully wrote file: {op_path}")

        except IOError as e: # Catch IOErrors from sftp.open or local open/makedirs
            # Log the error first regardless of which specific error it is
            logger.error(f"Error writing file '{path}' (resolved to '{op_path}'): {e}")
            
            if is_remote:
                # Check errno for sftp.open errors
                if getattr(e, 'errno', None) == errno.ENOENT or "no such file" in str(e).lower(): # Should be caught by _sftp_makedirs if parent, but could be file itself
                    raise FileNotFoundError(f"SFTP: File path not found or component missing for {op_path}: {e}") from e
                elif getattr(e, 'errno', None) == errno.EACCES or "permission denied" in str(e).lower():
                    # Add specific permission context for SFTP errors
                    host = conn_info[0].get_transport().getpeername()[0] if conn_info and conn_info[0] else "unknown_host"
                    username = "unknown_user"
                    for spec, (h, u, _, _) in self.path_mappings.items():
                        if h == host:
                            username = u
                            break
                    logger.error(f"Verify SFTP user '{username}' has write permissions in '{posixpath.dirname(op_path)}' on host '{host}'.")
                    raise PermissionError(f"SFTP: Permission denied for {op_path}: {e}") from e
                # Add check for EISDIR if trying to open a directory as a file for writing
                elif getattr(e, 'errno', None) == errno.EISDIR or "is a directory" in str(e).lower():
                     raise IsADirectoryError(f"SFTP: Cannot write to path, it is a directory: {op_path}: {e}") from e

            # For local errors, os.makedirs or open() usually raise specific enough exceptions (PermissionError, FileNotFoundError etc.)
            # If they raised a generic OSError/IOError not caught by specific handlers, it propagates here.
            # The original error `e` from local ops is often more specific than a generic wrapper.
            
            # Check for permission denied strings in the generic IOError case
            if is_remote and "permission denied" in str(e).lower():
                host = conn_info[0].get_transport().getpeername()[0] if conn_info and conn_info[0] else "unknown_host"
                username = "unknown_user"
                for spec, (h, u, _, _) in self.path_mappings.items():
                    if h == host:
                        username = u
                        break
                logger.error(f"Verify SFTP user '{username}' has write permissions in '{posixpath.dirname(op_path)}' on host '{host}'.")
            
            raise # Re-raise original or already translated error, or untranslated SFTP IOError
        except Exception as e:
            logger.error(f"Error writing file '{path}' (resolved to '{op_path}'): {e}")
            # Add specific SFTP permission context if possible (already in original code, good)
            if is_remote and isinstance(e, (IOError, OSError)) and ("Permission denied" in str(e) or "Operation not supported" in str(e)):
                 host = conn_info[0].get_transport().getpeername()[0] if conn_info and conn_info[0] else "unknown_host"
                 username = "unknown_user"
                 for spec, (h, u, _, _) in self.path_mappings.items():
                     if h == host:
                         username = u
                         break
                 logger.error(f"Verify SFTP user '{username}' has write permissions in '{posixpath.dirname(op_path)}' on host '{host}'.")
            raise # Re-raise other exceptions

    def remove_file(self, path: str) -> None:
        """
        Remove a file (local or remote).
        Handles tilde expansion.
        """
        is_remote, op_path, conn_info = self._resolve_path_for_operation(path)
        try:
            if is_remote:
                _, sftp = conn_info
                logger.debug(f"SFTP: Removing file {op_path}")
                sftp.remove(op_path)
                logger.info(f"SFTP: Successfully removed file: {op_path}")
            else:
                # Local remove
                logger.debug(f"Local: Removing file {op_path}")
                os.remove(op_path)
                logger.info(f"Local: Successfully removed file: {op_path}")

        except IOError as e: # Catches os.remove and sftp.remove IOErrors
            if is_remote:
                if getattr(e, 'errno', None) == errno.ENOENT or "no such file" in str(e).lower():
                    raise FileNotFoundError(f"SFTP: File not found for removal at {op_path}: {e}") from e
                elif getattr(e, 'errno', None) == errno.EACCES or "permission denied" in str(e).lower():
                    raise PermissionError(f"SFTP: Permission denied for removing {op_path}: {e}") from e
                # Paramiko might raise IOError with errno.EISDIR if trying to remove a non-empty directory with sftp.remove
                # However, sftp.remove is for files. For directories, it's sftp.rmdir().
                # If it's a directory, it might be an SFTP-specific error string or a generic IOError.
                elif getattr(e, 'errno', None) == errno.EISDIR or "is a directory" in str(e).lower():
                     raise IsADirectoryError(f"SFTP: Path is a directory, cannot remove with remove_file: {op_path}: {e}") from e

            # For local, os.remove raises FileNotFoundError, IsADirectoryError, PermissionError directly.
            # So if we get here for local, it's likely already specific.
            raise # Re-raise error (already specific for local, or translated/original for SFTP)
        except FileNotFoundError:
             # This explicit catch is mainly for local os.remove if it raised FileNotFoundError directly
             # And if the IOError above didn't catch & translate an SFTP FileNotFoundError first.
             # For SFTP, the IOError catch above should handle it.
            logger.warning(f"File not found for removal '{path}' (resolved to '{op_path}'). Raising FileNotFoundError.")
            raise FileNotFoundError(f"File not found for removal at {op_path}")
        except Exception as e:
            logger.error(f"Error removing file '{path}' (resolved to '{op_path}'): {e}")
            raise # Re-raise other exceptions

    def rename_file(self, old_path: str, new_path: str) -> None:
        """
        Rename/move a file (local or remote).
        Handles tilde expansion for both paths.
        Assumes both paths reside on the same target (both local or both remote on same host).
        """
        is_remote_old, op_old_path, conn_info_old = self._resolve_path_for_operation(old_path)
        is_remote_new, op_new_path, conn_info_new = self._resolve_path_for_operation(new_path)
        try:
            if is_remote_old != is_remote_new:
                raise IOError(f"Cannot rename/move between local and remote storage ({old_path} -> {new_path}).")

            if is_remote_old:
                if not conn_info_old or not conn_info_new or conn_info_old[0] != conn_info_new[0]:
                     raise ConnectionError("SFTP rename requires both paths to be on the same connected host.")
                _, sftp = conn_info_old
                
                new_parent_dir = posixpath.dirname(op_new_path)
                if new_parent_dir and new_parent_dir != '/':
                     # _sftp_makedirs will handle its own specific errors
                     self._sftp_makedirs(sftp, new_parent_dir)

                logger.debug(f"SFTP: Renaming {op_old_path} to {op_new_path}")
                sftp.rename(op_old_path, op_new_path) # sftp.rename can raise IOError for various reasons
                logger.info(f"SFTP: Successfully renamed {op_old_path} to {op_new_path}")
            else:
                # Local rename
                new_parent_dir = os.path.dirname(op_new_path)
                if new_parent_dir:
                     try:
                         os.makedirs(new_parent_dir, exist_ok=True)
                     except OSError as e:
                         logger.error(f"Local: Failed to create parent directory '{new_parent_dir}' for rename: {e}")
                         raise IOError(f"Could not create local directory '{new_parent_dir}': {e}") from e

                logger.debug(f"Local: Renaming {op_old_path} to {op_new_path}")
                os.rename(op_old_path, op_new_path) # os.rename raises specific errors
                logger.info(f"Local: Successfully renamed {op_old_path} to {op_new_path}")

        except IOError as e: # Catches IOErrors from sftp.rename, os.rename, or _sftp_makedirs (if it raises IOError not Perm/FNF)
            # Log the error first regardless of which specific error it is
            logger.error(f"Error renaming file '{old_path}' to '{new_path}' (resolved to '{op_old_path}' -> '{op_new_path}'): {e}")
            
            if is_remote_old:
                # paramiko sftp.rename can throw IOError for many reasons:
                # ENOENT if old_path doesn't exist or new_path parent component is missing (though _sftp_makedirs handles latter)
                # EACCES for permission issues
                # EEXIST or similar if new_path exists (though some servers might overwrite)
                # Check specific conditions for sftp.rename
                if getattr(e, 'errno', None) == errno.ENOENT or "no such file" in str(e).lower():
                    # Could be old_path not found, or a component of new_path not found (less likely due to makedirs)
                    raise FileNotFoundError(f"SFTP: Source path or component of destination path not found for rename {op_old_path} -> {op_new_path}: {e}") from e
                elif getattr(e, 'errno', None) == errno.EACCES or "permission denied" in str(e).lower():
                    # Add specific permission context for SFTP errors
                    host = conn_info_old[0].get_transport().getpeername()[0] if conn_info_old and conn_info_old[0] else "unknown_host"
                    username = "unknown_user"
                    for spec, (h, u, _, _) in self.path_mappings.items():
                        if h == host:
                            username = u
                            break
                    logger.error(f"Verify SFTP user '{username}' has write permissions in '{posixpath.dirname(op_new_path)}' on host '{host}'.")
                    raise PermissionError(f"SFTP: Permission denied for rename operation {op_old_path} -> {op_new_path}: {e}") from e
                # How sftp.rename handles existing new_path varies. Some servers might raise EEXIST, others overwrite.
                # For now, if it's an IOError not matching above, re-raise. Test plan should verify FileExistsError separately.
            
            # Check for permission denied strings in the generic IOError case
            if is_remote_old and "permission denied" in str(e).lower():
                host = conn_info_old[0].get_transport().getpeername()[0] if conn_info_old and conn_info_old[0] else "unknown_host"
                username = "unknown_user"
                for spec, (h, u, _, _) in self.path_mappings.items():
                    if h == host:
                        username = u
                        break
                logger.error(f"Verify SFTP user '{username}' has write permissions in '{posixpath.dirname(op_new_path)}' on host '{host}'.")
            
            # For local, os.rename() raises FileNotFoundError, PermissionError, FileExistsError, IsADirectoryError, NotADirectoryError etc.
            # These are usually specific enough.
            raise # Re-raise original or translated SFTP error, or specific local error
        except Exception as e:
            logger.error(f"Error renaming file '{old_path}' to '{new_path}' (resolved to '{op_old_path}' -> '{op_new_path}'): {e}")
            raise # Re-raise other exceptions

    def _get_remote_home_dir(self, sftp, host, username) -> Optional[str]:
        """Attempt to determine the user's home directory on the remote server."""
        # Try normalize('.') first, as it often returns the home directory
        try:
            # sftp.normalize('.') might return relative path, use pwd for absolute
            home_dir = sftp.getcwd()
            if home_dir:
                # Basic sanity check
                if f"/{username}" in home_dir: # Check with forward slash for remote paths
                    logger.debug(f"Determined remote home directory for {username}@{host} via getcwd: {home_dir}")
                    return posixpath.normpath(home_dir) # Use posixpath
                else:
                    # If getcwd doesn't look like a home path, try normalize
                    normalized_home = sftp.normalize('.')
                    if normalized_home and f"/{username}" in normalized_home: # Check with forward slash
                         logger.debug(f"Determined remote home directory for {username}@{host} via normalize('.'): {normalized_home}")
                         return posixpath.normpath(normalized_home) # Use posixpath

        except Exception as e:
            logger.warning(f"Could not reliably determine home dir via getcwd/normalize for {username}@{host}: {e}")

        # Fallback: Try common patterns if pwd/normalize didn't work
        common_homes = [f'/home/{username}', f'/Users/{username}']
        for potential_home in common_homes:
            try:
                sftp.stat(potential_home)
                logger.debug(f"Found potential home directory via stat for {username}@{host}: {potential_home}")
                return posixpath.normpath(potential_home) # Use posixpath
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning(f"Error checking potential home {potential_home}: {e}")

        logger.warning(f"Could not determine home directory for {username}@{host}. Directory creation might be less robust.")
        return None

    def _sftp_makedirs(self, sftp, absolute_remote_directory):
        """
        Recursively create remote directories over SFTP. Expects absolute path.
        Skips known base components like /home or /home/user.
        """
        if not absolute_remote_directory or not posixpath.isabs(absolute_remote_directory):
             logger.error(f"_sftp_makedirs called with non-absolute path: {absolute_remote_directory}")
             raise ValueError(f"SFTP makedirs internal function requires an absolute path, got: {absolute_remote_directory}")

        logger.debug(f"Ensuring remote directory structure exists: {absolute_remote_directory}")

        host_found = None
        username_found = None
        # client_found = None # Not used
        for h, (c, s) in self.connections.items():
            if s == sftp:
                 host_found = h
                 # client_found = c # Not used
                 for _, (map_host, map_user, _, _) in self.path_mappings.items():
                      if map_host == host_found:
                          username_found = map_user
                          break
                 break

        user_home_dir = None
        if host_found and username_found:
            user_home_dir = self._get_remote_home_dir(sftp, host_found, username_found)
            if user_home_dir:
                 user_home_dir = posixpath.normpath(user_home_dir)

        path_components = absolute_remote_directory.strip('/').split('/')
        current_path = "/" 

        for component in path_components:
            if not component: continue
            next_path = posixpath.join(current_path, component)

            if user_home_dir:
                 normalized_next_path = posixpath.normpath(next_path)
                 if normalized_next_path == user_home_dir or \
                    user_home_dir.startswith(normalized_next_path + '/'):
                      logger.debug(f"Skipping check/create for base directory component: {next_path}")
                      current_path = next_path
                      continue
            try:
                sftp.stat(next_path)
                logger.debug(f"Remote directory component exists: {next_path}")
            except IOError as e_stat: # Specifically for sftp.stat failure
                if getattr(e_stat, 'errno', None) == errno.ENOENT or "no such file" in str(e_stat).lower():
                    logger.info(f"Remote directory component {next_path} not found, creating.")
                    try:
                        sftp.mkdir(next_path)
                    except IOError as e_mkdir:
                        # Check if it's a race condition or genuine error
                        try: 
                            sftp.stat(next_path)
                            logger.warning(f"Directory {next_path} exists after mkdir failed (likely race condition), continuing.")
                        except IOError: # If stat still fails, then mkdir truly failed
                            if getattr(e_mkdir, 'errno', None) == errno.EACCES or "permission denied" in str(e_mkdir).lower():
                                logger.error(f"SFTP: Permission denied creating directory {next_path}. Verify user '{username_found}' has write permissions in '{current_path}' on host '{host_found}'.")
                                raise PermissionError(f"SFTP: Permission denied creating directory {next_path}: {e_mkdir}") from e_mkdir
                            raise IOError(f"SFTP: Failed to create directory {next_path} after stat confirmed non-existence: {e_mkdir}") from e_mkdir
                elif getattr(e_stat, 'errno', None) == errno.EACCES or "permission denied" in str(e_stat).lower():
                    # Permission error trying to stat the path component
                    logger.error(f"SFTP: Permission denied to access/stat directory component {next_path}. Verify user '{username_found}' has permissions in '{current_path}' on host '{host_found}'.")
                    raise PermissionError(f"SFTP: Permission denied to access/stat {next_path}: {e_stat}") from e_stat
                else:
                    # Other IOError from sftp.stat
                    raise IOError(f"SFTP: Error stating directory component {next_path}: {e_stat}") from e_stat
            except Exception as e_stat_other: # Catch other non-IOError from stat, if any
                raise IOError(f"SFTP: Unexpected error stating directory component {next_path}: {e_stat_other}") from e_stat_other
            current_path = next_path

    def _get_absolute_remote_path(self, path: str) -> Optional[str]:
        """Resolves any path (relative, absolute, ~) to an absolute remote path based on mappings."""
        original_path = path
        import posixpath # Use for remote path manipulation

        # 1. Handle tilde expansion first
        if path.startswith('~'):
            username, host, user_home = self._get_user_host_home_for_tilde()
            if username and user_home:
                relative_part = path[1:].lstrip('/')
                expanded_path = posixpath.join(user_home, relative_part)
                logger.debug(f"Expanded tilde path '{original_path}' to '{expanded_path}'")
                path = expanded_path # Update path to the absolute expanded version
            else:
                logger.error(f"Could not determine username/home directory to expand tilde in path: {original_path}")
                return None # Cannot proceed reliably

        # 2. Handle relative paths (after potential tilde expansion)
        if not posixpath.isabs(path):
            # Join relative path with the *normalized, absolute* version of the first available remote prefix
            found_mapping = False
            for spec, (host, username, remote_prefix, _) in self.path_mappings.items():
                try:
                    sftp = self.connections[host][1]
                    user_home = self._get_remote_home_dir(sftp, host, username)
                    temp_remote_prefix = remote_prefix
                    if temp_remote_prefix.startswith('~') and user_home:
                         temp_remote_prefix = posixpath.join(user_home, temp_remote_prefix[1:].lstrip('/'))

                    normalized_remote_prefix = posixpath.normpath(temp_remote_prefix)
                    if not posixpath.isabs(normalized_remote_prefix):
                         normalized_remote_prefix = sftp.normalize(temp_remote_prefix)

                    # Combine the normalized absolute prefix with the relative path
                    absolute_path = posixpath.join(normalized_remote_prefix, path)
                    logger.debug(f"Mapped relative path '{original_path}' to remote absolute path '{posixpath.normpath(absolute_path)}' using prefix '{normalized_remote_prefix}'")
                    path = posixpath.normpath(absolute_path)
                    found_mapping = True
                    break # Use first mapping
                except KeyError:
                    logger.warning(f"No active connection for host {host} while resolving relative path prefix '{remote_prefix}', skipping.")
                    continue
                except Exception as e:
                    logger.warning(f"Error normalizing prefix '{remote_prefix}' or joining path '{path}': {e}, skipping.")
                    continue

            if not found_mapping:
                logger.error(f"No SFTP mappings or active connections found to resolve relative path: {original_path}")
                return None

        # 3. Path should now be absolute, just normalize and return
        # Check if this absolute path falls under any managed prefix (sanity check)
        final_path = posixpath.normpath(path)
        is_managed = False
        for spec, (host, username, remote_prefix, _) in self.path_mappings.items():
             try:
                 sftp = self.connections[host][1]
                 user_home = self._get_remote_home_dir(sftp, host, username)
                 temp_remote_prefix = remote_prefix
                 if temp_remote_prefix.startswith('~') and user_home:
                      temp_remote_prefix = posixpath.join(user_home, temp_remote_prefix[1:].lstrip('/'))
                 normalized_remote_prefix = posixpath.normpath(temp_remote_prefix)
                 if not posixpath.isabs(normalized_remote_prefix):
                      normalized_remote_prefix = sftp.normalize(temp_remote_prefix)

                 if final_path == normalized_remote_prefix or final_path.startswith(normalized_remote_prefix + '/'):
                     is_managed = True
                     break
             except Exception:
                 continue # Ignore errors during this check

        if not is_managed:
             logger.warning(f"Resolved absolute path '{final_path}' does not fall under any managed SFTP prefix.")
             # Depending on strictness, could return None here, but let's return the path for now.

        return final_path

    def _get_user_host_home_for_tilde(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Helper to find user, host, and home dir for tilde expansion based on mappings."""
        for spec, (h, u, _, _) in self.path_mappings.items():
            # Use the first mapping found
            if h in self.connections:
                sftp = self.connections[h][1]
                home_dir = self._get_remote_home_dir(sftp, h, u)
                if home_dir:
                    return u, h, home_dir
            # Fallback if connection not ready or home dir not found
            fallback_home = f"/home/{u}" # Assume /home structure
            logger.warning(f"Using fallback home dir '{fallback_home}' for tilde expansion for {u}@{h}. Connection may not be ready or home detection failed.")
            return u, h, fallback_home
        return None, None, None # No mappings found

    def _resolve_path_for_operation(self, path: str) -> Tuple[bool, str, Optional[Tuple]]:
        """Resolves path, determines if remote, and returns the correct path for the operation (local or absolute remote)."""
        original_path = path
        # import posixpath # Already imported at top

        # Step 1: Get the absolute remote path equivalent, expanding tilde etc.
        absolute_remote_equivalent = self._get_absolute_remote_path(original_path)
        if not absolute_remote_equivalent:
             # If _get_absolute_remote_path returns None, it means resolution failed (e.g., no mapping for relative path)
             # In this case, we should probably treat it as a standard local path, assuming it wasn't meant for SFTP.
             logger.warning(f"Could not resolve '{original_path}' to an absolute remote path via mappings. Treating as potential local path.")
             local_op_path = os.path.normpath(original_path)
             logger.debug(f"Path '{original_path}' treated as standard LOCAL operation path: {local_op_path}")
             return False, local_op_path, None
             # Original logic: raise ValueError(f"Could not resolve path '{original_path}' to an absolute remote path.")

        # Step 2: Determine if this absolute remote path falls under any managed SFTP connection/prefix
        is_remote = False
        conn_info = None
        matching_local_prefix = None
        matching_remote_prefix = None

        for spec, (host, username, remote_prefix, local_prefix) in self.path_mappings.items():
            # IMPORTANT: Normalize the stored remote_prefix by expanding tilde *before* comparison
            try:
                sftp = self.connections[host][1]
                # Use normalize to resolve potential tilde or relative components in the stored prefix
                # We need an absolute, normalized version of the remote prefix for comparison.
                # Handle potential errors during normalization (e.g., connection closed)
                try:
                    # Get home dir for potential tilde expansion in prefix
                    user_home = self._get_remote_home_dir(sftp, host, username)
                    temp_remote_prefix = remote_prefix
                    if temp_remote_prefix.startswith('~') and user_home:
                         temp_remote_prefix = posixpath.join(user_home, temp_remote_prefix[1:].lstrip('/'))

                    normalized_remote_prefix = posixpath.normpath(temp_remote_prefix)
                    if not posixpath.isabs(normalized_remote_prefix):
                         # If prefix is still not absolute after potential tilde expansion, try normalizing with sftp
                         # This handles cases where the mapping itself might be like './subdir'
                         normalized_remote_prefix = sftp.normalize(temp_remote_prefix)

                except Exception as norm_err:
                     logger.warning(f"Could not normalize remote prefix '{remote_prefix}' for mapping '{spec}', skipping comparison: {norm_err}")
                     continue # Skip this mapping if prefix normalization fails

            except KeyError:
                logger.warning(f"No active connection for host {host} while checking mapping '{spec}', skipping comparison.")
                continue # Skip if not connected to this host

            # Now compare the fully resolved absolute_remote_equivalent with the normalized_remote_prefix
            if absolute_remote_equivalent == normalized_remote_prefix or \
               absolute_remote_equivalent.startswith(normalized_remote_prefix + '/'):
                # Match found!
                is_remote = True
                conn_info = self.connections[host]
                matching_local_prefix = local_prefix
                matching_remote_prefix = normalized_remote_prefix # Store the normalized one used for matching
                logger.debug(f"Absolute path '{absolute_remote_equivalent}' matches remote prefix '{normalized_remote_prefix}' for host '{host}'")
                break # Stop after first match

        # Step 3: Determine the final operation path
        if is_remote:
            # If remote, the operation path is the absolute remote path
            op_path = absolute_remote_equivalent
            logger.debug(f"Path '{original_path}' resolved to REMOTE operation path: {op_path}")
            return True, op_path, conn_info
        else:
            # If not remote, it must be a local path.
            # The operation path is the standard local normalization of the original path.
            local_op_path = os.path.normpath(original_path)
            logger.debug(f"Path '{original_path}' resolved to standard LOCAL operation path: {local_op_path}")
            return False, local_op_path, None

    def close_all(self) -> None:
        """Close all SFTP connections."""
        for host, (client, _) in self.connections.items():
            try:
                logger.info(f"Closing connection to {host}")
                client.close()
            except Exception as e:
                logger.error(f"Error closing connection to {host}: {e}")
        self.connections = {} 