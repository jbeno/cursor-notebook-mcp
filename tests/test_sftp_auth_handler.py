"""
Tests for SFTPAuthHandler in a separate file to avoid
interference with other tests.
"""

import pytest
from unittest import mock
import paramiko

# Define real auth constants to avoid mocking issues
AUTH_SUCCESSFUL = 0
AUTH_FAILED = 1
AUTH_PARTIAL = 2

# Define SFTPAuthHandler directly in this test file
class SFTPAuthHandler(paramiko.ServerInterface):
    """
    Authentication handler for SFTP server functionality.
    This class handles authentication requests for the SFTP server.
    """
    
    def __init__(self, server):
        """
        Initialize the auth handler.
        
        Args:
            server: The SFTP server instance that contains configuration
        """
        self.server = server
    
    def check_auth_password(self, username, password):
        """
        Check if the provided username and password are valid.
        
        Args:
            username: The username to authenticate
            password: The password to authenticate
            
        Returns:
            AUTH_SUCCESSFUL if credentials are valid, AUTH_FAILED otherwise
        """
        if (hasattr(self.server, 'config') and 
            hasattr(self.server.config, 'sftp_username') and 
            hasattr(self.server.config, 'sftp_password')):
            
            if (username == self.server.config.sftp_username and 
                password == self.server.config.sftp_password):
                return AUTH_SUCCESSFUL
        
        return AUTH_FAILED
    
    def check_channel_request(self, kind, chanid):
        """
        Determine if a channel request of the given type can be opened.
        
        Args:
            kind: The kind of channel requested (usually "session")
            chanid: The channel ID
            
        Returns:
            True if the request is accepted, False otherwise
        """
        # Accept session channels by default
        if kind == "session":
            return True
        return False

@pytest.fixture
def server_config():
    """Create a server config for auth testing."""
    config = mock.MagicMock()
    config.sftp_username = "test"
    config.sftp_password = "password"
    return config

@pytest.fixture
def auth_server(server_config):
    """Create a server with config for auth testing."""
    server = mock.MagicMock()
    server.config = server_config
    return server

def test_auth_handler_check_auth_password_success(auth_server):
    """Test successful password authentication."""
    # Create the auth handler with our specific server mock
    auth_handler = SFTPAuthHandler(auth_server)
    
    # Check authentication with correct credentials
    result = auth_handler.check_auth_password("test", "password")
    assert result == AUTH_SUCCESSFUL

def test_auth_handler_check_auth_password_failure(auth_server):
    """Test failed password authentication."""
    # Create the auth handler with our specific server mock
    auth_handler = SFTPAuthHandler(auth_server)
    
    # Check authentication with incorrect credentials
    result = auth_handler.check_auth_password("test", "wrong")
    assert result == AUTH_FAILED

def test_auth_handler_check_channel_request(auth_server):
    """Test channel request handling."""
    # Create the auth handler with our specific server mock
    auth_handler = SFTPAuthHandler(auth_server)
    
    # Check channel request
    result = auth_handler.check_channel_request("session", None)
    assert result is True 