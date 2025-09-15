"""
Agent Storage Provider - migrated from fiberwise-core-web/api/services  
Specialized storage provider for agent files and resources.
Handles agent-specific storage patterns and file management.
"""

import os
import logging
import tempfile
import json
import time
from typing import Dict, Any, Optional, BinaryIO, List
from pathlib import Path

# Note: Configuration should be passed to services rather than imported directly
from .storage_provider import StorageProvider, get_storage_provider, LocalStorageProvider, S3StorageProvider
from .scoped_storage_provider import ScopedStorageProvider

logger = logging.getLogger(__name__)

class AgentStorageProvider:
    """
    Agent-specific storage provider with scoped access to agent resources.
    Provides standardized access to agent code, dependencies, and state.
    """
    
    def __init__(self, 
                 app_id: str, 
                 agent_id: str, 
                 api_key: str = None, 
                 version_id: str = "latest"):
        """
        Initialize the agent storage provider.
        
        Args:
            app_id: The application ID
            agent_id: The agent ID
            api_key: Optional API key for authentication
            version_id: Agent version ID (defaults to "latest")
        """
        self.app_id = app_id
        self.agent_id = agent_id
        self.api_key = api_key
        self.version_id = version_id
        
        # Get base storage provider from system configuration
        base_provider = get_storage_provider()
        self.is_local = isinstance(base_provider, LocalStorageProvider)
        self.is_s3 = isinstance(base_provider, S3StorageProvider)
        
        # Set up base paths for agent storage
        self.base_path = self._get_base_path()
        
        # Create a scoped storage provider that restricts access to this agent's directory
        self.storage = ScopedStorageProvider(base_provider, self.base_path)
        
        # Define relative directory paths
        self.code_path = "code"
        self.cache_path = "cache" 
        self.deps_path = "dependencies"
        self.state_path = "state"
        
        # Ensure directories exist
        self._ensure_directories()
        
    def _get_base_path(self) -> str:
        """Get the base path for agent storage."""
        entity_bundles_dir = os.getenv('ENTITY_BUNDLES_DIR', 'entity_bundles')
        return os.path.join(
            entity_bundles_dir,
            "apps",
            str(self.app_id),
            "agent",
            str(self.agent_id),
            str(self.version_id)
        )
    
    async def _ensure_directories(self) -> None:
        """Ensure all required directories exist in the storage provider."""
        for path in [self.code_path, self.cache_path, self.deps_path, self.state_path]:
            # For local storage, just create directories
            if self.is_local:
                os.makedirs(os.path.join(self.base_path, path), exist_ok=True)
            else:
                # For cloud storage (S3, etc.), we need to check if directories exist
                # and create empty marker files if needed, since cloud storage doesn't 
                # have real directories
                try:
                    # Create an empty .keep file to ensure directory exists
                    keep_file = os.path.join(path, ".keep")
                    # Use synchronous method for initialization
                    full_path = os.path.join(self.base_path, keep_file)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    if not os.path.exists(full_path):
                        with open(full_path, 'w') as f:
                            f.write("")
                except Exception as e:
                    logger.warning(f"Could not ensure directory {path}: {e}")
    
    async def validate_access(self, path: str, operation: str) -> bool:
        """
        Validate if the agent has access to the specified path.
        
        Args:
            path: The path to validate
            operation: The operation to validate (read, write, delete)
            
        Returns:
            True if access is allowed, False otherwise
        """
        # Basic validation - check if path is within agent's allowed directories
        if not path:
            return False
            
        # Normalize path
        norm_path = os.path.normpath(path)
        
        # Check if path is within agent's base path
        return norm_path.startswith(self.base_path)
    
    #
    # Code Management Methods
    #
    
    async def get_code_file(self, file_path: str) -> str:
        """
        Get the contents of a code file.
        
        Args:
            file_path: Path to the file, relative to code directory
            
        Returns:
            String contents of the file
        """
        full_path = os.path.join(self.code_path, file_path)
        
        if not await self.validate_access(full_path, "read"):
            logger.warning(f"Access denied to file: {full_path}")
            raise PermissionError(f"Access denied to file: {file_path}")
            
        if not await self.storage.file_exists(full_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Create a temporary file to download contents
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Download the file using the base provider
            await self.storage.download_file(full_path, temp_path)
            
            # Read the contents
            with open(temp_path, 'r') as f:
                return f.read()
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    async def save_code_file(self, file_path: str, content: str) -> bool:
        """
        Save a code file.
        
        Args:
            file_path: Path to the file, relative to code directory
            content: String content to write
            
        Returns:
            True if successful
        """
        full_path = os.path.join(self.code_path, file_path)
        
        if not await self.validate_access(full_path, "write"):
            logger.warning(f"Access denied to file: {full_path}")
            raise PermissionError(f"Access denied to file: {file_path}")
        
        # Create a temporary file to store the content
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
            temp_path = temp_file.name
            temp_file.write(content)
            
        try:
            # Upload the file using base provider
            await self.storage.upload_file(temp_path, full_path)
            return True
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    async def list_code_files(self) -> List[str]:
        """
        List all code files for the agent.
        
        Returns:
            List of file paths relative to code directory
        """
        # First check if code path exists
        if not await self.storage.file_exists(self.code_path):
            return []
            
        # Get list of files from base provider
        files = await self.storage.list_files(self.code_path)
        
        # Filter out special files
        return [f for f in files if not f.startswith('.')]
    
    #
    # State Management Methods
    #
    
    async def get_state(self, key: str) -> Any:
        """
        Get state data for the agent.
        
        Args:
            key: State key
            
        Returns:
            State data or None if not found
        """
        state_file = os.path.join(self.state_path, f"{key}.json")
        
        if not await self.storage.file_exists(state_file):
            return None
        
        # Create a temporary file to download contents
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Download using base provider
            await self.storage.download_file(state_file, temp_path)
            
            # Read the JSON data
            with open(temp_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading state file {key}: {e}")
            return None
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    async def set_state(self, key: str, data: Any) -> bool:
        """
        Set state data for the agent.
        
        Args:
            key: State key
            data: State data (must be JSON serializable)
            
        Returns:
            True if successful, False otherwise
        """
        state_file = os.path.join(self.state_path, f"{key}.json")
        
        # Create temporary file with JSON content
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
            temp_path = temp_file.name
            json.dump(data, temp_file)
            
        try:
            # Upload using base provider
            await self.storage.upload_file(temp_path, state_file)
            return True
        except Exception as e:
            logger.error(f"Error writing state file {key}: {e}")
            return False
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    async def delete_state(self, key: str) -> bool:
        """
        Delete state data for the agent.
        
        Args:
            key: State key
            
        Returns:
            True if successful, False otherwise
        """
        state_file = os.path.join(self.state_path, f"{key}.json")
        
        if not await self.storage.file_exists(state_file):
            return True
            
        return await self.storage.delete_file(state_file)
    
    #
    # Cache Management Methods
    #
    
    async def get_cache(self, key: str) -> Any:
        """
        Get cached data for the agent.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found
        """
        cache_file = os.path.join(self.cache_path, f"{key}.json")
        
        if not await self.storage.file_exists(cache_file):
            return None
            
        try:
            # Create a temporary file to download contents
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
                
            try:
                # Download using base provider
                await self.storage.download_file(cache_file, temp_path)
                
                # Read the JSON data
                with open(temp_path, 'r') as f:
                    data = json.load(f)
                    
                # Check if cache has expired
                if "expiry" in data and data["expiry"] < time.time():
                    await self.storage.delete_file(cache_file)
                    return None
                    
                return data.get("value")
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        except Exception as e:
            logger.error(f"Error reading cache file {key}: {e}")
            return None
    
    async def set_cache(self, key: str, data: Any, ttl_seconds: int = 3600) -> bool:
        """
        Set cached data for the agent.
        
        Args:
            key: Cache key
            data: Cache data (must be JSON serializable)
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        cache_file = os.path.join(self.cache_path, f"{key}.json")
        
        try:
            cache_data = {
                "value": data,
                "created": time.time(),
                "expiry": time.time() + ttl_seconds
            }
            
            # Create temporary file with JSON content
            with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
                temp_path = temp_file.name
                json.dump(cache_data, temp_file)
                
            try:
                # Upload using base provider
                await self.storage.upload_file(temp_path, cache_file)
                return True
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        except Exception as e:
            logger.error(f"Error writing cache file {key}: {e}")
            return False
    
    async def clear_cache(self) -> bool:
        """
        Clear all cached data for the agent.
        
        Returns:
            True if successful, False otherwise
        """
        cache_files = await self.storage.list_files(self.cache_path)
        
        success = True
        for file_name in cache_files:
            file_path = os.path.join(self.cache_path, file_name)
            if not await self.storage.delete_file(file_path):
                success = False
                
        return success
    
    #
    # Helper Methods
    #
    
    async def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the agent storage.
        
        Returns:
            Dictionary with metadata
        """
        return {
            "app_id": self.app_id,
            "agent_id": self.agent_id,
            "version_id": self.version_id,
            "base_path": self.base_path,
            "storage_type": "local" if self.is_local else "s3" if self.is_s3 else "cloud",
            "provider": self.storage.base_provider.__class__.__name__
        }
    
    # Compatibility methods for common storage patterns
    async def store(self, key: str, data: Any) -> bool:
        """Store data using the state system - compatibility method"""
        return await self.set_state(key, data)
    
    async def put(self, key: str, data: Any) -> bool:
        """Put data using the state system - compatibility method"""
        return await self.set_state(key, data)
    
    async def get(self, key: str) -> Any:
        """Get data using the state system - compatibility method"""
        return await self.get_state(key)

def create_agent_storage_provider(
    app_id: str,
    agent_id: str, 
    api_key: str = None,
    version_id: str = "latest"
) -> AgentStorageProvider:
    """
    Create an agent storage provider instance.
    
    Args:
        app_id: The application ID
        agent_id: The agent ID
        api_key: Optional API key for authentication
        version_id: Agent version ID (defaults to "latest")
        
    Returns:
        AgentStorageProvider instance
    """
    return AgentStorageProvider(
        app_id=app_id,
        agent_id=agent_id,
        api_key=api_key,
        version_id=version_id
    )
