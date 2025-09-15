"""
Scoped Storage Provider - migrated from fiberwise-core-web/api/services
Storage provider with scoped access control and path isolation.
Ensures secure file operations within defined boundaries.
"""

import os
import logging
from typing import List, Dict, Any, Optional, BinaryIO
from pathlib import Path

from .storage_provider import StorageProvider
from ..utils.file_utils import normalize_path

logger = logging.getLogger(__name__)

class ScopedStorageProvider(StorageProvider):
    """
    A storage provider that restricts access to a specific root path.
    
    This provider wraps another storage provider and ensures all operations
    are scoped to a specific root path, preventing access to other areas
    of the storage system.
    """
    
    def __init__(self, base_provider: StorageProvider, root_path: str):
        """
        Initialize the scoped storage provider.
        
        Args:
            base_provider: The underlying storage provider to use
            root_path: The root path to restrict access to
        """
        self.base_provider = base_provider
        self.root_path = self._normalize_path(root_path)
        
    def _normalize_path(self, path: str) -> str:
        """Normalize a path to ensure consistent format."""
        return normalize_path(path)
        
    def _resolve_path(self, path: str) -> str:
        """
        Resolve a relative path to an absolute path within the root.
        
        Args:
            path: A path relative to the root
            
        Returns:
            Absolute path within the storage system
        """
        # Handle empty path
        if not path:
            return self.root_path
        
        # Normalize the path
        norm_path = self._normalize_path(path)
        
        # Check if already prefixed with root
        if norm_path.startswith(self.root_path):
            return norm_path
            
        # Join with root path
        full_path = os.path.join(self.root_path, norm_path.lstrip('/'))
        return self._normalize_path(full_path)
    
    def _validate_path(self, path: str) -> bool:
        """
        Validate that a path is within the allowed root.
        
        Args:
            path: Path to validate
            
        Returns:
            True if path is valid, False otherwise
        """
        # Normalize both paths for comparison
        norm_path = self._normalize_path(path)
        norm_root = self._normalize_path(self.root_path)
        
        # Path must start with root
        return norm_path.startswith(norm_root)
    
    async def upload_file(self, file_path: str, destination_path: str) -> str:
        """
        Upload a file to storage within the allowed root.
        
        Args:
            file_path: Local path to file
            destination_path: Path within storage (relative to root)
            
        Returns:
            URL or path to uploaded file
        """
        # Resolve destination to full path
        full_dest_path = self._resolve_path(destination_path)
        
        # Validate the destination is within allowed root
        if not self._validate_path(full_dest_path):
            raise PermissionError(f"Access denied: {destination_path} is outside allowed root")
        
        # Delegate to base provider
        return await self.base_provider.upload_file(file_path, full_dest_path)
    
    async def download_file(self, storage_path: str, local_path: str) -> str:
        """
        Download a file from storage within the allowed root.
        
        Args:
            storage_path: Path within storage (relative to root)
            local_path: Local path to save file
            
        Returns:
            Local path where file was saved
        """
        # Resolve source to full path
        full_source_path = self._resolve_path(storage_path)
        
        # Validate the source is within allowed root
        if not self._validate_path(full_source_path):
            raise PermissionError(f"Access denied: {storage_path} is outside allowed root")
        
        # Delegate to base provider
        return await self.base_provider.download_file(full_source_path, local_path)
    
    async def extract_archive(self, file_path: str, extract_dir: str) -> str:
        """
        Extract an archive file to a directory within the allowed root.
        
        Args:
            file_path: Path to archive file (relative to root)
            extract_dir: Directory to extract to (relative to root)
            
        Returns:
            Path to extracted directory
        """
        # Resolve paths to full paths
        full_file_path = self._resolve_path(file_path)
        full_extract_dir = self._resolve_path(extract_dir)
        
        # Validate paths are within allowed root
        if not self._validate_path(full_file_path):
            raise PermissionError(f"Access denied: {file_path} is outside allowed root")
        
        if not self._validate_path(full_extract_dir):
            raise PermissionError(f"Access denied: {extract_dir} is outside allowed root")
        
        # Delegate to base provider
        return await self.base_provider.extract_archive(full_file_path, full_extract_dir)
    
    async def file_exists(self, path: str) -> bool:
        """
        Check if a file exists in storage within the allowed root.
        
        Args:
            path: Path to check (relative to root)
            
        Returns:
            True if file exists, False otherwise
        """
        # Resolve path to full path
        full_path = self._resolve_path(path)
        
        # Validate the path is within allowed root
        if not self._validate_path(full_path):
            # For security, don't raise an error, just return False
            return False
        
        # Delegate to base provider
        return await self.base_provider.file_exists(full_path)
    
    async def get_file_info(self, path: str) -> Dict[str, Any]:
        """
        Get information about a file within the allowed root.
        
        Args:
            path: Path to file (relative to root)
            
        Returns:
            Dict with file information
        """
        # Resolve path to full path
        full_path = self._resolve_path(path)
        
        # Validate the path is within allowed root
        if not self._validate_path(full_path):
            return {"exists": False, "path": path, "error": "Access denied: outside allowed root"}
        
        # Delegate to base provider
        info = await self.base_provider.get_file_info(full_path)
        
        # Adjust the path in returned info to be relative to root
        if 'path' in info and info['path'].startswith(self.root_path):
            relative_path = os.path.relpath(info['path'], self.root_path)
            info['path'] = relative_path
            info['full_path'] = full_path
            
        return info
    
    async def delete_file(self, path: str) -> bool:
        """
        Delete a file from storage within the allowed root.
        
        Args:
            path: Path to file (relative to root)
            
        Returns:
            True if successful, False otherwise
        """
        # Resolve path to full path
        full_path = self._resolve_path(path)
        
        # Validate the path is within allowed root
        if not self._validate_path(full_path):
            raise PermissionError(f"Access denied: {path} is outside allowed root")
        
        # Delegate to base provider
        return await self.base_provider.delete_file(full_path)
    
    async def list_files(self, path: str) -> List[str]:
        """
        List files in a directory within the allowed root.
        
        Args:
            path: Directory path (relative to root)
            
        Returns:
            List of file paths relative to the specified directory
        """
        # Resolve path to full path
        full_path = self._resolve_path(path)
        
        # Validate the path is within allowed root
        if not self._validate_path(full_path):
            raise PermissionError(f"Access denied: {path} is outside allowed root")
        
        # Delegate to base provider
        files = await self.base_provider.list_files(full_path)
        
        # Return the files with their original relative paths
        return files
    
    async def is_local(self) -> bool:
        """
        Check if this is a local storage provider.
        
        Returns:
            bool: True if local, False if cloud
        """
        # Delegate to base provider
        return await self.base_provider.is_local()
    
    async def create_directory(self, path: str) -> bool:
        """
        Create a directory within the allowed root.
        
        Args:
            path: Directory path (relative to root)
            
        Returns:
            True if successful, False otherwise
        """
        # Resolve path to full path
        full_path = self._resolve_path(path)
        
        # Validate the path is within allowed root
        if not self._validate_path(full_path):
            raise PermissionError(f"Access denied: {path} is outside allowed root")
        
        # For local storage provider
        if await self.is_local():
            os.makedirs(full_path, exist_ok=True)
            return True
        
        # For cloud storage providers, create an empty marker file
        marker_file = os.path.join(full_path, ".keep")
        
        # Create a temporary empty file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create empty file
            with open(temp_path, 'w') as f:
                pass
            
            # Upload empty file as marker
            await self.base_provider.upload_file(temp_path, marker_file)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {path}: {e}")
            return False
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def get_relative_path(self, path: str) -> str:
        """
        Convert an absolute path to a path relative to the root.
        
        Args:
            path: Absolute path
            
        Returns:
            Path relative to root
        """
        norm_path = self._normalize_path(path)
        norm_root = self._normalize_path(self.root_path)
        
        if norm_path.startswith(norm_root):
            return os.path.relpath(norm_path, norm_root)
        
        # If path is not within root, return it unchanged
        return path
