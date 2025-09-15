"""
Storage Provider - migrated from fiberwise-core-web/api/services
Abstract base class and implementations for different storage backends.
Provides unified interface for local, cloud, and other storage systems.
"""

import os
import shutil
import tempfile
import logging
from abc import ABC, abstractmethod
from typing import BinaryIO, Dict, Any, Optional, Union
from pathlib import Path
from uuid import UUID

# Note: Configuration should be passed to services rather than imported directly
# This maintains better separation of concerns and testability

logger = logging.getLogger(__name__)

class StorageProvider(ABC):
    """Abstract base class for storage providers"""
    
    @abstractmethod
    async def upload_file(self, file_path: str, destination_path: str) -> str:
        """
        Upload a file to storage
        
        Args:
            file_path: Local path to file
            destination_path: Path within storage
            
        Returns:
            URL or path to uploaded file
        """
        pass
    
    @abstractmethod
    async def download_file(self, storage_path: str, local_path: str) -> str:
        """
        Download a file from storage
        
        Args:
            storage_path: Path within storage
            local_path: Local path to save file
            
        Returns:
            Local path where file was saved
        """
        pass
    
    @abstractmethod
    async def extract_archive(self, file_path: str, extract_dir: str) -> str:
        """
        Extract an archive file to a directory
        
        Args:
            file_path: Path to archive file
            extract_dir: Directory to extract to
            
        Returns:
            Path to extracted directory
        """
        pass
    
    @abstractmethod
    async def file_exists(self, path: str) -> bool:
        """
        Check if a file exists in storage
        
        Args:
            path: Path to check
            
        Returns:
            True if file exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_file_info(self, path: str) -> Dict[str, Any]:
        """
        Get information about a file
        
        Args:
            path: Path to file
            
        Returns:
            Dict with file information
        """
        pass
    
    @abstractmethod
    async def delete_file(self, path: str) -> bool:
        """
        Delete a file from storage
        
        Args:
            path: Path to file
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def list_files(self, path: str) -> list:
        """
        List files in a directory
        
        Args:
            path: Directory path
            
        Returns:
            List of file paths
        """
        pass
    
    async def is_local(self) -> bool:
        """
        Check if this is a local storage provider
        
        Returns:
            bool: True if local, False if cloud
        """
        return False


class LocalStorageProvider(StorageProvider):
    """Storage provider for local filesystem"""
    
    async def upload_file(self, file_path: str, destination_path: str) -> str:
        """Upload a file to local storage"""
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        # Copy file to destination
        shutil.copy2(file_path, destination_path)
        
        return destination_path
    
    async def download_file(self, storage_path: str, local_path: str) -> str:
        """Download a file from local storage"""
        # Ensure local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Copy file to local path
        shutil.copy2(storage_path, local_path)
        
        return local_path
    
    async def extract_archive(self, file_path: str, extract_dir: str) -> str:
        """Extract an archive file to a directory"""
        # Ensure extract directory exists
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract archive
        shutil.unpack_archive(file_path, extract_dir)
        
        return extract_dir
    
    async def file_exists(self, path: str) -> bool:
        """Check if a file exists in local storage"""
        return os.path.exists(path)
    
    async def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get information about a file in local storage"""
        if not os.path.exists(path):
            return {
                "exists": False,
                "path": path
            }
        
        stat_info = os.stat(path)
        
        return {
            "exists": True,
            "path": path,
            "size_bytes": stat_info.st_size,
            "created_at": stat_info.st_ctime,
            "modified_at": stat_info.st_mtime,
            "is_directory": os.path.isdir(path)
        }
    
    async def delete_file(self, path: str) -> bool:
        """Delete a file from local storage"""
        if not os.path.exists(path):
            return False
        
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        
        return True
    
    async def list_files(self, path: str) -> list:
        """List files in a directory in local storage"""
        if not os.path.exists(path) or not os.path.isdir(path):
            return []
        
        return os.listdir(path)
    
    async def is_local(self) -> bool:
        """
        Check if this is a local storage provider
        
        Returns:
            bool: True for LocalStorageProvider
        """
        return True


class S3StorageProvider(StorageProvider):
    """Storage provider for AWS S3"""
    
    def __init__(self, bucket_name: str = None, region: str = None, access_key_id: str = None, secret_access_key: str = None, endpoint_url: str = None):
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError
            
            # Use environment variables as fallback
            self.s3_bucket = bucket_name or os.getenv('S3_BUCKET_NAME', 'fiberwise-storage')
            self.s3_region = region or os.getenv('S3_REGION', 'us-east-1')
            
            # Configure boto3 session with credentials
            session_kwargs = {
                'region_name': self.s3_region
            }
            
            # Add credentials if provided (fallback to environment variables)
            access_key = access_key_id or os.getenv('S3_ACCESS_KEY_ID')
            secret_key = secret_access_key or os.getenv('S3_SECRET_ACCESS_KEY')
            
            if access_key and secret_key:
                session_kwargs['aws_access_key_id'] = access_key
                session_kwargs['aws_secret_access_key'] = secret_key
            
            # Create S3 client
            self.s3_client_kwargs = {}
            endpoint = endpoint_url or os.getenv('S3_ENDPOINT_URL')
            if endpoint:
                self.s3_client_kwargs['endpoint_url'] = endpoint
            
            self.session = boto3.session.Session(**session_kwargs)
            self.s3_client = self.session.client('s3', **self.s3_client_kwargs)
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.s3_bucket)
            logger.info(f"Successfully connected to S3 bucket: {self.s3_bucket}")
            
        except ImportError:
            logger.error("boto3 package not installed. Please install with: pip install boto3")
            raise
        except NoCredentialsError:
            logger.error("AWS credentials not found or invalid")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 storage provider: {str(e)}")
            raise
    
    async def upload_file(self, file_path: str, destination_path: str) -> str:
        """Upload a file to S3"""
        try:
            # Convert destination path to S3 key (without leading slash)
            s3_key = destination_path.lstrip('/')
            
            # Upload file to S3
            self.s3_client.upload_file(file_path, self.s3_bucket, s3_key)
            
            # Return S3 URI
            return f"s3://{self.s3_bucket}/{s3_key}"
        except Exception as e:
            logger.error(f"Error uploading file to S3: {str(e)}")
            raise
    
    async def download_file(self, storage_path: str, local_path: str) -> str:
        """Download a file from S3"""
        try:
            # Extract S3 key from storage path
            if storage_path.startswith('s3://'):
                parts = storage_path.replace('s3://', '').split('/', 1)
                if len(parts) < 2 or parts[0] != self.s3_bucket:
                    raise ValueError(f"Invalid S3 path: {storage_path}")
                s3_key = parts[1]
            else:
                s3_key = storage_path.lstrip('/')
            
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file from S3
            self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
            
            return local_path
        except Exception as e:
            logger.error(f"Error downloading file from S3: {str(e)}")
            raise
    
    async def extract_archive(self, file_path: str, extract_dir: str) -> str:
        """Extract an archive file to a directory"""
        try:
            # For S3, we need to download the file first
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
                temp_path = temp_file.name
            
            # Extract the S3 key from the file path
            if file_path.startswith('s3://'):
                parts = file_path.replace('s3://', '').split('/', 1)
                if len(parts) < 2 or parts[0] != self.s3_bucket:
                    raise ValueError(f"Invalid S3 path: {file_path}")
                s3_key = parts[1]
            else:
                s3_key = file_path.lstrip('/')
            
            # Download the file from S3
            self.s3_client.download_file(self.s3_bucket, s3_key, temp_path)
            
            # Ensure extract directory exists
            os.makedirs(extract_dir, exist_ok=True)
            
            # Extract the archive
            shutil.unpack_archive(temp_path, extract_dir)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return extract_dir
        except Exception as e:
            logger.error(f"Error extracting archive from S3: {str(e)}")
            raise
    
    async def file_exists(self, path: str) -> bool:
        """Check if a file exists in S3"""
        try:
            # Extract S3 key from path
            if path.startswith('s3://'):
                parts = path.replace('s3://', '').split('/', 1)
                if len(parts) < 2 or parts[0] != self.s3_bucket:
                    return False
                s3_key = parts[1]
            else:
                s3_key = path.lstrip('/')
            
            # Check if object exists
            try:
                self.s3_client.head_object(Bucket=self.s3_bucket, Key=s3_key)
                return True
            except self.s3_client.exceptions.ClientError as e:
                if e.response['Error']['Code'] == '404':
                    return False
                raise
        except Exception as e:
            logger.error(f"Error checking if file exists in S3: {str(e)}")
            return False
    
    async def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get information about a file in S3"""
        try:
            # Extract S3 key from path
            if path.startswith('s3://'):
                parts = path.replace('s3://', '').split('/', 1)
                if len(parts) < 2 or parts[0] != self.s3_bucket:
                    return {"exists": False, "path": path}
                s3_key = parts[1]
            else:
                s3_key = path.lstrip('/')
            
            # Get object metadata
            try:
                response = self.s3_client.head_object(Bucket=self.s3_bucket, Key=s3_key)
                return {
                    "exists": True,
                    "path": path,
                    "size_bytes": response.get('ContentLength', 0),
                    "created_at": response.get('LastModified', None),
                    "modified_at": response.get('LastModified', None),
                    "etag": response.get('ETag', ''),
                    "content_type": response.get('ContentType', '')
                }
            except self.s3_client.exceptions.ClientError as e:
                if e.response['Error']['Code'] == '404':
                    return {"exists": False, "path": path}
                raise
        except Exception as e:
            logger.error(f"Error getting file info from S3: {str(e)}")
            return {"exists": False, "path": path, "error": str(e)}
    
    async def delete_file(self, path: str) -> bool:
        """Delete a file from S3"""
        try:
            # Extract S3 key from path
            if path.startswith('s3://'):
                parts = path.replace('s3://', '').split('/', 1)
                if len(parts) < 2 or parts[0] != self.s3_bucket:
                    return False
                s3_key = parts[1]
            else:
                s3_key = path.lstrip('/')
            
            # Delete object
            self.s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
            return True
        except Exception as e:
            logger.error(f"Error deleting file from S3: {str(e)}")
            return False
    
    async def list_files(self, path: str) -> list:
        """List files in a directory in S3"""
        try:
            # Extract S3 prefix from path
            if path.startswith('s3://'):
                parts = path.replace('s3://', '').split('/', 1)
                if len(parts) < 2 or parts[0] != self.s3_bucket:
                    return []
                prefix = parts[1]
            else:
                prefix = path.lstrip('/')
            
            # Ensure prefix ends with a slash for directory semantics
            if prefix and not prefix.endswith('/'):
                prefix += '/'
            
            # List objects with prefix
            response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix=prefix, Delimiter='/')
            
            files = []
            
            # Add common prefixes (directories)
            for prefix_obj in response.get('CommonPrefixes', []):
                prefix_name = prefix_obj.get('Prefix', '')
                if prefix_name:
                    # Extract just the directory name
                    dir_name = prefix_name.rsplit('/', 2)[1] if '/' in prefix_name else prefix_name
                    files.append(dir_name)
            
            # Add objects (files)
            for content in response.get('Contents', []):
                key = content.get('Key', '')
                if key and key != prefix:
                    # Extract just the filename
                    file_name = key.rsplit('/', 1)[1] if '/' in key else key
                    files.append(file_name)
            
            return files
        except Exception as e:
            logger.error(f"Error listing files in S3: {str(e)}")
            return []
    
    async def is_local(self) -> bool:
        """
        Check if this is a local storage provider
        
        Returns:
            bool: False for S3StorageProvider
        """
        return False


# Factory function to get the configured storage provider
def get_storage_provider(provider_type: str = None) -> StorageProvider:
    """
    Get the storage provider based on configuration
    
    Args:
        provider_type: Storage provider type ('local', 's3'), defaults to environment variable
    
    Returns:
        StorageProvider instance
    """
    provider_type = (provider_type or os.getenv('STORAGE_PROVIDER', 'local')).lower()
    
    if provider_type == 'local':
        return LocalStorageProvider()
    elif provider_type == 's3':
        return S3StorageProvider()
    else:
        logger.warning(f"Unsupported storage provider: {provider_type}, using local storage instead")
        return LocalStorageProvider()
