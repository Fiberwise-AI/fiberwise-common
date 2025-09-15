"""
File utilities for FiberWise.

This module provides utilities for file operations including checksum calculation,
file validation, and other file-related operations used across the FiberWise system.
"""

import hashlib
import json
import os
import yaml
from pathlib import Path
from typing import Union, Optional, Dict, Any, Literal, Tuple, overload

# Define a constant for the chunk size for clarity and easy modification.
CHUNK_SIZE = 4096


def calculate_file_checksum(file_path: Union[str, Path]) -> str:
    """
    Calculates the SHA256 checksum of a file.

    This function reads the file in chunks to efficiently handle large files while
    computing their SHA256 hash. It follows the "fail-fast" principle by allowing
    exceptions to propagate to the caller.

    Args:
        file_path: The path to the file (can be a string or a Path object).

    Returns:
        The hex digest of the SHA256 checksum as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read due to permissions.
        IOError: For other general I/O errors.
        
    Example:
        >>> calculate_file_checksum("example.txt")
        "d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2"
        
        >>> from pathlib import Path
        >>> calculate_file_checksum(Path("example.txt"))
        "d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2"
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read and update hash in chunks to handle large files efficiently.
        while chunk := f.read(CHUNK_SIZE):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def calculate_file_checksum_safe(file_path: Union[str, Path]) -> Optional[str]:
    """
    Calculates the SHA256 checksum of a file, returning None on failure.

    This is a safe variant of calculate_file_checksum that handles exceptions
    internally and returns None if any I/O errors occur. Use this when you
    prefer error signaling via return values rather than exceptions.

    Args:
        file_path: The path to the file (can be a string or a Path object).

    Returns:
        The hex digest of the SHA256 checksum as a string, or None if an error occurs.
        
    Example:
        >>> calculate_file_checksum_safe("existing_file.txt")
        "d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2"
        
        >>> calculate_file_checksum_safe("nonexistent_file.txt")
        None
    """
    try:
        return calculate_file_checksum(file_path)
    except (IOError, OSError):
        return None


def verify_file_checksum(file_path: Union[str, Path], expected_checksum: str) -> bool:
    """
    Verifies that a file's checksum matches the expected value.

    Args:
        file_path: The path to the file to verify.
        expected_checksum: The expected SHA256 checksum in hexadecimal format.

    Returns:
        True if the file's checksum matches the expected value, False otherwise.
        Also returns False if the file cannot be read.
        
    Example:
        >>> verify_file_checksum("example.txt", "d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2")
        True
        
        >>> verify_file_checksum("example.txt", "wrong_checksum")
        False
    """
    actual_checksum = calculate_file_checksum_safe(file_path)
    if actual_checksum is None:
        return False
    return actual_checksum == expected_checksum


def get_file_info(file_path: Union[str, Path]) -> Optional[dict]:
    """
    Get comprehensive file information including size and checksum.
    
    Args:
        file_path: The path to the file.
        
    Returns:
        Dictionary with file information, or None if file cannot be accessed.
        Keys include: 'size', 'checksum', 'exists', 'readable'
        
    Example:
        >>> get_file_info("example.txt")
        {
            'size': 1024,
            'checksum': 'd2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2',
            'exists': True,
            'readable': True
        }
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                'size': None,
                'checksum': None,
                'exists': False,
                'readable': False
            }
        
        size = file_path.stat().st_size
        checksum = calculate_file_checksum_safe(file_path)
        
        return {
            'size': size,
            'checksum': checksum,
            'exists': True,
            'readable': checksum is not None
        }
    except (IOError, OSError):
        return None


def normalize_path(path: str) -> str:
    """
    Canonicalizes a path string.

    This function normalizes path separators, resolves '..' references,
    and ensures all separators are forward slashes ('/') for consistent,
    cross-platform path representation.

    Args:
        path: The file or directory path string.

    Returns:
        The normalized path string with forward slashes.
        
    Example:
        >>> normalize_path("folder\\..\\file.txt")
        "file.txt"
        
        >>> normalize_path("folder//subfolder/./file.txt")
        "folder/subfolder/file.txt"
    """
    if not isinstance(path, str):
        # Convert to string for consistent behavior
        path = str(path)
    return os.path.normpath(path).replace('\\', '/')


# Overloads provide excellent static type checking for different return types
@overload
def load_manifest(
    manifest_path: Path, return_format: Literal[False] = False
) -> Dict[str, Any]:
    ...


@overload
def load_manifest(
    manifest_path: Path, return_format: Literal[True]
) -> Tuple[Dict[str, Any], str]:
    ...


def load_manifest(
    manifest_path: Path, return_format: bool = False
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], str]]:
    """
    Loads and parses a manifest file (JSON or YAML) into a dictionary.

    This function combines the best practices from both original implementations:
    - Uses memory-efficient file object parsing
    - Supports both JSON (.json) and YAML (.yaml/.yml) formats
    - Provides flexible return types based on caller needs
    - Enhanced error handling with specific exception types

    Args:
        manifest_path: The path to the manifest file.
        return_format: If True, returns a tuple containing the data and the
                       file format ('json' or 'yaml'). Defaults to False.

    Returns:
        The parsed manifest data as a dictionary, or a tuple of (data, format)
        if return_format is True.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If the file format is unsupported or a parsing error occurs.
        
    Example:
        >>> # Basic usage (returns only data)
        >>> data = load_manifest(Path("config.json"))
        >>> print(data["version"])
        
        >>> # With format information
        >>> data, fmt = load_manifest(Path("config.yaml"), return_format=True)
        >>> print(f"Loaded {fmt} format: {data}")
    """
    suffix = manifest_path.suffix.lower()
    
    if suffix not in ('.json', '.yaml', '.yml'):
        raise ValueError(f"Unsupported manifest format for file: {manifest_path}")

    file_format = 'json' if suffix == '.json' else 'yaml'
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            if file_format == 'json':
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ValueError(f"Failed to parse manifest from {manifest_path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to load manifest from {manifest_path}: {e}") from e

    # Handle case where YAML/JSON is empty or null
    if data is None:
        data = {}

    if return_format:
        return data, file_format
    else:
        return data


def ensure_directory_exists(path: Union[str, Path]) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: The directory path to ensure exists (can be a string or Path object).
        
    Example:
        >>> ensure_directory_exists("/tmp/my_folder")
        >>> ensure_directory_exists(Path("/tmp/another_folder"))
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_json_loads(data: str, default: Any = None) -> Any:
    """
    Safely load JSON data with fallback.
    
    Args:
        data: JSON string to parse
        default: Default value to return if parsing fails
        
    Returns:
        Parsed JSON data or default value
        
    Example:
        >>> safe_json_loads('{"key": "value"}')
        {'key': 'value'}
        >>> safe_json_loads('invalid json', {})
        {}
    """
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """
    Safely dump data to JSON with fallback.
    
    Args:
        data: Data to serialize to JSON
        default: Default string to return if serialization fails
        
    Returns:
        JSON string or default value
        
    Example:
        >>> safe_json_dumps({"key": "value"})
        '{"key": "value"}'
        >>> safe_json_dumps(lambda x: x, "null")
        "null"
    """
    try:
        return json.dumps(data)
    except (TypeError, ValueError):
        return default