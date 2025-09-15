"""
Basic unit tests for fiberwise_common.utils.file_utils module.

This module tests the core file utility functions including checksum calculation,
file validation, path normalization, and manifest loading.
"""

import json
import tempfile
import yaml
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from fiberwise_common.utils.file_utils import (
    calculate_file_checksum,
    calculate_file_checksum_safe,
    verify_file_checksum,
    get_file_info,
    normalize_path,
    load_manifest,
    ensure_directory_exists,
    safe_json_loads,
    safe_json_dumps,
    CHUNK_SIZE,
)


class TestCalculateFileChecksum:
    """Test suite for calculate_file_checksum function."""
    
    def test_calculate_checksum_string_path(self, temp_dir):
        """Test checksum calculation with string path."""
        test_file = temp_dir / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)
        
        checksum = calculate_file_checksum(str(test_file))
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex length
        
    def test_calculate_checksum_path_object(self, temp_dir):
        """Test checksum calculation with Path object."""
        test_file = temp_dir / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)
        
        checksum = calculate_file_checksum(test_file)
        assert isinstance(checksum, str)
        assert len(checksum) == 64
        
    def test_calculate_checksum_consistent(self, temp_dir):
        """Test that checksum calculation is consistent."""
        test_file = temp_dir / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)
        
        checksum1 = calculate_file_checksum(test_file)
        checksum2 = calculate_file_checksum(test_file)
        assert checksum1 == checksum2
        
    def test_calculate_checksum_different_content(self, temp_dir):
        """Test that different content produces different checksums."""
        file1 = temp_dir / "test1.txt"
        file2 = temp_dir / "test2.txt"
        
        file1.write_text("Hello, World!")
        file2.write_text("Hello, Universe!")
        
        checksum1 = calculate_file_checksum(file1)
        checksum2 = calculate_file_checksum(file2)
        assert checksum1 != checksum2
        
    def test_calculate_checksum_empty_file(self, temp_dir):
        """Test checksum calculation for empty file."""
        test_file = temp_dir / "empty.txt"
        test_file.touch()
        
        checksum = calculate_file_checksum(test_file)
        assert isinstance(checksum, str)
        assert len(checksum) == 64
        
    def test_calculate_checksum_file_not_found(self, temp_dir):
        """Test that FileNotFoundError is raised for non-existent file."""
        non_existent = temp_dir / "does_not_exist.txt"
        
        with pytest.raises(FileNotFoundError):
            calculate_file_checksum(non_existent)
            
    @patch('builtins.open', side_effect=PermissionError("Access denied"))
    def test_calculate_checksum_permission_error(self, mock_file):
        """Test that PermissionError is propagated."""
        with pytest.raises(PermissionError):
            calculate_file_checksum("test.txt")
            
    def test_calculate_checksum_large_file_chunks(self, temp_dir):
        """Test that large files are handled properly using chunks."""
        test_file = temp_dir / "large.txt"
        # Create content larger than CHUNK_SIZE
        large_content = "x" * (CHUNK_SIZE * 2 + 100)
        test_file.write_text(large_content)
        
        checksum = calculate_file_checksum(test_file)
        assert isinstance(checksum, str)
        assert len(checksum) == 64


class TestCalculateFileChecksumSafe:
    """Test suite for calculate_file_checksum_safe function."""
    
    def test_safe_checksum_success(self, temp_dir):
        """Test safe checksum calculation success case."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")
        
        checksum = calculate_file_checksum_safe(test_file)
        assert isinstance(checksum, str)
        assert len(checksum) == 64
        
    def test_safe_checksum_file_not_found(self, temp_dir):
        """Test safe checksum returns None for non-existent file."""
        non_existent = temp_dir / "does_not_exist.txt"
        
        checksum = calculate_file_checksum_safe(non_existent)
        assert checksum is None
        
    @patch('builtins.open', side_effect=PermissionError("Access denied"))
    def test_safe_checksum_permission_error(self, mock_file):
        """Test safe checksum returns None on permission error."""
        checksum = calculate_file_checksum_safe("test.txt")
        assert checksum is None
        
    @patch('builtins.open', side_effect=IOError("General I/O error"))
    def test_safe_checksum_io_error(self, mock_file):
        """Test safe checksum returns None on I/O error."""
        checksum = calculate_file_checksum_safe("test.txt")
        assert checksum is None


class TestVerifyFileChecksum:
    """Test suite for verify_file_checksum function."""
    
    def test_verify_checksum_valid(self, temp_dir):
        """Test checksum verification with valid checksum."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")
        
        actual_checksum = calculate_file_checksum(test_file)
        result = verify_file_checksum(test_file, actual_checksum)
        assert result is True
        
    def test_verify_checksum_invalid(self, temp_dir):
        """Test checksum verification with invalid checksum."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")
        
        invalid_checksum = "0" * 64
        result = verify_file_checksum(test_file, invalid_checksum)
        assert result is False
        
    def test_verify_checksum_file_not_found(self, temp_dir):
        """Test checksum verification returns False for non-existent file."""
        non_existent = temp_dir / "does_not_exist.txt"
        checksum = "0" * 64
        
        result = verify_file_checksum(non_existent, checksum)
        assert result is False


class TestGetFileInfo:
    """Test suite for get_file_info function."""
    
    def test_get_info_existing_file(self, temp_dir):
        """Test file info for existing file."""
        test_file = temp_dir / "test.txt"
        content = "Hello, World!"
        test_file.write_text(content)
        
        info = get_file_info(test_file)
        assert info is not None
        assert info['exists'] is True
        assert info['readable'] is True
        assert info['size'] == len(content)
        assert isinstance(info['checksum'], str)
        assert len(info['checksum']) == 64
        
    def test_get_info_non_existent_file(self, temp_dir):
        """Test file info for non-existent file."""
        non_existent = temp_dir / "does_not_exist.txt"
        
        info = get_file_info(non_existent)
        assert info is not None
        assert info['exists'] is False
        assert info['readable'] is False
        assert info['size'] is None
        assert info['checksum'] is None
        
    def test_get_info_empty_file(self, temp_dir):
        """Test file info for empty file."""
        test_file = temp_dir / "empty.txt"
        test_file.touch()
        
        info = get_file_info(test_file)
        assert info is not None
        assert info['exists'] is True
        assert info['readable'] is True
        assert info['size'] == 0
        assert isinstance(info['checksum'], str)
        
    @patch('pathlib.Path.stat', side_effect=OSError("Stat failed"))
    def test_get_info_os_error(self, mock_stat, temp_dir):
        """Test file info returns None on OS error."""
        test_file = temp_dir / "test.txt"
        
        info = get_file_info(test_file)
        assert info is None


class TestNormalizePath:
    """Test suite for normalize_path function."""
    
    def test_normalize_forward_slashes(self):
        """Test normalization preserves forward slashes."""
        path = "folder/subfolder/file.txt"
        normalized = normalize_path(path)
        assert normalized == "folder/subfolder/file.txt"
        
    def test_normalize_backslashes(self):
        """Test normalization converts backslashes to forward slashes."""
        path = "folder\\subfolder\\file.txt"
        normalized = normalize_path(path)
        assert normalized == "folder/subfolder/file.txt"
        
    def test_normalize_mixed_slashes(self):
        """Test normalization handles mixed slashes."""
        path = "folder\\subfolder/file.txt"
        normalized = normalize_path(path)
        assert normalized == "folder/subfolder/file.txt"
        
    def test_normalize_parent_references(self):
        """Test normalization resolves parent directory references."""
        path = "folder/../file.txt"
        normalized = normalize_path(path)
        assert normalized == "file.txt"
        
    def test_normalize_current_references(self):
        """Test normalization resolves current directory references."""
        path = "folder/./subfolder/file.txt"
        normalized = normalize_path(path)
        assert normalized == "folder/subfolder/file.txt"
        
    def test_normalize_double_slashes(self):
        """Test normalization handles double slashes."""
        path = "folder//subfolder/file.txt"
        normalized = normalize_path(path)
        assert normalized == "folder/subfolder/file.txt"
        
    def test_normalize_non_string_input(self):
        """Test normalization handles non-string input."""
        path = Path("folder/file.txt")
        normalized = normalize_path(path)
        assert normalized == "folder/file.txt"
        
    def test_normalize_empty_path(self):
        """Test normalization handles empty path."""
        normalized = normalize_path("")
        assert normalized == "."


class TestSafeJsonLoads:
    """Test suite for safe_json_loads function."""
    
    def test_safe_json_loads_valid(self):
        """Test safe JSON loading with valid JSON."""
        json_str = '{"key": "value", "number": 42}'
        result = safe_json_loads(json_str)
        assert result == {"key": "value", "number": 42}
        
    def test_safe_json_loads_invalid_default_none(self):
        """Test safe JSON loading with invalid JSON and None default."""
        result = safe_json_loads("invalid json")
        assert result is None
        
    def test_safe_json_loads_invalid_custom_default(self):
        """Test safe JSON loading with invalid JSON and custom default."""
        result = safe_json_loads("invalid json", {})
        assert result == {}
        
    def test_safe_json_loads_none_input(self):
        """Test safe JSON loading with None input."""
        result = safe_json_loads(None, "default")
        assert result == "default"
        
    def test_safe_json_loads_empty_string(self):
        """Test safe JSON loading with empty string."""
        result = safe_json_loads("", [])
        assert result == []


class TestSafeJsonDumps:
    """Test suite for safe_json_dumps function."""
    
    def test_safe_json_dumps_valid(self):
        """Test safe JSON dumping with valid data."""
        data = {"key": "value", "number": 42}
        result = safe_json_dumps(data)
        assert result == '{"key": "value", "number": 42}'
        
    def test_safe_json_dumps_invalid_default(self):
        """Test safe JSON dumping with non-serializable data."""
        data = lambda x: x  # Functions are not JSON serializable
        result = safe_json_dumps(data)
        assert result == "{}"
        
    def test_safe_json_dumps_invalid_custom_default(self):
        """Test safe JSON dumping with custom default."""
        data = set([1, 2, 3])  # Sets are not JSON serializable
        result = safe_json_dumps(data, "null")
        assert result == "null"
        
    def test_safe_json_dumps_none(self):
        """Test safe JSON dumping with None."""
        result = safe_json_dumps(None)
        assert result == "null"
        
    def test_safe_json_dumps_list(self):
        """Test safe JSON dumping with list."""
        data = [1, 2, 3, "test"]
        result = safe_json_dumps(data)
        assert result == '[1, 2, 3, "test"]'


class TestEnsureDirectoryExists:
    """Test suite for ensure_directory_exists function."""
    
    def test_ensure_directory_new(self, temp_dir):
        """Test creating new directory."""
        new_dir = temp_dir / "new_folder"
        assert not new_dir.exists()
        
        ensure_directory_exists(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()
        
    def test_ensure_directory_existing(self, temp_dir):
        """Test with existing directory (should not fail)."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()
        
        # Should not raise an error
        ensure_directory_exists(existing_dir)
        assert existing_dir.exists()
        
    def test_ensure_directory_nested(self, temp_dir):
        """Test creating nested directories."""
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        
        ensure_directory_exists(nested_dir)
        assert nested_dir.exists()
        assert nested_dir.is_dir()
        
    def test_ensure_directory_string_path(self, temp_dir):
        """Test with string path."""
        new_dir = temp_dir / "string_path"
        
        ensure_directory_exists(str(new_dir))
        assert new_dir.exists()


@pytest.mark.parametrize("file_extension,expected_format", [
    (".json", "json"),
    (".yaml", "yaml"),
    (".yml", "yaml"),
])
class TestLoadManifest:
    """Test suite for load_manifest function."""
    
    def test_load_manifest_basic(self, temp_dir, file_extension, expected_format):
        """Test basic manifest loading."""
        test_data = {"name": "test", "version": "1.0.0"}
        manifest_file = temp_dir / f"test{file_extension}"
        
        if expected_format == "json":
            manifest_file.write_text(json.dumps(test_data))
        else:
            manifest_file.write_text(yaml.dump(test_data))
            
        result = load_manifest(manifest_file)
        assert result == test_data
        
    def test_load_manifest_with_format(self, temp_dir, file_extension, expected_format):
        """Test manifest loading with format return."""
        test_data = {"name": "test", "version": "1.0.0"}
        manifest_file = temp_dir / f"test{file_extension}"
        
        if expected_format == "json":
            manifest_file.write_text(json.dumps(test_data))
        else:
            manifest_file.write_text(yaml.dump(test_data))
            
        result, format_type = load_manifest(manifest_file, return_format=True)
        assert result == test_data
        assert format_type == expected_format
        
    def test_load_manifest_empty(self, temp_dir, file_extension, expected_format):
        """Test loading empty manifest."""
        manifest_file = temp_dir / f"empty{file_extension}"
        
        if expected_format == "json":
            manifest_file.write_text("")
        else:
            manifest_file.write_text("")
            
        if expected_format == "json":
            with pytest.raises(ValueError, match="Failed to parse manifest"):
                load_manifest(manifest_file)
        else:
            # Empty YAML should return empty dict
            result = load_manifest(manifest_file)
            assert result == {}


class TestLoadManifestEdgeCases:
    """Test edge cases for load_manifest function."""
    
    def test_load_manifest_unsupported_format(self, temp_dir):
        """Test loading manifest with unsupported format."""
        manifest_file = temp_dir / "test.txt"
        manifest_file.write_text("some content")
        
        with pytest.raises(ValueError, match="Unsupported manifest format"):
            load_manifest(manifest_file)
            
    def test_load_manifest_file_not_found(self, temp_dir):
        """Test loading non-existent manifest."""
        manifest_file = temp_dir / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError, match="Manifest file not found"):
            load_manifest(manifest_file)
            
    def test_load_manifest_invalid_json(self, temp_dir):
        """Test loading invalid JSON manifest."""
        manifest_file = temp_dir / "invalid.json"
        manifest_file.write_text("{invalid json")
        
        with pytest.raises(ValueError, match="Failed to parse manifest"):
            load_manifest(manifest_file)
            
    def test_load_manifest_invalid_yaml(self, temp_dir):
        """Test loading invalid YAML manifest."""
        manifest_file = temp_dir / "invalid.yaml"
        manifest_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(ValueError, match="Failed to parse manifest"):
            load_manifest(manifest_file)
            
    def test_load_manifest_null_yaml(self, temp_dir):
        """Test loading null YAML content."""
        manifest_file = temp_dir / "null.yaml"
        manifest_file.write_text("null")
        
        result = load_manifest(manifest_file)
        assert result == {}


if __name__ == "__main__":
    pytest.main([__file__])