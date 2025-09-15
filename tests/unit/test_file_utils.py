"""
Unit tests for fiberwise_common.utils.file_utils module.

Tests all file utility functions including:
- File checksum calculation and verification
- Path normalization
- Manifest loading (JSON/YAML)
- Directory creation
- JSON serialization utilities
"""
import pytest
import json
import yaml
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, Any

from fiberwise_common.utils.file_utils import (
    calculate_file_checksum,
    calculate_file_checksum_safe,
    verify_file_checksum,
    get_file_info,
    normalize_path,
    load_manifest,
    ensure_directory_exists,
    safe_json_loads,
    safe_json_dumps
)


class TestChecksumFunctions:
    """Test file checksum calculation and verification."""

    def test_calculate_file_checksum_valid_file(self, sample_json_file: Path):
        """Test calculating checksum for a valid file."""
        checksum = calculate_file_checksum(sample_json_file)
        
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex digest length
        assert all(c in '0123456789abcdef' for c in checksum)

    def test_calculate_file_checksum_with_path_object(self, temp_dir: Path):
        """Test checksum calculation with Path object."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")
        
        checksum = calculate_file_checksum(test_file)
        expected = hashlib.sha256(b"Hello, World!").hexdigest()
        
        assert checksum == expected

    def test_calculate_file_checksum_nonexistent_file(self, temp_dir: Path):
        """Test calculating checksum for nonexistent file raises exception."""
        nonexistent = temp_dir / "does_not_exist.txt"
        
        with pytest.raises(FileNotFoundError):
            calculate_file_checksum(nonexistent)

    def test_calculate_file_checksum_safe_valid_file(self, sample_json_file: Path):
        """Test safe checksum calculation for valid file."""
        checksum = calculate_file_checksum_safe(sample_json_file)
        
        assert isinstance(checksum, str)
        assert len(checksum) == 64

    def test_calculate_file_checksum_safe_nonexistent_file(self, temp_dir: Path):
        """Test safe checksum calculation returns None for nonexistent file."""
        nonexistent = temp_dir / "does_not_exist.txt"
        
        checksum = calculate_file_checksum_safe(nonexistent)
        assert checksum is None

    def test_verify_file_checksum_valid(self, temp_dir: Path):
        """Test verifying correct checksum."""
        test_file = temp_dir / "test.txt"
        content = "Test content for checksum verification"
        test_file.write_text(content)
        
        expected_checksum = hashlib.sha256(content.encode()).hexdigest()
        
        assert verify_file_checksum(test_file, expected_checksum) is True

    def test_verify_file_checksum_invalid(self, temp_dir: Path):
        """Test verifying incorrect checksum."""
        test_file = temp_dir / "test.txt" 
        test_file.write_text("Test content")
        
        wrong_checksum = "0" * 64
        
        assert verify_file_checksum(test_file, wrong_checksum) is False

    def test_verify_file_checksum_nonexistent_file(self, temp_dir: Path):
        """Test verifying checksum for nonexistent file returns False."""
        nonexistent = temp_dir / "does_not_exist.txt"
        
        assert verify_file_checksum(nonexistent, "any_checksum") is False


class TestFileInfo:
    """Test get_file_info function."""

    def test_get_file_info_existing_file(self, sample_json_file: Path):
        """Test getting info for existing file."""
        info = get_file_info(sample_json_file)
        
        assert info is not None
        assert info['exists'] is True
        assert info['readable'] is True
        assert isinstance(info['size'], int)
        assert info['size'] > 0
        assert isinstance(info['checksum'], str)
        assert len(info['checksum']) == 64

    def test_get_file_info_nonexistent_file(self, temp_dir: Path):
        """Test getting info for nonexistent file."""
        nonexistent = temp_dir / "does_not_exist.txt"
        info = get_file_info(nonexistent)
        
        assert info is not None
        assert info['exists'] is False
        assert info['readable'] is False
        assert info['size'] is None
        assert info['checksum'] is None

    def test_get_file_info_with_string_path(self, sample_json_file: Path):
        """Test getting info with string path."""
        info = get_file_info(str(sample_json_file))
        
        assert info is not None
        assert info['exists'] is True


class TestPathNormalization:
    """Test normalize_path function."""

    def test_normalize_path_forward_slashes(self):
        """Test normalizing path with forward slashes."""
        path = "folder/subfolder/file.txt"
        normalized = normalize_path(path)
        
        assert normalized == "folder/subfolder/file.txt"

    def test_normalize_path_backslashes(self):
        """Test normalizing path with backslashes."""
        path = "folder\\subfolder\\file.txt"
        normalized = normalize_path(path)
        
        assert normalized == "folder/subfolder/file.txt"

    def test_normalize_path_mixed_slashes(self):
        """Test normalizing path with mixed slashes."""
        path = "folder\\subfolder/file.txt"
        normalized = normalize_path(path)
        
        assert normalized == "folder/subfolder/file.txt"

    def test_normalize_path_with_dot_references(self):
        """Test normalizing path with . and .. references."""
        path = "folder/../other/./file.txt"
        normalized = normalize_path(path)
        
        assert normalized == "other/file.txt"

    def test_normalize_path_with_path_object(self):
        """Test normalizing Path object."""
        path = Path("folder") / "subfolder" / "file.txt"
        normalized = normalize_path(path)
        
        assert "folder/subfolder/file.txt" in normalized
        assert "\\" not in normalized

    @pytest.mark.parametrize("input_path,expected", [
        ("", "."),  # os.path.normpath("") returns "."
        (".", "."),
        ("folder//file.txt", "folder/file.txt"),
        ("folder/./file.txt", "folder/file.txt"),
        ("folder/../file.txt", "file.txt")
    ])
    def test_normalize_path_parametrized(self, input_path: str, expected: str):
        """Test various path normalization scenarios."""
        result = normalize_path(input_path)
        assert result == expected


class TestManifestLoading:
    """Test load_manifest function."""

    def test_load_manifest_json(self, sample_json_file: Path):
        """Test loading JSON manifest."""
        data = load_manifest(sample_json_file)
        
        assert isinstance(data, dict)
        assert "name" in data
        assert data["name"] == "test_agent"

    def test_load_manifest_yaml(self, sample_yaml_file: Path):
        """Test loading YAML manifest."""
        data = load_manifest(sample_yaml_file)
        
        assert isinstance(data, dict)
        assert "name" in data
        assert data["name"] == "test-app"

    def test_load_manifest_with_format_json(self, sample_json_file: Path):
        """Test loading manifest with format return."""
        data, file_format = load_manifest(sample_json_file, return_format=True)
        
        assert isinstance(data, dict)
        assert file_format == "json"

    def test_load_manifest_with_format_yaml(self, sample_yaml_file: Path):
        """Test loading manifest with format return."""
        data, file_format = load_manifest(sample_yaml_file, return_format=True)
        
        assert isinstance(data, dict) 
        assert file_format == "yaml"

    def test_load_manifest_unsupported_extension(self, temp_dir: Path):
        """Test loading manifest with unsupported extension."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("not json or yaml")
        
        with pytest.raises(ValueError, match="Unsupported manifest format"):
            load_manifest(txt_file)

    def test_load_manifest_nonexistent_file(self, temp_dir: Path):
        """Test loading nonexistent manifest file."""
        nonexistent = temp_dir / "does_not_exist.json"
        
        with pytest.raises(FileNotFoundError):
            load_manifest(nonexistent)

    def test_load_manifest_invalid_json(self, temp_dir: Path):
        """Test loading invalid JSON manifest."""
        invalid_json = temp_dir / "invalid.json"
        invalid_json.write_text("{ invalid json content")
        
        with pytest.raises(ValueError, match="Failed to parse manifest"):
            load_manifest(invalid_json)

    def test_load_manifest_empty_file(self, temp_dir: Path):
        """Test loading empty manifest file."""
        empty_yaml = temp_dir / "empty.yaml"
        empty_yaml.write_text("")
        
        data = load_manifest(empty_yaml)
        assert data == {}


class TestDirectoryCreation:
    """Test ensure_directory_exists function."""

    def test_ensure_directory_exists_new_dir(self, temp_dir: Path):
        """Test creating new directory."""
        new_dir = temp_dir / "new_directory"
        
        assert not new_dir.exists()
        ensure_directory_exists(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_directory_exists_nested_dirs(self, temp_dir: Path):
        """Test creating nested directories."""
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        
        ensure_directory_exists(nested_dir)
        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_ensure_directory_exists_existing_dir(self, temp_dir: Path):
        """Test with already existing directory."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()
        
        # Should not raise exception
        ensure_directory_exists(existing_dir)
        assert existing_dir.exists()

    def test_ensure_directory_exists_string_path(self, temp_dir: Path):
        """Test with string path."""
        new_dir_str = str(temp_dir / "string_path_dir")
        
        ensure_directory_exists(new_dir_str)
        assert Path(new_dir_str).exists()


class TestJSONUtilities:
    """Test safe JSON loading and dumping functions."""

    def test_safe_json_loads_valid_json(self):
        """Test safely loading valid JSON."""
        json_str = '{"key": "value", "number": 42}'
        result = safe_json_loads(json_str)
        
        assert result == {"key": "value", "number": 42}

    def test_safe_json_loads_invalid_json_with_default(self):
        """Test safely loading invalid JSON with default."""
        invalid_json = '{"invalid": json content}'
        default = {"default": "value"}
        
        result = safe_json_loads(invalid_json, default)
        assert result == default

    def test_safe_json_loads_invalid_json_no_default(self):
        """Test safely loading invalid JSON without default."""
        invalid_json = '{"invalid": json content}'
        
        result = safe_json_loads(invalid_json)
        assert result is None

    def test_safe_json_loads_none_input(self):
        """Test safely loading None input."""
        result = safe_json_loads(None)
        assert result is None

    def test_safe_json_dumps_valid_data(self):
        """Test safely dumping valid data."""
        data = {"key": "value", "number": 42}
        result = safe_json_dumps(data)
        
        assert isinstance(result, str)
        assert json.loads(result) == data

    def test_safe_json_dumps_invalid_data_with_default(self):
        """Test safely dumping invalid data with default."""
        # Functions are not JSON serializable
        invalid_data = {"func": lambda x: x}
        default = '{"error": "serialization_failed"}'
        
        result = safe_json_dumps(invalid_data, default)
        assert result == default

    def test_safe_json_dumps_invalid_data_no_default(self):
        """Test safely dumping invalid data without default."""
        invalid_data = {"func": lambda x: x}
        
        result = safe_json_dumps(invalid_data)
        assert result == "{}"

    @pytest.mark.parametrize("input_data,expected_type", [
        ({"key": "value"}, str),
        ([1, 2, 3], str),
        ("string", str),
        (42, str),
        (True, str),
        (None, str)
    ])
    def test_safe_json_dumps_various_types(self, input_data: Any, expected_type: type):
        """Test dumping various data types."""
        result = safe_json_dumps(input_data)
        assert isinstance(result, expected_type)


class TestIntegration:
    """Integration tests combining multiple file utilities."""

    def test_create_verify_and_load_manifest(self, temp_dir: Path):
        """Integration test: create, verify checksum, and load manifest."""
        # Create manifest data
        manifest_data = {
            "name": "integration-test",
            "version": "1.0.0",
            "description": "Integration test manifest"
        }
        
        # Create manifest file
        manifest_file = temp_dir / "manifest.json"
        manifest_json = safe_json_dumps(manifest_data)
        manifest_file.write_text(manifest_json)
        
        # Calculate and verify checksum
        original_checksum = calculate_file_checksum(manifest_file)
        assert verify_file_checksum(manifest_file, original_checksum)
        
        # Load and verify manifest content
        loaded_data = load_manifest(manifest_file)
        assert loaded_data == manifest_data
        
        # Get file info
        file_info = get_file_info(manifest_file)
        assert file_info['exists']
        assert file_info['checksum'] == original_checksum

    def test_directory_and_file_operations(self, temp_dir: Path):
        """Integration test: directory creation and file operations."""
        # Create nested directory structure
        project_dir = temp_dir / "project" / "src" / "agents"
        ensure_directory_exists(project_dir)
        
        # Create agent file
        agent_content = {
            "name": "test_agent",
            "type": "function",
            "code": "def run_agent(): return 'success'"
        }
        agent_file = project_dir / "agent.json"
        agent_file.write_text(safe_json_dumps(agent_content))
        
        # Verify everything exists and works
        assert project_dir.exists()
        assert agent_file.exists()
        
        # Load and verify agent content
        loaded_agent = load_manifest(agent_file)
        assert loaded_agent["name"] == "test_agent"
        
        # Test path normalization on the created structure
        normalized_path = normalize_path(str(agent_file))
        assert "project/src/agents/agent.json" in normalized_path