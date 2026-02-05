"""
FiberWise utility modules.

This package contains utility functions and classes used across the FiberWise system.
"""

from .agent_templates import (
    create_minimal_agent_code,
    create_function_agent_template,
    create_class_agent_template,
    create_llm_agent_config,
    create_agent_manifest,
    create_agent_manifest_yaml,
)
from .agent_utils import MetadataMixin, extract_agent_metadata
from .code_validators import validate_input, validate_code_snippet
from .file_utils import (
    calculate_file_checksum,
    calculate_file_checksum_safe,
    verify_file_checksum,
    get_file_info,
    normalize_path,
    load_manifest,
    ensure_directory_exists,
    safe_json_loads,
    safe_json_dumps,
)

__all__ = [
    # Agent templates
    "create_minimal_agent_code",
    "create_function_agent_template",
    "create_class_agent_template",
    "create_llm_agent_config",
    "create_agent_manifest",
    "create_agent_manifest_yaml",
    # Agent utilities
    "MetadataMixin",
    "extract_agent_metadata",
    # Code validation
    "validate_input",
    "validate_code_snippet",
    # File utilities
    "calculate_file_checksum",
    "calculate_file_checksum_safe",
    "verify_file_checksum",
    "get_file_info",
    "normalize_path",
    "load_manifest",
    "ensure_directory_exists",
    "safe_json_loads",
    "safe_json_dumps",
]