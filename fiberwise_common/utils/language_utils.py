"""
Language detection utilities for file extension analysis.

This module provides consolidated language detection functionality to replace
multiple duplicate implementations across the codebase. It resolves conflicts
between different default behaviors and provides extensible language support.
"""

from typing import Optional, Dict, Set
import logging

logger = logging.getLogger(__name__)

# Comprehensive extension mappings
EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    # Python
    '.py': 'python',
    '.pyw': 'python',
    '.pyi': 'python',
    
    # JavaScript/TypeScript
    '.js': 'javascript', 
    '.jsx': 'javascript',
    '.ts': 'javascript',
    '.tsx': 'javascript',
    '.mjs': 'javascript',
    '.cjs': 'javascript',
    
    # Go
    '.go': 'go',
    
    # Additional languages (extensible)
    '.java': 'java',
    '.cpp': 'cpp',
    '.c': 'c',
    '.rs': 'rust',
    '.rb': 'ruby',
    '.php': 'php',
    '.cs': 'csharp',
    '.swift': 'swift',
    '.kt': 'kotlin',
}

# Language groups for validation
SUPPORTED_LANGUAGES: Set[str] = set(EXTENSION_TO_LANGUAGE.values())


def detect_language_from_file(file_path: str, strict_mode: bool = True) -> Optional[str]:
    """
    Detect programming language from file extension.
    
    This function provides explicit language detection without making assumptions
    about defaults. Callers are responsible for handling None returns appropriately.
    
    Args:
        file_path: File path or filename
        strict_mode: If True, returns None for unknown extensions.
                    If False, logs warning and returns None.
                    
    Returns:
        Language identifier or None if not recognized
        
    Examples:
        >>> detect_language_from_file("script.py")
        'python'
        >>> detect_language_from_file("app.tsx") 
        'javascript'
        >>> detect_language_from_file("unknown.xyz")
        None
    """
    if not file_path:
        return None
        
    # Handle multiple dots and case-insensitive matching
    # (e.g., "file.test.JS" -> ".js")
    file_path_lower = file_path.lower()
    
    for ext in EXTENSION_TO_LANGUAGE:
        if file_path_lower.endswith(ext):
            return EXTENSION_TO_LANGUAGE[ext]
    
    if not strict_mode:
        logger.warning(f"Unknown file extension for: {file_path}")
    
    return None


def detect_language_with_fallback(file_path: str, fallback: str = 'python') -> str:
    """
    Detect language with explicit fallback for legacy compatibility.
    
    This method preserves the behavior expected by existing services
    while providing a clear API for fallback handling. It resolves the
    conflicting default behaviors from the original implementations.
    
    Args:
        file_path: File path or filename
        fallback: Default language to use if detection fails
        
    Returns:
        Detected language or fallback value
        
    Examples:
        >>> detect_language_with_fallback("script.py")
        'python'
        >>> detect_language_with_fallback("unknown.xyz", "javascript")
        'javascript'
    """
    detected = detect_language_from_file(file_path, strict_mode=True)
    return detected if detected is not None else fallback


def get_supported_extensions_for_language(language: str) -> list[str]:
    """
    Get all supported extensions for a language.
    
    Args:
        language: Language identifier (e.g., 'python', 'javascript')
        
    Returns:
        List of extensions that map to the specified language
        
    Example:
        >>> get_supported_extensions_for_language('javascript')
        ['.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs']
    """
    return [ext for ext, lang in EXTENSION_TO_LANGUAGE.items() if lang == language]


def is_language_supported(language: str) -> bool:
    """
    Check if a language is supported.
    
    Args:
        language: Language identifier to check
        
    Returns:
        True if the language is supported, False otherwise
        
    Example:
        >>> is_language_supported('python')
        True
        >>> is_language_supported('cobol')
        False
    """
    return language in SUPPORTED_LANGUAGES


def get_all_supported_languages() -> Set[str]:
    """
    Get all supported languages.
    
    Returns:
        Set of all supported language identifiers
    """
    return SUPPORTED_LANGUAGES.copy()


def add_language_mapping(extension: str, language: str) -> None:
    """
    Add a new language mapping (for runtime configuration).
    
    Args:
        extension: File extension (must start with '.')
        language: Language identifier
        
    Raises:
        ValueError: If extension doesn't start with '.'
    """
    if not extension.startswith('.'):
        raise ValueError(f"Extension must start with '.': {extension}")
    
    EXTENSION_TO_LANGUAGE[extension.lower()] = language.lower()
    SUPPORTED_LANGUAGES.add(language.lower())
    
    logger.info(f"Added language mapping: {extension} -> {language}")