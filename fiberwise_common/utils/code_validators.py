from typing import List, Tuple, Any


def validate_input(input_data: Any) -> bool:
    """
    Helper function to validate input data.
    
    This function performs basic validation on input data, checking for
    None values and empty containers. It can be extended with additional
    validation logic as needed.
    
    Args:
        input_data: The data to validate (can be any type)
        
    Returns:
        True if the input is considered valid, False otherwise
        
    Example:
        >>> validate_input("hello")
        True
        >>> validate_input("")
        False
        >>> validate_input(None)
        False
        >>> validate_input([1, 2, 3])
        True
        >>> validate_input([])
        False
    """
    if input_data is None:
        return False
    
    if isinstance(input_data, str):
        return len(input_data) > 0
    elif isinstance(input_data, (dict, list)):
        return len(input_data) > 0
    else:
        return True


def validate_code_snippet(code: str, language: str) -> Tuple[str, List[str]]:
    """
    Performs basic validation on a code snippet.

    This function checks for common issues like empty submissions. It is designed
    to be extensible with more validation rules in the future.

    Note: The 'language' parameter is currently unused but preserved for future
    language-specific validation logic.

    Args:
        code: The source code string to validate.
        language: The programming language of the code snippet.

    Returns:
        A tuple containing:
        - The original, unprocessed code string.
        - A list of warning messages.
    """
    warnings: List[str] = []

    if not code.strip():
        warnings.append("Code is empty")

    # Future validation rules can be added here.
    # e.g., check for max length, disallowed characters, etc.

    return code, warnings
