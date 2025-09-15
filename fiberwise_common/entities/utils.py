"""
Utility functions for schema handling and field metadata.
"""


def is_system_field(field_name: str) -> bool:
    """
    Check if a field is the system user_id field.
    
    Args:
        field_name: Name of the field to check
        
    Returns:
        bool: True if field is the system user_id field
    """
    return field_name.lower() == "user_id"


def get_system_user_field() -> str:
    """
    Get the name of the system user_id field.
    
    Returns:
        The field name for user identification
    """
    return "user_id"