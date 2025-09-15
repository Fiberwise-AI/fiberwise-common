"""
Resource loader for SQL scripts packaged with fiberwise_common.
"""

import sys
from typing import Optional

def load_sql_script(script_name: str) -> str:
    """
    Load a SQL script from the packaged sql directory.
    
    Args:
        script_name: Name of the SQL file (e.g., 'init.sql')
        
    Returns:
        The content of the SQL file
        
    Raises:
        FileNotFoundError: If the script is not found
        Exception: If there's an error reading the script
    """
    try:
        # Try using importlib.resources (Python 3.9+)
        if sys.version_info >= (3, 9):
            try:
                from importlib.resources import files
                sql_package = files('fiberwise_common.database.sql')
                sql_file = sql_package / script_name
                if sql_file.is_file():
                    return sql_file.read_text(encoding='utf-8')
            except ImportError:
                pass
        
        # Fallback to importlib_resources for older Python versions
        try:
            import importlib_resources
            sql_package = importlib_resources.files('fiberwise_common.database.sql')
            sql_file = sql_package / script_name
            if sql_file.is_file():
                return sql_file.read_text(encoding='utf-8')
        except ImportError:
            pass
        
        # Final fallback to pkg_resources
        try:
            import pkg_resources
            return pkg_resources.resource_string(
                'fiberwise_common.database.sql', 
                script_name
            ).decode('utf-8')
        except ImportError:
            pass
            
        # Last resort: try to find the file relative to this module
        from pathlib import Path
        sql_dir = Path(__file__).parent / 'sql'
        sql_file = sql_dir / script_name
        if sql_file.exists():
            return sql_file.read_text(encoding='utf-8')
        
        raise FileNotFoundError(f"SQL script '{script_name}' not found in package resources")
        
    except Exception as e:
        raise Exception(f"Failed to load SQL script '{script_name}': {e}")


def list_available_scripts() -> list:
    """
    List all available SQL scripts in the package.
    
    Returns:
        List of available SQL script names
    """
    try:
        # Try using importlib.resources (Python 3.9+)
        if sys.version_info >= (3, 9):
            try:
                from importlib.resources import files
                sql_package = files('fiberwise_common.database.sql')
                return [f.name for f in sql_package.iterdir() if f.name.endswith('.sql')]
            except ImportError:
                pass
        
        # Fallback to importlib_resources
        try:
            import importlib_resources
            sql_package = importlib_resources.files('fiberwise_common.database.sql')
            return [f.name for f in sql_package.iterdir() if f.name.endswith('.sql')]
        except ImportError:
            pass
            
        # Final fallback: list files from filesystem
        from pathlib import Path
        sql_dir = Path(__file__).parent / 'sql'
        if sql_dir.exists():
            return [f.name for f in sql_dir.glob('*.sql')]
        
        return []
        
    except Exception:
        return []
