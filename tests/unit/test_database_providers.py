"""
Unit tests for database providers in fiberwise-common.
Example of project-specific unit tests.
"""

import pytest
from pathlib import Path
import tempfile
import os


def test_database_providers_module_imports():
    """Test that database provider modules can be imported."""
    from fiberwise_common.database import providers
    from fiberwise_common.database.providers import SQLiteProvider
    
    assert hasattr(providers, 'SQLiteProvider')
    assert callable(SQLiteProvider)


def test_sqlite_provider_basic_functionality():
    """Test basic SQLite provider functionality."""
    from fiberwise_common.database.providers import SQLiteProvider
    
    # Create a temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Test provider creation
        provider = SQLiteProvider(f'sqlite:///{db_path}')
        assert provider.provider == 'sqlite'
        assert provider.connection_string == f'sqlite:///{db_path}'
        
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_migration_manager_imports():
    """Test that migration manager can be imported."""
    from fiberwise_common.database.migrations import MigrationManager
    
    assert callable(MigrationManager)


@pytest.mark.integration
def test_init_sql_file_accessibility():
    """Test that init.sql file can be accessed from fiberwise-common."""
    from fiberwise_common.database import sql_loader
    
    # This tests that the SQL loader can find the init.sql file
    # (Integration test because it touches file system)
    current_dir = Path(__file__).parent.parent.parent
    init_sql_path = current_dir / "fiberwise_common" / "database" / "sql" / "init.sql"
    
    assert init_sql_path.exists(), f"init.sql not found at {init_sql_path}"
    
    with open(init_sql_path, 'r') as f:
        content = f.read()
        assert 'CREATE TABLE' in content
        assert 'users' in content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])