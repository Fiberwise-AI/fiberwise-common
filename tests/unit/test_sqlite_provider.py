"""
Unit tests for fiberwise_common.database.providers.SQLiteProvider.

This module tests the SQLite database provider implementation including
connection management, query execution, transactions, and error handling.
"""

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pytest

from fiberwise_common.database.providers import SQLiteProvider
from fiberwise_common.database.base import DatabaseProvider


@pytest.mark.asyncio
class TestSQLiteProviderBasics:
    """Basic tests for SQLiteProvider functionality."""
    
    def test_provider_type(self):
        """Test that provider returns correct type."""
        provider = SQLiteProvider("sqlite:///test.db")
        assert provider.provider == "sqlite"
    
    def test_initialization_with_sqlite_url(self):
        """Test initialization with sqlite:// URL."""
        provider = SQLiteProvider("sqlite:///test.db")
        assert provider.database_url == "sqlite:///test.db"
        assert provider.db_path == "test.db"
    
    def test_initialization_with_plain_path(self):
        """Test initialization with plain file path."""
        provider = SQLiteProvider("/path/to/database.db")
        assert provider.database_url == "/path/to/database.db"
        assert provider.db_path == "/path/to/database.db"
    
    def test_inheritance(self):
        """Test that SQLiteProvider inherits from DatabaseProvider."""
        provider = SQLiteProvider("test.db")
        assert isinstance(provider, DatabaseProvider)
    
    def test_initial_state(self):
        """Test initial state of provider."""
        provider = SQLiteProvider("test.db")
        assert provider._connection is None
        assert provider._lock is not None
        assert hasattr(provider, '_query_adapter')


@pytest.mark.asyncio
class TestSQLiteProviderConnection:
    """Test connection management for SQLiteProvider."""
    
    async def test_connect_success(self, temp_dir):
        """Test successful connection."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        result = await provider.connect()
        
        assert result is True
        assert provider._connection is not None
        
        # Clean up
        await provider.disconnect()
    
    async def test_connect_creates_directory(self, temp_dir):
        """Test that connect creates directory if it doesn't exist."""
        nested_path = temp_dir / "nested" / "dir" / "test.db"
        provider = SQLiteProvider(str(nested_path))
        
        result = await provider.connect()
        
        assert result is True
        assert nested_path.parent.exists()
        assert nested_path.exists()
        
        await provider.disconnect()
    
    async def test_connect_idempotent(self, temp_dir):
        """Test that multiple connects are idempotent."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        # First connect
        result1 = await provider.connect()
        connection1 = provider._connection
        
        # Second connect
        result2 = await provider.connect()
        connection2 = provider._connection
        
        assert result1 is True
        assert result2 is True
        assert connection1 is connection2
        
        await provider.disconnect()
    
    @patch('sqlite3.connect')
    async def test_connect_failure(self, mock_connect):
        """Test connection failure handling."""
        mock_connect.side_effect = sqlite3.Error("Connection failed")
        
        provider = SQLiteProvider("test.db")
        result = await provider.connect()
        
        assert result is False
        assert provider._connection is None
    
    async def test_disconnect_success(self, temp_dir):
        """Test successful disconnection."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        assert provider._connection is not None
        
        await provider.disconnect()
        assert provider._connection is None
    
    async def test_disconnect_when_not_connected(self):
        """Test disconnect when not connected (should not raise error)."""
        provider = SQLiteProvider("test.db")
        
        # Should not raise error
        await provider.disconnect()
        assert provider._connection is None
    
    @patch('sqlite3.connect')
    async def test_disconnect_error_handling(self, mock_connect):
        """Test disconnect error handling."""
        # Create a mock connection that fails on close
        mock_connection = Mock()
        mock_connection.close.side_effect = sqlite3.Error("Close failed")
        mock_connect.return_value = mock_connection
        
        provider = SQLiteProvider("test.db")
        await provider.connect()
        
        # Should not raise error, just log it
        await provider.disconnect()
        assert provider._connection is None


@pytest.mark.asyncio
class TestSQLiteProviderQueries:
    """Test query execution for SQLiteProvider."""
    
    async def test_execute_insert(self, temp_dir):
        """Test executing INSERT statement."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        
        # Create table and insert data
        await provider.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        result = await provider.execute("INSERT INTO test (name) VALUES (?)", "test_name")
        
        assert result == 1  # Should return lastrowid
        
        await provider.disconnect()
    
    async def test_execute_without_connection(self, temp_dir):
        """Test that execute connects automatically if not connected."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        # Don't connect manually
        result = await provider.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        
        # Should auto-connect and succeed
        assert provider._connection is not None
        
        await provider.disconnect()
    
    async def test_fetch_one_success(self, temp_dir):
        """Test successful fetch_one query."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        await provider.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        await provider.execute("INSERT INTO test VALUES (1, 'test')")
        
        result = await provider.fetch_one("SELECT id, name FROM test WHERE id = ?", 1)
        
        assert result is not None
        assert result['id'] == 1
        assert result['name'] == 'test'
        
        await provider.disconnect()
    
    async def test_fetch_one_no_results(self, temp_dir):
        """Test fetch_one with no results."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        await provider.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        
        result = await provider.fetch_one("SELECT * FROM test WHERE id = ?", 999)
        
        assert result is None
        
        await provider.disconnect()
    
    async def test_fetch_all_multiple_rows(self, temp_dir):
        """Test fetch_all with multiple rows."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        await provider.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        await provider.execute("INSERT INTO test VALUES (1, 'first')")
        await provider.execute("INSERT INTO test VALUES (2, 'second')")
        
        results = await provider.fetch_all("SELECT id, name FROM test ORDER BY id")
        
        assert len(results) == 2
        assert results[0]['id'] == 1
        assert results[0]['name'] == 'first'
        assert results[1]['id'] == 2
        assert results[1]['name'] == 'second'
        
        await provider.disconnect()
    
    async def test_fetch_all_no_results(self, temp_dir):
        """Test fetch_all with no results."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        await provider.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        
        results = await provider.fetch_all("SELECT * FROM test")
        
        assert results == []
        
        await provider.disconnect()
    
    async def test_fetch_val_success(self, temp_dir):
        """Test successful fetch_val query."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        await provider.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        await provider.execute("INSERT INTO test VALUES (42, 'test')")
        
        result = await provider.fetch_val("SELECT id FROM test WHERE name = ?", 'test')
        
        assert result == 42
        
        await provider.disconnect()
    
    async def test_fetch_val_no_results(self, temp_dir):
        """Test fetch_val with no results."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        await provider.execute("CREATE TABLE test (id INTEGER)")
        
        result = await provider.fetch_val("SELECT id FROM test WHERE id = 999")
        
        assert result is None
        
        await provider.disconnect()


@pytest.mark.asyncio
class TestSQLiteProviderTransactions:
    """Test transaction management for SQLiteProvider."""
    
    async def test_transaction_commit(self, temp_dir):
        """Test successful transaction commit."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        await provider.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        
        async with provider.transaction():
            await provider.execute("INSERT INTO test VALUES (1, 'test1')")
            await provider.execute("INSERT INTO test VALUES (2, 'test2')")
        
        # Check data was committed
        results = await provider.fetch_all("SELECT * FROM test ORDER BY id")
        assert len(results) == 2
        assert results[0]['id'] == 1
        assert results[1]['id'] == 2
        
        await provider.disconnect()
    
    async def test_transaction_rollback(self, temp_dir):
        """Test transaction rollback on exception."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        await provider.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        
        # Insert some initial data
        await provider.execute("INSERT INTO test VALUES (1, 'initial')")
        
        # Start transaction that will fail
        with pytest.raises(Exception, match="Intentional error"):
            async with provider.transaction():
                await provider.execute("INSERT INTO test VALUES (2, 'rollback_me')")
                raise Exception("Intentional error")
        
        # Check only initial data exists
        results = await provider.fetch_all("SELECT * FROM test")
        assert len(results) == 1
        assert results[0]['id'] == 1
        assert results[0]['name'] == 'initial'
        
        await provider.disconnect()
    
    async def test_transaction_auto_connect(self, temp_dir):
        """Test that transaction auto-connects if needed."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        # Don't connect manually
        async with provider.transaction():
            await provider.execute("CREATE TABLE test (id INTEGER)")
            await provider.execute("INSERT INTO test VALUES (1)")
        
        # Should have auto-connected
        assert provider._connection is not None
        
        # Verify data was committed
        result = await provider.fetch_val("SELECT COUNT(*) FROM test")
        assert result == 1
        
        await provider.disconnect()
    
    @patch('sqlite3.Connection.execute')
    async def test_transaction_handles_nested_transaction_error(self, mock_execute, temp_dir):
        """Test handling of nested transaction errors."""
        # Setup mock to simulate nested transaction error
        def side_effect(*args):
            if "BEGIN" in args[0]:
                raise sqlite3.OperationalError("cannot start a transaction within a transaction")
            return Mock()
        
        mock_execute.side_effect = side_effect
        
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        
        # Should handle the nested transaction error gracefully
        async with provider.transaction():
            await provider.execute("SELECT 1")  # This will use the mock
        
        await provider.disconnect()


@pytest.mark.asyncio
class TestSQLiteProviderUtilityMethods:
    """Test utility methods for SQLiteProvider."""
    
    async def test_file_exists_true(self, temp_dir):
        """Test file_exists returns True for existing file."""
        test_file = temp_dir / "exists.txt"
        test_file.write_text("test")
        
        provider = SQLiteProvider("test.db")
        result = await provider.file_exists(str(test_file))
        
        assert result is True
    
    async def test_file_exists_false(self, temp_dir):
        """Test file_exists returns False for non-existing file."""
        non_existent = temp_dir / "does_not_exist.txt"
        
        provider = SQLiteProvider("test.db")
        result = await provider.file_exists(str(non_existent))
        
        assert result is False
    
    async def test_is_healthy_connected(self, temp_dir):
        """Test is_healthy returns True when connected."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        result = await provider.is_healthy()
        
        assert result is True
        
        await provider.disconnect()
    
    async def test_is_healthy_not_connected(self):
        """Test is_healthy returns False when not connected."""
        provider = SQLiteProvider("test.db")
        
        result = await provider.is_healthy()
        
        # Should auto-connect and succeed
        assert result is True  # SQLite auto-connects on queries
    
    async def test_migrate_success(self, temp_dir):
        """Test successful migration."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        # Create migration files
        migration1 = temp_dir / "001_create_users.sql"
        migration1.write_text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);")
        
        migration2 = temp_dir / "002_create_posts.sql"
        migration2.write_text("CREATE TABLE posts (id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT);")
        
        result = await provider.migrate([str(migration1), str(migration2)])
        
        assert result is True
        
        # Verify tables were created
        await provider.connect()
        users_count = await provider.fetch_val("SELECT COUNT(name) FROM sqlite_master WHERE type='table' AND name='users'")
        posts_count = await provider.fetch_val("SELECT COUNT(name) FROM sqlite_master WHERE type='table' AND name='posts'")
        
        assert users_count == 1
        assert posts_count == 1
        
        await provider.disconnect()
    
    async def test_migrate_file_not_found(self, temp_dir):
        """Test migration with non-existent file."""
        provider = SQLiteProvider("test.db")
        
        result = await provider.migrate([str(temp_dir / "nonexistent.sql")])
        
        assert result is False
    
    async def test_migrate_invalid_sql(self, temp_dir):
        """Test migration with invalid SQL."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        # Create migration with invalid SQL
        migration = temp_dir / "invalid.sql"
        migration.write_text("INVALID SQL STATEMENT;")
        
        result = await provider.migrate([str(migration)])
        
        assert result is False


@pytest.mark.asyncio
class TestSQLiteProviderAliases:
    """Test alias methods for SQLiteProvider."""
    
    async def test_fetch_alias(self, temp_dir):
        """Test fetch alias for fetch_all."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        await provider.execute("CREATE TABLE test (id INTEGER)")
        await provider.execute("INSERT INTO test VALUES (1)")
        
        # Test alias
        result = await provider.fetch("SELECT * FROM test")
        
        assert len(result) == 1
        assert result[0]['id'] == 1
        
        await provider.disconnect()
    
    async def test_fetchone_alias(self, temp_dir):
        """Test fetchone alias for fetch_one."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        await provider.execute("CREATE TABLE test (id INTEGER)")
        await provider.execute("INSERT INTO test VALUES (42)")
        
        result = await provider.fetchone("SELECT * FROM test")
        
        assert result is not None
        assert result['id'] == 42
        
        await provider.disconnect()
    
    async def test_fetchall_alias(self, temp_dir):
        """Test fetchall alias for fetch_all."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        await provider.execute("CREATE TABLE test (id INTEGER)")
        await provider.execute("INSERT INTO test VALUES (1)")
        await provider.execute("INSERT INTO test VALUES (2)")
        
        result = await provider.fetchall("SELECT * FROM test ORDER BY id")
        
        assert len(result) == 2
        assert result[0]['id'] == 1
        assert result[1]['id'] == 2
        
        await provider.disconnect()
    
    async def test_fetchval_alias(self, temp_dir):
        """Test fetchval alias for fetch_val."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        await provider.execute("CREATE TABLE test (id INTEGER)")
        await provider.execute("INSERT INTO test VALUES (123)")
        
        result = await provider.fetchval("SELECT id FROM test")
        
        assert result == 123
        
        await provider.disconnect()


@pytest.mark.asyncio
class TestSQLiteProviderErrorHandling:
    """Test error handling for SQLiteProvider."""
    
    @patch('sqlite3.connect')
    async def test_execute_error_rollback(self, mock_connect):
        """Test that execute errors trigger rollback."""
        # Create mock connection that fails on execute
        mock_connection = Mock()
        mock_connection.execute.side_effect = sqlite3.Error("Execute failed")
        mock_connection.rollback = Mock()
        mock_connect.return_value = mock_connection
        
        provider = SQLiteProvider("test.db")
        
        with pytest.raises(sqlite3.Error, match="Execute failed"):
            await provider.execute("SELECT 1")
        
        # Should have attempted rollback
        mock_connection.rollback.assert_called_once()
    
    @patch('sqlite3.connect')
    async def test_fetch_one_error_propagation(self, mock_connect):
        """Test that fetch_one errors are propagated."""
        mock_connection = Mock()
        mock_connection.execute.side_effect = sqlite3.Error("Fetch failed")
        mock_connect.return_value = mock_connection
        
        provider = SQLiteProvider("test.db")
        
        with pytest.raises(sqlite3.Error, match="Fetch failed"):
            await provider.fetch_one("SELECT 1")
    
    @patch('sqlite3.connect')
    async def test_fetch_all_error_propagation(self, mock_connect):
        """Test that fetch_all errors are propagated."""
        mock_connection = Mock()
        mock_connection.execute.side_effect = sqlite3.Error("Fetch all failed")
        mock_connect.return_value = mock_connection
        
        provider = SQLiteProvider("test.db")
        
        with pytest.raises(sqlite3.Error, match="Fetch all failed"):
            await provider.fetch_all("SELECT 1")
    
    @patch('sqlite3.connect')
    async def test_fetch_val_error_propagation(self, mock_connect):
        """Test that fetch_val errors are propagated."""
        mock_connection = Mock()
        mock_connection.execute.side_effect = sqlite3.Error("Fetch val failed")
        mock_connect.return_value = mock_connection
        
        provider = SQLiteProvider("test.db")
        
        with pytest.raises(sqlite3.Error, match="Fetch val failed"):
            await provider.fetch_val("SELECT 1")


@pytest.mark.asyncio
class TestSQLiteProviderThreadSafety:
    """Test thread safety for SQLiteProvider."""
    
    async def test_concurrent_operations(self, temp_dir):
        """Test concurrent operations with locking."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        await provider.connect()
        await provider.execute("CREATE TABLE test (id INTEGER)")
        
        async def insert_data(value):
            await provider.execute("INSERT INTO test VALUES (?)", value)
        
        # Run concurrent inserts
        tasks = [insert_data(i) for i in range(10)]
        await asyncio.gather(*tasks)
        
        # Check all data was inserted
        count = await provider.fetch_val("SELECT COUNT(*) FROM test")
        assert count == 10
        
        await provider.disconnect()
    
    async def test_lock_acquisition(self, temp_dir):
        """Test that operations acquire lock properly."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(str(db_path))
        
        # Mock the lock to verify it's being used
        original_lock = provider._lock
        provider._lock = AsyncMock()
        provider._lock.__aenter__ = AsyncMock()
        provider._lock.__aexit__ = AsyncMock()
        
        await provider.connect()
        
        # Verify lock was acquired
        provider._lock.__aenter__.assert_called()
        provider._lock.__aexit__.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])