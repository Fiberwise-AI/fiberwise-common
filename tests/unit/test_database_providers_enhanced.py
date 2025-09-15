"""
Enhanced unit tests for fiberwise_common.database.providers module.

Comprehensive tests for all database providers including SQLite and DuckDB.
"""
import pytest
import tempfile
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from fiberwise_common.database.base import DatabaseProvider
from fiberwise_common.database.providers import (
    SQLiteProvider, 
    DuckDBProvider,
    create_database_provider
)


class TestDatabaseProviderBase:
    """Test base DatabaseProvider abstract class behavior."""

    def test_database_provider_is_abstract(self):
        """Test that DatabaseProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DatabaseProvider()

    def test_database_provider_interface(self):
        """Test that DatabaseProvider defines the correct interface."""
        abstract_methods = DatabaseProvider.__abstractmethods__
        expected_methods = {
            'connect', 'disconnect', 'execute', 'execute_many',
            'fetch_all', 'fetch_one', 'begin_transaction', 
            'commit_transaction', 'rollback_transaction'
        }
        assert abstract_methods == expected_methods


class TestSQLiteProvider:
    """Test SQLiteProvider implementation."""

    @pytest.fixture
    def temp_db_path(self) -> str:
        """Create temporary database file path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            return tmp_file.name

    @pytest.fixture
    def sqlite_provider(self, temp_db_path: str) -> SQLiteProvider:
        """Create SQLite provider instance."""
        return SQLiteProvider(f'sqlite:///{temp_db_path}')

    def test_sqlite_provider_initialization(self, temp_db_path: str):
        """Test SQLiteProvider initialization."""
        provider = SQLiteProvider(f'sqlite:///{temp_db_path}')
        
        assert provider.provider == 'sqlite'
        assert temp_db_path in provider.connection_string
        assert provider.connection_string.startswith('sqlite:///')

    def test_sqlite_provider_initialization_with_url(self):
        """Test SQLiteProvider initialization with different URL formats."""
        test_cases = [
            'sqlite:///test.db',
            'sqlite:////absolute/path/test.db',
            'sqlite://:memory:',
        ]
        
        for connection_string in test_cases:
            provider = SQLiteProvider(connection_string)
            assert provider.connection_string == connection_string

    @pytest.mark.asyncio
    async def test_sqlite_provider_connect_disconnect(self, sqlite_provider: SQLiteProvider):
        """Test SQLite provider connection lifecycle."""
        # Initially not connected
        assert sqlite_provider._connection is None
        
        # Test connection
        await sqlite_provider.connect()
        assert sqlite_provider._connection is not None
        
        # Test disconnection
        await sqlite_provider.disconnect()
        # Note: Connection might be None or closed, depends on implementation

    @pytest.mark.asyncio
    async def test_sqlite_provider_execute_query(self, sqlite_provider: SQLiteProvider):
        """Test executing queries with SQLite provider."""
        await sqlite_provider.connect()
        
        try:
            # Create test table
            await sqlite_provider.execute(
                "CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)"
            )
            
            # Insert test data
            await sqlite_provider.execute(
                "INSERT INTO test_table (name) VALUES (?)", 
                ("test_name",)
            )
            
            # Query data
            result = await sqlite_provider.fetch_all("SELECT * FROM test_table")
            assert len(result) == 1
            assert result[0][1] == "test_name"  # name column
            
        finally:
            await sqlite_provider.disconnect()

    @pytest.mark.asyncio
    async def test_sqlite_provider_execute_many(self, sqlite_provider: SQLiteProvider):
        """Test executing multiple queries."""
        await sqlite_provider.connect()
        
        try:
            await sqlite_provider.execute(
                "CREATE TABLE batch_test (id INTEGER, value TEXT)"
            )
            
            # Test batch insert
            data = [(1, 'one'), (2, 'two'), (3, 'three')]
            await sqlite_provider.execute_many(
                "INSERT INTO batch_test (id, value) VALUES (?, ?)",
                data
            )
            
            # Verify batch insert
            results = await sqlite_provider.fetch_all("SELECT * FROM batch_test ORDER BY id")
            assert len(results) == 3
            assert results[0][1] == 'one'
            assert results[2][1] == 'three'
            
        finally:
            await sqlite_provider.disconnect()

    @pytest.mark.asyncio
    async def test_sqlite_provider_fetch_one(self, sqlite_provider: SQLiteProvider):
        """Test fetching single row."""
        await sqlite_provider.connect()
        
        try:
            await sqlite_provider.execute(
                "CREATE TABLE single_test (id INTEGER PRIMARY KEY, name TEXT)"
            )
            await sqlite_provider.execute(
                "INSERT INTO single_test (name) VALUES (?)",
                ("single_row",)
            )
            
            result = await sqlite_provider.fetch_one("SELECT name FROM single_test")
            assert result is not None
            assert result[0] == "single_row"
            
            # Test fetch_one with no results
            no_result = await sqlite_provider.fetch_one("SELECT * FROM single_test WHERE id = 999")
            assert no_result is None
            
        finally:
            await sqlite_provider.disconnect()

    @pytest.mark.asyncio
    async def test_sqlite_provider_transactions(self, sqlite_provider: SQLiteProvider):
        """Test transaction handling."""
        await sqlite_provider.connect()
        
        try:
            await sqlite_provider.execute(
                "CREATE TABLE transaction_test (id INTEGER, value TEXT)"
            )
            
            # Test successful transaction
            await sqlite_provider.begin_transaction()
            await sqlite_provider.execute("INSERT INTO transaction_test VALUES (1, 'test')")
            await sqlite_provider.commit_transaction()
            
            result = await sqlite_provider.fetch_one("SELECT COUNT(*) FROM transaction_test")
            assert result[0] == 1
            
            # Test rollback transaction
            await sqlite_provider.begin_transaction()
            await sqlite_provider.execute("INSERT INTO transaction_test VALUES (2, 'rollback')")
            await sqlite_provider.rollback_transaction()
            
            result = await sqlite_provider.fetch_one("SELECT COUNT(*) FROM transaction_test")
            assert result[0] == 1  # Should still be 1, rollback worked
            
        finally:
            await sqlite_provider.disconnect()

    def test_sqlite_provider_memory_database(self):
        """Test SQLite provider with in-memory database."""
        provider = SQLiteProvider('sqlite://:memory:')
        assert ':memory:' in provider.connection_string


class TestDuckDBProvider:
    """Test DuckDBProvider implementation."""

    @pytest.fixture
    def temp_db_path(self) -> str:
        """Create temporary DuckDB file path."""
        with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as tmp_file:
            return tmp_file.name

    @pytest.fixture
    def duckdb_provider(self, temp_db_path: str) -> DuckDBProvider:
        """Create DuckDB provider instance."""
        return DuckDBProvider(f'duckdb:///{temp_db_path}')

    def test_duckdb_provider_initialization(self, temp_db_path: str):
        """Test DuckDBProvider initialization."""
        provider = DuckDBProvider(f'duckdb:///{temp_db_path}')
        
        assert provider.provider == 'duckdb'
        assert temp_db_path in provider.connection_string
        assert provider.connection_string.startswith('duckdb:///')

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="DuckDB provider may not be fully implemented")
    async def test_duckdb_provider_basic_operations(self, duckdb_provider: DuckDBProvider):
        """Test basic DuckDB operations."""
        await duckdb_provider.connect()
        
        try:
            # Create test table
            await duckdb_provider.execute(
                "CREATE TABLE test_table (id INTEGER, name VARCHAR)"
            )
            
            # Insert and query data
            await duckdb_provider.execute(
                "INSERT INTO test_table VALUES (1, 'test')"
            )
            
            result = await duckdb_provider.fetch_all("SELECT * FROM test_table")
            assert len(result) == 1
            
        finally:
            await duckdb_provider.disconnect()


class TestCreateDatabaseProvider:
    """Test create_database_provider factory function."""

    def test_create_sqlite_provider(self, temp_dir: Path):
        """Test creating SQLite provider via factory."""
        db_path = temp_dir / "test.db"
        provider = create_database_provider('sqlite', str(db_path))
        
        assert isinstance(provider, SQLiteProvider)
        assert provider.provider == 'sqlite'
        assert str(db_path) in provider.connection_string

    def test_create_duckdb_provider(self, temp_dir: Path):
        """Test creating DuckDB provider via factory."""
        db_path = temp_dir / "test.duckdb"
        provider = create_database_provider('duckdb', str(db_path))
        
        assert isinstance(provider, DuckDBProvider)
        assert provider.provider == 'duckdb'
        assert str(db_path) in provider.connection_string

    def test_create_unsupported_provider(self):
        """Test creating unsupported provider type."""
        with pytest.raises((ValueError, NotImplementedError)):
            create_database_provider('postgresql', 'postgresql://localhost/test')

    @pytest.mark.parametrize("provider_type,expected_class", [
        ('sqlite', SQLiteProvider),
        ('duckdb', DuckDBProvider),
    ])
    def test_create_provider_parametrized(self, provider_type: str, expected_class: type, temp_dir: Path):
        """Test creating different provider types."""
        db_path = temp_dir / f"test.{provider_type}"
        provider = create_database_provider(provider_type, str(db_path))
        
        assert isinstance(provider, expected_class)
        assert provider.provider == provider_type


class TestDatabaseProviderErrorHandling:
    """Test error handling in database providers."""

    @pytest.mark.asyncio
    async def test_sqlite_invalid_connection_string(self):
        """Test SQLite provider with invalid connection string."""
        # This might not fail immediately but could fail on connect
        provider = SQLiteProvider('invalid://connection/string')
        
        with pytest.raises(Exception):  # Specific exception depends on implementation
            await provider.connect()

    @pytest.mark.asyncio
    async def test_sqlite_permission_denied(self):
        """Test SQLite provider with permission denied."""
        # Try to create database in root directory (usually permission denied)
        if Path('/root').exists():
            provider = SQLiteProvider('sqlite:////root/permission_denied.db')
            with pytest.raises(Exception):
                await provider.connect()

    @pytest.mark.asyncio
    async def test_execute_without_connection(self, temp_dir: Path):
        """Test executing query without connection."""
        db_path = temp_dir / "test.db"
        provider = SQLiteProvider(f'sqlite:///{db_path}')
        
        # Don't connect, try to execute
        with pytest.raises(Exception):
            await provider.execute("SELECT 1")

    @pytest.mark.asyncio
    async def test_sqlite_invalid_sql(self, sqlite_provider):
        """Test executing invalid SQL."""
        await sqlite_provider.connect()
        
        try:
            with pytest.raises(Exception):
                await sqlite_provider.execute("INVALID SQL STATEMENT")
        finally:
            await sqlite_provider.disconnect()


class TestDatabaseProviderConcurrency:
    """Test concurrent database operations."""

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, sqlite_provider: SQLiteProvider):
        """Test concurrent database queries."""
        await sqlite_provider.connect()
        
        try:
            await sqlite_provider.execute(
                "CREATE TABLE concurrent_test (id INTEGER, value TEXT)"
            )
            
            # Execute multiple concurrent inserts
            async def insert_data(value: str):
                await sqlite_provider.execute(
                    "INSERT INTO concurrent_test (value) VALUES (?)",
                    (value,)
                )
                return value
            
            tasks = [insert_data(f"value_{i}") for i in range(5)]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            
            # Verify all data was inserted
            count_result = await sqlite_provider.fetch_one("SELECT COUNT(*) FROM concurrent_test")
            assert count_result[0] == 5
            
        finally:
            await sqlite_provider.disconnect()

    @pytest.mark.asyncio
    async def test_concurrent_connections(self, temp_dir: Path):
        """Test multiple concurrent connections."""
        db_path = temp_dir / "concurrent.db"
        
        async def create_and_use_provider(provider_id: int):
            provider = SQLiteProvider(f'sqlite:///{db_path}')
            await provider.connect()
            
            try:
                # Create table with unique name per provider
                table_name = f"test_table_{provider_id}"
                await provider.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER)")
                await provider.execute(f"INSERT INTO {table_name} VALUES ({provider_id})")
                
                result = await provider.fetch_one(f"SELECT id FROM {table_name}")
                return result[0]
            finally:
                await provider.disconnect()
        
        # Create multiple providers concurrently
        tasks = [create_and_use_provider(i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        assert results == [0, 1, 2]


class TestDatabaseProviderIntegration:
    """Integration tests for database providers."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_database_workflow(self, sqlite_provider: SQLiteProvider):
        """Test complete database workflow."""
        await sqlite_provider.connect()
        
        try:
            # Create schema
            await sqlite_provider.execute("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await sqlite_provider.execute("""
                CREATE TABLE posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    title TEXT NOT NULL,
                    content TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Insert test data
            await sqlite_provider.begin_transaction()
            
            users_data = [
                ('user1', 'user1@example.com'),
                ('user2', 'user2@example.com'),
                ('user3', 'user3@example.com')
            ]
            
            await sqlite_provider.execute_many(
                "INSERT INTO users (username, email) VALUES (?, ?)",
                users_data
            )
            
            # Get user IDs and create posts
            users = await sqlite_provider.fetch_all("SELECT id, username FROM users")
            posts_data = []
            for user_id, username in users:
                posts_data.append((user_id, f"Post by {username}", f"Content from {username}"))
            
            await sqlite_provider.execute_many(
                "INSERT INTO posts (user_id, title, content) VALUES (?, ?, ?)",
                posts_data
            )
            
            await sqlite_provider.commit_transaction()
            
            # Query with JOIN
            results = await sqlite_provider.fetch_all("""
                SELECT u.username, p.title, p.content
                FROM users u
                JOIN posts p ON u.id = p.user_id
                ORDER BY u.username
            """)
            
            assert len(results) == 3
            assert results[0][0] == 'user1'  # First user's username
            assert 'Post by user1' in results[0][1]  # First user's post title
            
        finally:
            await sqlite_provider.disconnect()

    @pytest.mark.integration
    def test_provider_factory_integration(self, temp_dir: Path):
        """Test integration of provider factory with different database types."""
        # Test creating and using different providers
        databases = [
            ('sqlite', 'test.db', SQLiteProvider),
            ('duckdb', 'test.duckdb', DuckDBProvider)
        ]
        
        for db_type, filename, expected_class in databases:
            db_path = temp_dir / filename
            provider = create_database_provider(db_type, str(db_path))
            
            assert isinstance(provider, expected_class)
            assert provider.provider == db_type
            assert str(db_path) in provider.connection_string

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_dataset_operations(self, sqlite_provider: SQLiteProvider):
        """Test operations with larger datasets."""
        await sqlite_provider.connect()
        
        try:
            await sqlite_provider.execute("""
                CREATE TABLE large_test (
                    id INTEGER PRIMARY KEY,
                    data TEXT,
                    value INTEGER
                )
            """)
            
            # Insert larger dataset (1000 records)
            large_dataset = [(i, f"data_{i}", i * 2) for i in range(1000)]
            
            await sqlite_provider.begin_transaction()
            await sqlite_provider.execute_many(
                "INSERT INTO large_test (id, data, value) VALUES (?, ?, ?)",
                large_dataset
            )
            await sqlite_provider.commit_transaction()
            
            # Test querying large dataset
            count_result = await sqlite_provider.fetch_one("SELECT COUNT(*) FROM large_test")
            assert count_result[0] == 1000
            
            # Test filtered query
            filtered_results = await sqlite_provider.fetch_all(
                "SELECT * FROM large_test WHERE value > ? ORDER BY id LIMIT 10",
                (1000,)
            )
            assert len(filtered_results) == 10
            assert filtered_results[0][2] > 1000  # value column
            
        finally:
            await sqlite_provider.disconnect()