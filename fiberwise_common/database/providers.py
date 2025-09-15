"""
Database provider implementations for FiberWise applications.
"""

import asyncio
import logging
import sqlite3
from .base import DatabaseProvider
from .postgres import PostgresProvider
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import threading

from .base import DatabaseProvider

logger = logging.getLogger(__name__)


class SQLiteProvider(DatabaseProvider):
    """SQLite database provider implementation - simplified for reliability."""
    
    @property
    def provider(self) -> str:
        return "sqlite"
    
    def __init__(self, database_url: str):
        super().__init__()  # Initialize parent class with query adapter
        self.database_url = database_url
        # Extract database path from URL
        if database_url.startswith("sqlite:///"):
            self.db_path = database_url[10:]  # Remove "sqlite:///"
        else:
            self.db_path = database_url
        
        self._connection = None
        self._lock = asyncio.Lock()
    
    def _create_connection(self):
        """Create a new SQLite connection."""
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        connection = sqlite3.connect(
            str(db_path), 
            check_same_thread=False,
            timeout=10.0,  # 10 second timeout
            isolation_level='DEFERRED'  # Use deferred transactions for proper rollback
        )
        connection.row_factory = sqlite3.Row
        # Enable foreign keys
        connection.execute("PRAGMA foreign_keys=ON")
        connection.commit()
        
        return connection
    
    async def connect(self) -> bool:
        """Establish connection to SQLite database."""
        try:
            async with self._lock:
                if self._connection is None:
                    loop = asyncio.get_event_loop()
                    self._connection = await loop.run_in_executor(None, self._create_connection)
            
            logger.info(f"Connected to SQLite database: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"SQLite connection error: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close database connection."""
        try:
            async with self._lock:
                if self._connection:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._connection.close)
                    self._connection = None
            logger.info("Disconnected from SQLite database")
        except Exception as e:
            logger.error(f"Error disconnecting from SQLite: {e}")
    
    async def execute(self, query: str, *args) -> Any:
        """Execute a query."""
        try:
            # Adapt query and parameters for SQLite
            adapted_query, adapted_params = self.adapt_query(query, args)
            
            async with self._lock:
                if not self._connection:
                    await self.connect()
                
                def _execute():
                    cursor = self._connection.execute(adapted_query, adapted_params)
                    self._connection.commit()
                    return cursor.lastrowid
                
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, _execute)
                
        except Exception as e:
            logger.error(f"SQLite execute error: {e}")
            if self._connection:
                try:
                    self._connection.rollback()
                    logger.info("Transaction rolled back")
                except:
                    pass
            raise
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch one row."""
        try:
            # Adapt query and parameters for SQLite
            adapted_query, adapted_params = self.adapt_query(query, args)
            
            async with self._lock:
                if not self._connection:
                    await self.connect()
                
                def _fetchone():
                    cursor = self._connection.execute(adapted_query, adapted_params)
                    row = cursor.fetchone()
                    return dict(row) if row else None
                
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, _fetchone)
                
        except Exception as e:
            logger.error(f"SQLite fetchone error: {e}")
            raise
    
    async def fetch_all(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetch all rows."""
        try:
            # Adapt query and parameters for SQLite
            adapted_query, adapted_params = self.adapt_query(query, args)
            
            async with self._lock:
                if not self._connection:
                    await self.connect()
                
                def _fetchall():
                    cursor = self._connection.execute(adapted_query, adapted_params)
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows] if rows else []
                
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, _fetchall)
                
        except Exception as e:
            logger.error(f"SQLite fetchall error: {e}")
            raise
    
    async def fetch_val(self, query: str, *args) -> Any:
        """Fetch a single value from the first row."""
        try:
            # Adapt query and parameters for SQLite
            adapted_query, adapted_params = self.adapt_query(query, args)
            
            async with self._lock:
                if not self._connection:
                    await self.connect()
                
                def _fetchval():
                    cursor = self._connection.execute(adapted_query, adapted_params)
                    row = cursor.fetchone()
                    return row[0] if row else None
                
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, _fetchval)
                
        except Exception as e:
            logger.error(f"SQLite fetchval error: {e}")
            raise
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        # Ensure connection exists
        async with self._lock:
            if not self._connection:
                await self.connect()
        
        def _begin():
            # Check if we're already in a transaction to avoid nesting
            try:
                self._connection.execute("BEGIN")
            except sqlite3.OperationalError as e:
                if "cannot start a transaction within a transaction" in str(e):
                    # Already in a transaction, that's fine - just continue
                    pass
                else:
                    # Some other error, re-raise it
                    raise
        
        def _commit():
            # Only commit if we're in a transaction
            try:
                self._connection.commit()
            except sqlite3.OperationalError as e:
                if "cannot commit - no transaction is active" not in str(e):
                    # Some other error, re-raise it
                    raise
        
        def _rollback():
            # Only rollback if we're in a transaction  
            try:
                self._connection.rollback()
            except sqlite3.OperationalError as e:
                if "cannot rollback - no transaction is active" not in str(e):
                    # Some other error, re-raise it
                    raise
        
        loop = asyncio.get_event_loop()
        
        # Begin transaction (acquire lock only for this operation)
        async with self._lock:
            await loop.run_in_executor(None, _begin)
        
        try:
            yield self
            # Commit transaction (acquire lock only for this operation)
            async with self._lock:
                await loop.run_in_executor(None, _commit)
        except Exception as e:
            # Rollback transaction (acquire lock only for this operation)
            async with self._lock:
                await loop.run_in_executor(None, _rollback)
            logger.error(f"Transaction rolled back: {e}")
            raise
    
    async def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        import os
        return os.path.exists(path)

    async def is_healthy(self) -> bool:
        """Check if the database connection is healthy."""
        try:
            await self.fetchone("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def migrate(self, migration_files: List[str]) -> bool:
        """Run database migrations from files."""
        try:
            for migration_file in migration_files:
                with open(migration_file, 'r') as f:
                    sql = f.read()
                    await self.execute(sql)
            return True
        except Exception as e:
            logger.error(f"Migration error: {e}")
            return False

    # Alias methods for compatibility
    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Alias for fetch_all."""
        return await self.fetch_all(query, *args)
    
    async def fetchone(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Alias for fetch_one."""
        return await self.fetch_one(query, *args)
    
    async def fetchall(self, query: str, *args) -> List[Dict[str, Any]]:
        """Alias for fetch_all."""
        return await self.fetch_all(query, *args)
    
    async def fetchval(self, query: str, *args) -> Any:
        """Alias for fetch_val."""
        return await self.fetch_val(query, *args)



class DuckDBProvider(DatabaseProvider):
    """DuckDB database provider implementation."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._executor = None
        self._local = threading.local()
        self._parse_url()
    """DuckDB database provider implementation."""
    
    def __init__(self, database_url: str):
        # Extract database path from URL
        if database_url.startswith("duckdb:///"):
            self.db_path = database_url[10:]  # Remove "duckdb:///"
        else:
            self.db_path = database_url
        
        self._executor = None
        self._local = threading.local()
    
    def _get_connection(self):
        """Get thread-local DuckDB connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            try:
                import duckdb
                
                db_path = Path(self.db_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                
                self._local.connection = duckdb.connect(str(db_path))
                logger.debug(f"Created new DuckDB connection for thread: {threading.current_thread().name}")
            except ImportError:
                raise RuntimeError("duckdb package not installed. Install with: pip install duckdb")
        
        return self._local.connection
    
    async def connect(self) -> bool:
        """Establish connection to DuckDB database."""
        try:
            if self._executor is None:
                from concurrent.futures import ThreadPoolExecutor
                self._executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="duckdb-")
            
            # Test connection
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._test_connection)
            
            logger.info(f"Connected to DuckDB database: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"DuckDB connection error: {e}")
            return False
    
    def _test_connection(self):
        """Test DuckDB connection (sync)."""
        conn = self._get_connection()
        conn.execute("SELECT 1").fetchone()
        return True
    
    async def disconnect(self) -> None:
        """Close database connection."""
        try:
            if hasattr(self._local, 'connection') and self._local.connection:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._local.connection.close)
                self._local.connection = None
            
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None
            
            logger.info("Disconnected from DuckDB database")
        except Exception as e:
            logger.error(f"Error disconnecting from DuckDB: {e}")
    
    async def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute a query."""
        if not self._executor:
            raise RuntimeError("Database not connected")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._execute_sync, query, params)
    
    def _execute_sync(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute query synchronously."""
        conn = self._get_connection()
        try:
            if params:
                return conn.execute(query, params)
            else:
                return conn.execute(query)
        except Exception as e:
            logger.error(f"DuckDB execute error: {e}")
            raise
    
    
    async def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        import os
        return os.path.exists(path)
    
    def _fetchone_sync(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """Fetch one row synchronously."""
        conn = self._get_connection()
        try:
            if params:
                result = conn.execute(query, params).fetchone()
            else:
                result = conn.execute(query).fetchone()
            
            if result:
                # Get column names
                columns = [desc[0] for desc in conn.description]
                return dict(zip(columns, result))
            return None
        except Exception as e:
            logger.error(f"DuckDB fetchone error: {e}")
            raise
    
    async def fetchall(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Fetch all rows."""
        if not self._executor:
            raise RuntimeError("Database not connected")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._fetchall_sync, query, params)
    
    def _fetchall_sync(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Fetch all rows synchronously."""
        conn = self._get_connection()
        try:
            if params:
                result = conn.execute(query, params).fetchall()
            else:
                result = conn.execute(query).fetchall()
            
            if result:
                # Get column names
                columns = [desc[0] for desc in conn.description]
                return [dict(zip(columns, row)) for row in result]
            return []
        except Exception as e:
            logger.error(f"DuckDB fetchall error: {e}")
            raise
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        if not self._executor:
            raise RuntimeError("Database not connected")
        
        loop = asyncio.get_event_loop()
        
        # Start transaction
        await loop.run_in_executor(self._executor, self._begin_transaction)
        
        try:
            yield self
            # Commit transaction
            await loop.run_in_executor(self._executor, self._commit_transaction)
        except Exception as e:
            # Rollback transaction
            await loop.run_in_executor(self._executor, self._rollback_transaction)
            logger.error(f"DuckDB transaction rolled back: {e}")
            raise
    
    def _begin_transaction(self):
        """Begin transaction synchronously."""
        conn = self._get_connection()
        conn.execute("BEGIN TRANSACTION")
    
    def _commit_transaction(self):
        """Commit transaction synchronously."""
        conn = self._get_connection()
        conn.execute("COMMIT")
    
    def _rollback_transaction(self):
        """Rollback transaction synchronously."""
        conn = self._get_connection()
        conn.execute("ROLLBACK")
    
    async def is_healthy(self) -> bool:
        """Check if the database connection is healthy."""
        try:
            if not self._executor:
                return False
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._test_connection)
            return True
        except Exception:
            return False

    async def migrate(self, migration_files: List[str]) -> bool:
        """Run database migrations from files."""
        try:
            for migration_file in migration_files:
                with open(migration_file, 'r') as f:
                    sql = f.read()
                    await self.execute(sql)
            return True
        except Exception as e:
            logger.error(f"Migration error: {e}")
            return False


__all__ = ["SQLiteProvider", "DuckDBProvider", "PostgresProvider", "create_database_provider"]

def create_database_provider(database_url: str) -> DatabaseProvider:
    """Factory function to create the appropriate database provider."""
    if database_url.startswith("sqlite"):
        return SQLiteProvider(database_url)
    elif database_url.startswith("duckdb"):
        return DuckDBProvider(database_url)
    elif database_url.startswith("postgres"):
        return PostgresProvider(database_url)
    else:
        raise ValueError(f"Unsupported database URL: {database_url}")
