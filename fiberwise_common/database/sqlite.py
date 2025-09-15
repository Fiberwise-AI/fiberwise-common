import sqlite3
import asyncio
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from contextlib import asynccontextmanager
from .base import DatabaseProvider


class SQLiteProvider(DatabaseProvider):
    """SQLite database provider implementation"""
    
    def __init__(self):
        super().__init__()  # Initialize the parent DatabaseProvider
        self.db_path: Optional[str] = None
        self.connection: Optional[sqlite3.Connection] = None
    
    @property
    def provider(self) -> str:
        return "sqlite"
    
    def set_db_path(self, db_path: str):
        """Set the database file path"""
        self.db_path = db_path
    
    async def connect(self) -> None:
        """Establish connection to SQLite database"""
        if not self.db_path:
            raise ValueError("Database path not set")
        
        # Run in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(
            None, self._sync_connect
        )
    
    def _sync_connect(self) -> None:
        """Synchronous connection method"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(
                self.db_path, 
                check_same_thread=False,
                isolation_level=None  # Autocommit mode
            )
            self.connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self.connection.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error as e:
            raise ConnectionError(f"SQLite connection error: {e}")
    
    async def disconnect(self) -> None:
        """Close database connection"""
        if self.connection:
            await asyncio.get_event_loop().run_in_executor(
                None, self.connection.close
            )
            self.connection = None
    
    @asynccontextmanager
    async def transaction(self):
        """Create a database transaction context manager"""
        if not self.connection:
            raise ConnectionError("Database not connected")
        
        # For SQLite, we need to handle transactions more carefully
        # Set isolation level to defer commits
        old_isolation = self.connection.isolation_level
        self.connection.isolation_level = ""  # Start transaction mode
        
        try:
            # Explicitly start transaction
            await asyncio.get_event_loop().run_in_executor(
                None, self.connection.execute, "BEGIN IMMEDIATE"
            )
            
            # Yield self so the same DatabaseProvider interface can be used
            yield self
            
            # Commit the transaction
            await asyncio.get_event_loop().run_in_executor(
                None, self.connection.execute, "COMMIT"
            )
        except Exception as e:
            # Rollback on any error
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.connection.execute, "ROLLBACK"
                )
            except:
                pass  # Ignore rollback errors
            raise e
        finally:
            # Restore original isolation level
            self.connection.isolation_level = old_isolation
    
    async def execute(self, query: str, *args) -> Optional[str]:
        """Execute a query and return affected rows"""
        # Use query adapter to convert PostgreSQL queries to SQLite
        adapted_query, adapted_params = self.adapt_query(query, args)
        
        if not self.connection:
            raise ConnectionError("Database not connected")
        
        def _execute():
            cursor = self.connection.cursor()
            cursor.execute(adapted_query, adapted_params)
            rowcount = cursor.rowcount
            # Ensure changes are committed if not in explicit transaction
            if self.connection.isolation_level is None:
                self.connection.commit()
            return f"COMMAND EXECUTED {rowcount}"
        
        return await asyncio.get_event_loop().run_in_executor(None, _execute)
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch a single row"""
        # Use query adapter to convert PostgreSQL queries to SQLite
        adapted_query, adapted_params = self.adapt_query(query, args)
        
        if not self.connection:
            raise ConnectionError("Database not connected")
        
        def _fetch_one():
            cursor = self.connection.cursor()
            cursor.execute(adapted_query, adapted_params)
            row = cursor.fetchone()
            return dict(row) if row else None
        
        return await asyncio.get_event_loop().run_in_executor(None, _fetch_one)
    
    async def fetch_all(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetch all rows"""
        # Use query adapter to convert PostgreSQL queries to SQLite
        adapted_query, adapted_params = self.adapt_query(query, args)
        
        if not self.connection:
            raise ConnectionError("Database not connected")
        
        def _fetch_all():
            cursor = self.connection.cursor()
            cursor.execute(adapted_query, adapted_params)
            return [dict(row) for row in cursor.fetchall()]
        
        return await asyncio.get_event_loop().run_in_executor(None, _fetch_all)
    
    async def execute_script(self, script: str) -> None:
        """Execute multiple SQL statements"""
        if not self.connection:
            raise ConnectionError("Database not connected")
        
        await asyncio.get_event_loop().run_in_executor(
            None, self.connection.executescript, script
        )
    
    async def fetch_val(self, query: str, *args) -> Any:
        """Fetch a single value"""
        # Use query adapter to convert PostgreSQL queries to SQLite
        adapted_query, adapted_params = self.adapt_query(query, args)
        
        if not self.connection:
            raise ConnectionError("Database not connected")
        
        def _fetch_val():
            cursor = self.connection.cursor()
            cursor.execute(adapted_query, adapted_params)
            row = cursor.fetchone()
            return row[0] if row else None
        
        return await asyncio.get_event_loop().run_in_executor(None, _fetch_val)
    
    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Execute a query and fetch a single row (alias for fetch_one)"""
        return await self.fetch_one(query, *args)
    
    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a query and fetch multiple rows (alias for fetch_all)"""
        return await self.fetch_all(query, *args)
    
    async def fetchval(self, query: str, *args) -> Any:
        """Execute a query and fetch a single value (alias for fetch_val)"""
        return await self.fetch_val(query, *args)
    
    async def file_exists(self, path: str) -> bool:
        """Check if a file exists"""
        return Path(path).exists()
    
    async def migrate(self, migration_files: List[str]) -> bool:
        """Run database migrations from files.
        
        Args:
            migration_files: List of file paths containing SQL migration scripts
            
        Returns:
            bool: True if all migrations succeeded, False otherwise
        """
        if not self.connection:
            raise ConnectionError("Database not connected")
        
        try:
            for migration_file in migration_files:
                migration_path = Path(migration_file)
                if not migration_path.exists():
                    raise FileNotFoundError(f"Migration file not found: {migration_file}")
                
                # Read and execute migration script
                script_content = migration_path.read_text(encoding='utf-8')
                await self.execute_script(script_content)
            
            return True
        except Exception as e:
            print(f"Migration failed: {e}")
            return False
