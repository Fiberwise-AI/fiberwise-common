"""
NexusQL-based database provider for FiberWise applications.

Wraps nexusql.DatabaseManager with an async interface. All queries use
NexusQL's native :param_name style with dict parameters.
"""

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from nexusql import DatabaseManager as NexusDB

logger = logging.getLogger(__name__)


class NexusQLProvider:
    """
    Database provider backed by NexusQL.

    Supports PostgreSQL, SQLite, MySQL, and MSSQL through a single interface.
    All methods are async and internally delegate to NexusQL's sync engine
    via run_in_executor to avoid blocking the event loop.
    """

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._db = NexusDB(database_url)
        self._connected = False
        self._lock = asyncio.Lock()
        # Single-thread executor ensures all DB calls run on the same thread,
        # which is required by SQLite (connection affinity).
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def provider(self) -> str:
        """Return the provider type string."""
        url = self.database_url.lower()
        if url.startswith('postgresql') or url.startswith('postgres'):
            return "postgresql"
        elif url.startswith('mysql'):
            return "mysql"
        elif url.startswith('mssql'):
            return "mssql"
        return "sqlite"

    async def connect(self) -> bool:
        """Establish database connection."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self._executor, self._db.connect)
            self._connected = bool(result)
            if self._connected:
                logger.info(f"Connected to {self.provider} database via NexusQL")
            return self._connected
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Close database connection."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._db.disconnect)
            self._connected = False
            logger.info("Disconnected from database")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")

    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a query (INSERT, UPDATE, DELETE, CREATE, etc.)."""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                self._executor, self._db.execute, query, params
            )
        except Exception as e:
            logger.error(f"Execute failed: {e}")
            raise

    async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Execute query and return single row as dict."""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self._executor, self._db.fetch_one, query, params
            )
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Fetch one failed: {e}")
            raise

    async def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute query and return all rows as list of dicts."""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self._executor, self._db.fetch_all, query, params
            )
            return [dict(row) for row in result] if result else []
        except Exception as e:
            logger.error(f"Fetch all failed: {e}")
            raise

    async def fetch_val(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute query and return single value from first row."""
        row = await self.fetch_one(query, params)
        if row:
            return next(iter(row.values()))
        return None

    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor, self._db.execute, "BEGIN TRANSACTION", None
        )
        try:
            yield self
            await loop.run_in_executor(
                self._executor, self._db.execute, "COMMIT", None
            )
        except Exception:
            await loop.run_in_executor(
                self._executor, self._db.execute, "ROLLBACK", None
            )
            raise

    async def execute_script(self, script: str) -> Any:
        """
        Execute a multi-statement SQL script.

        Uses NexusQL's SQL translation to automatically adapt the script
        for the target database (PostgreSQL, SQLite, MySQL, MSSQL).
        """
        loop = asyncio.get_event_loop()

        def _run():
            translated = self._db._translate_sql(script)
            if self._db.config.database_type.value == "sqlite":
                self._db._connection.executescript(translated)
                self._db._connection.commit()
            else:
                statements = self._db._split_sql_statements(translated)
                for stmt in statements:
                    # Skip empty/comment-only statements
                    test = re.sub(r'--[^\n]*', '', stmt)
                    test = re.sub(r'/\*.*?\*/', '', test, flags=re.DOTALL)
                    if test.strip():
                        self._db.execute(stmt)

        await loop.run_in_executor(self._executor, _run)

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists (sync)."""
        return self._db.table_exists(table_name)

    async def is_healthy(self) -> bool:
        """Check if the database connection is healthy."""
        try:
            await self.fetch_one("SELECT 1")
            return True
        except Exception:
            return False

    async def file_exists(self, path: str) -> bool:
        """Check if a file exists (for storage providers)."""
        import os
        return os.path.exists(path)

    async def migrate(self, migration_files: List[str]) -> bool:
        """Run database migrations from SQL files."""
        try:
            for migration_file in migration_files:
                with open(migration_file, 'r') as f:
                    sql = f.read()
                await self.execute_script(sql)
            return True
        except Exception as e:
            logger.error(f"Migration error: {e}")
            return False

    async def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> Any:
        """Execute a query multiple times with different parameter dicts."""
        for params in params_list:
            await self.execute(query, params)


# Backward-compatible type alias so existing code importing DatabaseProvider keeps working
DatabaseProvider = NexusQLProvider


def create_database_provider(database_url: str) -> NexusQLProvider:
    """Factory function to create a NexusQL-backed database provider."""
    return NexusQLProvider(database_url)
