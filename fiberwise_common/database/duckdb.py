import os
import logging
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import duckdb
from fastapi.concurrency import run_in_threadpool

from .base import DatabaseProvider

logger = logging.getLogger(__name__)

class DuckDBProvider(DatabaseProvider):
    """DuckDB database provider."""

    @property
    def provider(self) -> str:
        return "duckdb"

    def __init__(self, database_url: Optional[str] = None):
        self._connection = None
        self._database_url = database_url or os.environ.get("DATABASE_URL", ":memory:")
        logger.info(f"Initializing DuckDBProvider with database: {self._database_url}")

    async def connect(self) -> None:
        """Connect to the database."""
        logger.info(f"Connecting to DuckDB database: {self._database_url}")
        try:
            self._connection = await run_in_threadpool(
                duckdb.connect, self._database_url, read_only=False
            )
            logger.info("Connected to DuckDB database")
        except Exception as e:
            logger.error(f"Error connecting to DuckDB database: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the database."""
        if self._connection:
            logger.info("Disconnecting from DuckDB database")
            await run_in_threadpool(self._connection.close)
            self._connection = None

    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Execute a query and fetch a single row."""
        if not self._connection:
            raise RuntimeError("Database not connected.")
        return await run_in_threadpool(self._sync_fetch_one, query, *args)

    async def fetch_all(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a query and fetch all rows."""
        if not self._connection:
            raise RuntimeError("Database not connected.")
        return await run_in_threadpool(self._sync_fetch_all, query, *args)

    async def fetch_val(self, query: str, *args) -> Any:
        """Execute a query and fetch a single value."""
        if not self._connection:
            raise RuntimeError("Database not connected.")
        return await run_in_threadpool(self._sync_fetch_val, query, *args)

    async def execute(self, query: str, *args) -> Optional[str]:
        if not self._connection:
            raise RuntimeError("Database not connected.")
        await run_in_threadpool(self._sync_execute, query, *args)
        return None

    @asynccontextmanager
    async def transaction(self):
        """Create a database transaction."""
        if not self._connection:
            raise RuntimeError("Database not connected.")
        
        # DuckDB doesn't have explicit transaction management in this way
        # We'll use a simple context manager to maintain API compatibility
        try:
            await run_in_threadpool(self._connection.begin)
            yield self
            await run_in_threadpool(self._connection.commit)
        except Exception as e:
            await run_in_threadpool(self._connection.rollback)
            raise e

    # Alias methods for compatibility
    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Alias for fetch_one."""
        return await self.fetch_one(query, *args)
    
    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Alias for fetch_all."""
        return await self.fetch_all(query, *args)
    
    async def fetchval(self, query: str, *args) -> Any:
        """Alias for fetch_val."""
        return await self.fetch_val(query, *args)
    
    async def file_exists(self, path: str) -> bool:
        """Check if a file exists (for storage providers)."""
        return os.path.exists(path)

    # Synchronous methods to be run in thread pool
    def _sync_fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Synchronously execute a query and fetch a single row."""
        try:
            cursor = self._connection.cursor()
            result = cursor.execute(query, args if args else None)
            columns = [desc[0] for desc in result.description]
            row = result.fetchone()
            if row:
                return dict(zip(columns, row))
            return None
        except Exception as e:
            logger.error(f"Error in _sync_fetch_one: {e}, query: {query}, args: {args}")
            raise

    def _sync_fetch_all(self, query: str, *args) -> List[Dict[str, Any]]:
        """Synchronously execute a query and fetch all rows."""
        try:
            cursor = self._connection.cursor()
            result = cursor.execute(query, args if args else None)
            columns = [desc[0] for desc in result.description]
            return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Error in _sync_fetch_all: {e}, query: {query}, args: {args}")
            raise

    def _sync_fetch_val(self, query: str, *args) -> Any:
        """Synchronously execute a query and fetch a single value."""
        try:
            cursor = self._connection.cursor()
            result = cursor.execute(query, args if args else None)
            row = result.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"Error in _sync_fetch_val: {e}, query: {query}, args: {args}")
            raise

    def _sync_execute(self, query: str, *args) -> None:
        """Synchronously execute a command without returning data."""
        try:
            cursor = self._connection.cursor()
            cursor.execute(query, args if args else None)
        except Exception as e:
            logger.error(f"Error in _sync_execute: {e}, query: {query}, args: {args}")
            raise
