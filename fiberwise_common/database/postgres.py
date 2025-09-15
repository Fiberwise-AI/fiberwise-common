import os
import logging
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import asyncpg

from .base import DatabaseProvider

logger = logging.getLogger(__name__)

class PostgresProvider(DatabaseProvider):
    """PostgreSQL database provider."""

    @property
    def provider(self) -> str:
        return "postgres"

    def __init__(self, database_url: Optional[str] = None):
        self._connection_pool = None
        self._connection = None
        self._database_url = database_url or os.environ.get("DATABASE_URL")
        logger.info(f"Initializing PostgresProvider with database: {self._database_url}")

    async def connect(self) -> None:
        """Connect to the database."""
        logger.info("Connecting to PostgreSQL database")
        try:
            self._connection_pool = await asyncpg.create_pool(
                dsn=self._database_url,
                min_size=1,
                max_size=10
            )
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL database: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the database."""
        if self._connection_pool:
            logger.info("Disconnecting from PostgreSQL database")
            await self._connection_pool.close()
            self._connection_pool = None

    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Execute a query and fetch a single row."""
        if not self._connection_pool:
            raise RuntimeError("Database not connected.")
        async with self._connection_pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def fetch_all(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a query and fetch all rows."""
        if not self._connection_pool:
            raise RuntimeError("Database not connected.")
        async with self._connection_pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def fetch_val(self, query: str, *args) -> Any:
        """Execute a query and fetch a single value."""
        if not self._connection_pool:
            raise RuntimeError("Database not connected.")
        async with self._connection_pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def execute(self, query: str, *args) -> Optional[str]:
        """Execute a command without returning data."""
        if not self._connection_pool:
            raise RuntimeError("Database not connected.")
        async with self._connection_pool.acquire() as conn:
            return await conn.execute(query, *args)

    @asynccontextmanager
    async def transaction(self):
        """Create a database transaction."""
        if not self._connection_pool:
            raise RuntimeError("Database not connected.")
        
        async with self._connection_pool.acquire() as connection:
            transaction = connection.transaction()
            try:
                await transaction.start()
                yield connection
                await transaction.commit()
            except:
                await transaction.rollback()
                raise


    async def file_exists(self, path: str) -> bool:
        """Check if a file exists (for storage providers)."""
        return os.path.exists(path)
