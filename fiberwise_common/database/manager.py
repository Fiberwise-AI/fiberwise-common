"""
Database initialization and management for FiberWise applications.

Uses NexusQL as the underlying database engine. Handles connection lifecycle,
schema initialization, and migration tracking via NexusQL's MigrationRunner.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from nexusql import DatabaseManager as NexusDB
from nexusql.migrations import MigrationRunner

from .provider import NexusQLProvider, create_database_provider

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database initialization, migrations, and lifecycle."""

    def __init__(self, database_url: str, migrations_dir: Optional[Path] = None):
        self.database_url = database_url
        self.provider = create_database_provider(database_url)
        self.migrations_dir = migrations_dir or self._get_default_migrations_dir()

    @classmethod
    def create_from_settings(cls, settings, migrations_dir: Optional[Path] = None):
        """Create DatabaseManager from settings object."""
        if not hasattr(settings, 'DATABASE_URL'):
            raise ValueError("Settings must have DATABASE_URL attribute")
        return cls(settings.DATABASE_URL, migrations_dir)

    def _get_default_migrations_dir(self) -> Path:
        """Get the default migrations directory (the sql/ directory in this package)."""
        return Path(__file__).parent / "sql"

    async def initialize(self) -> bool:
        """Initialize the database connection."""
        try:
            success = await self.provider.connect()
            if success:
                logger.info("Database initialized successfully via NexusQL")
                return True
            else:
                logger.error("Failed to initialize database")
                return False
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the database connection."""
        try:
            await self.provider.disconnect()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

    async def apply_migrations(self) -> bool:
        """Apply database migrations using NexusQL MigrationRunner.

        Creates a dedicated nexusql DatabaseManager on the current thread
        so that SQLite thread-affinity is satisfied (the NexusQLProvider's
        inner connection lives on a separate executor thread).
        """
        loop = asyncio.get_event_loop()

        def _run_migrations():
            db = NexusDB(self.database_url)
            db.connect()
            try:
                runner = MigrationRunner(
                    db,
                    migration_path=self.migrations_dir,
                    migration_type="system",
                )
                # MigrationRunner methods are async def but internally synchronous,
                # so we run the coroutine in a new event loop on this thread.
                inner_loop = asyncio.new_event_loop()
                try:
                    return inner_loop.run_until_complete(runner.run_pending_migrations())
                finally:
                    inner_loop.close()
            finally:
                db.disconnect()

        try:
            result = await loop.run_in_executor(
                self.provider._executor, _run_migrations
            )
            if not result:
                logger.error("Database migrations failed")
                return False

            logger.info("All migrations applied successfully")
            return True

        except Exception as e:
            logger.error(f"Migration process failed: {e}")
            return False

    async def health_check(self) -> bool:
        """Perform a health check on the database connection."""
        try:
            return await self.provider.is_healthy()
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def get_provider(self) -> NexusQLProvider:
        """Get the database provider instance."""
        return self.provider
