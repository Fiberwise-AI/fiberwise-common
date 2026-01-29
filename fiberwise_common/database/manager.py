"""
Database initialization and management for FiberWise applications.

Uses NexusQL as the underlying database engine. Handles connection lifecycle,
schema initialization, and migration tracking.
"""

import logging
from pathlib import Path
from typing import Optional

from .provider import NexusQLProvider, create_database_provider
from .sql_loader import load_sql_script

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
        """Apply database migrations (schema init + numbered migrations)."""
        try:
            # Create migration tracking table
            await self.provider.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Get already-applied migrations
            applied = []
            try:
                rows = await self.provider.fetch_all(
                    "SELECT version FROM schema_migrations ORDER BY version"
                )
                applied = [r['version'] for r in rows]
            except Exception:
                pass

            # Apply initial schema if not yet applied
            if 'init' not in applied:
                logger.info("Applying initial schema...")
                try:
                    schema = load_sql_script("init.sql")
                    await self.provider.execute_script(schema)
                    await self._mark_applied('init')
                    logger.info("Initial schema applied successfully")
                except FileNotFoundError:
                    logger.warning("init.sql not found in package resources")
                except Exception as e:
                    logger.error(f"Failed to apply initial schema: {e}")
                    raise
                # Refresh applied list
                rows = await self.provider.fetch_all(
                    "SELECT version FROM schema_migrations ORDER BY version"
                )
                applied = [r['version'] for r in rows]

            # Apply numbered migration files from the sql/ directory
            if self.migrations_dir.exists():
                migration_files = sorted(self.migrations_dir.glob("*.sql"))
                for migration_file in migration_files:
                    version = migration_file.stem
                    if version == "init":
                        continue
                    if version not in applied:
                        logger.info(f"Applying migration: {version}")
                        try:
                            sql = migration_file.read_text(encoding='utf-8')
                            await self.provider.execute_script(sql)
                            await self._mark_applied(version)
                            logger.info(f"Applied migration: {version}")
                        except Exception as e:
                            logger.error(f"Migration {version} failed: {e}")
                            raise

            logger.info("All migrations applied successfully")
            return True

        except Exception as e:
            logger.error(f"Migration process failed: {e}")
            return False

    async def _mark_applied(self, version: str):
        """Record a migration as applied."""
        existing = await self.provider.fetch_one(
            "SELECT version FROM schema_migrations WHERE version = :version",
            {"version": version},
        )
        if not existing:
            await self.provider.execute(
                "INSERT INTO schema_migrations (version) VALUES (:version)",
                {"version": version},
            )

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
