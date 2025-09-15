"""
Database initialization and management utilities.
"""

import logging
from pathlib import Path
from typing import Optional

from .providers import DatabaseProvider, create_database_provider
from .migrations import MigrationManager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database initialization, migrations, and lifecycle."""
    
    def __init__(self, database_url: str, migrations_dir: Optional[Path] = None):
        self.database_url = database_url
        self.provider = create_database_provider(database_url)
        self.migrations_dir = migrations_dir
        self.migration_manager = MigrationManager(self.provider, migrations_dir)
    
    @classmethod
    def create_from_settings(cls, settings, migrations_dir: Optional[Path] = None):
        """Create DatabaseManager from settings object.
        
        Args:
            settings: Settings object with DATABASE_URL attribute
            migrations_dir: Optional custom migrations directory
            
        Returns:
            DatabaseManager instance
        """
        if not hasattr(settings, 'DATABASE_URL'):
            raise ValueError("Settings must have DATABASE_URL attribute")
        
        return cls(settings.DATABASE_URL, migrations_dir)
    
    async def initialize(self) -> bool:
        """Initialize the database connection."""
        try:
            success = await self.provider.connect()
            if success:
                logger.info("Database initialized successfully")
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
        """Apply database migrations using MigrationManager."""
        try:
            await self.migration_manager.apply_migrations()
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
    
    def get_provider(self) -> DatabaseProvider:
        """Get the database provider instance."""
        return self.provider
