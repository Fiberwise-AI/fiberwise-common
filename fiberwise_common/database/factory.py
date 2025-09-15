from ..config import BaseWebSettings
from .base import DatabaseProvider
from .postgres import PostgresProvider
from .duckdb import DuckDBProvider
from .sqlite import SQLiteProvider

def get_database_provider(settings_instance: BaseWebSettings = None) -> DatabaseProvider:
    """
    Factory function to get the configured database provider.
    """
    if settings_instance is None:
        raise ValueError("Settings instance is required")
        
    provider_name = settings_instance.DB_PROVIDER.lower()
    if provider_name == "postgres":
        return PostgresProvider()
    elif provider_name == "duckdb":
        return DuckDBProvider()
    elif provider_name == "sqlite":
        provider = SQLiteProvider()
        # Set the database path from settings
        print(f"[DEBUG] Settings instance: {settings_instance}")
        print(f"[DEBUG] Has DATABASE_URL: {hasattr(settings_instance, 'DATABASE_URL')}")
        if hasattr(settings_instance, 'DATABASE_URL'):
            print(f"[DEBUG] DATABASE_URL: {settings_instance.DATABASE_URL}")
            
        if hasattr(settings_instance, 'DATABASE_URL') and settings_instance.DATABASE_URL.startswith('sqlite:///'):
            db_path = settings_instance.DATABASE_URL.replace('sqlite:///', '')
            print(f"[DEBUG] Setting database path: {db_path}")
            provider.set_db_path(db_path)
        return provider
    else:
        raise ValueError(f"Unsupported database provider: {provider_name}")
