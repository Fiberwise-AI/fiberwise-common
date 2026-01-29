"""
Database provider factory - backward compatibility shim.

Uses NexusQL under the hood. The DATABASE_URL determines the database type
automatically (no need for a separate DB_PROVIDER setting).
"""

from .provider import DatabaseProvider, create_database_provider


def get_database_provider(settings_instance=None) -> DatabaseProvider:
    """
    Factory function to get the configured database provider.

    With NexusQL, the database type is determined from the DATABASE_URL.
    """
    if settings_instance is None:
        raise ValueError("Settings instance is required")

    if not hasattr(settings_instance, 'DATABASE_URL'):
        raise ValueError("Settings must have DATABASE_URL attribute")

    return create_database_provider(settings_instance.DATABASE_URL)
