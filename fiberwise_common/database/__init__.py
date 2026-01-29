"""
Database components for FiberWise applications.

Backed by NexusQL for multi-database support (PostgreSQL, SQLite, MySQL, MSSQL).
"""

from .provider import NexusQLProvider, DatabaseProvider, create_database_provider
from .manager import DatabaseManager

__all__ = [
    'NexusQLProvider',
    'DatabaseProvider',
    'create_database_provider',
    'DatabaseManager',
]
