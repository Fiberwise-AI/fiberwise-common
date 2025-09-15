from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union
from .query_adapter import QueryAdapter, ParameterStyle, create_query_adapter
from .security import SecurityLevel, create_secure_query_adapter

class DatabaseProvider(ABC):
    """Abstract Base Class for a database provider."""

    def __init__(self, security_level: SecurityLevel = SecurityLevel.MODERATE):
        """Initialize database provider with secure query adapter."""
        self._query_adapter = None
        self._security_level = security_level
    
    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider type."""
        pass
    
    @property
    def query_adapter(self) -> QueryAdapter:
        """Get or create secure query adapter for this provider."""
        if self._query_adapter is None:
            self._query_adapter = create_secure_query_adapter(self.provider, self._security_level)
        return self._query_adapter
    
    def adapt_query(self, query: str, params: Any = None, 
                   source_style: ParameterStyle = ParameterStyle.POSTGRESQL) -> tuple:
        """
        Adapt query and parameters for this database provider.
        
        Args:
            query: SQL query string
            params: Query parameters
            source_style: Source parameter style (default: PostgreSQL)
            
        Returns:
            Tuple of (adapted_query, adapted_params)
        """
        return self.query_adapter.adapt_query_and_params(query, params, source_style)

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the database."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the database."""
        pass

    @abstractmethod
    @asynccontextmanager
    async def transaction(self):
        """Create a database transaction."""
        yield None

    @abstractmethod
    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Execute a query and fetch a single row."""
        pass

    @abstractmethod
    async def fetch_all(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a query and fetch all rows."""
        pass

    @abstractmethod
    async def fetch_val(self, query: str, *args) -> Any:
        """Execute a query and fetch a single value."""
        pass

    @abstractmethod
    async def execute(self, query: str, *args) -> Optional[str]:
        """Execute a command (e.g., INSERT, UPDATE) without returning data."""
        pass
    
    @abstractmethod
    async def file_exists(self, path: str) -> bool:
        """Check if a file exists (for storage providers)."""
        pass

    @abstractmethod
    async def migrate(self, migration_files: List[str]) -> bool:
        """Run database migrations from files.
        
        Args:
            migration_files: List of file paths containing SQL migration scripts
            
        Returns:
            bool: True if all migrations succeeded, False otherwise
        """
        pass
