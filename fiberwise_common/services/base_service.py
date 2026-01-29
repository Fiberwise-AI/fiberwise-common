"""
Base service class and utilities for FiberWise services.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class BaseService(ABC):
    """
    Base class for all FiberWise services.
    Provides common functionality like logging, error handling, and database access.
    """
    
    def __init__(self, db_provider, logger_name: Optional[str] = None):
        """
        Initialize the service with a database provider.
        
        Args:
            db_provider: Database provider instance
            logger_name: Custom logger name, defaults to class name
        """
        self.db = db_provider
        self.logger = logging.getLogger(logger_name or self.__class__.__name__)

    async def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a database query with error handling.

        Args:
            query: SQL query string with :param_name placeholders
            params: Dict of named parameters

        Returns:
            Query result

        Raises:
            ServiceError: If query execution fails
        """
        try:
            return await self.db.execute(query, params)
        except Exception as e:
            self.logger.error(f"Query execution failed: {query} - {e}")
            raise ServiceError(f"Database operation failed: {e}") from e

    async def _fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row from database.

        Args:
            query: SQL query string with :param_name placeholders
            params: Dict of named parameters

        Returns:
            Single row as dict or None
        """
        try:
            return await self.db.fetch_one(query, params)
        except Exception as e:
            self.logger.error(f"Fetch one failed: {query} - {e}")
            raise ServiceError(f"Database fetch failed: {e}") from e

    async def _fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Fetch all rows from database.

        Args:
            query: SQL query string with :param_name placeholders
            params: Dict of named parameters

        Returns:
            List of rows as dicts
        """
        try:
            return await self.db.fetch_all(query, params)
        except Exception as e:
            self.logger.error(f"Fetch all failed: {query} - {e}")
            raise ServiceError(f"Database fetch failed: {e}") from e

    async def _execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a database query (INSERT, UPDATE, DELETE).

        Args:
            query: SQL query string with :param_name placeholders
            params: Dict of named parameters

        Returns:
            Query result
        """
        try:
            return await self.db.execute(query, params)
        except Exception as e:
            self.logger.error(f"Execute failed: {query} - {e}")
            raise ServiceError(f"Database execute failed: {e}") from e

    async def _execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> Any:
        """
        Execute a query multiple times with different parameters.

        Args:
            query: SQL query string with :param_name placeholders
            params_list: List of parameter dicts

        Returns:
            Query result
        """
        try:
            return await self.db.execute_many(query, params_list)
        except Exception as e:
            self.logger.error(f"Execute many failed: {query} - {e}")
            raise ServiceError(f"Database execute many failed: {e}") from e


class ServiceError(Exception):
    """
    Base exception for service layer errors.
    """
    pass


class ValidationError(ServiceError):
    """
    Exception raised when input validation fails.
    """
    pass


class NotFoundError(ServiceError):
    """
    Exception raised when requested resource is not found.
    """
    pass


class AuthorizationError(ServiceError):
    """
    Exception raised when user lacks permission for operation.
    """
    pass


# Service registry for dependency injection
class ServiceRegistry:
    """
    Registry for service instances to enable dependency injection.
    """
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
    
    def register(self, name: str, instance: Any) -> None:
        """
        Register a service instance.
        
        Args:
            name: Service name
            instance: Service instance
        """
        self._services[name] = instance
        logger.debug(f"Registered service: {name}")
    
    def get(self, name: str) -> Any:
        """
        Get a service instance.
        
        Args:
            name: Service name
            
        Returns:
            Service instance
            
        Raises:
            ServiceError: If service not found
        """
        if name not in self._services:
            raise ServiceError(f"Service not found: {name}")
        return self._services[name]
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all registered services.
        
        Returns:
            Dict of all services
        """
        return self._services.copy()
    
    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()


# Global service registry instance
service_registry = ServiceRegistry()
