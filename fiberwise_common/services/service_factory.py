"""
Service Factory and Dependency Injection for FiberWise.
Manages service instances and provides dependency injection capabilities.
"""

import logging
from typing import Any, Dict, Optional, Type
from pathlib import Path

from ..database import DatabaseManager
from .base_service import ServiceError
from .agent_service import AgentService
from .user_service import UserService

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    Service initialization manager.
    Handles the initialization state of the service system.
    """
    
    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize the service factory.
        
        Args:
            database_manager: Database manager instance
        """
        self.db_manager = database_manager
        self.db_provider = database_manager.get_provider()
        self._initialized = False
        self._service_registry = None

    async def initialize_services(self) -> None:
        """
        Initialize the service registry and register default services.
        """
        if self._initialized:
            return
        
        try:
            # Create service registry
            from .service_registry import ServiceRegistry
            self._service_registry = ServiceRegistry(self.db_provider)
            
            # Register core services in the new registry
            self._service_registry.register_service('db_provider', self.db_provider)
            self._service_registry.register_service('db_manager', self.db_manager)
            
            self._initialized = True
            logger.info("Service registry initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize service registry: {e}")
            raise ServiceError(f"Service initialization failed: {e}") from e

    def get_service(self, service_name: str) -> Any:
        """
        Get a service instance by name.
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            Service instance
            
        Raises:
            ServiceError: If services not initialized
        """
        if not self._initialized:
            raise ServiceError("Services not initialized. Call initialize_services() first.")
        
        return self._service_registry.get_service(service_name)

    def get_all_services(self) -> Dict[str, Any]:
        """
        Get all registered services.
        
        Returns:
            Dictionary of all services
        """
        if not self._initialized:
            return {}
        
        # Return a dict of available service names (not instances)
        return {name: name for name in self._service_registry.get_available_services()}

    async def shutdown_services(self) -> None:
        """
        Shutdown services and cleanup resources.
        """
        if not self._initialized:
            return
        
        try:
            # Close database connections
            await self.db_manager.close()
            
            # Clear service registry
            if self._service_registry:
                self._service_registry.clear_singletons()
            
            self._initialized = False
            
            logger.info("Services shutdown successfully")
            
        except Exception as e:
            logger.error(f"Error during service shutdown: {e}")


class ServiceContainer:
    """
    Service container for dependency injection in web applications.
    Provides a simple interface for services in request handlers.
    """
    
    def __init__(self, service_factory: ServiceFactory):
        """
        Initialize the service container.
        
        Args:
            service_factory: Factory instance
        """
        self.factory = service_factory

    @property
    def agents(self) -> AgentService:
        """Get the agent service."""
        return self.factory.get_service('agent_service')

    @property
    def activations(self) -> Any:
        """Get the activation service."""
        return self.factory.get_service('activation_service')

    @property
    def users(self) -> UserService:
        """Get the user service.""" 
        return self.factory.get_service('user_service')

    @property
    def db(self):
        """Get the database provider."""
        return self.factory.get_service('db_provider')


def create_service_factory(
    database_url: str,
    migrations_dir: Optional[Path] = None
) -> ServiceFactory:
    """
    Create a service factory with database initialization.
    
    Args:
        database_url: Database connection URL
        migrations_dir: Directory containing migrations
        
    Returns:
        Initialized service factory
    """
    # Use default migrations directory if not provided
    if migrations_dir is None:
        migrations_dir = Path(__file__).parent.parent / "database" / "migrations"
    
    db_manager = DatabaseManager(database_url, migrations_dir)
    return ServiceFactory(db_manager)


# Dependency injection helpers for FastAPI
def get_service_container() -> ServiceContainer:
    """
    Dependency function to get service container in FastAPI routes.
    This should be configured during app startup.
    """
    if not hasattr(get_service_container, '_container'):
        raise ServiceError("Service container not initialized")
    
    return get_service_container._container


def set_service_container(container: ServiceContainer) -> None:
    """
    Set the global service container instance.
    
    Args:
        container: Service container instance
    """
    get_service_container._container = container
