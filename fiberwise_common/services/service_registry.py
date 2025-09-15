"""
Service Registry for Dependency Injection
Central registry for service dependency injection in FiberWise agents.
"""

import logging
from typing import Dict, Type, Any, Optional, Callable, List
from abc import ABC, abstractmethod

from .base_service import BaseService

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """
    Central registry for service dependency injection.
    
    This registry manages service instances and factories for dependency injection
    into FiberAgent instances and other components that require services.
    """
    
    def __init__(self, db_provider):
        self.db_provider = db_provider
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._lazy_factories: Dict[str, Callable] = {}  # New: Store lazy factories separately
        self._setup_default_services()
    
    def register_service(self, service_name: str, service_instance: Any = None, 
                        factory: Callable = None, singleton: bool = True) -> None:
        """
        Register a service instance or factory.
        
        Args:
            service_name: Service name for dependency injection
            service_instance: Pre-created service instance
            factory: Factory function that creates service instances
            singleton: Whether to cache instances (default: True)
        """
        if service_instance is not None:
            self._services[service_name] = service_instance
            if singleton:
                self._singletons[service_name] = service_instance
        elif factory is not None:
            self._factories[service_name] = factory
        else:
            raise ValueError("Must provide either service_instance or factory")
        
        logger.debug(f"Registered service: {service_name}")
    
    def register_factory(self, service_name: str, factory: Callable, lazy: bool = False):
        """
        Register a factory for a service, optionally as lazy.
        
        Args:
            service_name: Service name for dependency injection
            factory: Factory function that creates service instances
            lazy: Whether to create service only on first access (default: False)
        """
        if lazy:
            self._lazy_factories[service_name] = factory
        else:
            self._factories[service_name] = factory
        
        logger.debug(f"Registered {'lazy ' if lazy else ''}factory for service: {service_name}")
    
    def register_service_class(self, service_name: str, service_class: Type[BaseService], 
                             singleton: bool = True) -> None:
        """
        Register a service class that will be instantiated with db_provider.
        
        Args:
            service_name: Service name for dependency injection
            service_class: Service class that extends BaseService
            singleton: Whether to cache instances (default: True)
        """
        def factory(db_provider):
            return service_class(db_provider)
        
        self.register_service(service_name, factory=factory, singleton=singleton)
    
    def get_service(self, service_name: str) -> Any:
        """
        Get a service instance by name. Handles lazy creation.
        
        Args:
            service_name: Service name
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service is not registered
        """
        # Check if we have a direct instance
        if service_name in self._services:
            return self._services[service_name]
        
        # Check if we have a cached singleton
        if service_name in self._singletons:
            return self._singletons[service_name]
        
        # Check and resolve lazy factories (New Logic)
        if service_name in self._lazy_factories:
            factory = self._lazy_factories.pop(service_name)  # Remove to prevent re-creation
            # Create, cache, and return the service
            service = self._create_from_factory(service_name, factory)
            self._singletons[service_name] = service
            return service
        
        # Create from regular factory
        if service_name in self._factories:
            factory = self._factories[service_name]
            service = self._create_from_factory(service_name, factory)
            # Cache as singleton if needed
            if service_name not in self._services:  # Only cache if not explicitly registered as instance
                self._singletons[service_name] = service
            
            logger.debug(f"Created service instance: {service_name}")
            return service
        
        raise KeyError(f"Service '{service_name}' not found in registry")
    
    def _create_from_factory(self, service_name: str, factory: Callable) -> Any:
        """
        Create a service instance from a factory function.
        
        Args:
            service_name: Service name
            factory: Factory function
            
        Returns:
            Service instance
            
        Raises:
            Exception: If factory fails to create service
        """
        try:
            # Try to call factory with db_provider (for BaseService subclasses)
            return factory(self.db_provider)
        except TypeError:
            # Fall back to calling factory without arguments
            return factory()
    
    def is_registered(self, service_name: str) -> bool:
        """Check if a service, factory, or singleton is registered with the container."""
        return (service_name in self._services or 
                service_name in self._factories or 
                service_name in self._singletons or
                service_name in self._lazy_factories)
    
    def inject_dependencies(self, target_instance, dependencies: List[str]) -> Dict[str, Any]:
        """
        Inject requested dependencies into a target instance.
        
        Args:
            target_instance: The object that needs dependencies injected
            dependencies: List of service names to inject
            
        Returns:
            Dict mapping service names to service instances
        """
        injected = {}
        
        for dep_name in dependencies:
            try:
                service = self.get_service(dep_name)
                injected[dep_name] = service
                logger.debug(f"Injected dependency '{dep_name}' into {type(target_instance).__name__}")
            except KeyError:
                logger.warning(f"Dependency '{dep_name}' not available for injection into {type(target_instance).__name__}")
                # Don't fail - just log the warning and continue
        
        return injected
    
    def get_available_services(self) -> List[str]:
        """Get list of all available service names."""
        all_services = set()
        all_services.update(self._services.keys())
        all_services.update(self._factories.keys())
        all_services.update(self._singletons.keys())
        all_services.update(self._lazy_factories.keys())
        return sorted(list(all_services))
    
    def clear_singletons(self) -> None:
        """Clear all cached singleton instances."""
        self._singletons.clear()
        logger.info("Cleared all singleton service instances")
    
    def unregister_service(self, service_name: str) -> bool:
        """
        Unregister a service.
        
        Args:
            service_name: Service name to unregister
            
        Returns:
            True if service was found and removed
        """
        removed = False
        
        if service_name in self._services:
            del self._services[service_name]
            removed = True
        
        if service_name in self._factories:
            del self._factories[service_name]
            removed = True
            
        if service_name in self._singletons:
            del self._singletons[service_name]
            removed = True
            
        if service_name in self._lazy_factories:
            del self._lazy_factories[service_name]
            removed = True
        
        if removed:
            logger.info(f"Unregistered service: {service_name}")
        
        return removed
    
    def _setup_default_services(self):
        """
        Skip default service registration at startup.
        Services will be created at runtime with proper user/app context.
        """
        logger.info("Skipping default service registration - services will be created at runtime with proper context")


class Injectable(ABC):
    """
    Base class for objects that support dependency injection.
    
    Classes that extend this can declare their dependencies and have them
    automatically injected by the ServiceRegistry.
    """
    
    def __init__(self):
        self._injected_services: Dict[str, Any] = {}
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """
        Return list of service names this object depends on.
        
        Returns:
            List of service dependency names
        """
        return []
    
    def inject_services(self, services: Dict[str, Any]) -> None:
        """
        Inject services into this object.
        
        Args:
            services: Dict mapping service names to service instances
        """
        self._injected_services.update(services)
        
        # Set services as attributes for easy access
        for name, service in services.items():
            setattr(self, name, service)
    
    def get_service(self, service_name: str) -> Any:
        """
        Get an injected service by name.
        
        Args:
            service_name: Name of the service to get
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service was not injected
        """
        if service_name in self._injected_services:
            return self._injected_services[service_name]
        
        raise KeyError(f"Service '{service_name}' was not injected into {type(self).__name__}")
    
    def was_injected(self, service_name: str) -> bool:
        """Check if a service was successfully injected into this instance."""
        return service_name in self._injected_services
    
    def get_injected_services(self) -> Dict[str, Any]:
        """Get all injected services."""
        return self._injected_services.copy()


def create_default_registry(db_provider) -> ServiceRegistry:
    """
    Create a ServiceRegistry with default services registered.
    
    Args:
        db_provider: Database provider instance
        
    Returns:
        ServiceRegistry with default services
    """
    return ServiceRegistry(db_provider)


# Global registry instance - can be set by applications
_global_registry: Optional[ServiceRegistry] = None


def set_global_registry(registry: ServiceRegistry) -> None:
    """Set the global service registry instance."""
    global _global_registry
    _global_registry = registry
    logger.info("Set global service registry")


def get_global_registry() -> Optional[ServiceRegistry]:
    """Get the global service registry instance."""
    return _global_registry


def inject_services(target_instance, registry: Optional[ServiceRegistry] = None) -> Dict[str, Any]:
    """
    Convenience function to inject services into an Injectable instance.
    
    Args:
        target_instance: Object that implements Injectable interface
        registry: Optional registry to use (defaults to global registry)
        
    Returns:
        Dict of injected services
    """
    if not isinstance(target_instance, Injectable):
        raise TypeError(f"Target instance must implement Injectable interface")
    
    service_registry = registry or get_global_registry()
    if not service_registry:
        raise RuntimeError("No service registry available for injection")
    
    dependencies = target_instance.get_dependencies()
    injected_services = service_registry.inject_dependencies(target_instance, dependencies)
    target_instance.inject_services(injected_services)
    
    return injected_services