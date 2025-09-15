"""
Local service for handling local database operations in FiberWise SDK.
Provides a unified interface for local data access without REST API calls.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from ..database.providers import DatabaseProvider
from ..database.query_adapter import create_query_adapter, QueryAdapter
from .base_service import BaseService
from .agent_service import AgentService
from .user_service import UserService
from .pipeline_service import PipelineService

logger = logging.getLogger(__name__)


class LocalService(BaseService):
    """
    Service for handling local database operations.
    Provides unified access to local data without REST API dependencies.
    """
    
    def __init__(self, database_provider: DatabaseProvider):
        """
        Initialize LocalService with a database provider.
        
        Args:
            database_provider: Database provider instance to use for operations
        """
        super().__init__(database_provider)
        self.db_provider = database_provider
        
        # Create query adapter for database compatibility
        provider_type = self._get_provider_type(database_provider)
        self.query_adapter = create_query_adapter(provider_type)
        
        # Initialize services
        self._agent_service = None
        self._user_service = None
        self._pipeline_service = None
        
        logger.info(f"LocalService initialized with database provider: {type(database_provider).__name__}")
        logger.info(f"Query adapter created for provider type: {provider_type}")
    
    def _get_provider_type(self, provider: DatabaseProvider) -> str:
        """
        Determine the database provider type from the provider instance.
        
        Args:
            provider: Database provider instance
            
        Returns:
            Provider type string (sqlite, postgresql, mysql, etc.)
        """
        provider_class_name = type(provider).__name__.lower()
        
        if 'sqlite' in provider_class_name:
            return 'sqlite'
        elif 'postgres' in provider_class_name or 'pg' in provider_class_name:
            return 'postgresql'
        elif 'mysql' in provider_class_name:
            return 'mysql'
        elif 'mssql' in provider_class_name or 'sqlserver' in provider_class_name:
            return 'mssql'
        else:
            # Default to postgresql for unknown providers
            logger.warning(f"Unknown provider type {provider_class_name}, defaulting to postgresql")
            return 'postgresql'
    
    async def initialize(self) -> bool:
        """Initialize the local service and database connection."""
        try:
            # Database provider should already be initialized by the caller
            is_healthy = await self.db_provider.is_healthy()
            if is_healthy:
                logger.info("LocalService database connection verified")
                return True
            else:
                logger.error("LocalService database connection is not healthy")
                return False
        except Exception as e:
            logger.error(f"LocalService initialization error: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the local service and database connections."""
        try:
            # Don't close the database provider as it may be shared
            # The caller is responsible for managing the provider lifecycle
            logger.info("LocalService shutdown complete")
        except Exception as e:
            logger.error(f"Error during LocalService shutdown: {e}")
    
    @property
    def agent_service(self) -> AgentService:
        """Get the agent service instance."""
        if self._agent_service is None:
            self._agent_service = AgentService(self.db_provider)
        return self._agent_service
    
    @property
    def user_service(self) -> UserService:
        """Get the user service instance."""
        if self._user_service is None:
            self._user_service = UserService(self.db_provider)
        return self._user_service
    
    @property
    def pipeline_service(self) -> PipelineService:
        """Get the pipeline service instance."""
        if self._pipeline_service is None:
            self._pipeline_service = PipelineService(self.db_provider)
        return self._pipeline_service
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the local service."""
        try:
            db_healthy = await self.db_provider.is_healthy()
            return {
                "status": "healthy" if db_healthy else "unhealthy",
                "database": "connected" if db_healthy else "disconnected",
                "services": {
                    "agent_service": "available",
                    "user_service": "available"
                }
            }
        except Exception as e:
            logger.error(f"LocalService health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def execute_query(self, query: str, params: Any = None, 
                          source_style: str = 'postgresql') -> Any:
        """
        Execute a query with automatic parameter style conversion.
        
        Args:
            query: SQL query string
            params: Query parameters
            source_style: Source parameter style ('postgresql', 'sqlite', 'mysql')
            
        Returns:
            Query result
        """
        from ..database.query_adapter import ParameterStyle
        
        # Convert parameter style name to enum
        style_map = {
            'postgresql': ParameterStyle.POSTGRESQL,
            'sqlite': ParameterStyle.SQLITE,
            'mysql': ParameterStyle.MYSQL,
            'mssql': ParameterStyle.MSSQL,
            'named': ParameterStyle.NAMED
        }
        
        source_enum = style_map.get(source_style.lower(), ParameterStyle.POSTGRESQL)
        converted_query, converted_params = self.query_adapter.adapt_query_and_params(
            query, params, source_enum
        )
        
        logger.debug(f"Query adapted from {source_style}: {converted_query}")
        logger.debug(f"Parameters adapted: {converted_params}")
        
        return await self.db_provider.execute(converted_query, converted_params)
    
    # Agent operations
    async def get_agents(self, **kwargs) -> List[Dict[str, Any]]:
        """Get agents using local database."""
        return await self.agent_service.get_agents(**kwargs)
    
    async def get_agent_by_id(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get specific agent by ID."""
        return await self.agent_service.get_agent_by_id(agent_id)
    
    async def create_agent(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new agent."""
        return await self.agent_service.create_agent(agent_data)
    
    async def update_agent(self, agent_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update existing agent."""
        return await self.agent_service.update_agent(agent_id, update_data)
    
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete agent."""
        try:
            await self.agent_service.delete_agent(agent_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting agent {agent_id}: {e}")
            return False
    
    async def create_agent_activation(self, activation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new agent activation."""
        return await self.agent_service.create_activation(activation_data)

    async def get_agent_by_filepath(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get an agent by its file path."""
        return await self.agent_service.get_agent_by_filepath(file_path)

    # Pipeline operations
    async def get_pipeline_by_filepath(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get a pipeline by its file path."""
        return await self.pipeline_service.get_pipeline_by_filepath(file_path)

    async def create_pipeline_execution(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new pipeline execution record."""
        return await self.pipeline_service.create_execution(execution_data)
    
    # App operations
    # App service methods - disabled since AppService was removed
    async def get_apps(self, **kwargs) -> List[Dict[str, Any]]:
        """Get apps using local database."""
        raise NotImplementedError("AppService functionality has been moved to web layer")
    
    async def get_app_by_id(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Get specific app by ID."""
        raise NotImplementedError("AppService functionality has been moved to web layer")
    
    # Generic request handler for SDK compatibility
    async def request(self, method: str, path: str, **kwargs) -> Any:
        """
        Handle generic requests by routing to appropriate service methods.
        This provides compatibility with the SDK's client interface.
        """
        method = method.upper()
        path = path.strip('/')
        
        # Parse path to determine service and operation
        parts = path.split('/')
        
        if len(parts) >= 2 and parts[0] == 'apps':
            app_id = parts[1]
            
            if len(parts) >= 3 and parts[2] == 'agents':
                # Agent operations
                if len(parts) == 3:
                    if method == 'GET':
                        # List agents for app
                        params = kwargs.get('params', {})
                        return await self.get_agents(app_id=app_id, **params)
                    elif method == 'POST':
                        # Create agent
                        agent_data = kwargs.get('json', {})
                        agent_data['app_id'] = app_id
                        return await self.create_agent(agent_data)
                
                elif len(parts) == 4:
                    agent_id = parts[3]
                    if method == 'GET':
                        # Get specific agent
                        return await self.get_agent_by_id(agent_id)
                    elif method == 'PUT':
                        # Update agent
                        update_data = kwargs.get('json', {})
                        return await self.update_agent(agent_id, update_data)
                    elif method == 'DELETE':
                        # Delete agent
                        await self.delete_agent(agent_id)
                        return {"message": "Agent deleted successfully"}
        
        # Default response for unhandled paths
        raise NotImplementedError(f"LocalService does not support {method} {path}")
    
    async def handle_request(self, method: str, path: str, **kwargs) -> Any:
        """
        Handle request - alias for request method to maintain compatibility.
        This is used by LocalClient.
        """
        return await self.request(method, path, **kwargs)
