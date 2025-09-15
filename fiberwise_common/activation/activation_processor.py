"""
Common activation processing logic for FiberWise agents.

This module contains the core ActivationProcessor that handles agent execution
in both CLI (instant) and web (worker-based) contexts.

Enhanced with per-activation LLM service creation for clean provider isolation.
"""

import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Union, Callable

from fiberwise_common.database.base import DatabaseProvider
from ..services.service_registry import ServiceRegistry
from ..database.query_adapter import create_query_adapter

logger = logging.getLogger(__name__)


class ActivationProcessor:
    """
    Common activation processor that handles agent execution.
    
    This processor can be used in different contexts:
    - CLI: For instant execution after queueing
    - Web: For worker-based processing of queued activations
    
    Enhanced with per-activation LLM service creation.
    """
    
    def __init__(self, db_provider: DatabaseProvider, context: str = "unknown", base_url: Optional[str] = None):
        """
        Initialize the activation processor.
        
        Args:
            db_provider: Database provider instance
            context: Context string ("cli" or "web") for logging
            base_url: Base URL for activation service (defaults to environment variable or localhost:5555)
        """
        self.db = db_provider
        self.context = context
        
        provider_type = self._get_provider_type(db_provider)
        self.query_adapter = create_query_adapter(provider_type)
        self._injected_services = {}
        self._notification_callback: Optional[Callable] = None
        
        # Set base URL - use FIBER_API_BASE_URL (includes /api/v1) or fallback to BASE_URL + /api/v1
        fiber_api_url = os.getenv('FIBER_API_BASE_URL')
        if fiber_api_url:
            self.base_url = fiber_api_url
        else:
            raw_base_url = base_url or os.getenv('BASE_URL', 'http://localhost:5555')
            self.base_url = raw_base_url.rstrip('/') + '/api/v1'
        
        # Initialize activation service for dependency injection
        # This will handle creating context-specific services at activation time
        self.activation_service = None  # Will be created lazily per activation
        
        # Create a unique instance ID for tracking
        import uuid
        self.instance_id = str(uuid.uuid4())[:8]
        logger.info(f"🔧 ACTIVATION PROCESSOR [{self.instance_id}] - Created with context: {context}, base_url: {self.base_url}")
        
    def _get_provider_type(self, provider: DatabaseProvider) -> str:
        """
        Determine the database provider type from the provider instance.
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
        
    def inject_services(self, **services):
        """
        Inject services that will be available to agents.
        
        Args:
            **services: Named services to inject (llm_service, storage_service, etc.)
        """
        self._injected_services.update(services)
        logger.info(f"[{self.context}] Injected services: {list(services.keys())}")
    
    async def _create_activation_service(self, activation_context: Dict[str, Any], activation: Dict[str, Any] = None) -> 'ServiceRegistry':
        """
        Create a ServiceRegistry instance for this specific activation.
        
        This allows for context-specific service creation at activation time
        rather than global service creation at startup.
        
        Args:
            activation_context: The activation context containing app_id, etc.
            activation: Optional full activation record for additional context
            
        Returns:
            ServiceRegistry instance configured for this activation
        """
        try:
            from ..services.service_registry import ServiceRegistry
            
            # Get app_id properly from the activation
            if not activation:
                raise ValueError("Activation record is required to determine app_id")
            
            app_id = await self._extract_app_id_from_activation(activation)
            agent_id = activation.get('agent_id')
            created_by = activation.get('created_by')
            organization_id = activation.get('organization_id')

            if not organization_id:
                logger.error(f"No organization_id found in agent activation record")
                raise ValueError("organization_id is required for agent activation")

            # Get or create an API key for this agent activation
            api_key = await self._get_or_create_agent_api_key(app_id, agent_id, created_by, organization_id)
            if not api_key:
                logger.error(f"Failed to get or create API key for agent {agent_id}")
                raise ValueError(f"Unable to obtain API key for agent {agent_id}")
            
            # Create a config object for this activation
            config = type('ActivationConfig', (), {
                'app_id': app_id,
                'api_key': api_key,
                'base_url': self.base_url,
            })()
            
            # Create service registry for this specific activation
            service_registry = ServiceRegistry(self.db)
            
            # Create only the services we need with proper user/app context
            user_id = activation.get('created_by')
            
            if user_id and app_id:
                # Create LLM service with user context
                from ..services.llm_provider_service import LLMProviderService
                from ..services.llm_service_factory import LLMServiceFactory
                
                llm_service_with_context = LLMProviderService(
                    db_provider=self.db,
                    user_id=user_id,
                    llm_service_factory=LLMServiceFactory()
                )
                service_registry.register_service("llm_service", llm_service_with_context)
                
                # Create OAuth service with user context (if needed)
                from ..services.oauth_service import OAuthService
                oauth_service_with_context = OAuthService(db_provider=self.db)
                service_registry.register_service("oauth_service", oauth_service_with_context)
                
                logger.info(f"Registered context-aware services with user_id: {user_id}, app_id: {app_id}")
            else:
                logger.warning(f"Missing user_id ({user_id}) or app_id ({app_id}) - services not registered")
            
            # Register the fiber service (FiberApp instance) for this activation
            try:
                # Import FiberApp from SDK
                from fiberwise_sdk import FiberApp
                
                # Create fiber config for this activation
                fiber_config = {
                    'app_id': app_id,
                    'api_key': api_key,
                    'base_url': self.base_url,
                    'agent_api_key': api_key  # Use the agent API key
                }
                
                # Create FiberApp instance
                fiber_app = FiberApp(config=fiber_config)
                
                # Register it as the 'fiber' service
                service_registry.register_service('fiber', fiber_app)
                
                logger.debug(f"Registered 'fiber' service with app_id={app_id}, base_url={self.base_url}")
                
            except ImportError as e:
                logger.warning(f"Could not register fiber service: {e}")
                # Register a None service so it doesn't fail completely
                service_registry.register_service('fiber', None)
            
            return service_registry
            
        except Exception as e:
            logger.error(f"Failed to create service registry: {e}")
            raise

    async def _create_pipeline_service_registry(self, execution: Dict[str, Any]) -> 'ServiceRegistry':
        """
        Create a ServiceRegistry instance for a specific pipeline execution.
        """
        try:
            from ..services.service_registry import ServiceRegistry
            
            app_id = await self._extract_app_id_from_pipeline_execution(execution)
            pipeline_id = execution.get('pipeline_id')
            created_by = execution.get('created_by')
            organization_id = execution.get('organization_id')

            logger.info(f"Pipeline service registry creation: app_id={app_id}, pipeline_id={pipeline_id}, created_by={created_by}, organization_id={organization_id}")

            # Get or create an execution key for this pipeline execution
            execution_key = await self._get_or_create_pipeline_execution_key(app_id, pipeline_id, created_by, organization_id)
            if not execution_key:
                raise ValueError(f"Unable to obtain execution key for pipeline {pipeline_id}")
            
            # Create service registry for this specific execution
            service_registry = ServiceRegistry(self.db)
            
            # Create context-aware services
            if created_by and app_id:
                from ..services.llm_provider_service import LLMProviderService
                from ..services.llm_service_factory import LLMServiceFactory
                llm_service = LLMProviderService(
                    db_provider=self.db, user_id=created_by, llm_service_factory=LLMServiceFactory()
                )
                service_registry.register_service("llm_service", llm_service)
                
                from ..services.oauth_service import OAuthService
                oauth_service = OAuthService(db_provider=self.db)
                service_registry.register_service("oauth_service", oauth_service)
                logger.info(f"Registered context-aware services for pipeline execution with user_id: {created_by}")
            
            # Register the fiber service (FiberApp instance)
            try:
                from fiberwise_sdk import FiberApp
                fiber_config = {
                    'app_id': app_id,
                    'api_key': execution_key,
                    'base_url': self.base_url,
                }

                # Include organization_id if available for WebSocket connections
                if organization_id:
                    fiber_config['organization_id'] = organization_id
                    logger.info(f"Including organization_id {organization_id} in FiberApp config")
                else:
                    logger.warning(f"No organization_id found in execution record for WebSocket connection")

                fiber_app = FiberApp(config=fiber_config)
                service_registry.register_service('fiber', fiber_app)
                logger.debug(f"Registered 'fiber' service for pipeline with app_id={app_id}")
            except ImportError as e:
                logger.warning(f"Could not register fiber service for pipeline: {e}")
                service_registry.register_service('fiber', None)
            
            return service_registry
            
        except Exception as e:
            logger.error(f"Failed to create service registry for pipeline: {e}")
            raise
    
    async def _get_or_create_agent_api_key(self, app_id: str, agent_id: str, created_by: int, organization_id: int) -> Optional[str]:
        """
        Get or create an API key for the agent activation.

        Args:
            app_id: The application ID
            agent_id: The agent ID
            created_by: The user ID who created the activation
            organization_id: The organization ID (from auth middleware)

        Returns:
            API key string
        """
        try:
            # First, try to get an existing active API key for this agent
            query = """
                SELECT key_value FROM agent_api_keys 
                WHERE app_id = $1 AND agent_id = $2 AND is_active = 1 AND is_revoked = 0
                ORDER BY created_at DESC LIMIT 1
            """
            
            adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(query, (app_id, agent_id))
            result = await self.db.fetch_one(adapted_query, *adapted_params)
            
            if result:
                logger.info(f"Using existing API key for agent {agent_id}")
                return result['key_value']
            
            # No existing key found, create a new one
            logger.info(f"Creating new API key for agent {agent_id}")
            
            import secrets
            
            # Generate a secure API key with agent_ prefix (compatible with auth middleware)
            api_key = f"agent_{secrets.token_urlsafe(32)}"
            
            # Insert the new API key
            insert_query = """
                INSERT INTO agent_api_keys (
                    key_id, app_id, agent_id, organization_id, key_value, is_active, is_revoked, 
                    created_by, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, 1, 0, $6, NOW(), NOW())
            """
            
            import uuid
            key_id = str(uuid.uuid4())
            
            adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(
                insert_query, (key_id, app_id, agent_id, organization_id, api_key, created_by)
            )
            await self.db.execute(adapted_query, *adapted_params)
            
            logger.info(f"Created new API key for agent {agent_id}: {api_key[:10]}...")
            return api_key
            
        except Exception as e:
            logger.error(f"Failed to get or create agent API key: {e}")
            # Return None to indicate failure - let the caller handle the error
            return None

    async def _get_or_create_pipeline_execution_key(self, app_id: str, pipeline_id: str, created_by: int, organization_id: int) -> Optional[str]:
        """
        Create a temporary execution key for the pipeline execution.
        This key is single-use and tied to the execution.
        """
        try:
            import secrets
            import uuid
            
            key_value = f"exec_{secrets.token_urlsafe(32)}"
            key_id = str(uuid.uuid4())
            
            # Use provided organization_id from current user context
            
            # Use datetime arithmetic compatible with both SQLite and PostgreSQL
            insert_query = """
                INSERT INTO execution_api_keys (
                    key_id, app_id, organization_id, key_value, executor_type_id, executor_id,
                    created_by, expiration
                ) VALUES ($1, $2, $3, $4, 'pipeline', $5, $6, $7)
            """
            
            # Calculate expiration time (1 hour from now)
            from datetime import datetime, timedelta, timezone
            expiration_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(
                insert_query, (key_id, app_id, organization_id, key_value, pipeline_id, created_by, expiration_time)
            )
            await self.db.execute(adapted_query, *adapted_params)
            
            logger.info(f"Created execution key for pipeline {pipeline_id}")
            return key_value
            
        except Exception as e:
            logger.error(f"Failed to create pipeline execution key: {e}")
            return None

    async def _extract_app_id_from_pipeline_execution(self, execution: Dict[str, Any]) -> str:
        """Extract app_id from pipeline execution by joining to pipelines table"""
        pipeline_id = execution.get('pipeline_id')
        if not pipeline_id:
            raise ValueError("Missing pipeline_id in execution record")
        
        pipeline_query = "SELECT app_id FROM pipelines WHERE pipeline_id = $1"
        adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(pipeline_query, (pipeline_id,))
        pipeline_result = await self.db.fetch_one(adapted_query, *adapted_params)
        if not pipeline_result:
            raise ValueError(f"Pipeline {pipeline_id} not found in pipelines table")
        
        app_id = pipeline_result.get('app_id')
        if not app_id:
            raise ValueError(f"Pipeline {pipeline_id} has no app_id in pipelines table")
        
        return app_id
        
    def set_notification_callback(self, callback: Optional[Callable]):
        """
        Set a callback function to be called when activations complete.
        
        Args:
            callback: Function to call with (activation_id, status, app_id) when status changes
        """
        self._notification_callback = callback
        logger.info(f"[{self.context}] ACTIVATION PROCESSOR [{self.instance_id}] - Notification callback {'set' if callback else 'cleared'}")
        
    def _inject_services_into_module(self, module):
        """
        Inject services into the agent module as global variables.
        
        Args:
            module: The agent module to inject services into
        """
        injected_count = 0
        
        # Inject each service as a global variable in the module
        for service_name, service_instance in self._injected_services.items():
            setattr(module, service_name, service_instance)
            injected_count += 1
            logger.debug(f"[{self.context}] Injected {service_name} into agent module")
        
        # Also inject a services registry for backwards compatibility
        setattr(module, 'fiberwise_services', self._injected_services)
        
        logger.info(f"[{self.context}] Injected {injected_count} services into agent module")
    
    async def _extract_app_id_from_activation(self, activation: Dict[str, Any]) -> str:
        """Extract app_id from activation by joining to agents table"""
        agent_id = activation.get('agent_id')
        if not agent_id:
            raise ValueError(f"Missing agent_id in activation record")
        
        agent_query = "SELECT app_id FROM agents WHERE agent_id = $1"
        adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(agent_query, (agent_id,))
        agent_result = await self.db.fetch_one(adapted_query, *adapted_params)
        if not agent_result:
            raise ValueError(f"Agent {agent_id} not found in agents table")
        
        app_id = agent_result.get('app_id')
        if not app_id:
            raise ValueError(f"Agent {agent_id} has no app_id in agents table")
        
        return app_id
        
    def _prepare_comprehensive_dependencies(self, parameters) -> Dict[str, Any]:
        """
        Prepare comprehensive dependencies for agent execution based on function signature.
        Uses the same pattern as ServiceRegistry for consistency.
        
        Args:
            parameters: Function parameters from inspect.signature().parameters
            
        Returns:
            Dict of kwargs to inject into the agent function
        """
        dependencies = {}
        
        import inspect
        
        for param_name, param in parameters.items():
            if param_name == 'input_data':
                continue  # Skip the input_data parameter
            
            # Skip **kwargs parameters (VAR_KEYWORD)
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            
            # Map parameter names to injected services using the established pattern
            if param_name in ['fiber', 'fiber_app']:
                dependencies[param_name] = self._injected_services.get('fiber')
            elif param_name in ['llm_service', 'llm_provider_service', 'llm']:
                dependencies[param_name] = self._injected_services.get('llm_service')
            elif param_name in ['storage', 'agent_storage', 'storage_provider']:
                dependencies[param_name] = self._injected_services.get('storage')
            elif param_name in ['oauth_service', 'oauth', 'credentials']:
                dependencies[param_name] = self._injected_services.get('oauth_service')
            else:
                # Check if any other service matches this parameter name
                dependencies[param_name] = self._injected_services.get(param_name)
            
            # Log service injection status
            if dependencies[param_name] is not None:
                logger.debug(f"[{self.context}] Injecting {param_name}: {type(dependencies[param_name]).__name__}")
            else:
                logger.debug(f"[{self.context}] Service '{param_name}' not available, using None")
        
        return dependencies
        
    async def get_next_work_item(self) -> Optional[Dict[str, Any]]:
        """
        Get the next queued work item (agent activation or pipeline execution).
        
        Returns:
            Work item record with 'work_type' field indicating 'agent' or 'pipeline'.
        """
        try:
            # Prioritize agent activations
            agent_query = """
                SELECT * FROM agent_activations 
                WHERE status = 'queued'
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            """
            adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(agent_query)
            agent_result = await self.db.fetch_one(adapted_query, *adapted_params)

            if agent_result:
                work_item = dict(agent_result)
                work_item['work_type'] = 'agent'
                # Parse JSON fields
                for field in ['input_data', 'metadata', 'context', 'notes']:
                    if work_item.get(field) and isinstance(work_item[field], str):
                        try:
                            work_item[field] = json.loads(work_item[field])
                        except json.JSONDecodeError:
                            work_item[field] = {}
                return work_item

            # Then check for pipeline executions
            pipeline_query = """
                SELECT * FROM pipeline_executions
                WHERE status = 'queued'
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            """
            adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(pipeline_query)
            pipeline_result = await self.db.fetch_one(adapted_query, *adapted_params)

            if pipeline_result:
                work_item = dict(pipeline_result)
                work_item['work_type'] = 'pipeline'
                # Parse JSON fields
                for field in ['input_data', 'context', 'results']:
                    if work_item.get(field) and isinstance(work_item[field], str):
                        try:
                            work_item[field] = json.loads(work_item[field])
                        except json.JSONDecodeError:
                            work_item[field] = {}
                return work_item

            return None
            
        except Exception as e:
            logger.error(f"[{self.context}] Error getting next work item: {str(e)}", exc_info=True)
            return None

    async def get_next_activation(self) -> Optional[Dict[str, Any]]:
        """
        Get the next queued activation to process.
        
        Returns:
            Activation record or None if none available
        """
        try:
            query = """
                SELECT activation_id, agent_id, agent_type_id, status, started_at, 
                       completed_at, duration_ms, input_data, output_data, context, 
                       metadata, created_by, created_at FROM agent_activations 
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
            """
            adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(query)
            result = await self.db.fetch_one(adapted_query, *adapted_params)
            
            if result:
                activation = dict(result)
                
                # Parse JSON fields
                for field in ['input_data', 'metadata', 'context', 'notes']:
                    if activation.get(field) and isinstance(activation[field], str):
                        try:
                            activation[field] = json.loads(activation[field])
                        except json.JSONDecodeError:
                            activation[field] = {}
                
                return activation
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting next queued activation: {e}")
            return None
        
    async def process_activation(self, activation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an activation by executing the associated agent.
        
        Args:
            activation: Activation record from database
            
        Returns:
            Updated activation record with execution results
        """
        activation_id = activation.get('activation_id')
        agent_id = activation.get('agent_id')
        
        logger.info(f"[{self.context}] Processing activation {activation_id} for agent {agent_id}")
        
        try:
            # Update status to 'running'
            await self._update_activation_status(activation_id, 'running')
            
            # Get agent details
            agent = await self._get_agent_details(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Parse input data
            input_data = activation.get('input_data', {})
            if isinstance(input_data, str):
                try:
                    input_data = json.loads(input_data)
                except json.JSONDecodeError:
                    input_data = {}
            
            # Route based on agent type
            # Use agent_type_id if available, fallback to type field
            agent_type = agent.get('agent_type_id') or agent.get('type', '')
            agent_type = agent_type.lower()
            logger.info(f"[{self.context}] Agent type: {agent_type}")
            
            if agent_type == 'llm':
                # Route to LLM API provider with per-activation service
                execution_result = await self._execute_llm_agent(agent, input_data, activation)
            else:
                # Route to custom code execution
                version = await self._get_agent_version(agent_id, activation.get('version', 'latest'))
                if not version:
                    raise ValueError(f"No version found for agent {agent_id}")
                    
                file_path = version.get('file_path')
                if not file_path:
                    raise ValueError(f"No file path found for agent version")
                
                # Construct entity bundle path using IDs
                # Real path structure: apps/{app_id}/agent/{agent_id}/{version_id}/{filename}
                
                # Extract app_id using the centralized method
                app_id = await self._extract_app_id_from_activation(activation)
                
                version_id = version.get('version_id')  # Get version_id from agent_versions
                
                logger.info(f"[{self.context}] Debug - activation keys: {list(activation.keys())}")
                logger.info(f"[{self.context}] Debug - version keys: {list(version.keys())}")
                logger.info(f"[{self.context}] Debug - app_id: {app_id}, version_id: {version_id}")
                
                if not app_id:
                    raise ValueError(f"Missing app_id - not found in activation record, context, or agents table")
                if not version_id:
                    raise ValueError(f"Missing version_id in agent version record")
                
                # Extract just the filename from file_path
                import os
                filename = os.path.basename(file_path)
                # Construct the full entity bundle path
                entity_bundle_path = f"apps\\{app_id}\\agent\\{agent_id}\\{version_id}\\{filename}"
                logger.info(f"[{self.context}] Constructed entity bundle path: {entity_bundle_path}")
                
                # Execute the custom agent
                execution_result = await self._execute_custom_agent(entity_bundle_path, input_data, activation)
            
            # Update activation with results
            await self._update_activation_status(
                activation_id, 
                'completed',
                output_data=execution_result.get('output_data'),
                execution_time_ms=execution_result.get('execution_time_ms')
            )
            
            logger.info(f"[{self.context}] Successfully completed activation {activation_id}")
            
            # Return updated activation
            return await self._get_activation(activation_id)
            
        except Exception as e:
            logger.error(f"[{self.context}] Error processing activation {activation_id}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Update activation with error
            await self._update_activation_status(
                activation_id, 
                'failed',
                error=str(e)
            )
            
            # Return updated activation with error
            return await self._get_activation(activation_id)
    
    async def _execute_llm_agent(self, agent: Dict[str, Any], input_data: Dict[str, Any], activation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an LLM agent by creating a fresh LLM service per activation.
        
        Args:
            agent: Agent record from database
            input_data: Input data to pass to the agent
            activation: Full activation record for context
            
        Returns:
            Dictionary with execution results
        """
        start_time = datetime.now()
        
        try:
            from fiberwise_common.services.llm_provider_service import LLMProviderService
            
            # Import the LLMServiceFactory from common and create per-activation instance
            from fiberwise_common.services.llm_service_factory import LLMServiceFactory

            logger.info(f"[{self.context}] === EXECUTING LLM AGENT ===")
            
            # Get agent configuration
            agent_config = agent.get('config', {}) or {}
            
            # Check activation metadata and context for provider information
            activation_metadata = activation.get('metadata', {}) or {}
            activation_context = activation.get('context', {}) or {}
            
            # Look for provider information in activation metadata, then context, then agent config
            provider_id = activation_metadata.get('provider_id') or \
                          activation_context.get('provider_id') or \
                          agent_config.get('provider_id')
            
            # Create an LLMProviderService instance with the factory
            llm_service = LLMProviderService(
                db_provider=self.db,
                user_id=activation.get('created_by'),
                llm_service_factory=LLMServiceFactory
            )
            logger.info(f"[{self.context}] Created LLMProviderService with injected factory.")
            
            # Get the agent's configuration or use defaults
            system_message = agent_config.get('system_message', agent.get('description', ''))
            model = agent_config.get('model')
            
            # Prepare the prompt
            user_message = input_data.get('message', input_data.get('prompt', str(input_data)))
            if system_message:
                combined_prompt = f"System: {system_message}\n\nUser: {user_message}"
            else:
                combined_prompt = user_message
            
            # Execute the request. The LLMProviderService will handle provider selection
            # based on the provider_id (or use default if None).
            result = await llm_service.execute_llm_request(
                provider_id=provider_id,
                prompt=combined_prompt,
                model=model
            )
            
            response_text = result.get('text', str(result))

            # Calculate execution time
            end_time = datetime.now()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            logger.info(f"[{self.context}] LLM agent execution completed in {execution_time_ms}ms")
            
            return {
                'output_data': {
                    'response': response_text,
                    'model_used': model,
                    'agent_type': 'llm'
                },
                'execution_time_ms': execution_time_ms,
                'status': 'completed'
            }
            
        except Exception as e:
            end_time = datetime.now()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            logger.error(f"[{self.context}] LLM agent execution failed after {execution_time_ms}ms: {str(e)}", exc_info=True)
            
            return {
                'output_data': None,
                'execution_time_ms': execution_time_ms,
                'status': 'failed',
                'error': str(e)
            }
    
    
    async def _get_provider_config(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """Get provider configuration from database."""
        try:
            query = """
                SELECT provider_id, name, provider_type, api_endpoint, configuration, is_active
                FROM llm_providers 
                WHERE provider_id = $1 AND is_active = 1
            """
            adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(query, (provider_id,))
            result = await self.db.fetch_one(adapted_query, *adapted_params)
            
            if not result:
                return None
            
            provider_config = dict(result)
            
            # Parse configuration from JSON string
            if 'configuration' in provider_config and isinstance(provider_config['configuration'], str):
                try:
                    provider_config['configuration'] = json.loads(provider_config['configuration'])
                except json.JSONDecodeError:
                    logger.error(f"[{self.context}] Failed to parse JSON configuration for provider {provider_id}")
                    provider_config['configuration'] = {}
            
            return provider_config
            
        except Exception as e:
            logger.error(f"[{self.context}] Error getting provider config for {provider_id}: {e}")
            return None
    
    async def _execute_custom_agent(self, file_path: str, input_data: Dict[str, Any], activation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a custom agent file with the given input data.
        
        Args:
            file_path: Path to the agent Python file (relative or absolute)
            input_data: Input data to pass to the agent
            activation: Full activation record for context
            
        Returns:
            Dictionary with execution results
        """
        start_time = datetime.now()
        
        try:
            # Resolve file path - handle both absolute and relative paths
            resolved_file_path = file_path
            
            # If path is relative, resolve it using ENTITY_BUNDLES_DIR
            if not os.path.isabs(file_path):
                entity_bundles_dir = os.getenv('ENTITY_BUNDLES_DIR')
                if not entity_bundles_dir:
                    raise ValueError("ENTITY_BUNDLES_DIR environment variable not set")
                
                resolved_file_path = os.path.join(entity_bundles_dir, file_path)
                logger.info(f"[{self.context}] Resolved relative path '{file_path}' to '{resolved_file_path}'")
            
            # Validate file exists
            if not os.path.exists(resolved_file_path):
                raise FileNotFoundError(f"Agent file not found: {resolved_file_path} (original path: {file_path})")
            
            logger.info(f"[{self.context}] Executing agent file: {resolved_file_path}")
            
            # Prepare execution environment
            agent_dir = os.path.dirname(os.path.abspath(resolved_file_path))
            agent_filename = os.path.basename(resolved_file_path)
            
            # Add agent directory to Python path
            if agent_dir not in sys.path:
                sys.path.insert(0, agent_dir)
            
            try:
                # Import the agent module
                module_name = os.path.splitext(agent_filename)[0]
                
                # Remove module from cache if it exists to get fresh import
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
                # Dynamic import
                import importlib.util
                spec = importlib.util.spec_from_file_location(module_name, resolved_file_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load module from {resolved_file_path}")
                
                agent_module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = agent_module
                spec.loader.exec_module(agent_module)
                
                # Inject services into the agent module
                self._inject_services_into_module(agent_module)
                
                # Look for different execution patterns
                output_data = None
                
                # Pattern 1: run_agent function
                if hasattr(agent_module, 'run_agent'):
                    logger.info(f"[{self.context}] Executing run_agent function")
                    
                    # Use signature inspection for parameter-based injection
                    import inspect
                    run_agent_func = agent_module.run_agent
                    sig = inspect.signature(run_agent_func)
                    
                    # Use activation service for dependency preparation with context
                    service_registry = await self._create_activation_service(activation.get('context', {}), activation)
                    if service_registry:
                        # Extract app_id using the centralized method
                        app_id = await self._extract_app_id_from_activation(activation)
                        
                        # Create comprehensive activation context
                        activation_context = activation.get('context', {}).copy()
                        activation_context['app_id'] = app_id
                        activation_context['user_id'] = activation.get('created_by')
                        
                        logger.info(f"[{self.context}] Retrieved app_id={app_id} for agent_id={activation.get('agent_id')}")
                        
                        # Populate _injected_services from service_registry
                        self._injected_services = {
                            'fiber': service_registry.get_service('fiber'),
                            'llm_service': service_registry.get_service('llm_service'),
                            'storage': service_registry.get_service('storage'),
                            'oauth_service': service_registry.get_service('oauth_service')
                        }
                        
                        dependencies = self._prepare_comprehensive_dependencies(sig.parameters)
                    else:
                        # Fallback to old method if activation service creation fails
                        dependencies = self._prepare_comprehensive_dependencies(sig.parameters)
                    
                    logger.info(f"[{self.context}] Injecting services: {list(dependencies.keys())}")
                    
                    # Extend input_data with context and metadata for agent access
                    extended_input_data = input_data.copy()
                    extended_input_data['_context'] = activation.get('context', {})
                    extended_input_data['_metadata'] = activation.get('metadata', {})
                    
                    # Execute with parameter injection
                    result = run_agent_func(extended_input_data, **dependencies)
                    
                    # Handle async results
                    if asyncio.iscoroutine(result):
                        output_data = await result
                    else:
                        output_data = result
                        
                # Pattern 2: Any class that inherits from FiberAgent with run method
                else:
                    # Look for any class that inherits from FiberAgent
                    agent_class = None
                    
                    # Try to import FiberAgent for inheritance checking
                    try:
                        from fiberwise_sdk import FiberAgent
                        fiber_agent_available = True
                    except ImportError:
                        fiber_agent_available = False
                        logger.debug(f"[{self.context}] FiberAgent not available, falling back to exact class name matching")
                    
                    # Scan module for classes
                    for attr_name in dir(agent_module):
                        attr = getattr(agent_module, attr_name)
                        
                        # Check if it's a class
                        if isinstance(attr, type):
                            # If FiberAgent is available, check inheritance
                            if fiber_agent_available:
                                if issubclass(attr, FiberAgent) and attr != FiberAgent:
                                    agent_class = attr
                                    logger.info(f"[{self.context}] Found FiberAgent subclass: {attr_name}")
                                    break
                            # Fallback: check for exact "Agent" class name
                            elif attr_name == 'Agent':
                                agent_class = attr
                                logger.info(f"[{self.context}] Found Agent class: {attr_name}")
                                break
                    
                    if agent_class:
                        agent_instance = agent_class()
                        
                        # Check for run_agent method (preferred for FiberAgent classes)
                        if hasattr(agent_instance, 'run_agent'):
                            logger.info(f"[{self.context}] Executing agent class run_agent method")
                            
                            # Use signature inspection for parameter-based injection
                            import inspect
                            run_agent_method = getattr(agent_instance, 'run_agent')
                            sig = inspect.signature(run_agent_method)
                            
                            # Use activation service for dependency preparation with context
                            service_registry = await self._create_activation_service(activation.get('context', {}), activation)
                            if service_registry:
                                # Extract app_id using the centralized method
                                app_id = await self._extract_app_id_from_activation(activation)
                                
                                # Create comprehensive activation context
                                activation_context = activation.get('context', {}).copy()
                                activation_context['app_id'] = app_id
                                activation_context['user_id'] = activation.get('created_by')
                                
                                # Populate _injected_services from service_registry - only get services that exist
                                self._injected_services = {}
                                
                                # Add only context-aware services that are registered
                                for service_name in ['fiber', 'llm_service', 'oauth_service']:
                                    if service_registry.is_registered(service_name):
                                        try:
                                            service_instance = service_registry.get_service(service_name)
                                            # Only inject if we got a valid service instance
                                            if service_instance is not None:
                                                self._injected_services[service_name] = service_instance
                                                logger.debug(f"Injected context-aware service: {service_name}")
                                            else:
                                                logger.warning(f"Service '{service_name}' returned None - not injecting")
                                        except KeyError as e:
                                            logger.warning(f"Service '{service_name}' registered but failed to retrieve: {e}")
                                    else:
                                        logger.debug(f"Service '{service_name}' not registered - skipping")
                                
                                dependencies = self._prepare_comprehensive_dependencies(sig.parameters)
                            else:
                                # Fallback to old method if activation service creation fails
                                dependencies = self._prepare_comprehensive_dependencies(sig.parameters)
                            
                            logger.info(f"[{self.context}] Injecting services into run_agent method: {list(dependencies.keys())}")
                            
                            # Extend input_data with context and metadata for agent access
                            extended_input_data = input_data.copy()
                            extended_input_data['_context'] = activation.get('context', {})
                            extended_input_data['_metadata'] = activation.get('metadata', {})
                            
                            logger.info(f"[{self.context}] Extended input_data with context provider_id: {extended_input_data['_context'].get('provider_id')}")
                            logger.info(f"[{self.context}] Extended input_data with metadata provider_id: {extended_input_data['_metadata'].get('provider_id')}")
                            
                            # Execute with parameter injection
                            result = run_agent_method(extended_input_data, **dependencies)
                            
                            # Handle async results
                            if asyncio.iscoroutine(result):
                                output_data = await result
                            else:
                                output_data = result
                        # Fallback to run method for compatibility
                        elif hasattr(agent_instance, 'run'):
                            logger.info(f"[{self.context}] Executing agent class run method")
                            
                            # Use signature inspection for parameter-based injection
                            import inspect
                            run_method = getattr(agent_instance, 'run')
                            sig = inspect.signature(run_method)
                            
                            # Use activation service for dependency preparation with context
                            service_registry = await self._create_activation_service(activation.get('context', {}), activation)
                            if service_registry:
                                # Get app_id using the centralized method
                                app_id = await self._extract_app_id_from_activation(activation)
                                
                                # Create comprehensive activation context
                                activation_context = activation.get('context', {}).copy()
                                activation_context['app_id'] = app_id
                                activation_context['user_id'] = activation.get('created_by')
                                
                                # Populate _injected_services from service_registry - only get services that exist
                                self._injected_services = {}
                                
                                # Add only context-aware services that are registered
                                for service_name in ['fiber', 'llm_service', 'oauth_service']:
                                    if service_registry.is_registered(service_name):
                                        try:
                                            service_instance = service_registry.get_service(service_name)
                                            # Only inject if we got a valid service instance
                                            if service_instance is not None:
                                                self._injected_services[service_name] = service_instance
                                                logger.debug(f"Injected context-aware service: {service_name}")
                                            else:
                                                logger.warning(f"Service '{service_name}' returned None - not injecting")
                                        except KeyError as e:
                                            logger.warning(f"Service '{service_name}' registered but failed to retrieve: {e}")
                                    else:
                                        logger.debug(f"Service '{service_name}' not registered - skipping")
                                
                                dependencies = self._prepare_comprehensive_dependencies(sig.parameters)
                            else:
                                # Fallback to old method if activation service creation fails
                                dependencies = self._prepare_comprehensive_dependencies(sig.parameters)
                            
                            logger.info(f"[{self.context}] Injecting services into run method: {list(dependencies.keys())}")
                            
                            # Extend input_data with context and metadata for agent access
                            extended_input_data = input_data.copy()
                            extended_input_data['_context'] = activation.get('context', {})
                            extended_input_data['_metadata'] = activation.get('metadata', {})
                            
                            logger.info(f"[{self.context}] Extended input_data with context provider_id: {extended_input_data['_context'].get('provider_id')}")
                            logger.info(f"[{self.context}] Extended input_data with metadata provider_id: {extended_input_data['_metadata'].get('provider_id')}")
                            
                            # Execute with parameter injection
                            result = run_method(extended_input_data, **dependencies)
                            
                            # Handle async results
                            if asyncio.iscoroutine(result):
                                output_data = await result
                            else:
                                output_data = result
                        else:
                            raise AttributeError(f"Agent class {agent_class.__name__} must have either a 'run_agent' or 'run' method")
                    
                    # Pattern 3: Look for any callable that takes input_data
                    if output_data is None:
                        # Try to find a main function or similar
                        for attr_name in ['main', 'execute', 'process']:
                            if hasattr(agent_module, attr_name):
                                func = getattr(agent_module, attr_name)
                                if callable(func):
                                    logger.info(f"[{self.context}] Executing {attr_name} function")
                                    result = func(input_data)
                                    
                                    # Handle async results
                                    if asyncio.iscoroutine(result):
                                        output_data = await result
                                    else:
                                        output_data = result
                                    break
                        
                        if output_data is None:
                            raise AttributeError(
                                "Agent must have either a 'run_agent' function, "
                                "a FiberAgent subclass with 'run_agent' method, or a 'main'/'execute'/'process' function"
                            )
                
                # Calculate execution time
                end_time = datetime.now()
                execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
                
                logger.info(f"[{self.context}] Agent execution completed in {execution_time_ms}ms")
                
                return {
                    'output_data': output_data,
                    'execution_time_ms': execution_time_ms,
                    'status': 'completed'
                }
                
            finally:
                # Clean up sys.path
                if agent_dir in sys.path:
                    sys.path.remove(agent_dir)
                    
        except Exception as e:
            end_time = datetime.now()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            logger.error(f"[{self.context}] Agent execution failed after {execution_time_ms}ms: {str(e)}")
            
            return {
                'output_data': None,
                'execution_time_ms': execution_time_ms,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _get_agent_details(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent details from database."""
        query = "SELECT * FROM agents WHERE agent_id = $1"
        adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(query, (agent_id,))
        result = await self.db.fetch_one(adapted_query, *adapted_params)
        
        if not result:
            return None
            
        agent = dict(result)
        
        # Parse config from JSON string
        if 'config' in agent and isinstance(agent.get('config'), str):
            try:
                config_str = agent.get('config', '').strip()
                if config_str:
                    agent['config'] = json.loads(config_str)
                else:
                    agent['config'] = {}
            except json.JSONDecodeError:
                logger.warning(f"[{self.context}] Failed to parse JSON config for agent {agent_id}")
                agent['config'] = {}
        
        return agent
    
    async def _get_agent_version(self, agent_id: str, version: str = 'latest') -> Optional[Dict[str, Any]]:
        """Get agent version details from database."""
        if version == 'latest':
            query = """
                SELECT av.*
                FROM agent_versions av
                WHERE av.agent_id = $1 AND av.is_active = 1
                ORDER BY av.created_at DESC 
                LIMIT 1
            """
            adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(query, (agent_id,))
            result = await self.db.fetch_one(adapted_query, *adapted_params)
        else:
            query = """
                SELECT av.*
                FROM agent_versions av
                WHERE av.agent_id = $1 AND av.version = $2 AND av.is_active = 1
            """
            adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(query, (agent_id, version))
            result = await self.db.fetch_one(adapted_query, *adapted_params)
        
        return dict(result) if result else None
    
    async def _get_activation(self, activation_id: str) -> Dict[str, Any]:
        """Get activation record from database."""
        query = """SELECT activation_id, agent_id, agent_type_id, status, started_at, 
                          completed_at, duration_ms, input_data, output_data, context, 
                          metadata, created_by, created_at FROM agent_activations WHERE activation_id = $1"""
        adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(query, (activation_id,))
        result = await self.db.fetch_one(adapted_query, *adapted_params)
        
        if not result:
            return {}
        
        activation = dict(result)
        
        # Parse JSON fields
        for field in ['input_data', 'output_data', 'metadata', 'context', 'notes']:
            if activation.get(field) and isinstance(activation[field], str):
                try:
                    activation[field] = json.loads(activation[field])
                except json.JSONDecodeError:
                    activation[field] = {}
        
        return activation
    
    async def process_work_item(self, work_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for processing any type of work item in the FiberWise system.
        
        This method routes work items to their appropriate processing handlers based on
        the 'work_type' field. It supports three main work types:
        
        - 'agent': Agent activation (LLM or custom code execution)
        - 'pipeline': Pipeline execution (structured workflow processing)  
        - 'function': Function execution (serverless code execution)
        
        The method ensures consistent processing patterns across all work types while
        delegating to specialized handlers for each type's unique requirements.
        
        Args:
            work_item: Work item dictionary with required 'work_type' field and
                      type-specific data (activation_id, execution_id, etc.)
                      
        Returns:
            Updated work item record with execution results, status, and timing
            
        Raises:
            Routes to appropriate handler - exceptions handled by individual processors
            
        Example:
            # Agent activation
            work_item = {
                'work_type': 'agent',
                'activation_id': 'uuid',
                'agent_id': 'uuid',
                'input_data': {...}
            }
            
            # Pipeline execution  
            work_item = {
                'work_type': 'pipeline',
                'execution_id': 'uuid',
                'pipeline_id': 'uuid',
                'input_data': {...}
            }
        """
        work_type = work_item.get('work_type')
        
        if work_type == 'agent':
            return await self.process_activation(work_item)
        elif work_type == 'pipeline':
            return await self._process_pipeline_execution(work_item)
        elif work_type == 'function':
            return await self._process_function_execution(work_item)
        else:
            logger.error(f"Unknown work_type '{work_type}' received. Defaulting to agent activation.")
            return await self.process_activation(work_item)
        work_type = work_item.get('work_type')
        
        if work_type == 'agent':
            return await self.process_activation(work_item)
        elif work_type == 'pipeline':
            return await self._process_pipeline_execution(work_item)
        elif work_type == 'function':
            return await self._process_function_execution(work_item)
        else:
            logger.error(f"Unknown work_type '{work_type}' received. Defaulting to agent activation.")
            return await self.process_activation(work_item)
    
    async def _process_pipeline_execution(self, execution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a pipeline execution.
        
        Args:
            execution: Pipeline execution record from database
            
        Returns:
            Updated execution record with results
        """
        execution_id = execution.get('execution_id')
        pipeline_id = execution.get('pipeline_id')
        
        logger.info(f"[{self.context}] Processing pipeline execution {execution_id} for pipeline {pipeline_id}")
        
        try:
            # Update status to 'running'
            await self._update_pipeline_execution_status(execution_id, 'running')
            
            # Get pipeline details
            pipeline = await self._get_pipeline_details(pipeline_id)
            if not pipeline:
                raise ValueError(f"Pipeline {pipeline_id} not found")
            
            # Parse input data
            input_data = execution.get('input_data', {})
            if isinstance(input_data, str):
                try:
                    input_data = json.loads(input_data)
                except json.JSONDecodeError:
                    input_data = {}
            
            # Execute the pipeline
            start_time = datetime.now()
            result = await self._execute_pipeline(pipeline, input_data, execution)
            end_time = datetime.now()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Update execution record with success
            await self._update_pipeline_execution_status(
                execution_id, 
                'completed',
                result,
                execution_time_ms
            )
            
            logger.info(f"[{self.context}] Pipeline execution {execution_id} completed successfully")
            
            return {
                'execution_id': execution_id,
                'pipeline_id': pipeline_id,
                'status': 'completed',
                'result': result,
                'execution_time_ms': execution_time_ms
            }
            
        except Exception as e:
            logger.error(f"[{self.context}] Pipeline execution {execution_id} failed: {str(e)}", exc_info=True)
            
            # Update execution record with failure
            await self._update_pipeline_execution_status(execution_id, 'failed', error=str(e))
            
            return {
                'execution_id': execution_id,
                'pipeline_id': pipeline_id,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _process_function_execution(self, execution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a function execution.
        
        Args:
            execution: Function execution record from database
            
        Returns:
            Updated execution record with results
        """
        execution_id = execution.get('execution_id')
        function_id = execution.get('function_id')
        
        logger.info(f"[{self.context}] Processing function execution {execution_id} for function {function_id}")
        
        try:
            # Update status to 'running'
            await self._update_function_execution_status(execution_id, 'running')
            
            # Get function details
            function = await self._get_function_details(function_id)
            if not function:
                raise ValueError(f"Function {function_id} not found")
            
            # Parse input data
            input_data = execution.get('input_data', {})
            if isinstance(input_data, str):
                try:
                    input_data = json.loads(input_data)
                except json.JSONDecodeError:
                    input_data = {}
            
            # Execute the function
            start_time = datetime.now()
            result = await self._execute_function(function, input_data)
            end_time = datetime.now()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Update execution record with success
            await self._update_function_execution_status(
                execution_id, 
                'completed',
                result,
                execution_time_ms
            )
            
            logger.info(f"[{self.context}] Function execution {execution_id} completed successfully")
            
            return {
                'execution_id': execution_id,
                'function_id': function_id,
                'status': 'completed',
                'result': result,
                'execution_time_ms': execution_time_ms
            }
            
        except Exception as e:
            logger.error(f"[{self.context}] Function execution {execution_id} failed: {str(e)}", exc_info=True)
            
            # Update execution record with failure
            await self._update_function_execution_status(execution_id, 'failed', error=str(e))
            
            return {
                'execution_id': execution_id,
                'function_id': function_id,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _execute_pipeline(self, pipeline: Dict[str, Any], input_data: Dict[str, Any], execution: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a pipeline with structured definition following agent execution patterns."""
        
        # Get pipeline definition (required for new structured pipelines)
        definition = pipeline.get('definition')
        if not definition:
            # Fallback to legacy file-based execution
            return await self._execute_legacy_pipeline(pipeline, input_data, execution)
        
        try:
            structure = json.loads(definition) if isinstance(definition, str) else definition
            if not structure or 'steps' not in structure:
                raise ValueError("Invalid pipeline structure - missing 'steps'")
            
            return await self._execute_structured_pipeline(structure, input_data, execution, pipeline)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in pipeline definition: {str(e)}")
    
    async def _execute_structured_pipeline(self, structure: Dict[str, Any], input_data: Dict[str, Any], execution: Dict[str, Any], pipeline: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a pipeline as a directed graph with conditional branching."""
        steps_map = {step['id']: step for step in structure.get('steps', [])}
        paths = structure.get('flow', {}).get('paths', [])
        
        # Load step classes using agent execution patterns
        step_instances = await self._load_pipeline_step_classes(steps_map, pipeline, execution)
        step_results = {}
        
        # Graph traversal logic
        current_step_id = structure.get('flow', {}).get('start_at')
        current_data = input_data.copy()

        while current_step_id and current_step_id not in ['end', 'end_with_success']:
            if current_step_id in ['end_with_failure']:
                raise Exception("Pipeline flow ended in a failure state.")

            step_def = steps_map.get(current_step_id)
            if not step_def:
                raise ValueError(f"Step '{current_step_id}' not found in pipeline definition.")

            logger.info(f"[{self.context}] Executing step {current_step_id}")
            
            # Handle Human Input Steps
            if step_def.get('type') == 'human_input':
                logger.info(f"[{self.context}] Pausing pipeline for human input at step {current_step_id}.")
                await self._pause_for_human_input(execution['id'], step_def)
                # Return a special status indicating the pipeline is paused
                return {'status': 'PAUSED_FOR_INPUT', 'wait_on_step': current_step_id}

            step_instance = step_instances[current_step_id]
            
            try:
                # Use same service creation pattern as agents
                service_registry = await self._create_pipeline_service_registry(execution)
                fiber = service_registry.get_service('fiber')
                
                # Resolve dynamic parameters
                resolved_params = self._resolve_dynamic_parameters(
                    step_def.get('parameters', {}), 
                    {'pipeline_input': input_data, 'steps': step_results}
                )
                
                # Execute step with same signature as agent execution
                result = await step_instance.execute(resolved_params, fiber)
                step_results[current_step_id] = result
                
                if not result.get('success'):
                    error_msg = result.get('error', 'Unknown error')
                    raise Exception(f"Step {current_step_id} failed: {error_msg}")
                
                # The output of this step becomes available for next decisions
                step_output = result
                current_data = result.get('result', {})
                
                # Find the next step based on evaluating path conditions
                next_step_id = await self._find_next_step(current_step_id, paths, step_output, fiber, pipeline)
                current_step_id = next_step_id
                
            except Exception as e:
                logger.error(f"[{self.context}] Step {current_step_id} execution error: {str(e)}", exc_info=True)
                step_results[current_step_id] = {'success': False, 'error': str(e)}
                raise

        return {
            'success': True,
            'final_result': current_data,
            'step_results': step_results
        }
    
    async def _execute_legacy_pipeline(self, pipeline: Dict[str, Any], input_data: Dict[str, Any], execution: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a legacy file-based pipeline (fallback)."""
        file_path = pipeline.get('file_path')
        if not file_path:
            raise ValueError("Pipeline has no file_path or definition")

        start_time = datetime.now()
        
        try:
            resolved_file_path = file_path
            if not os.path.isabs(file_path):
                entity_bundles_dir = os.getenv('ENTITY_BUNDLES_DIR')
                if not entity_bundles_dir:
                    raise ValueError("ENTITY_BUNDLES_DIR environment variable not set")
                resolved_file_path = os.path.join(entity_bundles_dir, file_path)
            
            if not os.path.exists(resolved_file_path):
                raise FileNotFoundError(f"Pipeline file not found: {resolved_file_path}")

            logger.info(f"[{self.context}] Executing legacy pipeline file: {resolved_file_path}")

            pipeline_dir = os.path.dirname(os.path.abspath(resolved_file_path))
            pipeline_filename = os.path.basename(resolved_file_path)
            
            if pipeline_dir not in sys.path:
                sys.path.insert(0, pipeline_dir)

            try:
                module_name = os.path.splitext(pipeline_filename)[0]
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
                import importlib.util
                spec = importlib.util.spec_from_file_location(module_name, resolved_file_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load module from {resolved_file_path}")
                
                pipeline_module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = pipeline_module
                spec.loader.exec_module(pipeline_module)

                if not hasattr(pipeline_module, 'execute'):
                    raise AttributeError("Pipeline module must have an 'execute' function.")

                execute_func = getattr(pipeline_module, 'execute')
                
                # Use signature inspection for parameter-based injection
                import inspect
                sig = inspect.signature(execute_func)
                
                # Create service registry and prepare dependencies
                service_registry = await self._create_pipeline_service_registry(execution)
                if service_registry:
                    self._injected_services = {
                        'fiber': service_registry.get_service('fiber'),
                        'llm_service': service_registry.get_service('llm_service'),
                        'storage': service_registry.get_service('storage'),
                        'oauth_service': service_registry.get_service('oauth_service')
                    }
                    dependencies = self._prepare_comprehensive_dependencies(sig.parameters)
                else:
                    dependencies = {}
                
                logger.info(f"[{self.context}] Injecting services into pipeline: {list(dependencies.keys())}")
                
                result = execute_func(input_data, **dependencies)
                
                if asyncio.iscoroutine(result):
                    output_data = await result
                else:
                    output_data = result

                return output_data

            finally:
                if pipeline_dir in sys.path:
                    sys.path.remove(pipeline_dir)

        except Exception as e:
            logger.error(f"[{self.context}] Error executing legacy pipeline: {str(e)}", exc_info=True)
            raise
    
    async def _load_pipeline_step_classes(self, steps_map: Dict[str, Dict], pipeline: Dict[str, Any], execution: Dict[str, Any]) -> Dict[str, Any]:
        """Load all step classes using agent execution patterns - entity bundles and dynamic loading."""
        step_instances = {}
        
        # Get pipeline app_id and version info
        app_id = pipeline.get('app_id')
        pipeline_id = pipeline.get('pipeline_id')
        
        if not app_id or not pipeline_id:
            raise ValueError("Pipeline missing required app_id or pipeline_id")
        
        for step_id, step_def in steps_map.items():
            if step_def.get('type') == 'human_input':
                continue  # Human input steps don't have classes
                
            step_class_name = step_def.get('step_class')
            if not step_class_name:
                raise ValueError(f"Step {step_id} missing step_class")
            
            step_instance = await self._load_pipeline_step_class(step_class_name, app_id, pipeline_id)
            step_instances[step_id] = step_instance
        
        return step_instances

    async def _load_pipeline_step_class(self, step_class_name: str, app_id: str, pipeline_id: str):
        """Load a pipeline step class from pipeline_code table (following agent_code pattern)."""
        try:
            # Query pipeline_code table to get step implementation
            query = """
                SELECT step_id, step_class, implementation_code, language 
                FROM pipeline_code 
                WHERE pipeline_id = $1 AND step_class = $2 AND is_active = true
                ORDER BY created_at DESC 
                LIMIT 1
            """
            adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(
                query, (pipeline_id, step_class_name)
            )
            step_record = await self.db.fetch_one(adapted_query, *adapted_params)
            
            if not step_record:
                raise ValueError(f"Step class '{step_class_name}' not found in pipeline_code table for pipeline {pipeline_id}")
            
            implementation_code = step_record['implementation_code']
            step_id = step_record['step_id']
            language = step_record['language']
            
            logger.info(f"Loading step class {step_class_name} for step {step_id} from pipeline_code table")
            
            if language != 'python':
                raise ValueError(f"Only Python step implementations are supported, got: {language}")
            
            # Execute the implementation code to define the class (following agent execution pattern)
            namespace = {
                '__name__': f'pipeline_step_{step_id}',
                '__builtins__': __builtins__
            }
            
            try:
                # Execute the step class code
                exec(implementation_code, namespace)
                
                # Find the step class in the namespace
                if step_class_name in namespace:
                    step_class = namespace[step_class_name]
                    step_instance = step_class()
                    logger.info(f"Successfully loaded and instantiated step class: {step_class_name}")
                    return step_instance
                else:
                    raise AttributeError(f"Step class '{step_class_name}' not found in executed code")
                    
            except Exception as e:
                logger.error(f"Error executing step implementation code: {str(e)}")
                logger.error(f"Code that failed: {implementation_code[:500]}...")
                raise ValueError(f"Failed to execute step implementation: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error loading pipeline step class {step_class_name}: {str(e)}")
            raise

    def _resolve_dynamic_parameters(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve dynamic parameter values using template substitution."""
        resolved = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and '{' in value:
                # Simple template resolution - can be enhanced
                try:
                    resolved[key] = value.format(**context)
                except (KeyError, ValueError):
                    # If template resolution fails, use original value
                    resolved[key] = value
            else:
                resolved[key] = value
        
        return resolved

    async def _find_next_step(self, current_step_id: str, paths: List[Dict], step_output: Dict, fiber, pipeline: Dict[str, Any]) -> Optional[str]:
        """Evaluate outgoing paths from the current step to find the next one."""
        outgoing_paths = [p for p in paths if p.get('from') == current_step_id]
        
        for path in outgoing_paths:  # Order in manifest is the priority
            condition = path.get('condition', {'type': 'always'})
            
            is_path_taken = await self._evaluate_path_condition(condition, step_output, fiber, pipeline)
            
            if is_path_taken:
                logger.info(f"[{self.context}] Condition met for path from '{current_step_id}' to '{path.get('to')}'.")
                return path.get('to')
                
        logger.warning(f"[{self.context}] No outgoing path condition met from step '{current_step_id}'. Ending pipeline.")
        return None

    async def _evaluate_path_condition(self, condition: Dict, previous_step_output: Dict, fiber, pipeline: Dict[str, Any]) -> bool:
        """Dynamically evaluates a condition of any type."""
        condition_type = condition.get('type', 'always')
        config = condition.get('config', {})

        if condition_type == 'always':
            return True
        
        elif condition_type == 'expression':
            # Simple expression evaluation
            source = config.get('source', '')
            operator = config.get('operator', '')
            value = config.get('value')
            
            # Extract value using dot notation
            actual_value = self._get_nested_value(previous_step_output, source)
            
            if operator == 'equals':
                return actual_value == value
            elif operator == 'greater_than':
                return float(actual_value) > float(value)
            elif operator == 'greater_than_or_equal':
                return float(actual_value) >= float(value)
            elif operator == 'less_than':
                return float(actual_value) < float(value)
            elif operator == 'equals_ignore_case':
                return str(actual_value).lower() == str(value).lower()
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False
        
        elif condition_type == 'agent':
            # AI-driven routing decision
            agent_id = config.get('agent_id')
            input_config = config.get('input', {})
            output_config = config.get('output', {})
            
            if not agent_id or 'prompt' not in input_config:
                raise ValueError("Agent condition requires 'agent_id' and 'input.prompt' in config.")

            # Format the agent's prompt using data from the previous pipeline step
            prompt_template = input_config.get('prompt')
            prompt = self._resolve_dynamic_parameters({'prompt': prompt_template}, {'output': previous_step_output})['prompt']

            # Activate the agent
            agent_service = fiber.get_service("agent")
            agent_output = await agent_service.activate(agent_id, prompt)

            # Check if there's a nested evaluation to perform
            if "evaluation" in output_config:
                eval_config = output_config['evaluation']
                value_from_agent = self._get_nested_value(agent_output, eval_config.get('source'))
                return self._perform_comparison(value_from_agent, eval_config.get('operator'), eval_config.get('value'))
            else:
                # Default behavior: check if agent's raw output is 'true'
                raw_text_output = agent_output.get("result", {}).get("text", "")
                return raw_text_output.strip().lower() == "true"
        
        elif condition_type == 'function':
            # Custom function-based routing using bundled classes
            function_class = config.get('function_class')
            function_method = config.get('function_method')
            input_mapping = config.get('input_mapping')
            expected_result = config.get('expected_result')
            
            if not function_class or not function_method:
                raise ValueError("Function condition requires 'function_class' and 'function_method'")
            
            # Load and execute the function from pipeline bundle
            function_result = await self._execute_condition_function(
                function_class, function_method, previous_step_output, input_mapping, 
                pipeline.get('app_id'), pipeline.get('pipeline_id')
            )
            
            return function_result == expected_result
        
        return False

    def _get_nested_value(self, obj: Dict, path: str):
        """Extract nested value using dot notation (e.g., 'result.score')."""
        try:
            keys = path.split('.')
            current = obj
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key)
                else:
                    return None
            return current
        except:
            return None

    def _perform_comparison(self, actual_value, operator: str, expected_value):
        """Perform comparison operation."""
        if operator == 'equals':
            return actual_value == expected_value
        elif operator == 'greater_than':
            return float(actual_value) > float(expected_value)
        elif operator == 'equals_ignore_case':
            return str(actual_value).lower() == str(expected_value).lower()
        # Add more operators as needed
        return False

    async def _execute_condition_function(self, function_class: str, function_method: str, step_output: Dict, input_mapping: str, app_id: str, pipeline_id: str):
        """Execute a custom condition function using same bundling pattern as steps."""
        # Load function class from pipeline bundle (same as step loading)
        function_instance = await self._load_pipeline_step_class(function_class, app_id, pipeline_id)
        
        # Extract input value based on mapping
        input_value = self._extract_value_from_mapping(step_output, input_mapping)
        
        # Call the specified method
        if hasattr(function_instance, function_method):
            method = getattr(function_instance, function_method)
            return await method(input_value) if asyncio.iscoroutinefunction(method) else method(input_value)
        else:
            raise AttributeError(f"Method '{function_method}' not found in function class '{function_class}'")

    def _extract_value_from_mapping(self, step_output: Dict, input_mapping: str):
        """Extract value from step output using mapping string."""
        return self._get_nested_value(step_output, input_mapping)

    # HITL (Human-in-the-Loop) Support Methods
    async def _pause_for_human_input(self, execution_id: str, step_def: Dict[str, Any]):
        """Pause pipeline execution and store human input requirements."""
        ui_schema = step_def.get('parameters', {}).get('ui_schema', {})
        
        # Store the UI schema and step info in database for the web UI to retrieve
        update_query = """
            UPDATE pipeline_executions 
            SET status = 'paused_for_input', 
                human_input_config = $1,
                waiting_step_id = $2
            WHERE execution_id = $3
        """
        
        adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(
            update_query, (json.dumps(ui_schema), step_def.get('id'), execution_id)
        )
        await self.db.execute(adapted_query, *adapted_params)
        
        # Optionally send notification to user
        if self._notification_callback:
            await self._notification_callback({
                'type': 'human_input_required',
                'execution_id': execution_id,
                'step_id': step_def.get('id'),
                'ui_schema': ui_schema
            })

    async def resume_pipeline_execution(self, execution_id: str, human_input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resume a paused pipeline with human input data."""
        # Get the execution record
        execution_query = "SELECT * FROM pipeline_executions WHERE execution_id = $1"
        adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(execution_query, (execution_id,))
        execution = await self.db.fetch_one(adapted_query, *adapted_params)
        
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")
        
        if execution['status'] != 'paused_for_input':
            raise ValueError(f"Execution {execution_id} is not paused for input")
        
        # Update status and store human input
        update_query = """
            UPDATE pipeline_executions 
            SET status = 'running',
                human_input_data = $1,
                human_input_config = NULL,
                waiting_step_id = NULL
            WHERE execution_id = $2
        """
        
        adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(
            update_query, (json.dumps(human_input_data), execution_id)
        )
        await self.db.execute(adapted_query, *adapted_params)
        
        # Continue pipeline execution from where it left off
        # This would typically be called by a web API endpoint
        return await self._process_pipeline_execution(dict(execution))

    async def _execute_function(self, function: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function with the given input data."""
        implementation = function.get('implementation', '')
        
        if not implementation:
            raise ValueError("Function has no implementation")
        
        # Use the same execution approach as agents
        try:
            # Create a safe namespace
            namespace = {
                "input_data": input_data, 
                "asyncio": asyncio, 
                "json": json,
                "datetime": datetime
            }
            
            # Add injected services to namespace
            namespace.update(self._injected_services)
            
            # Add run function if not present
            if "async def run" not in implementation and "def run" not in implementation:
                implementation = f"""
async def run(input_data):
    # Default implementation
    return {{"result": "No implementation provided"}}

{implementation}
                """
            
            # Execute the code
            exec(implementation, namespace)
            
            # Call the run function
            run_func = namespace.get("run")
            if asyncio.iscoroutinefunction(run_func):
                result = await run_func(input_data)
            else:
                result = run_func(input_data)
            
            return result
        except Exception as e:
            logger.error(f"[{self.context}] Error executing function implementation: {str(e)}", exc_info=True)
            raise ValueError(f"Error in function implementation: {str(e)}")
    
    async def _get_pipeline_details(self, pipeline_id: str) -> Dict[str, Any]:
        """Get pipeline details from database."""
        query = "SELECT * FROM pipelines WHERE pipeline_id = $1"
        adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(query, (pipeline_id,))
        result = await self.db.fetch_one(adapted_query, *adapted_params)
        
        if not result:
            return {}
        
        return dict(result)

    async def _get_function_details(self, function_id: str) -> Dict[str, Any]:
        """Get function details from database."""
        query = "SELECT * FROM functions WHERE function_id = $1"
        adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(query, (function_id,))
        result = await self.db.fetch_one(adapted_query, *adapted_params)
        
        if not result:
            return {}
        
        function = dict(result)
        
        # Parse JSON fields
        for field in ['input_schema', 'output_schema']:
            if function.get(field) and isinstance(function[field], str):
                try:
                    function[field] = json.loads(function[field])
                except json.JSONDecodeError:
                    function[field] = {}
        
        return function
    
    async def _update_pipeline_execution_status(self, execution_id: str, status: str, 
                                              output_data: Dict[str, Any] = None,
                                              execution_time_ms: int = None,
                                              error: str = None):
        """Update pipeline execution status in database."""
        update_fields = ["status = $2"]
        values = [execution_id, status]
        param_index = 3
        
        if status in ['completed', 'failed']:
            update_fields.append("completed_at = CURRENT_TIMESTAMP")
        
        if output_data is not None:
            update_fields.append(f"results = ${param_index}") # pipelines table uses 'results'
            values.append(json.dumps(output_data))
            param_index += 1
            
        if error is not None:
            update_fields.append(f"error = ${param_index}")
            values.append(error)
            param_index += 1
        
        query = f"""
            UPDATE pipeline_executions 
            SET {', '.join(update_fields)}
            WHERE execution_id = $1
        """
        
        try:
            adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(query, values)
            await self.db.execute(adapted_query, *adapted_params)
            logger.debug(f"[{self.context}] Updated pipeline execution {execution_id} status to {status}")
        except Exception as e:
            logger.error(f"[{self.context}] Failed to update pipeline execution status: {str(e)}")

    async def _update_function_execution_status(self, execution_id: str, status: str, 
                                              output_data: Dict[str, Any] = None,
                                              execution_time_ms: int = None,
                                              error: str = None):
        """Update function execution status in database."""
        update_fields = ["status = $2"]
        values = [execution_id, status]
        param_index = 3
        
        if status in ['completed', 'failed']:
            update_fields.append("completed_at = CURRENT_TIMESTAMP")
        
        if output_data is not None:
            update_fields.append(f"output_data = ${param_index}")
            values.append(json.dumps(output_data))
            param_index += 1
            
        if error is not None:
            update_fields.append(f"error = ${param_index}")
            values.append(error)
            param_index += 1
        
        query = f"""
            UPDATE function_executions 
            SET {', '.join(update_fields)}
            WHERE execution_id = $1
        """
        
        try:
            adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(query, values)
            await self.db.execute(adapted_query, *adapted_params)
            logger.debug(f"[{self.context}] Updated function execution {execution_id} status to {status}")
        except Exception as e:
            logger.error(f"[{self.context}] Failed to update function execution status: {str(e)}")

    async def _update_activation_status(self, activation_id: str, status: str, 
                                      output_data: Optional[Dict[str, Any]] = None,
                                      error: Optional[str] = None,
                                      execution_time_ms: Optional[int] = None) -> bool:
        """
        Update activation status in database with comprehensive status tracking.
        
        This method handles all activation status updates including:
        - Status changes (queued → running → completed/failed)
        - Output data storage (JSON serialized)
        - Error message recording
        - Execution timing tracking
        - Timestamp management (started_at, completed_at, updated_at)
        - Notification callbacks for real-time updates
        
        Args:
            activation_id: UUID of the activation to update
            status: New status ('queued', 'running', 'completed', 'failed')
            output_data: Optional result data to store (will be JSON serialized)
            error: Optional error message for failed activations
            execution_time_ms: Optional execution duration in milliseconds
            
        Returns:
            bool: True if update successful, False if failed
            
        Side Effects:
            - Updates database record
            - Triggers notification callback for completed/failed status
            - Logs status changes and notifications
        """
        try:
            # Build update query dynamically
            set_clauses = ["status = $1", "updated_at = NOW()"]
            values = [status]
            param_index = 2
            
            if output_data is not None:
                set_clauses.append(f"output_data = ${param_index}")
                values.append(json.dumps(output_data))
                param_index += 1
            
            if error is not None:
                set_clauses.append(f"error = ${param_index}")
                values.append(error)
                param_index += 1
            
            if execution_time_ms is not None:
                set_clauses.append(f"duration_ms = ${param_index}")
                values.append(execution_time_ms)
                param_index += 1
            
            if status in ('completed', 'failed'):
                set_clauses.append("completed_at = NOW()")
            elif status == 'running':
                set_clauses.append("started_at = NOW()")
            
            query = f"""
                UPDATE agent_activations 
                SET {', '.join(set_clauses)}
                WHERE activation_id = ${param_index}
            """
            values.append(activation_id)
            
            adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(query, values)
            await self.db.execute(adapted_query, *adapted_params)
            
            # Send notification if callback is set and activation is completed
            if self._notification_callback and status in ('completed', 'failed'):
                try:
                    logger.info(f"[{self.context}] 🔔 NOTIFICATION CALLBACK TRIGGERED on PROCESSOR [{self.instance_id}]! activation_id={activation_id}, status={status}")
                    
                    # Get activation details to find app_id
                    activation = await self._get_activation(activation_id)
                    
                    logger.info(f"[{self.context}] 📋 Activation metadata: {activation.get('metadata', {})}")
                    logger.info(f"[{self.context}] 📋 Activation context: {activation.get('context', {})}")
                    
                    # Use centralized method to get app_id
                    try:
                        app_id = await self._extract_app_id_from_activation(activation)
                        logger.info(f"[{self.context}] ✅ Found app_id: {app_id}")
                    except Exception as e:
                        logger.error(f"[{self.context}] ❌ Failed to extract app_id: {e}")
                        app_id = None
                    
                    logger.info(f"[{self.context}] 🎯 Final extracted app_id: {app_id}")
                    
                    if app_id:
                        # Call notification callback
                        logger.info(f"[{self.context}] 📞 CALLING NOTIFICATION CALLBACK for app_id: {app_id}")
                        await self._notification_callback(activation_id, status, app_id)
                        logger.info(f"[{self.context}] ✅ NOTIFICATION CALLBACK COMPLETED")
                    else:
                        logger.warning(f"[{self.context}] ❌ No app_id found in activation metadata/context, skipping notification")
                        
                except Exception as e:
                    logger.error(f"[{self.context}] 💥 Error calling notification callback: {str(e)}", exc_info=True)
            elif not self._notification_callback:
                logger.warning(f"[{self.context}] ⚠️ No notification callback set on PROCESSOR [{self.instance_id}] - WebSocket broadcasting not configured!")
            else:
                logger.debug(f"[{self.context}] Status {status} doesn't trigger notifications (only 'completed' and 'failed' do)")
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.context}] Error updating activation {activation_id}: {str(e)}")
            return False
        try:
            # Build update query dynamically
            set_clauses = ["status = $1", "updated_at = NOW()"]
            values = [status]
            param_index = 2
            
            if output_data is not None:
                set_clauses.append(f"output_data = ${param_index}")
                values.append(json.dumps(output_data))
                param_index += 1
            
            if error is not None:
                set_clauses.append(f"error = ${param_index}")
                values.append(error)
                param_index += 1
            
            if execution_time_ms is not None:
                set_clauses.append(f"duration_ms = ${param_index}")
                values.append(execution_time_ms)
                param_index += 1
            
            if status in ('completed', 'failed'):
                set_clauses.append("completed_at = NOW()")
            elif status == 'running':
                set_clauses.append("started_at = NOW()")
            
            query = f"""
                UPDATE agent_activations 
                SET {', '.join(set_clauses)}
                WHERE activation_id = ${param_index}
            """
            values.append(activation_id)
            
            adapted_query, adapted_params = self.query_adapter.adapt_query_and_params(query, values)
            await self.db.execute(adapted_query, *adapted_params)
            
            # Send notification if callback is set and activation is completed
            if self._notification_callback and status in ('completed', 'failed'):
                try:
                    logger.info(f"[{self.context}] 🔔 NOTIFICATION CALLBACK TRIGGERED on PROCESSOR [{self.instance_id}]! activation_id={activation_id}, status={status}")
                    
                    # Get activation details to find app_id
                    activation = await self._get_activation(activation_id)
                    
                    logger.info(f"[{self.context}] 📋 Activation metadata: {activation.get('metadata', {})}")
                    logger.info(f"[{self.context}] 📋 Activation context: {activation.get('context', {})}")
                    
                    # Use centralized method to get app_id
                    try:
                        app_id = await self._extract_app_id_from_activation(activation)
                        logger.info(f"[{self.context}] ✅ Found app_id: {app_id}")
                    except Exception as e:
                        logger.error(f"[{self.context}] ❌ Failed to extract app_id: {e}")
                        app_id = None
                    
                    logger.info(f"[{self.context}] 🎯 Final extracted app_id: {app_id}")
                    
                    if app_id:
                        # Call notification callback
                        logger.info(f"[{self.context}] 📞 CALLING NOTIFICATION CALLBACK for app_id: {app_id}")
                        await self._notification_callback(activation_id, status, app_id)
                        logger.info(f"[{self.context}] ✅ NOTIFICATION CALLBACK COMPLETED")
                    else:
                        logger.warning(f"[{self.context}] ❌ No app_id found in activation metadata/context, skipping notification")
                        
                except Exception as e:
                    logger.error(f"[{self.context}] 💥 Error calling notification callback: {str(e)}", exc_info=True)
            elif not self._notification_callback:
                logger.warning(f"[{self.context}] ⚠️ No notification callback set on PROCESSOR [{self.instance_id}] - WebSocket broadcasting not configured!")
            else:
                logger.debug(f"[{self.context}] Status {status} doesn't trigger notifications (only 'completed' and 'failed' do)")
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.context}] Error updating activation {activation_id}: {str(e)}")
            return False
