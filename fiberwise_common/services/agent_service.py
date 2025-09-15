"""
Agent Service - moved from core-web to fiberwise_common
Adapted to work with the common database providers and dynamic agents.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from .base_service import BaseService, ServiceError, NotFoundError, ValidationError
from ..database.query_adapter import QueryAdapter, ParameterStyle

logger = logging.getLogger(__name__)


class AgentService(BaseService):
    """
    Service for managing AI agents and their operations.
    Works with dynamic agents stored in the database.
    """
    
    def __init__(self, db_provider):
        """
        Initialize with a database provider.
        
        Args:
            db_provider: Database provider (SQLiteProvider, PostgreSQLProvider, etc.)
        """
        super().__init__(db_provider)
        # Initialize query adapter for SQLite (since we're using SQLite)
        self.query_adapter = QueryAdapter(ParameterStyle.SQLITE)

    async def get_agents(
        self,
        agent_type: Optional[str] = None,
        is_active: Optional[bool] = True,
        created_by: Optional[int] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all agents with filtering, formatted for AgentSummary schema
        
        Args:
            agent_type: Filter by agent type
            is_active: Filter by active status
            created_by: Filter by creator user ID
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of agent records formatted for AgentSummary schema
        """
        # Join with agent_types to get version, is_system, and capabilities
        query_parts = ["""
            SELECT 
                a.agent_id,
                a.name,
                a.description,
                a.agent_type_id,
                a.is_active,
                a.created_by,
                a.created_at,
                a.updated_at,
                COALESCE(at.version, '1.0.0') as version,
                COALESCE(at.is_system, 0) as is_system,
                COALESCE(at.capabilities, '[]') as capabilities
            FROM agents a
            LEFT JOIN agent_types at ON a.agent_type_id = at.id
            WHERE 1=1
        """]
        params = []
        
        if agent_type:
            query_parts.append("AND a.agent_type_id = ?")
            params.append(agent_type)
        
        if is_active is not None:
            query_parts.append("AND a.is_active = ?")
            params.append(1 if is_active else 0)
        
        if created_by:
            query_parts.append("AND a.created_by = ?")
            params.append(created_by)
        
        query_parts.append("ORDER BY a.name ASC")
        query_parts.append(f"LIMIT {limit} OFFSET {offset}")
        
        query = " ".join(query_parts)
        agents = await self.db.fetchall(query, *params)
        
        # Process results to match AgentSummary schema
        result = []
        for agent in agents:
            agent_dict = dict(agent)
            
            # Parse capabilities from JSON string if needed
            capabilities = agent_dict.get('capabilities', '[]')
            if isinstance(capabilities, str):
                try:
                    capabilities = json.loads(capabilities)
                    # Ensure it's a list, convert dict to empty list
                    if isinstance(capabilities, dict):
                        capabilities = []
                except (json.JSONDecodeError, TypeError):
                    capabilities = []
            elif isinstance(capabilities, dict):
                # If capabilities is a dict, convert to empty list
                capabilities = []
            elif not isinstance(capabilities, list):
                capabilities = []
            
            # Format for AgentSummary schema
            formatted_agent = {
                'id': str(agent_dict['agent_id']),  # Map agent_id to id
                'name': agent_dict['name'],
                'version': agent_dict['version'],
                'description': agent_dict.get('description'),
                'is_system': bool(agent_dict.get('is_system', False)),
                'capabilities': capabilities
            }
            
            result.append(formatted_agent)
        
        return result

    async def get_agent_by_id(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific agent by ID
        
        Args:
            agent_id: The ID of the agent to retrieve
            
        Returns:
            Agent record or None if not found
        """
        query = "SELECT * FROM agents WHERE agent_id = ?"
        agent = await self.db.fetchone(query, agent_id)
        
        if not agent:
            return None
        
        agent_dict = dict(agent)
        
        # Parse JSON fields
        for field in ['input_schema', 'output_schema']:
            if agent_dict.get(field):
                try:
                    if isinstance(agent_dict[field], str):
                        agent_dict[field] = json.loads(agent_dict[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        
        # Parse required_services array
        if agent_dict.get('required_services'):
            try:
                if isinstance(agent_dict['required_services'], str):
                    agent_dict['required_services'] = json.loads(agent_dict['required_services'])
            except (json.JSONDecodeError, TypeError):
                pass
        
        return agent_dict

    async def create_agent(
        self,
        agent_data: Dict[str, Any],
        created_by: int
    ) -> Dict[str, Any]:
        """
        Create a new dynamic agent
        
        Args:
            agent_data: Dict containing agent details
            created_by: ID of the user creating the agent
            
        Returns:
            Created agent record
        """
        # Generate agent_id if not provided
        agent_id = agent_data.get('agent_id', str(uuid.uuid4()))
        
        # Prepare JSON fields
        config = json.dumps({
            'input_schema': agent_data.get('input_schema', {}),
            'output_schema': agent_data.get('output_schema', {}),
            'required_services': agent_data.get('required_services', [])
        })
        capabilities = json.dumps(agent_data.get('capabilities', []))
        
        query = """
            INSERT INTO agents (
                agent_id, name, description, agent_type_id, agent_code,
                type, config, capabilities,
                is_active, created_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        now = datetime.now().isoformat()
        
        await self.db.execute(query, (
            agent_id,
            agent_data.get('name'),
            agent_data.get('description', ''),
            agent_data.get('agent_type_id', 'function'),
            agent_data.get('code', agent_data.get('agent_code', '')),
            agent_data.get('type', 'function'),
            config,
            capabilities,
            1 if agent_data.get('is_active', True) else 0,
            created_by,
            now,
            now
        ))
        
        return await self.get_agent_by_id(agent_id)

    async def update_agent(
        self,
        agent_id: str,
        update_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update an agent
        
        Args:
            agent_id: ID of the agent to update
            update_data: Dict containing fields to update
            
        Returns:
            Updated agent record or None if not found
        """
        # Build dynamic update query
        update_fields = []
        params = []
        
        for field in ['name', 'description', 'agent_type_id', 'agent_code', 'type', 'is_active']:
            if field in update_data:
                update_fields.append(f"{field} = ?")
                value = update_data[field]
                
                # Handle boolean for SQLite
                if field == 'is_active' and isinstance(value, bool):
                    value = 1 if value else 0
                
                params.append(value)
        
        # Handle JSON fields - store in config and capabilities
        json_updates = {}
        for field in ['input_schema', 'output_schema', 'required_services']:
            if field in update_data:
                json_updates[field] = update_data[field]
        
        if 'capabilities' in update_data:
            update_fields.append("capabilities = ?")
            params.append(json.dumps(update_data['capabilities']))
        
        if json_updates or 'config' in update_data:
            # Merge JSON updates into config
            current_agent = await self.get_agent_by_id(agent_id)
            if current_agent:
                current_config = json.loads(current_agent.get('config', '{}'))
                current_config.update(json_updates)
                if 'config' in update_data:
                    current_config.update(update_data['config'])
                update_fields.append("config = ?")
                params.append(json.dumps(current_config))
        
        if not update_fields:
            return await self.get_agent_by_id(agent_id)
        
        # Add updated_at and agent_id
        update_fields.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(agent_id)
        
        query = f"""
            UPDATE agents 
            SET {', '.join(update_fields)}
            WHERE agent_id = ?
        """
        
        await self.db.execute(query, *params)
        
        return await self.get_agent_by_id(agent_id)

    async def delete_agent(self, agent_id: str) -> bool:
        """
        Delete an agent
        
        Args:
            agent_id: ID of the agent to delete
            
        Returns:
            True if deleted, False if not found
        """
        # Check if agent exists
        existing = await self.get_agent_by_id(agent_id)
        if not existing:
            return False
        
        # Delete agent (activations will be handled by foreign key constraints)
        query = "DELETE FROM agents WHERE agent_id = ?"
        await self.db.execute(query, (agent_id,))
        
        return True

    async def execute_agent(
        self,
        agent_id: str,
        input_data: Dict[str, Any],
        created_by: int
    ) -> Dict[str, Any]:
        """
        Execute an agent and return the result
        
        Args:
            agent_id: ID of the agent to execute
            input_data: Input data for the agent
            created_by: ID of the user executing the agent
            
        Returns:
            Execution result with activation details
        """
        # Get the agent
        agent = await self.get_agent_by_id(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")
        
        if not agent.get('is_active'):
            raise ValueError(f"Agent is disabled: {agent_id}")
        
        # Create activation record
        activation_data = {
            'agent_id': agent_id,
            'agent_type_id': agent.get('agent_type_id', 'default'),
            'status': 'running',
            'input_data': input_data
        }
        
        activation = await self.create_activation(activation_data, str(created_by))
        activation_id = activation['activation_id']
        
        try:
            # Handle different agent types
            if agent.get('agent_type_id') == 'llm':
                # Use LLM provider for LLM agents
                result, error = await self._execute_llm_agent(agent, input_data, created_by)
            else:
                # Execute code for custom agents
                result, error = await self._execute_agent_code(agent['code'], input_data)
            
            # Update activation with result
            if error:
                await self.update_activation(activation_id, {
                    'status': 'failed',
                    'error': error,
                    'completed_at': datetime.now().isoformat()
                })
                
                # Fetch the complete activation record from database
                activation_record = await self._get_activation_by_id(activation_id)
                return activation_record
            else:
                await self.update_activation(activation_id, {
                    'status': 'completed',
                    'output_data': result,
                    'completed_at': datetime.now().isoformat()
                })
                
                # Fetch the complete activation record from database
                activation_record = await self._get_activation_by_id(activation_id)
                return activation_record
                
        except Exception as e:
            # Update activation with error
            await self.update_activation(activation_id, {
                'status': 'failed',
                'error': str(e),
                'completed_at': datetime.now().isoformat()
            })
            
            # Fetch the complete activation record from database
            activation_record = await self._get_activation_by_id(activation_id)
            return activation_record

    async def _get_activation_by_id(self, activation_id: str) -> Dict[str, Any]:
        """Get a complete activation record by ID."""
        query = """SELECT activation_id, agent_id, agent_type_id, status, started_at,
                          completed_at, duration_ms, input_data, output_data, context,
                          metadata, created_by, organization_id, created_at FROM agent_activations WHERE activation_id = ?"""
        
        try:
            rows = await self.db.fetch_all(query, activation_id)
            if not rows:
                return None
            
            row_dict = dict(rows[0])
            
            # Parse JSON string fields to dictionaries
            for json_field in ['input_data', 'output_data', 'context', 'metadata']:
                if json_field in row_dict and row_dict[json_field]:
                    try:
                        if isinstance(row_dict[json_field], str):
                            row_dict[json_field] = json.loads(row_dict[json_field])
                    except json.JSONDecodeError:
                        row_dict[json_field] = {}
                else:
                    row_dict[json_field] = {}
            
            return row_dict
            
        except Exception as e:
            logger.error(f"Error fetching activation {activation_id}: {e}")
            return None

    async def _execute_llm_agent(self, agent: Dict[str, Any], activation_request: Dict[str, Any], created_by: int):
        """
        Execute LLM agent using provider service
        
        Args:
            agent: Agent record with LLM configuration  
            activation_request: Full activation request with input, context, metadata
            created_by: User ID for LLM provider service context
            
        Returns:
            Tuple of (result, error)
        """
        try:
            from .llm_provider_service import LLMProviderService
            from .llm_service_factory import LLMServiceFactory
            
            # Extract data from activation request structure
            input_data = activation_request.get('input', {})
            context = activation_request.get('context', {})
            metadata = activation_request.get('metadata', {})
            
            # Get the user prompt
            user_prompt = input_data.get('prompt', '')
            if not user_prompt:
                return None, "No prompt provided"
            
            # Get system prompt from context or agent
            system_prompt = context.get('system_prompt') or agent.get('system_prompt', 'You are a helpful AI assistant.')
            full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            
            # Get provider_id from context or metadata
            provider_id = context.get('provider_id') or metadata.get('provider_id')
            
            # Get parameters from metadata or agent defaults
            temperature = metadata.get('temperature') or agent.get('temperature', 0.7)
            max_tokens = metadata.get('max_tokens') or agent.get('max_tokens', 2048)
            model = agent.get('model')  # Let provider handle default model
            
            # Create LLM service instance with factory and user context
            llm_service = LLMProviderService(self.db, user_id=created_by, llm_service_factory=LLMServiceFactory())
            
            # Execute the LLM request - only pass model if it exists
            request_params = {
                'prompt': full_prompt,
                'provider_id': provider_id,
                'temperature': temperature,
                'max_tokens': max_tokens
            }
            if model:
                request_params['model'] = model
            
            response = await llm_service.generate_completion(**request_params)
            
            if response.get('error'):
                return None, response['error']
            
            # Debug: log the full response to see the structure
            logger.info(f"LLM response structure: {response}")
            
            # Try different field names for the response content
            content = (response.get('text') or 
                      response.get('content') or 
                      response.get('message') or 
                      response.get('response') or 
                      response.get('output', ''))
            
            return {'response': content}, None
            
        except Exception as e:
            logger.error(f"Error executing LLM agent: {e}")
            return None, str(e)

    async def _execute_agent_code(self, agent_code: str, input_data: Dict[str, Any]):
        """
        Execute agent code safely
        
        Args:
            agent_code: Python code to execute
            input_data: Input data for the agent
            
        Returns:
            Tuple of (result, error)
        """
        try:
            # Create a safe execution environment
            namespace = {
                '__builtins__': {
                    'len': len,
                    'str': str,
                    'int': int, 
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'sorted': sorted,
                    'print': print,
                    '__import__': __import__,  # Allow imports
                }
            }
            
            # Allow datetime import for agents
            import datetime
            namespace['datetime'] = datetime
            
            # Execute the agent code
            exec(agent_code, namespace)
            
            # Call the run function
            if 'run' in namespace:
                result = namespace['run'](input_data)
                return result, None
            else:
                return None, "Agent code must define a 'run' function"
                
        except Exception as e:
            return None, str(e)

    async def get_agent_activations(
        self, 
        agent_id: str, 
        created_by: int,
        limit: int = 10,
        context_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get activations for a specific agent filtered by the user who created them
        
        Args:
            agent_id: ID of the agent
            created_by: ID of the user who created the activations (required)
            limit: Maximum number of results
            context_filter: Dict to filter activations by context fields (e.g., {"chat_id": "123"})
            
        Returns:
            List of activation records
        """
        # Build context filter clause if provided
        context_clause = ""
        context_params = []
        
        if context_filter:
            context_conditions = []
            for key, value in context_filter.items():
                # Filter on context.{key} using JSON extraction
                context_conditions.append(f"JSON_EXTRACT(context, '$.{key}') = ?")
                context_params.append(str(value))
            
            if context_conditions:
                context_clause = " AND " + " AND ".join(context_conditions)
        
        # Get activations for specific agent filtered by created_by
        query = f"""
        SELECT activation_id, agent_id, agent_type_id, status, started_at, 
               completed_at, duration_ms, input_data, output_data, context, 
               metadata, created_by, created_at FROM agent_activations 
        WHERE agent_id = ? AND created_by = ?{context_clause}
        ORDER BY created_at DESC 
        LIMIT ?
        """
        
        params = [str(agent_id), created_by] + context_params + [limit]
        
        try:
            logger.debug(f"Query: {query}")
            logger.debug(f"Params: {params} (count: {len(params)})")
            # Convert params list to tuple for database execution
            params_tuple = tuple(params)
            rows = await self.db.fetch_all(query, *params_tuple)
            
            # Convert rows to dicts and parse JSON fields
            result = []
            for row in rows:
                row_dict = dict(row)
                
                # Parse JSON string fields to dictionaries
                for json_field in ['input_data', 'output_data', 'context', 'metadata']:
                    if json_field in row_dict and row_dict[json_field]:
                        try:
                            if isinstance(row_dict[json_field], str):
                                row_dict[json_field] = json.loads(row_dict[json_field])
                        except json.JSONDecodeError:
                            # If JSON parsing fails, keep as empty dict
                            row_dict[json_field] = {}
                    else:
                        # Ensure field exists as empty dict if missing
                        row_dict[json_field] = {}
                
                result.append(row_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching activations for agent {agent_id}: {e}")
            logger.error(f"Query was: {query}")
            logger.error(f"Params were: {params} (count: {len(params)})")
            return []

    async def get_agent_activations_by_user(
        self,
        agent_id: str,
        user_id: int,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get activations for a specific agent filtered by user
        
        Args:
            agent_id: ID of the agent
            user_id: ID of the user to filter by
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of activation records for the current user
        """
        query = """
        SELECT activation_id, agent_id, agent_type_id, status, started_at, 
               completed_at, duration_ms, input_data, output_data, context, 
               metadata, created_by, created_at FROM agent_activations 
        WHERE agent_id = ? AND created_by = ?
        ORDER BY created_at DESC 
        LIMIT ? OFFSET ?
        """
        
        try:
            # Ensure parameters are correct types
            safe_limit = int(limit) if limit is not None else 10
            safe_offset = int(offset) if offset is not None else 0
            params = (str(agent_id), str(user_id), safe_limit, safe_offset)
            
            logger.debug(f"Fetching activations for agent {agent_id} by user {user_id}")
            logger.debug(f"Query: {query}")
            logger.debug(f"Params: {params}")
            
            rows = await self.db.fetch_all(query, *params)
            
            # Convert rows to dicts and parse JSON fields
            result = []
            for row in rows:
                row_dict = dict(row)
                
                # Parse JSON string fields to dictionaries
                for json_field in ['input_data', 'output_data', 'context', 'metadata']:
                    if json_field in row_dict and row_dict[json_field]:
                        try:
                            if isinstance(row_dict[json_field], str):
                                row_dict[json_field] = json.loads(row_dict[json_field])
                        except json.JSONDecodeError:
                            # If JSON parsing fails, keep as empty dict
                            row_dict[json_field] = {}
                    else:
                        # Ensure field exists as empty dict if missing
                        row_dict[json_field] = {}
                
                result.append(row_dict)
            
            logger.info(f"Found {len(result)} activations for agent {agent_id} by user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching activations for agent {agent_id} by user {user_id}: {e}")
            return []

    async def get_app_activations_by_organization(
        self,
        app_id: str,
        organization_id: str,
        limit: int = 10,
        context_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get activations for an app filtered by organization ID for proper user isolation
        
        Args:
            app_id: ID of the app
            organization_id: ID of the organization
            limit: Maximum number of results
            context_filter: Optional context filter
            
        Returns:
            List of activation records for the organization's app installation
        """
        # Build context filter conditions
        context_conditions = []
        context_params = []
        
        if context_filter:
            for key, value in context_filter.items():
                if isinstance(value, str):
                    # Use JSON_EXTRACT for string values
                    context_conditions.append("JSON_EXTRACT(context, ?) = ?")
                    context_params.extend([f"$.{key}", value])
                elif isinstance(value, (int, float)):
                    # For numeric values
                    context_conditions.append("JSON_EXTRACT(context, ?) = ?")
                    context_params.extend([f"$.{key}", value])
        
        context_clause = ""
        if context_conditions:
            context_clause = " AND " + " AND ".join(context_conditions)
        
        query = f"""
        SELECT aa.activation_id, aa.agent_id, aa.agent_type_id, aa.status, aa.started_at, 
               aa.completed_at, aa.duration_ms, aa.input_data, aa.output_data, aa.context, 
               aa.metadata, aa.created_by, aa.created_at, aa.app_id
        FROM agent_activations aa
        JOIN app_installations ai ON aa.app_id = ai.app_id
        WHERE aa.app_id = ? AND ai.organization_id = ?{context_clause}
        ORDER BY aa.created_at DESC 
        LIMIT ?
        """
        
        params = [str(app_id), str(organization_id)] + context_params + [limit]
        
        try:
            logger.debug(f"Query: {query}")
            logger.debug(f"Params: {params} (count: {len(params)})")
            # Convert params list to tuple for database execution
            params_tuple = tuple(params)
            rows = await self.db.fetch_all(query, *params_tuple)
            
            # Convert rows to dicts and parse JSON fields
            result = []
            for row in rows:
                row_dict = dict(row)
                
                # Parse JSON string fields to dictionaries
                for json_field in ['input_data', 'output_data', 'context', 'metadata']:
                    if json_field in row_dict and row_dict[json_field]:
                        try:
                            if isinstance(row_dict[json_field], str):
                                row_dict[json_field] = json.loads(row_dict[json_field])
                        except json.JSONDecodeError:
                            # If JSON parsing fails, keep as empty dict
                            row_dict[json_field] = {}
                    else:
                        # Ensure field exists as empty dict if missing
                        row_dict[json_field] = {}
                
                result.append(row_dict)
            
            logger.info(f"Found {len(result)} activations for app {app_id} in organization {organization_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching activations for app {app_id} in organization {organization_id}: {e}")
            logger.error(f"Query was: {query}")
            logger.error(f"Params were: {params} (count: {len(params)})")
            return []

    async def get_activations_by_app_id(
        self,
        app_id: str,
        user_id: int,
        organization_id: str,
        limit: int = 10,
        context_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all activations created by user for a specific app with organization isolation
        
        Args:
            app_id: ID of the app
            user_id: ID of the user who created the activations
            organization_id: Organization ID for isolation
            limit: Maximum number of results
            context_filter: Optional context filter
            
        Returns:
            List of activation records created by the user for the app in the organization
        """
        # Build context filter conditions
        context_conditions = []
        context_params = []
        
        if context_filter:
            for key, value in context_filter.items():
                if isinstance(value, str):
                    context_conditions.append("JSON_EXTRACT(aa.context, ?) = ?")
                    context_params.extend([f"$.{key}", value])
                elif isinstance(value, (int, float)):
                    context_conditions.append("JSON_EXTRACT(aa.context, ?) = ?")
                    context_params.extend([f"$.{key}", value])
        
        context_clause = ""
        if context_conditions:
            context_clause = " AND " + " AND ".join(context_conditions)
        
        query = f"""
        SELECT aa.activation_id, aa.agent_id, aa.agent_type_id, aa.status, aa.started_at, 
               aa.completed_at, aa.duration_ms, aa.input_data, aa.output_data, aa.context, 
               aa.metadata, aa.created_by, aa.created_at
        FROM agent_activations aa
        JOIN app_installations ai ON ai.app_id = ?
        WHERE aa.created_by = ? AND ai.organization_id = ?{context_clause}
        ORDER BY aa.created_at DESC 
        LIMIT ?
        """
        
        params = [str(app_id), int(user_id), str(organization_id)] + context_params + [limit]
        
        try:
            logger.debug(f"Getting activations for app {app_id}, user {user_id}, org {organization_id}")
            logger.debug(f"Query: {query}")
            logger.debug(f"Params: {params}")
            
            rows = await self.db.fetch_all(query, *params)
            
            # Convert rows to dicts and parse JSON fields
            result = []
            for row in rows:
                row_dict = dict(row)
                
                # Parse JSON string fields to dictionaries
                for json_field in ['input_data', 'output_data', 'context', 'metadata']:
                    if json_field in row_dict and row_dict[json_field]:
                        try:
                            if isinstance(row_dict[json_field], str):
                                row_dict[json_field] = json.loads(row_dict[json_field])
                        except json.JSONDecodeError:
                            row_dict[json_field] = {}
                    else:
                        row_dict[json_field] = {}
                
                result.append(row_dict)
            
            logger.info(f"Found {len(result)} activations for app {app_id}, user {user_id}, org {organization_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching activations for app {app_id}, user {user_id}, org {organization_id}: {e}")
            return []

    async def get_activations_by_agent_id(
        self,
        agent_id: str,
        user_id: int,
        organization_id: str,
        limit: int = 10,
        context_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get activations created by user for a specific agent with organization isolation
        
        Args:
            agent_id: ID of the agent
            user_id: ID of the user who created the activations
            organization_id: Organization ID for isolation
            limit: Maximum number of results
            context_filter: Optional context filter
            
        Returns:
            List of activation records created by the user for the agent in the organization
        """
        # Build context filter conditions
        context_conditions = []
        context_params = []
        
        if context_filter:
            for key, value in context_filter.items():
                if isinstance(value, str):
                    context_conditions.append("JSON_EXTRACT(aa.context, ?) = ?")
                    context_params.extend([f"$.{key}", value])
                elif isinstance(value, (int, float)):
                    context_conditions.append("JSON_EXTRACT(aa.context, ?) = ?")
                    context_params.extend([f"$.{key}", value])
        
        context_clause = ""
        if context_conditions:
            context_clause = " AND " + " AND ".join(context_conditions)
        
        query = f"""
        SELECT aa.activation_id, aa.agent_id, aa.agent_type_id, aa.status, aa.started_at, 
               aa.completed_at, aa.duration_ms, aa.input_data, aa.output_data, aa.context, 
               aa.metadata, aa.created_by, aa.created_at
        FROM agent_activations aa
        JOIN app_installations ai ON ai.app_id = (
            SELECT app_id FROM agents WHERE agent_id = ?
        )
        WHERE aa.agent_id = ? AND aa.created_by = ? AND ai.organization_id = ?{context_clause}
        ORDER BY aa.created_at DESC 
        LIMIT ?
        """
        
        params = [str(agent_id), str(agent_id), int(user_id), str(organization_id)] + context_params + [limit]
        
        try:
            logger.debug(f"Getting activations for agent {agent_id}, user {user_id}, org {organization_id}")
            logger.debug(f"Query: {query}")
            logger.debug(f"Params: {params}")
            
            rows = await self.db.fetch_all(query, *params)
            
            # Convert rows to dicts and parse JSON fields
            result = []
            for row in rows:
                row_dict = dict(row)
                
                # Parse JSON string fields to dictionaries
                for json_field in ['input_data', 'output_data', 'context', 'metadata']:
                    if json_field in row_dict and row_dict[json_field]:
                        try:
                            if isinstance(row_dict[json_field], str):
                                row_dict[json_field] = json.loads(row_dict[json_field])
                        except json.JSONDecodeError:
                            row_dict[json_field] = {}
                    else:
                        row_dict[json_field] = {}
                
                result.append(row_dict)
            
            logger.info(f"Found {len(result)} activations for agent {agent_id}, user {user_id}, org {organization_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching activations for agent {agent_id}, user {user_id}, org {organization_id}: {e}")
            return []

    async def create_activation(self, activation_data: Dict[str, Any], created_by: str) -> Dict[str, Any]:
        """Create a new activation record."""
        activation_id = str(uuid.uuid4())
        
        query = """
        INSERT INTO agent_activations (
            activation_id, agent_id, agent_type_id, status, started_at,
            input_data, context, metadata, created_by, organization_id, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            started_at = datetime.now().isoformat()
            created_at = datetime.now().isoformat()

            organization_id = activation_data.get('organization_id')
            logger.info(f"Creating activation with organization_id: {organization_id} from activation_data: {activation_data}")

            params = (
                activation_id,
                activation_data['agent_id'],
                activation_data.get('agent_type_id', 'default'),
                activation_data['status'],
                started_at,
                json.dumps(activation_data.get('input_data', {})),  # Convert dict to JSON string for database
                json.dumps(activation_data.get('context', {})),    # Store context separately
                json.dumps(activation_data.get('metadata', {})),   # Store metadata separately
                created_by,
                organization_id,
                created_at
            )
            await self.db.execute(query, *params)
            
            # Return complete activation record for ActivationResponse validation
            return {
                'activation_id': activation_id,
                'agent_id': activation_data['agent_id'],
                'agent_type_id': activation_data.get('agent_type_id', 'default'),
                'status': activation_data['status'],
                'started_at': started_at,
                'created_by': created_by,
                'organization_id': activation_data.get('organization_id'),
                'created_at': created_at,
                'input_data': activation_data.get('input_data', {}),
                'completed_at': None,
                'duration_ms': None,
                'output_data': None,
                'input_summary': None,
                'output_summary': None,
                'error': None,
                'notes': None,
                'metadata': activation_data.get('metadata', {}),
                'context': activation_data.get('context', {}),
                'updated_at': created_at,
                'dependencies': None,
                'input_schema': None,
                'output_schema': None,
                'execution_metadata': None
            }
        except Exception as e:
            logger.error(f"Error creating activation: {e}")
            raise
    
    async def update_activation(self, activation_id: str, update_data: Dict[str, Any]) -> None:
        """Update an activation record."""
        set_clauses = []
        params = []
        
        for key, value in update_data.items():
            if key in ['output_data', 'error']:
                set_clauses.append(f"{key} = ?")
                params.append(json.dumps(value) if isinstance(value, (dict, list)) else value)
            else:
                set_clauses.append(f"{key} = ?")
                params.append(value)
        
        params.append(activation_id)
        
        query = f"UPDATE agent_activations SET {', '.join(set_clauses)} WHERE activation_id = ?"
        
        try:
            await self.db.execute(query, *params)
        except Exception as e:
            logger.error(f"Error updating activation {activation_id}: {e}")
            raise

    async def delete_activation(self, activation_id: str) -> bool:
        """
        Delete an activation record permanently.
        
        Args:
            activation_id: ID of the activation to delete
            
        Returns:
            True if deleted successfully
            
        Raises:
            ServiceError: If deletion fails
        """
        try:
            # Check if activation exists
            existing = await self._get_activation_by_id(activation_id)
            if not existing:
                return False
            
            # Delete the activation
            query = "DELETE FROM agent_activations WHERE activation_id = ?"
            await self.db.execute(query, activation_id)
            
            logger.info(f"Deleted activation {activation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting activation {activation_id}: {e}")
            raise ServiceError(f"Failed to delete activation: {e}")

    async def create_agent(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new agent.
        
        Args:
            agent_data: Agent data dictionary
            
        Returns:
            Created agent record
            
        Raises:
            ValidationError: If validation fails
        """
        # Validate required fields
        required_fields = ['name', 'app_id']
        for field in required_fields:
            if not agent_data.get(field):
                raise ValidationError(f"Field '{field}' is required")
        
        # Generate agent_id if not provided
        if not agent_data.get('agent_id'):
            agent_data['agent_id'] = str(uuid.uuid4())
        
        now = datetime.now().isoformat()
        
        query = """
            INSERT INTO agents (
                agent_id, app_id, name, description, version, capabilities,
                config, entrypoint_file, class_name, language, is_active,
                is_system, created_by, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        """
        
        converted_query = self.query_adapter.convert_query(query)
        await self.db.execute(
            converted_query,
            agent_data['agent_id'],
            agent_data['app_id'],
            agent_data['name'],
            agent_data.get('description'),
            agent_data.get('version', '1.0.0'),
            json.dumps(agent_data.get('capabilities', {})),
            json.dumps(agent_data.get('config', {})),
            agent_data.get('entrypoint_file'),
            agent_data.get('class_name'),
            agent_data.get('language', 'python'),
            agent_data.get('is_active', True),
            agent_data.get('is_system', False),
            agent_data.get('created_by'),
            now,
            now
        )
        
        # Return the created agent
        created_agent = await self.get_agent_by_id(agent_data['agent_id'])
        if not created_agent:
            raise ServiceError("Failed to create agent")
        
        return created_agent

    async def update_agent(self, agent_id: str, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing agent.
        
        Args:
            agent_id: Agent ID to update
            agent_data: Updated agent data
            
        Returns:
            Updated agent record
            
        Raises:
            NotFoundError: If agent not found
        """
        # Check if agent exists
        existing_agent = await self.get_agent_by_id(agent_id)
        if not existing_agent:
            raise NotFoundError(f"Agent with id {agent_id} not found")
        
        # Build update query dynamically
        update_fields = []
        params = []
        
        updateable_fields = [
            'name', 'description', 'version', 'capabilities', 'config',
            'entrypoint_file', 'class_name', 'language', 'is_active'
        ]
        
        for field in updateable_fields:
            if field in agent_data:
                if field in ['capabilities', 'config']:
                    update_fields.append(f"{field} = $1")
                    params.append(json.dumps(agent_data[field]))
                else:
                    update_fields.append(f"{field} = $1")
                    params.append(agent_data[field])
        
        if update_fields:
            update_fields.append("updated_at = $1")
            params.append(datetime.now().isoformat())
            params.append(agent_id)
            
            query = f"UPDATE agents SET {', '.join(update_fields)} WHERE agent_id = $1"
            converted_query = self.query_adapter.convert_query(query)
            await self.db.execute(converted_query, *params)
        
        return await self.get_agent_by_id(agent_id)

    async def delete_agent(self, agent_id: str) -> bool:
        """
        Soft delete an agent (set is_active = False).
        
        Args:
            agent_id: Agent ID to delete
            
        Returns:
            True if deleted successfully
            
        Raises:
            NotFoundError: If agent not found
        """
        # Check if agent exists
        existing_agent = await self.get_agent_by_id(agent_id)
        if not existing_agent:
            raise NotFoundError(f"Agent with id {agent_id} not found")
        
        query = "UPDATE agents SET is_active = $1, updated_at = $2 WHERE agent_id = $3"
        converted_query = self.query_adapter.convert_query(query)
        await self.db.execute(converted_query, False, datetime.now().isoformat(), agent_id)
        return True

    async def get_agents_by_app(self, app_id: str) -> List[Dict[str, Any]]:
        """
        Get all agents for a specific app.
        
        Args:
            app_id: App ID
            
        Returns:
            List of agent records
        """
        query = """
            SELECT agent_id, app_id, name, description, agent_type_id, config,
                   agent_code, is_active, is_enabled, created_by, created_at, updated_at,
                   checksum, input_schema, output_schema, dependencies
            FROM agents 
            WHERE app_id = $1 AND is_active = $2
            ORDER BY name
        """
        
        converted_query = self.query_adapter.convert_query(query)
        agents = await self.db.fetch_all(converted_query, app_id, True)
        
        # Process results to parse JSON fields
        result = []
        for agent in agents:
            agent_dict = dict(agent)
            
            # Parse JSON fields
            for field in ['config', 'input_schema', 'output_schema']:
                if agent_dict.get(field):
                    try:
                        if isinstance(agent_dict[field], str):
                            agent_dict[field] = json.loads(agent_dict[field])
                    except (json.JSONDecodeError, TypeError):
                        agent_dict[field] = {}
            
            result.append(agent_dict)
        
        return result

    async def search_agents(self, search_term: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search agents by name or description.
        
        Args:
            search_term: Search term
            limit: Maximum results
            
        Returns:
            List of matching agent records
        """
        query = """
            SELECT agent_id, app_id, name, description, version, capabilities,
                   config, entrypoint_file, class_name, language, is_active,
                   is_system, created_by, created_at, updated_at
            FROM agents 
            WHERE is_active = $1 
            AND (
                name LIKE $2 OR 
                description LIKE $2
            )
            ORDER BY name
            LIMIT $3
        """
        
        converted_query = self.query_adapter.convert_query(query)
        search_pattern = f"%{search_term}%"
        agents = await self.db.fetch_all(converted_query, True, search_pattern, limit)
        
        # Process results to parse JSON fields
        result = []
        for agent in agents:
            agent_dict = dict(agent)
            
            # Parse JSON fields
            for field in ['capabilities', 'config']:
                if agent_dict.get(field):
                    try:
                        if isinstance(agent_dict[field], str):
                            agent_dict[field] = json.loads(agent_dict[field])
                    except (json.JSONDecodeError, TypeError):
                        agent_dict[field] = {}
            
            result.append(agent_dict)
        
        return result

    async def get_activation_by_id(self, activation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific activation by ID.
        
        Args:
            activation_id: The activation ID to retrieve
            
        Returns:
            Activation record dict or None if not found
        """
        return await self._get_activation_by_id(activation_id)
