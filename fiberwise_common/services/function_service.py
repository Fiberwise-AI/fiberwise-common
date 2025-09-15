import uuid
import json
import logging
import asyncio
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import secrets

from pydantic import BaseModel, ValidationError

# Temporary fix for schema dependencies
from typing import Dict as User
from fiberwise_common.services import get_storage_provider
# from .agent_code_validator import detect_language_from_file  # Still in web project
from ..utils.language_utils import detect_language_with_fallback
from fiberwise_common import DatabaseProvider
from fiberwise_common.config import BaseWebSettings

# Initialize settings from common configuration
settings = BaseWebSettings()
from fastapi import HTTPException, status
# FiberApp import moved to avoid circular dependency
# OAuth services moved to fiberwise-common for shared use between functions and agents
from .agent_key_service import AgentKeyService

# Create compatibility functions
# OAuth service creation moved to fiberwise-common for shared use
import os
import shutil
logger = logging.getLogger(__name__)

class SupportAgentRequest(BaseModel):
    query: str
    project_id: str
    session_id: Optional[str] = None
    dynamic_inputs: Optional[Dict[str, str]] = None

class SupportAgentResponse(BaseModel):
    status: str
    results: Dict
    error: Optional[str] = None

class FunctionService:
    """Service for managing and executing functions in the system"""
    
    def __init__(self, db: DatabaseProvider):
        self.db = db

    async def get_functions(
        self, 
        search=None, 
        function_type=None, 
        limit=100, 
        offset=0
    ) -> List[Dict[str, Any]]:
        """Get functions with optional filtering"""
        query_parts = ["SELECT * FROM functions"]
        params = []
        param_index = 1
        
        # Build WHERE clause
        where_clauses = []
        if search:
            where_clauses.append(f"(name LIKE ? OR description LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])
        
        if function_type:
            where_clauses.append(f"function_type = ?")
            params.append(function_type)
        
        if where_clauses:
            query_parts.append("WHERE " + " AND ".join(where_clauses))
        
        # Add order and pagination
        query_parts.append("ORDER BY name ASC")
        query_parts.append(f"LIMIT {limit} OFFSET {offset}")
        
        query = " ".join(query_parts)
        
        try:
            result = await self.db.fetch_all(query, *params)
            functions = []
            for row in result:
                function_dict = dict(row)
                # Parse JSON strings back to dictionaries
                if function_dict.get('input_schema'):
                    try:
                        function_dict['input_schema'] = json.loads(function_dict['input_schema'])
                        logger.debug(f"Parsed input_schema: {type(function_dict['input_schema'])}")
                    except Exception as e:
                        logger.error(f"Error parsing input_schema: {e}")
                        function_dict['input_schema'] = {}
                if function_dict.get('output_schema'):
                    try:
                        function_dict['output_schema'] = json.loads(function_dict['output_schema'])
                        logger.debug(f"Parsed output_schema: {type(function_dict['output_schema'])}")
                    except Exception as e:
                        logger.error(f"Error parsing output_schema: {e}")
                        function_dict['output_schema'] = {}
                functions.append(function_dict)
            return functions
        except Exception as e:
            logger.error(f"Error fetching functions: {str(e)}")
            return []

    async def get_function_by_id(self, app_id: uuid, function_id: str) -> Optional[Dict[str, Any]]:
        """Get function by ID"""
        query = "SELECT * FROM functions WHERE function_id = ?"
        try:
            result = await self.db.fetch_one(query, function_id)
            if result:
                function_dict = dict(result)
                # Parse JSON strings back to dictionaries
                if function_dict.get('input_schema'):
                    try:
                        function_dict['input_schema'] = json.loads(function_dict['input_schema'])
                        logger.debug(f"Parsed input_schema: {type(function_dict['input_schema'])}")
                    except Exception as e:
                        logger.error(f"Error parsing input_schema: {e}")
                        function_dict['input_schema'] = {}
                if function_dict.get('output_schema'):
                    try:
                        function_dict['output_schema'] = json.loads(function_dict['output_schema'])
                        logger.debug(f"Parsed output_schema: {type(function_dict['output_schema'])}")
                    except Exception as e:
                        logger.error(f"Error parsing output_schema: {e}")
                        function_dict['output_schema'] = {}
                return function_dict
            return None
        except Exception as e:
            logger.error(f"Error fetching function {function_id}: {str(e)}")
            return None

    async def get_function_by_name(self, app_id: str, function_name: str) -> Optional[Dict[str, Any]]:
        """Get function by name"""
        query = """
            SELECT f.* 
            FROM functions f
            JOIN functions_app fa ON f.function_id = fa.function_id
            WHERE fa.app_id = $1 AND f.name = $2
        """
        try:
            result = await self.db.fetch_one(query, str(app_id), function_name)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error fetching function by name {function_name}: {str(e)}")
            return None

    async def create_function(self, function_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new function"""
        function_id = str(uuid.uuid4())
        
        # Validate input/output schemas
        try:
            # Validate that input_schema and output_schema are valid JSON
            input_schema = json.dumps(function_data.get("input_schema", {}))
            output_schema = json.dumps(function_data.get("output_schema", {}))
        except Exception as e:
            logger.error(f"Invalid JSON schema: {e}")
            raise ValueError("Invalid JSON schema provided")
        
        query = """
        INSERT INTO functions (
            function_id, name, description, function_type, 
            input_schema, output_schema, implementation,
            is_async, is_system
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """
        
        values = (
            function_id,
            function_data.get("name"),
            function_data.get("description", ""),
            function_data.get("function_type", "utility"),
            input_schema,
            output_schema,
            function_data.get("implementation", ""),
            function_data.get("is_async", False),
            function_data.get("is_system", False)
        )
        
        try:
            await self.db.execute(query, *values)
            # SQLite doesn't support RETURNING, so we need to fetch the created function
            result = await self.db.fetch_one("SELECT * FROM functions WHERE function_id = ?", function_id)
            if result:
                function_dict = dict(result)
                # Parse JSON strings back to dictionaries
                if function_dict.get('input_schema'):
                    try:
                        function_dict['input_schema'] = json.loads(function_dict['input_schema'])
                        logger.debug(f"Parsed input_schema: {type(function_dict['input_schema'])}")
                    except Exception as e:
                        logger.error(f"Error parsing input_schema: {e}")
                        function_dict['input_schema'] = {}
                if function_dict.get('output_schema'):
                    try:
                        function_dict['output_schema'] = json.loads(function_dict['output_schema'])
                        logger.debug(f"Parsed output_schema: {type(function_dict['output_schema'])}")
                    except Exception as e:
                        logger.error(f"Error parsing output_schema: {e}")
                        function_dict['output_schema'] = {}
                return function_dict
            return {}
        except Exception as e:
            logger.error(f"Error creating function: {str(e)}")
            raise

    async def update_function(self, function_id: str, function_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing function"""
        # Check if function exists
        existing = await self.get_function_by_id(None, function_id)
        if not existing:
            return None
        
        update_fields = []
        update_values = []
        param_index = 1
        
        # Only update fields that are provided
        field_mappings = {
            "name": "name",
            "description": "description",
            "function_type": "function_type",
            "implementation": "implementation",
            "is_async": "is_async"
        }
        
        for field, db_field in field_mappings.items():
            if field in function_data:
                update_fields.append(f"{db_field} = ${param_index}")
                update_values.append(function_data[field])
                param_index += 1
        
        # Handle JSON fields separately
        if "input_schema" in function_data:
            update_fields.append(f"input_schema = ${param_index}")
            update_values.append(json.dumps(function_data["input_schema"]))
            param_index += 1
        
        if "output_schema" in function_data:
            update_fields.append(f"output_schema = ${param_index}")
            update_values.append(json.dumps(function_data["output_schema"]))
            param_index += 1
        
        # Always update updated_at
        update_fields.append("updated_at = NOW()")
        
        query = f"""
        UPDATE functions
        SET {", ".join(update_fields)}
        WHERE function_id = ${param_index}
        RETURNING *
        """
        update_values.append(function_id)
        
        try:
            result = await self.db.fetch_one(query, *update_values)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error updating function {function_id}: {str(e)}")
            raise

    async def delete_function(self, function_id: str) -> bool:
        """Delete a function"""
        # Don't allow deleting system functions
        check_query = "SELECT is_system FROM functions WHERE function_id = ?"
        is_system = await self.db.fetch_val(check_query, function_id)
        if is_system:
            raise ValueError("Cannot delete system functions")
        
        query = "DELETE FROM functions WHERE function_id = ? AND is_system = 0"
        
        try:
            await self.db.execute(query, function_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting function {function_id}: {str(e)}")
            return False

    async def execute_function(self, app_id: str, function_id: str, input_data: Dict[str, Any], user: User, organization_id: int = None) -> Dict[str, Any]:
        """Execute a function with the provided input data"""
        # Check if function_id is a UUID or name
        function = None
        try:
            # First, try to parse as UUID
            uuid_obj = uuid.UUID(function_id)
            # If successful, it's a UUID - retrieve by ID
            function = await self.get_function_by_id(app_id, function_id)
        except ValueError:
            # Not a valid UUID, assume it's a function name
            function = await self.get_function_by_name(app_id, function_id)
            
        if not function:
            raise ValueError(f"Function '{function_id}' not found for app {app_id}")
        
        # Store actual function_id for the execution record
        actual_function_id = function["function_id"]
        return await self._execute_function_directly(app_id, actual_function_id, function, input_data, user, organization_id)

        # # Check if worker is enabled for async execution
        # if settings.WORKER_ENABLED:
        #     # Queue for worker processing
        #     return await self._queue_function_execution(app_id, actual_function_id, function["name"], input_data, user)
        # else:
        #     # Execute directly (CLI/local mode)
        #     return await self._execute_function_directly(app_id, actual_function_id, function, input_data, user)

    async def _queue_function_execution(self, app_id: str, function_id: str, function_name: str, input_data: Dict[str, Any], user: User) -> Dict[str, Any]:
        """Queue function for worker processing"""
        execution_id = str(uuid.uuid4())
        
        # Create execution record in 'queued' status
        exec_query = """
        INSERT INTO function_executions (
            execution_id, function_id, input_data, status, started_at, created_by
        ) VALUES (
            ?, ?, ?, ?, datetime('now'), ?
        )
        """
        
        created_by = user.id if user else None
        try:
            # Ensure input_data is properly serialized
            serialized_input = json.dumps(input_data)
            logger.info(f"Queuing function execution {execution_id} for function {function_name}")
            
            await self.db.execute(
                exec_query, 
                execution_id,
                function_id,
                serialized_input,
                "queued",
                created_by
            )
            
            logger.info(f"Function {function_name} queued successfully with execution ID {execution_id}")
            
            return {
                "execution_id": execution_id,
                "function_id": function_id,
                "function_name": function_name,
                "status": "queued",
                "message": f"Function '{function_name}' queued for execution",
                "started_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error queuing function execution: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to queue function execution: {str(e)}")

    async def _execute_function_directly(self, app_id: str, function_id: str, function: Dict[str, Any], input_data: Dict[str, Any], user: User, organization_id: int = None) -> Dict[str, Any]:
        """Execute function directly (local/CLI mode) - original logic"""
        
        # Create execution record
        execution_id = str(uuid.uuid4())
        
        # Check schema - function_executions table might not have app_id column
        # Correct query per schema from 0004_functions.sql
        exec_query = """
        INSERT INTO function_executions (
            execution_id, function_id, input_data, status, started_at, created_by
        ) VALUES (
            ?, ?, ?, ?, datetime('now'), ?
        )
        """
        
        created_by = user.id if user else None
        try:
            # Ensure input_data is properly serialized and stored as JSONB
            serialized_input = json.dumps(input_data)
            logger.info(f"Creating function execution record with input: {serialized_input[:100]}...")
            
            await self.db.execute(
                exec_query, 
                execution_id,                # execution_id
                function_id,                 # function_id
                serialized_input,            # input_data 
                "running",                   # status
                created_by                   # created_by
            )
            # Fetch the created execution record
            execution = await self.db.fetch_one("SELECT * FROM function_executions WHERE execution_id = ?", execution_id)
            execution_dict = dict(execution) if execution else {}
            logger.info(f"Created execution record: {execution_id}, data stored: {bool(execution_dict.get('input_data'))}")
        except Exception as e:
            logger.error(f"Error creating function execution record: {str(e)}", exc_info=True)
            # Continue without execution record if database operation fails
            execution_dict = {"started_at": datetime.now().isoformat()}
        
        # Execute function based on implementation method - use function_code table consistently
        try:
            # Check for active implementation in function_code table (both file and inline)
            impl_query = """
                SELECT implementation_type, file_path, content, storage_provider, language, name
                FROM function_code
                WHERE function_id = ? AND is_active = 1
                ORDER BY created_at DESC
                LIMIT 1
            """
            
            function_impl = await self.db.fetch_one(impl_query, function_id)
            
            if function_impl:
                impl_data = dict(function_impl)
                implementation_type = impl_data["implementation_type"]
                language = impl_data["language"]
                storage_provider = impl_data.get("storage_provider", "local")
                
                if implementation_type == "file":
                    # File-based implementation
                    file_path = impl_data["file_path"]
                    logger.info(f"Found file-based implementation for function {function_id}: {file_path} (storage: {storage_provider})")
                    
                    if language == "python":
                        result = await self._execute_python_file_from_storage(function_id, file_path, input_data, user.id, app_id, storage_provider, organization_id)
                    else:
                        raise ValueError(f"Execution of {language} functions not yet supported")
                        
                elif implementation_type == "inline":
                    # Inline implementation
                    content = impl_data["content"]
                    logger.info(f"Found inline implementation for function {function_id}")
                    
                    if not content:
                        raise ValueError("Inline implementation has no content")
                    
                    if language == "python":
                        result = await self._execute_python_code(content, input_data)
                    else:
                        raise ValueError(f"Execution of {language} functions not yet supported")
                        
                else:
                    raise ValueError(f"Unsupported implementation type: {implementation_type}")
                    
            else:
                # Fallback: check if function has inline implementation in legacy 'implementation' field
                implementation_code = function.get("implementation")
                
                if not implementation_code:
                    raise ValueError("No implementation found in function_code table or legacy implementation field")
                
                logger.info("Using legacy inline implementation from functions table")
                result = await self._execute_python_code(implementation_code, input_data)
            
            # Update execution record with success
            await self._update_execution(
                execution_id, 
                "completed", 
                result
            )
            
            return {
                "execution_id": execution_id,
                "function_id": function_id,
                "function_name": function["name"],
                "status": "completed",
                "result": result,
                "started_at": function["created_at"],
                "completed_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error executing function {function_id}: {str(e)}", exc_info=True)
            
            # Update execution record with error
            await self._update_execution(
                execution_id, 
                "failed", 
                None, 
                str(e)
            )
            
            return {
                "execution_id": execution_id,
                "function_id": function_id,
                "status": "failed",
                "error": str(e),
                "started_at": execution_dict.get("started_at", datetime.now().isoformat()),
                "completed_at": datetime.now().isoformat()
            }

    async def _update_execution(
        self,
        execution_id: str, 
        status: str, 
        result: Optional[Dict[str, Any]] = None, 
        error: Optional[str] = None
    ) -> None:
        """Update function execution record"""
        update_fields = ["status = ?", "completed_at = datetime('now')"]
        values = [status]
        
        if result is not None:
            # Ensure result is properly serialized as JSON
            update_fields.append("output_data = ?")
            serialized_result = json.dumps(result)
            values.append(serialized_result)
            logger.info(f"Updating execution with result: {serialized_result[:100]}...")
        
        if error is not None:
            update_fields.append("error = ?")
            values.append(error)
        
        # Add execution_id to the end for the WHERE clause
        values.append(execution_id)
        
        query = f"""
        UPDATE function_executions
        SET {", ".join(update_fields)}
        WHERE execution_id = ?
        """
        
        try:
            logger.info(f"Updating execution record {execution_id} with status {status}")
            result = await self.db.execute(query, *values)
            logger.info(f"Update result: {result}")
        except Exception as e:
            logger.error(f"Error updating execution record {execution_id}: {str(e)}", exc_info=True)

    @staticmethod
    async def _execute_python_code(code: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code with the given input data"""
        # SECURITY RISK: Executing arbitrary Python code is dangerous
        # In a production environment, use a sandbox or isolation mechanism
        
        # For now, we'll use a simple approach for demonstration purposes
        try:
            # Create a safe namespace
            namespace = {"input_data": input_data, "asyncio": asyncio, "json": json}
            
            # Add run function if not present
            if "async def run" not in code and "def run" not in code:
                code = f"""
async def run(input_data):
    # Default implementation
    return {{"result": "No implementation provided"}}

{code}
                """
            
            # Execute the code
            exec(code, namespace)
            
            # Call the run function
            run_func = namespace.get("run")
            if asyncio.iscoroutinefunction(run_func):
                result = await run_func(input_data)
            else:
                result = run_func(input_data)
            
            return result
        except Exception as e:
            logger.error(f"Error executing Python code: {str(e)}", exc_info=True)
            raise ValueError(f"Error in function implementation: {str(e)}")

    # Support Agent functionality
    @staticmethod
    async def run_support_agent(agent_details: Dict[str, Any], context_data: Dict[str, Any], context_type: str, notes=None) -> Dict[str, Any]:
        """
        Executes the support agent.
        
        Args:
            agent_details: Details about the agent, including configuration.
            context_data: Data from the project or notebook.
            context_type: Type of context ('project' or 'notebook').
            notes: WorkerNotes object for logging.
            
        Returns:
            A dictionary containing the results of the agent's execution.
        """
        logger.info(f"Starting support agent with agent_details: {agent_details}, context_type: {context_type}")
        
        try:
            # Extract relevant information from the arguments
            project_id = context_data.get('id') if context_type == 'project' else None
            parameters = agent_details.get('config', {})
            
            # Simulate agent processing
            await asyncio.sleep(2)  # Simulate work
            
            # Generate a response
            response = f"Generated support response for project {project_id} based on parameters: {parameters}"
            
            # Simulate token usage
            input_tokens = 120
            output_tokens = 180
            
            result = {
                "response": response,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
            
            logger.info(f"Support agent completed successfully for project: {project_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error running support agent for project {project_id}: {e}", exc_info=True)
            raise

    @staticmethod
    async def get_plugin_config(project_id: str) -> dict:
        """Get the plugin configuration for this project"""
        # In a real implementation, this would call Fiberwise's config service
        # For now, return default configuration
        return {
            "enable_guided_prompts": True,
            "enable_rag": True,
            "max_tokens": 1000,
            "temperature": 0.7
        }

    @staticmethod
    async def try_guided_prompt(request: SupportAgentRequest, notes=None) -> Optional[str]:
        """Attempt to match and execute a guided prompt"""
        # This is a simplified implementation for demonstration
        # A real implementation would search for project notebooks tagged as guided prompts
        
        # Example hardcoded patterns - in reality these would come from project notebooks
        patterns = [
            {"keywords": ["install", "setup", "start"], 
             "response": "To install the application, run `npm install` then `npm start`."},
            {"keywords": ["login", "sign in", "password"], 
             "response": "To reset your password, go to the Account page and click 'Reset Password'."},
            {"keywords": ["api", "documentation", "endpoints"], 
             "response": "API documentation can be found at https://docs.narratr.com/api"}
        ]
        
        query_lower = request.query.lower()
        
        for pattern in patterns:
            if any(keyword in query_lower for keyword in pattern["keywords"]):
                if notes:
                    await notes.add_reasoning(
                        f"Query matched guided prompt pattern with keywords: {', '.join(pattern['keywords'])}",
                        confidence=0.9,
                        tags=["pattern_match"]
                    )
                return pattern["response"]
        
        return None

    @staticmethod
    async def try_rag_search(request: SupportAgentRequest, notes=None) -> Optional[str]:
        """Attempt to answer using RAG over project documents"""
        # This is a simplified implementation for demonstration
        # A real implementation would:
        # 1. Generate query embedding
        # 2. Search vector DB for relevant chunks
        # 3. Construct LLM prompt with context
        # 4. Call LLM and return response
        
        # Simulate a RAG search result - in reality this would come from project documents
        if "api" in request.query.lower():
            if notes:
                await notes.add_reasoning(
                    "Found API documentation in project documents",
                    confidence=0.85,
                    tags=["document_match"]
                )
            return "The API endpoints for the project include: /users, /items, and /transactions. You can find detailed documentation at https://docs.narratr.com/api."
        
        if "error" in request.query.lower() and "code" in request.query.lower():
            if notes:
                await notes.add_reasoning(
                    "Found error code documentation in project knowledge base",
                    confidence=0.8,
                    tags=["document_match"]
                )
            return "Error codes starting with 4xx indicate client errors, while codes starting with 5xx indicate server errors. The most common error code 404 means the resource was not found."
        
        return None

    @staticmethod
    async def basic_llm_fallback(query: str) -> str:
        """Fallback response when no better option available"""
        # In a real implementation, this would call a LLM API
        return f"I understand you're asking about '{query}'. While I don't have specific project information about this, I'd suggest checking the project documentation or reaching out to the development team for more detailed assistance."

    @staticmethod
    async def _execute_support_agent(function: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a support agent function"""
        try:
            # Parse the request
            try:
                request = SupportAgentRequest(**input_data)
            except ValidationError as e:
                logger.error(f"Invalid input data for support agent: {e}")
                raise ValueError(f"Invalid input data: {e}")
                
            # Get plugin configuration
            config = await FunctionService.get_plugin_config(request.project_id)
            
            # Try guided prompt first if enabled
            if config.get("enable_guided_prompts", True):
                guided_response = await FunctionService.try_guided_prompt(request)
                if guided_response:
                    return {
                        "status": "success",
                        "results": {
                            "response": guided_response,
                            "source": "guided_prompt",
                            "confidence": 0.9
                        }
                    }
            
            # Try RAG if enabled and guided prompt didn't work
            if config.get("enable_rag", True):
                rag_response = await FunctionService.try_rag_search(request)
                if rag_response:
                    return {
                        "status": "success",
                        "results": {
                            "response": rag_response,
                            "source": "rag_search",
                            "confidence": 0.85
                        }
                    }
            
            # Fallback to basic LLM response
            fallback_response = await FunctionService.basic_llm_fallback(request.query)
            return {
                "status": "success",
                "results": {
                    "response": fallback_response,
                    "source": "llm_fallback",
                    "confidence": 0.6
                }
            }
                
        except Exception as e:
            logger.error(f"Error executing support agent: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "results": {}}

    
    async def execute_function_from_file(self, function_id: str, input_data: Dict[str, Any], user: User, app_id: uuid) -> Dict[str, Any]:
        """Execute a function using its file implementation"""
        # Get function implementation details
        query = """
            SELECT fc.file_path, fc.storage_provider, fc.language, fc.name,
                   f.name as function_name, f.function_type
            FROM function_code fc
            JOIN functions f ON fc.function_id = f.function_id
            WHERE fc.function_id = $1 AND fc.is_active = 1 AND fc.implementation_type = 'file'
            ORDER BY fc.created_at DESC
            LIMIT 1
        """
        
        try:
            function_impl = await self.db.fetch_one(query, function_id)
            
            if not function_impl:
                raise ValueError(f"No active implementation found for function {function_id}")
                
            impl_data = dict(function_impl)
            
            # Create execution record
            execution_id = str(uuid.uuid4())
            exec_query = """
            INSERT INTO function_executions (
                execution_id, function_id, input_data, status, started_at, created_by
            ) VALUES (
                $1, $2, $3::jsonb, $4, NOW(), $5
            ) RETURNING *
            """
            
            created_by = user.id if user else None
            serialized_input = json.dumps(input_data)
            
            execution = await self.db.fetch_one(
                exec_query, 
                execution_id,
                function_id,
                serialized_input,
                "running",
                created_by
            )
                        
            implementation_path = impl_data["implementation_path"]
            entrypoint_file = impl_data["entrypoint_file"]
            language = impl_data["language"]
            
            # Full path to the implementation directory
            full_path = os.path.join(settings.ENTITY_BUNDLES_DIR, implementation_path)
            
            # Path to the entry point file
            entry_point_path = os.path.join(full_path, entrypoint_file)
            
            if language == "python":
                # For Python, import and execute the module
                result = await self._execute_python_file(entry_point_path, input_data, user.id, app_id)
            else:
                # For other languages, not yet implemented
                raise ValueError(f"Execution of {language} functions not yet supported")
            
            # Update execution record with success
            await self._update_execution(
                execution_id, 
                "completed", 
                result
            )
            
            return {
                "execution_id": execution_id,
                "function_id": function_id,
                "function_name": impl_data["function_name"],
                "status": "completed",
                "result": result,
                "started_at": execution["started_at"],
                "completed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing function from file {function_id}: {str(e)}", exc_info=True)
            
            # Update execution record with error if it was created
            if locals().get('execution_id'):
                await self._update_execution(
                    execution_id, 
                    "failed", 
                    None, 
                    str(e)
                )
                
                return {
                    "execution_id": execution_id,
                    "function_id": function_id,
                    "status": "failed",
                    "error": str(e),
                    "started_at": locals().get('execution', {}).get("started_at", datetime.now().isoformat()),
                    "completed_at": datetime.now().isoformat()
                }
            
            raise ValueError(f"Error executing function from file: {str(e)}")
    
    async def _execute_python_file_from_storage(self, function_id: str, storage_path: str, input_data: Dict[str, Any], user_id, app_id, storage_provider_type: str = "local", organization_id: int = None) -> Dict[str, Any]:
        """Execute a Python file from storage provider"""
        try:
            # Get storage provider instance
            storage_provider = get_storage_provider()
            
            # Normalize path to use forward slashes consistently
            normalized_path = storage_path.replace("\\", "/")
            
            # If storage_path is already an absolute path, use it directly (legacy support)
            if os.path.isabs(storage_path):
                file_path = storage_path
            else:
                # This is a storage provider relative path - resolve based on provider type
                if storage_provider_type == "local" or await storage_provider.is_local():
                    file_path = os.path.join(settings.ENTITY_BUNDLES_DIR, normalized_path)
                else:
                    # For cloud storage, we would need to download the file first
                    # TODO: Implement cloud storage download logic
                    file_path = os.path.join(settings.ENTITY_BUNDLES_DIR, normalized_path)
            
            # Check if file_path is a directory containing Python files
            logger.info(f"DEBUG: Checking if {file_path} is a directory: {os.path.isdir(file_path)}")
            if os.path.isdir(file_path):
                logger.info(f"DEBUG: Directory found, listing contents...")
                # Find Python files in the directory
                python_files = []
                for file in os.listdir(file_path):
                    logger.info(f"DEBUG: Found file in directory: {file}")
                    if file.endswith('.py') and not file.startswith('__'):
                        python_files.append(file)
                
                logger.info(f"DEBUG: Found {len(python_files)} Python files: {python_files}")
                if len(python_files) == 0:
                    raise FileNotFoundError(f"No Python files found in directory: {file_path}")
                elif len(python_files) == 1:
                    # Use the single Python file
                    old_file_path = file_path
                    file_path = os.path.join(file_path, python_files[0])
                    logger.info(f"DEBUG: Changed path from {old_file_path} to {file_path}")
                else:
                    # Multiple Python files - look for main.py or __init__.py first
                    if 'main.py' in python_files:
                        file_path = os.path.join(file_path, 'main.py')
                        logger.info(f"Using main.py from {len(python_files)} Python files")
                    elif '__init__.py' in python_files:
                        file_path = os.path.join(file_path, '__init__.py')
                        logger.info(f"Using __init__.py from {len(python_files)} Python files")
                    else:
                        # Use the first alphabetically
                        python_files.sort()
                        file_path = os.path.join(file_path, python_files[0])
                        logger.info(f"Using {python_files[0]} (first alphabetically) from {len(python_files)} Python files")
            
            logger.info(f"Resolved storage path {storage_path} ({storage_provider_type}) to local path: {file_path}")
            return await self._execute_python_file(function_id, file_path, input_data, user_id, app_id, organization_id)
        except Exception as e:
            logger.error(f"Error executing Python file from storage {storage_path}: {str(e)}", exc_info=True)
            raise ValueError(f"Error in function implementation: {str(e)}")

    async def _execute_python_file(self, function_id: str, file_path: str, input_data: Dict[str, Any], user_id, app_id, organization_id: int = None) -> Dict[str, Any]:
        """Execute a Python file with the given input data"""
        try:
            import importlib.util
            import sys
            import os
            import inspect
            
            # Get the directory and file name
            directory = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            module_name = os.path.splitext(file_name)[0]
            
            # Add the directory to sys.path temporarily
            sys.path.insert(0, directory)
            
            try:
                # If the file is __init__.py, import the directory as a module
                if file_name == "__init__.py":
                    module_name = os.path.basename(directory)
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                else:
                    # Otherwise import the file as a module
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                
                if spec is None:
                    raise ImportError(f"Could not find module spec for {file_path}")
                
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                # Check if the module has a run function
                if not hasattr(module, 'run'):
                    raise ValueError(f"Module {module_name} does not have a run function")
                
                # Get the run function and inspect its parameters
                run_func = getattr(module, 'run')
                params = inspect.signature(run_func).parameters
                
                
                # Prepare dependencies based on function signature
                kwargs = await self._prepare_dependencies(function_id, params, input_data, user_id, app_id, organization_id)
                
                # Call the run function with dependencies
                if asyncio.iscoroutinefunction(run_func):
                    result = await run_func(input_data, **kwargs)
                else:
                    result = run_func(input_data, **kwargs)
                
                return result
            finally:
                # Remove the directory from sys.path
                if directory in sys.path:
                    sys.path.remove(directory)
                
                # Remove the module from sys.modules
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
        except Exception as e:
            logger.error(f"Error executing Python file {file_path}: {str(e)}", exc_info=True)
            raise ValueError(f"Error in function implementation: {str(e)}")
    
    async def _prepare_dependencies(self, function_id, params, input_data, user_id, app_id, organization_id=None):
        logger.info(f"DEBUG: _prepare_dependencies called with organization_id={organization_id}, user_id={user_id}, app_id={app_id}")
        """
        Prepare dependencies for function execution based on parameter inspection.
        
        Args:
            function_id: The ID of the function being executed
            params: Function parameters from inspect.signature
            input_data: The input data for the function
            user_id: Optional user ID for context
            app_id: Optional app ID for context
            
        Returns:
            Dict of kwargs to inject into the function
        """
        kwargs = {}
        
        # Skip the first param which is assumed to be input_data
        param_items = list(params.items())
        if len(param_items) <= 1:
            return kwargs  # No extra params needed
        
        # Initialize variable to track API key
        function_api_key = None
        
        # Extract user_id and app_id if not provided from input_data
        if user_id is None:
            user_id = input_data.get('user_id')
        if app_id is None:
            app_id = input_data.get('app_id')
            
        # Process remaining parameters to inject dependencies
        for name, param in param_items[1:]:
            # Check both parameter name and type annotation (if available)
            param_type = str(param.annotation) if param.annotation != param.empty else ""
            
            logger.info(f"Processing parameter: {name}, type: {param_type}")
            
            # Handle FiberWise SDK injection
            if name == 'fiber' or 'FiberApp' in param_type:
                try:
                    # Create function execution API key
                    if not function_api_key:
                        logger.info(f"DEBUG: Creating execution API key for app_id={app_id}, function_id={function_id}, user_id={user_id}")
                        function_api_key = await self._create_execution_api_key(
                            app_id=app_id,
                            function_id=function_id,
                            user_id=user_id,
                            organization_id=organization_id,
                            description="Function execution key",
                            scopes=["data:read", "data:write", "functions:activate"],
                            expiration_hours=1
                        )
                        logger.info(f"DEBUG: Created API key result: {function_api_key is not None}")
                    
                    # Import FiberWiseConfig for context-specific configuration
                    from fiberwise_sdk import FiberWiseConfig, FiberApp
                    
                    # Create context-specific config (same pattern as activation service)
                    context_config = FiberWiseConfig({
                        'app_id': app_id,  # Use the function's app_id
                        'api_key': function_api_key,  # Use the api_key as-is
                        'base_url': os.getenv('FIBER_API_BASE_URL', f"{settings.BASE_URL}/api/v1"),  # Use configured API base URL
                        'version': '1.0.0'
                    })
                    
                    fiber = FiberApp(context_config)
                    kwargs[name] = fiber
                    logger.info(f"Injected FiberWise SDK with execution API key and context (app_id: {app_id}, user_id: {user_id})")
                except Exception as e:
                    logger.error(f"Failed to create FiberWise SDK: {e}")
                    raise ValueError(f"Cannot execute function without FiberWise SDK: {e}")
            
            # Handle Credential Service injection
            elif name in ['cred_service', 'credentials'] or any(x in param_type for x in ['BaseCredentialService', 'CredentialAgentService']):
                try:
                    # Create API key for credential service if needed
                    cred_api_key = function_api_key
                    if not cred_api_key:
                        from .agent_key_service import AgentKeyService
                        agent_key_service = AgentKeyService(self.db)
                        cred_api_key = await agent_key_service.create_agent_key(
                            app_id=str(app_id),
                            agent_id=str(function_id),
                            description="Function credential API key",
                            scopes=["credentials:read", "credentials:write"],
                            expiration_hours=1,
                            resource_pattern="*",
                            created_by=str(user_id) if user_id else "system",
                            metadata={
                                "function_execution": True,
                                "function_id": str(function_id)
                            }
                        )
                    
                    # Use the common OAuth injection service for both functions and agents
                    from fiberwise_common.services.oauth_injection_service import create_oauth_credential_service_for_injection
                    
                    cred_service = await create_oauth_credential_service_for_injection(
                        db=self.db,
                        app_id=str(app_id),
                        user_id=user_id
                    )
                    
                    kwargs[name] = cred_service
                    logger.info(f"Injected OAuth Credential Service for app {app_id}, user {user_id}")
                except Exception as e:
                    logger.error(f"Failed to create Credential Service: {e}")
                    raise ValueError(f"Cannot execute function without Credential Service: {e}")
            
            # Handle LLM Provider Service injection
            elif name in ['llm_service', 'llm_provider_service'] or 'LLMProviderService' in param_type:
                try:
                    from fiberwise_sdk.llm_provider_service import create_llm_provider_service
                    
                    # Get API key (use the one from previous creations if available)
                    api_key = function_api_key
                    if not api_key:
                        # Create a short-lived key with appropriate scopes
                        api_key = await self._create_execution_api_key(
                            app_id=app_id,
                            function_id=function_id,
                            user_id=user_id,
                            description="Function LLM service key",
                            scopes=["llm:read", "llm:write"],
                            expiration_hours=1
                        )
                        function_api_key = api_key
                    
                    # Create a LLM provider service with the API key
                    llm_service = create_llm_provider_service(
                        providers={},  # Will be populated dynamically when used
                        config_options={
                            "base_url": settings.BASE_URL,
                            "api_key": api_key
                        }
                    )
                    
                    kwargs[name] = llm_service
                    logger.info(f"Injected LLM Provider Service with API key")
                except Exception as e:
                    logger.error(f"Failed to create LLM Provider Service: {e}")
                    raise ValueError(f"Cannot execute function without LLM Provider Service: {e}")
        
        return kwargs

    # Add new method to create execution API keys
    async def _create_execution_api_key(
        self,
        app_id: str,
        function_id: str,
        user_id: Optional[str] = None,
        organization_id: Optional[int] = None,
        description: str = "Function execution key",
        scopes: List[str] = None,
        expiration_hours: int = 1,
        resource_pattern: str = "*"
    ) -> Optional[str]:
        """
        Create a temporary API key for function execution using the ExecutionKeyService
        
        Args:
            app_id: App ID this key is associated with
            function_id: ID of the function using this key
            user_id: Optional user ID context
            description: Description of this key's purpose
            scopes: List of permission scopes
            expiration_hours: Hours until key expires
            resource_pattern: Resource pattern for access control
            
        Returns:
            The generated API key value if successful, None otherwise
        """
        try:
            # Import ExecutionKeyService
            from fiberwise_common.services import ExecutionKeyService
            
            # Create service instance
            execution_key_service = ExecutionKeyService(self.db)
            logger.info(f"DEBUG: ExecutionKeyService created successfully")
            
            # Default scopes if not provided
            if not scopes:
                scopes = ["data:read", "data:write"]
                
            # Convert metadata 
            metadata = {
                "function_execution": True,
                "function_id": str(function_id),
                "created_at": datetime.now().isoformat()
            }
            
            if user_id:
                metadata["user_id"] = str(user_id)
            
            # Use organization_id parameter or lookup from app installation
            if not organization_id:
                logger.info(f"DEBUG: No organization_id provided, looking up from app installation for app_id={app_id}, user_id={user_id}")
                # Lookup organization_id from app_installations table
                try:
                    query = """
                        SELECT ai.organization_id 
                        FROM app_installations ai
                        WHERE ai.app_id = $1 AND ai.user_id = $2 
                        AND ai.status = 'active'
                        LIMIT 1
                    """
                    from fiberwise_common.database.query_adapter import create_query_adapter, ParameterStyle
                    
                    # Convert query for current database type
                    db_type = getattr(self.db, 'provider_type', 'sqlite')
                    query_adapter = create_query_adapter(db_type)
                    converted_query, converted_params = query_adapter.adapt_query_and_params(
                        query, (str(app_id), user_id), ParameterStyle.POSTGRESQL
                    )
                    
                    result = await self.db.fetch_one(converted_query, *converted_params)
                    if result:
                        organization_id = result['organization_id']
                        logger.info(f"DEBUG: Found organization_id={organization_id} from app installation")
                    else:
                        logger.error(f"No app installation found for app_id={app_id}, user_id={user_id}")
                        return None
                except Exception as e:
                    logger.error(f"Error looking up organization_id: {str(e)}")
                    return None
            
            if not organization_id:
                logger.error(f"No organization_id available for execution key creation")
                return None
            
            # Create the execution key using the service
            logger.info(f"DEBUG: Creating execution key with params - app_id={app_id}, function_id={function_id}, user_id={user_id}, organization_id={organization_id}, scopes={scopes}")
            key_data = await execution_key_service.create_execution_key(
                app_id=str(app_id) if app_id else None,
                organization_id=organization_id,
                executor_type_id='function',
                executor_id=str(function_id),
                created_by=int(user_id) if user_id else None,
                scopes=scopes,
                expiration_minutes=expiration_hours * 60,  # Convert hours to minutes
                resource_pattern=resource_pattern,
                metadata=metadata
            )
            logger.info(f"DEBUG: Execution key creation result: {key_data}")
            
            if key_data:
                logger.info(f"Created execution API key {key_data['key_id']} for function {function_id}")
                return key_data['key_value']
            else:
                logger.error(f"Failed to create execution API key for function {function_id}")
                return None
            
        except Exception as e:
            logger.error(f"Error creating execution API key: {str(e)}")
            return None