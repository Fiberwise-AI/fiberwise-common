"""
Clean Function Processor - Immediate execution with async support

This processor handles immediate function execution with support for:
1. Synchronous functions - Direct execution
2. Asynchronous functions - HTTP calls, database queries, etc.

Key principles:
- Immediate execution (no background jobs)
- Support for both sync and async functions
- Clean dependency injection
- Proper error handling and logging
"""

import asyncio
import inspect
import json
import logging
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Callable

logger = logging.getLogger(__name__)


class FunctionProcessor:
    """
    Clean function processor for immediate execution.
    
    Handles immediate execution of functions with support for:
    - Sync functions (direct execution)
    - Async functions (HTTP calls, database queries, etc.)
    - Dependency injection
    - Proper error handling
    """
    
    def __init__(self, db_provider, context: str = "unknown", base_url: str = "http://localhost:5757/api/v1"):
        """
        Initialize the function processor.
        
        Args:
            db_provider: Database provider instance
            context: Context string for logging ("api", "cli", etc.)
            base_url: Base URL for API calls
        """
        self.db = db_provider
        self.context = context
        self.base_url = base_url
        self._injected_services = {}
        
        self.instance_id = str(uuid.uuid4())[:8]
        logger.info(f"🔧 FUNCTION PROCESSOR [{self.instance_id}] - Initialized with context: {context}")
    
    def inject_services(self, **services):
        """Inject services that will be available to functions."""
        self._injected_services.update(services)
        logger.info(f"[{self.context}] Injected services: {list(services.keys())}")
    
    async def execute_function(self, function_id: str, input_data: Dict[str, Any], 
                              app_id: str = None, user_id: int = None) -> Dict[str, Any]:
        """
        Execute a function immediately with the given input data.
        
        Args:
            function_id: Function ID to execute
            input_data: Input data for the function
            app_id: Optional app ID for context
            user_id: Optional user ID for context
            
        Returns:
            Dictionary with execution results
        """
        execution_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"[{self.context}] Executing function {function_id} - execution_id: {execution_id}")
        
        try:
            # Get function details
            function = await self._get_function(function_id)
            if not function:
                raise ValueError(f"Function {function_id} not found")
            
            logger.info(f"[{self.context}] Function name: {function.get('name')}")
            logger.info(f"[{self.context}] Function type: {function.get('function_type', 'unknown')}")
            
            # Execute the function
            result = await self._execute_function_code(function, input_data, app_id, user_id)
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            logger.info(f"[{self.context}] Function {function_id} completed in {execution_time_ms}ms")
            
            return {
                'execution_id': execution_id,
                'function_id': function_id,
                'status': 'completed',
                'result': result,
                'execution_time_ms': execution_time_ms,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            end_time = datetime.now()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            logger.error(f"[{self.context}] Function {function_id} failed after {execution_time_ms}ms: {e}")
            logger.error(traceback.format_exc())
            
            return {
                'execution_id': execution_id,
                'function_id': function_id,
                'status': 'failed',
                'error': str(e),
                'execution_time_ms': execution_time_ms,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_function_code(self, function: Dict[str, Any], input_data: Dict[str, Any],
                                   app_id: str = None, user_id: int = None) -> Any:
        """
        Execute the function code with proper dependency injection.
        
        Supports:
        1. Sync functions - Direct execution
        2. Async functions - HTTP calls, database operations, etc.
        """
        implementation = function.get('implementation', '')
        
        if not implementation:
            raise ValueError("Function has no implementation")
        
        try:
            # Create execution namespace with services
            namespace = {
                "input_data": input_data,
                "asyncio": asyncio,
                "json": json,
                "datetime": datetime,
                "app_id": app_id,
                "user_id": user_id
            }
            
            # Add injected services to namespace
            namespace.update(self._injected_services)
            
            # Add FiberApp if app_id is provided
            if app_id:
                fiber_app = await self._get_fiber_app(app_id, user_id)
                if fiber_app:
                    namespace['fiber'] = fiber_app
                    namespace['fiber_app'] = fiber_app
            
            # Ensure there's a run function
            if "def run(" not in implementation:
                # Wrap the implementation in a run function
                implementation = f"""
async def run(input_data):
    # Function implementation
{implementation}
    
    # Return result or input_data if no explicit return
    return locals().get('result', input_data)
"""
            
            # Execute the code to define functions
            exec(implementation, namespace)
            
            # Get the run function
            run_func = namespace.get("run")
            if not run_func:
                raise ValueError("Function must define a 'run' function")
            
            # Execute the function with dependency injection
            if asyncio.iscoroutinefunction(run_func):
                # Async function - can make HTTP calls, database queries, etc.
                logger.debug(f"[{self.context}] Executing async function")
                result = await self._call_with_dependencies(run_func, input_data, function)
            else:
                # Sync function - direct execution
                logger.debug(f"[{self.context}] Executing sync function")
                result = self._call_with_dependencies_sync(run_func, input_data, function)
            
            return result
            
        except Exception as e:
            logger.error(f"[{self.context}] Error executing function code: {e}")
            logger.error(f"[{self.context}] Implementation: {implementation[:200]}...")
            raise ValueError(f"Error in function implementation: {str(e)}")
    
    async def _call_with_dependencies(self, func, input_data: Dict[str, Any], function: Dict[str, Any]) -> Any:
        """Call async function with dependency injection based on signature."""
        sig = inspect.signature(func)
        dependencies = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'input_data':
                continue  # Will be passed as first argument
            
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                continue  # Skip **kwargs
            
            # Map parameter names to available services
            if param_name in ['fiber', 'fiber_app']:
                dependencies[param_name] = self._injected_services.get('fiber_app')
            elif param_name in ['llm_service', 'llm_provider_service', 'llm']:
                dependencies[param_name] = self._injected_services.get('llm_service')
            elif param_name in ['storage', 'storage_service']:
                dependencies[param_name] = self._injected_services.get('storage')
            elif param_name in ['oauth_service', 'oauth']:
                dependencies[param_name] = self._injected_services.get('oauth_service')
            elif param_name == 'db' or param_name == 'database':
                dependencies[param_name] = self.db
            else:
                # Check for direct service name match
                dependencies[param_name] = self._injected_services.get(param_name)
            
            if dependencies[param_name] is not None:
                logger.debug(f"[{self.context}] Injecting {param_name}: {type(dependencies[param_name]).__name__}")
        
        # Call the async function
        return await func(input_data, **dependencies)
    
    def _call_with_dependencies_sync(self, func, input_data: Dict[str, Any], function: Dict[str, Any]) -> Any:
        """Call sync function with dependency injection based on signature."""
        sig = inspect.signature(func)
        dependencies = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'input_data':
                continue  # Will be passed as first argument
            
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                continue  # Skip **kwargs
            
            # Map parameter names to available services
            if param_name in ['fiber', 'fiber_app']:
                dependencies[param_name] = self._injected_services.get('fiber_app')
            elif param_name in ['llm_service', 'llm_provider_service', 'llm']:
                dependencies[param_name] = self._injected_services.get('llm_service')
            elif param_name in ['storage', 'storage_service']:
                dependencies[param_name] = self._injected_services.get('storage')
            elif param_name in ['oauth_service', 'oauth']:
                dependencies[param_name] = self._injected_services.get('oauth_service')
            elif param_name == 'db' or param_name == 'database':
                dependencies[param_name] = self.db
            else:
                # Check for direct service name match
                dependencies[param_name] = self._injected_services.get(param_name)
            
            if dependencies[param_name] is not None:
                logger.debug(f"[{self.context}] Injecting {param_name}: {type(dependencies[param_name]).__name__}")
        
        # Call the sync function
        return func(input_data, **dependencies)
    
    async def _get_fiber_app(self, app_id: str, user_id: int = None):
        """Create a FiberApp instance for function execution."""
        try:
            # For functions, we might need a different API key strategy
            # Could use user API key, app API key, or function-specific key
            
            # For now, try to get an app-level API key or user API key
            api_key = None
            
            # Try to get user API key first if user_id provided
            if user_id:
                query = "SELECT api_key FROM users WHERE user_id = $1"
                result = await self.db.fetch_one(query, user_id)
                if result:
                    api_key = result.get('api_key')
            
            # Fallback to app-level key or create one
            if not api_key:
                # This would need to be implemented based on your API key strategy
                logger.warning(f"[{self.context}] No API key available for FiberApp creation")
                return None
            
            # Create FiberApp
            from fiberwise_sdk import FiberApp
            
            fiber_app = FiberApp(
                api_key=api_key,
                base_url=self.base_url,
                app_id=str(app_id)
            )
            
            logger.info(f"[{self.context}] Created FiberApp for function execution")
            return fiber_app
            
        except Exception as e:
            logger.error(f"[{self.context}] Failed to create FiberApp: {e}")
            return None
    
    # Database helper methods
    
    async def _get_function(self, function_id: str) -> Optional[Dict[str, Any]]:
        """Get function details from database."""
        query = "SELECT * FROM functions WHERE function_id = $1"
        result = await self.db.fetch_one(query, function_id)
        
        if not result:
            return None
        
        function = dict(result)
        
        # Parse JSON fields
        for field in ['input_schema', 'output_schema', 'metadata']:
            if function.get(field) and isinstance(function[field], str):
                try:
                    function[field] = json.loads(function[field])
                except json.JSONDecodeError:
                    function[field] = {}
        
        return function
    
    async def list_functions(self, app_id: str = None) -> list[Dict[str, Any]]:
        """List available functions, optionally filtered by app_id."""
        if app_id:
            query = "SELECT * FROM functions WHERE app_id = $1 ORDER BY name"
            results = await self.db.fetch_all(query, app_id)
        else:
            query = "SELECT * FROM functions ORDER BY name"
            results = await self.db.fetch_all(query)
        
        functions = []
        for result in results:
            function = dict(result)
            
            # Parse JSON fields
            for field in ['input_schema', 'output_schema', 'metadata']:
                if function.get(field) and isinstance(function[field], str):
                    try:
                        function[field] = json.loads(function[field])
                    except json.JSONDecodeError:
                        function[field] = {}
            
            functions.append(function)
        
        return functions
    
    async def get_function_info(self, function_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed function information including schema and metadata."""
        return await self._get_function(function_id)
