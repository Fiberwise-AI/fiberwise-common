"""
Base FiberAgent class for FiberWise SDK.
All custom agents should inherit from this class.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
import asyncio

# Import from fiberwise_common directly since we're already in it
from ..utils.agent_utils import MetadataMixin

logger = logging.getLogger(__name__)


class FiberInjectable(ABC):
    """
    Base class for SDK objects that support dependency injection.
    
    Classes that extend this can declare their dependencies and have them
    automatically injected. In the SDK, only FiberApp, OAuth, and LLM services
    are injectable.
    """
    
    def __init__(self):
        self._injected_services: Dict[str, Any] = {}
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """
        Return list of service names this object depends on.
        
        Valid dependencies for SDK agents:
        - 'fiber_app': FiberApp instance
        - 'oauth_service': OAuth service
        - 'llm_service': LLM provider service
        
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
    
    def has_service(self, service_name: str) -> bool:
        """Check if a service was injected."""
        return service_name in self._injected_services
    
    def get_injected_services(self) -> Dict[str, Any]:
        """Get all injected services."""
        return self._injected_services.copy()


class FiberAgent(MetadataMixin, FiberInjectable):
    """
    Base class for all FiberWise SDK agents with dependency injection support.
    
    This class provides:
    - Dependency injection for FiberApp, OAuth, and LLM services
    - Input/output schema validation
    - Execution lifecycle management
    - Error handling and logging
    - Agent metadata and configuration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent with optional configuration"""
        super().__init__()
        self.config = config or {}
        self.agent_id: Optional[str] = None
        self.agent_type: Optional[str] = None
        self._description = "FiberWise SDK Agent"
        self._version = "1.0.0"
        self.metadata: Dict[str, Any] = {}

    def get_dependencies(self) -> List[str]:
        """
        Override to specify required service dependencies.
        
        Valid dependencies for SDK agents:
        - 'fiber_app': FiberApp instance
        - 'oauth_service': OAuth service  
        - 'llm_service': LLM provider service
        
        Returns:
            List of service names required by this agent
        """
        return []

    @abstractmethod
    def run_agent(self, input_data: Any, **kwargs) -> Any:
        """
        Main execution method for the agent with dependency injection.
        
        Args:
            input_data: The input data to process
            **kwargs: Injected dependencies (fiber_app, oauth_service, llm_service, etc.)
            
        Returns:
            Agent execution results
        """
        raise NotImplementedError("Agent must implement run_agent() method")

    async def run_agent_async(self, input_data: Any, **kwargs) -> Any:
        """
        Async version of run_agent for async agents.
        
        Args:
            input_data: The input data to process  
            **kwargs: Injected dependencies (fiber_app, oauth_service, llm_service, etc.)
            
        Returns:
            Agent execution results
        """
        # Default implementation calls sync version
        return self.run_agent(input_data, **kwargs)

    # Legacy execute method for compatibility with common patterns
    async def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent with input data and injected dependencies.
        Legacy method that calls run_agent_async for compatibility.
        
        Args:
            input_data: Input data for the agent
            **kwargs: Additional execution parameters
            
        Returns:
            Dict containing the agent's output
        """
        result = await self.run_agent_async(input_data, **kwargs)
        
        # Ensure result is a dict
        if not isinstance(result, dict):
            result = {"result": result}
            
        return result

    @property
    def description(self) -> str:
        """Get agent description"""
        return self._description

    @property 
    def version(self) -> str:
        """Get agent version"""
        return self._version
    
    def get_input_schema(self) -> Dict[str, Any]:
        """
        Override to provide input schema for validation.
        
        Returns:
            JSON schema dict for input validation
        """
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Override to provide output schema for validation.
        
        Returns:
            JSON schema dict for output validation
        """
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get agent information including metadata, schemas, and dependencies.
        
        Returns:
            Dict containing agent information
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type or self.__class__.__name__,
            "version": self.version,
            "description": self.description,
            "dependencies": self.get_dependencies(),
            "input_schema": self.get_input_schema(),
            "output_schema": self.get_output_schema(),
            "metadata": self.metadata,
            "class_name": self.__class__.__name__,
            "module": self.__class__.__module__
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data against the agent's input schema.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            Dict with validation results
        """
        try:
            schema = self.get_input_schema()
            return self._validate_against_schema(input_data, schema, "input")
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Input validation error: {str(e)}"],
                "warnings": []
            }
    
    def validate_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate output data against the agent's output schema.
        
        Args:
            output_data: Output data to validate
            
        Returns:
            Dict with validation results
        """
        try:
            schema = self.get_output_schema()
            return self._validate_against_schema(output_data, schema, "output")
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Output validation error: {str(e)}"],
                "warnings": []
            }
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any], 
                               data_type: str) -> Dict[str, Any]:
        """
        Validate data against a JSON schema.
        
        Args:
            data: Data to validate
            schema: JSON schema
            data_type: Type description for error messages
            
        Returns:
            Dict with validation results
        """
        errors = []
        warnings = []
        
        # Basic type validation
        if schema.get("type") == "object" and not isinstance(data, dict):
            errors.append(f"{data_type} must be an object")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required {data_type} field: {field}")
        
        # Check properties
        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in data:
                field_type = field_schema.get("type")
                field_value = data[field]
                
                if field_type == "string" and not isinstance(field_value, str):
                    errors.append(f"{data_type} field '{field}' must be a string")
                elif field_type == "number" and not isinstance(field_value, (int, float)):
                    errors.append(f"{data_type} field '{field}' must be a number")
                elif field_type == "boolean" and not isinstance(field_value, bool):
                    errors.append(f"{data_type} field '{field}' must be a boolean")
                elif field_type == "array" and not isinstance(field_value, list):
                    errors.append(f"{data_type} field '{field}' must be an array")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def execute_with_validation(self, input_data: Dict[str, Any], 
                                    validate_input: bool = True,
                                    validate_output: bool = True,
                                    **kwargs) -> Dict[str, Any]:
        """
        Execute agent with optional input/output validation.
        
        Args:
            input_data: Input data for the agent
            validate_input: Whether to validate input data
            validate_output: Whether to validate output data
            **kwargs: Additional execution parameters
            
        Returns:
            Dict containing execution result with validation info
        """
        execution_start = datetime.now()
        
        # Input validation
        if validate_input:
            input_validation = self.validate_input(input_data)
            if not input_validation["valid"]:
                return {
                    "success": False,
                    "error": "Input validation failed",
                    "input_validation": input_validation,
                    "execution_time": 0
                }
        else:
            input_validation = {"valid": True, "errors": [], "warnings": []}
        
        try:
            # Execute the agent
            logger.info(f"Executing agent {self.__class__.__name__}")
            output_data = await self.execute(input_data, **kwargs)
            
            # Output validation
            if validate_output:
                output_validation = self.validate_output(output_data)
                if not output_validation["valid"]:
                    logger.warning(f"Output validation failed for {self.__class__.__name__}: {output_validation['errors']}")
            else:
                output_validation = {"valid": True, "errors": [], "warnings": []}
            
            execution_time = (datetime.now() - execution_start).total_seconds()
            
            return {
                "success": True,
                "output": output_data,
                "input_validation": input_validation,
                "output_validation": output_validation,
                "execution_time": execution_time,
                "agent_info": {
                    "class_name": self.__class__.__name__,
                    "dependencies_used": list(self._injected_services.keys())
                }
            }
            
        except Exception as e:
            execution_time = (datetime.now() - execution_start).total_seconds()
            logger.error(f"Agent execution failed: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "input_validation": input_validation,
                "execution_time": execution_time,
                "agent_info": {
                    "class_name": self.__class__.__name__,
                    "dependencies_used": list(self._injected_services.keys())
                }
            }
    
    
    @classmethod
    def from_module(cls, module_path: str) -> 'FiberAgent':
        """
        Create agent instance from module path.
        
        Args:
            module_path: Python module path to agent class
            
        Returns:
            Agent instance
        """
        try:
            parts = module_path.split('.')
            module_name = '.'.join(parts[:-1])
            class_name = parts[-1]
            
            import importlib
            module = importlib.import_module(module_name)
            agent_class = getattr(module, class_name)
            
            if not issubclass(agent_class, FiberAgent):
                raise TypeError(f"Class {class_name} must inherit from FiberAgent")
            
            return agent_class()
            
        except Exception as e:
            logger.error(f"Failed to load agent from {module_path}: {e}")
            raise
