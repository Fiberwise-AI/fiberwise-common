"""
Simplified FiberAgent - extends ia_modules BaseAgent.
ONE class handles all agent types (CLASS, FUNCTION, LLM) with minimal complexity.
"""

import asyncio
import inspect
import logging
from abc import abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime

# Import ia_modules components
from ia_modules.agents import BaseAgent, AgentRole, StateManager

# Import MetadataMixin from utils
from ..utils.agent_utils import MetadataMixin

logger = logging.getLogger(__name__)


class FiberInjectable:
    """
    Base class for FiberWise components that support service injection.

    Provides dependency declaration and injection capabilities.
    """

    def __init__(self):
        """Initialize injectable with empty service dict."""
        self._injected_services: Dict[str, Any] = {}

    def get_dependencies(self) -> List[str]:
        """
        Override to declare dependencies.

        Returns:
            List of service names this component needs
        """
        return []

    def inject_services(self, services: Dict[str, Any]) -> None:
        """
        Inject services into this component.

        Args:
            services: Dict of service_name -> service_instance
        """
        self._injected_services.update(services)
        for name, service in services.items():
            setattr(self, name, service)

    def get_service(self, service_name: str) -> Any:
        """
        Get an injected service by name.

        Args:
            service_name: Name of the service

        Returns:
            Service instance or None
        """
        return self._injected_services.get(service_name)

    def has_service(self, service_name: str) -> bool:
        """
        Check if a service is available.

        Args:
            service_name: Name of the service

        Returns:
            True if service is injected
        """
        return service_name in self._injected_services


class FiberAgent(BaseAgent, MetadataMixin, FiberInjectable):
    """
    FiberWise agent extending ia_modules BaseAgent.

    Design decisions:
    - AgentRole: Maps FiberWise config (system_message → system_prompt, allowed_tools)
    - StateManager: Empty placeholder (not used - we use activations + DynamicData)
    - execute(): Only method - no run_agent_async (use ia_modules way)
    - Handles all agent types: CUSTOM, FUNCTION, LLM in a single unified class

    FiberWise Features:
    - Schema validation (get_input_schema, get_output_schema, validate_input, validate_output)
    - Service injection (get_dependencies, inject_services)
    - Config management
    - Metadata support
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        agent_function: Optional[Callable] = None
    ):
        """
        Initialize FiberAgent.

        Args:
            config: FiberWise agent config (from DB/manifest)
                - system_message: System prompt for agent
                - allowed_tools: List of allowed tool names
                - name: Agent name
                - description: Agent description
                - agent_type: 'class', 'function', or 'llm_chat'
                - input_schema: JSON schema for input validation
                - output_schema: JSON schema for output validation
            agent_function: For FUNCTION type agents (optional)
        """
        self.config = config or {}

        # Map FiberWise config → ia_modules AgentRole
        role = AgentRole(
            name=self.config.get('name', 'Agent'),
            description=self.config.get('description', ''),
            allowed_tools=self.config.get('allowed_tools', []),
            system_prompt=self.config.get('system_message', '')  # FiberWise → ia_modules mapping
        )

        # StateManager: Empty (BaseAgent requires it, but we don't use it)
        # We use: activations table + DynamicData instead
        state_manager = StateManager(thread_id="fiber-agent-unused")

        # Initialize BaseAgent
        super().__init__(role, state_manager)

        # Initialize MetadataMixin
        MetadataMixin.__init__(self)

        # Initialize FiberInjectable
        FiberInjectable.__init__(self)

        # FiberWise properties
        self.agent_id: Optional[str] = None
        self.agent_type = self.config.get('agent_type', 'custom')
        self._description = self.config.get('description', 'FiberWise SDK Agent')
        self._version = self.config.get('version', '1.0.0')
        self.metadata: Dict[str, Any] = self.config.get('metadata', {})

        # For FUNCTION type agents
        self._agent_function = agent_function

    # === Service Injection ===

    def get_dependencies(self) -> List[str]:
        """
        Override in subclass to declare dependencies.

        Valid dependencies for SDK agents:
        - 'fiber_app': FiberApp instance
        - 'oauth_service': OAuth service
        - 'llm_service': LLM provider service
        - 'db': Database provider

        Returns:
            List of service names required by this agent
        """
        return []

    def inject_services(self, services: Dict[str, Any]) -> None:
        """
        Inject services into this agent.

        Args:
            services: Dict mapping service names to service instances
        """
        self._injected_services.update(services)

        # Set services as attributes for easy access
        for name, service in services.items():
            setattr(self, name, service)

    def get_injected_services(self) -> Dict[str, Any]:
        """Get all injected services."""
        return self._injected_services.copy()

    # === Schema Validation ===

    def get_input_schema(self) -> Dict[str, Any]:
        """
        Get input schema for validation.
        Override in subclass to provide custom schema.

        Returns:
            JSON schema dict for input validation
        """
        return self.config.get('input_schema', {
            "type": "object",
            "properties": {},
            "required": []
        })

    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get output schema for validation.
        Override in subclass to provide custom schema.

        Returns:
            JSON schema dict for output validation
        """
        return self.config.get('output_schema', {
            "type": "object",
            "properties": {},
            "required": []
        })

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

    # === Agent Properties ===

    @property
    def description(self) -> str:
        """Get agent description"""
        return self._description

    @property
    def version(self) -> str:
        """Get agent version"""
        return self._version

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get agent information including metadata, schemas, and dependencies.

        Returns:
            Dict containing agent information
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "version": self.version,
            "description": self.description,
            "dependencies": self.get_dependencies(),
            "input_schema": self.get_input_schema(),
            "output_schema": self.get_output_schema(),
            "metadata": self.metadata,
            "class_name": self.__class__.__name__,
            "module": self.__class__.__module__
        }

    # === Execution (ia_modules way - no run_agent_async!) ===

    async def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent (ia_modules BaseAgent method).
        Routes based on agent type (class/function/llm_chat).

        NO run_agent_async() - just use execute() directly!

        Args:
            input_data: Input data for the agent
            **kwargs: Additional execution parameters and injected services

        Returns:
            Dict containing the agent's output
        """
        # Merge injected services into kwargs
        merged_kwargs = {**kwargs, **self._injected_services}

        agent_type = self.agent_type.lower()

        if agent_type == 'llm_chat' or agent_type == 'llm':
            return await self._execute_llm(input_data, merged_kwargs)

        elif agent_type == 'function':
            return await self._execute_function(input_data, merged_kwargs)

        else:  # 'custom' or default
            return await self._execute_custom(input_data, merged_kwargs)

    async def _execute_custom(self, input_data: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute CUSTOM type agent.

        CUSTOM agents are Python classes that extend FiberAgent and override execute().
        This method should never be called directly - subclasses override execute().

        Args:
            input_data: Input data
            kwargs: Execution kwargs with services

        Returns:
            Agent execution result

        Raises:
            NotImplementedError: If called on base FiberAgent (must be overridden)
        """
        # If this is called, it means execute() wasn't overridden in subclass
        # This should only happen if base FiberAgent is instantiated directly
        raise NotImplementedError(
            "CUSTOM agents must override execute() method. "
            "Do NOT implement run_agent_async() - use execute() instead!"
        )

    async def _execute_function(self, input_data: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute FUNCTION type agent.

        FUNCTION agents are standalone Python functions wrapped by FiberAgent.

        Args:
            input_data: Input data
            kwargs: Execution kwargs with services

        Returns:
            Agent execution result

        Raises:
            ValueError: If no agent function was provided
        """
        if not self._agent_function:
            raise ValueError("FUNCTION type requires agent_function parameter")

        # Call function (handle sync/async)
        if inspect.iscoroutinefunction(self._agent_function):
            result = await self._agent_function(input_data, **kwargs)
        else:
            result = self._agent_function(input_data, **kwargs)

        # Ensure result is a dict
        if not isinstance(result, dict):
            result = {"result": result}

        return result

    async def _execute_llm(self, input_data: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute LLM type agent (config-only, no code).

        LLM agents are configuration-only - they use an LLM provider
        without custom Python code.

        Args:
            input_data: Input data
            kwargs: Execution kwargs with services (must include llm_service)

        Returns:
            Agent execution result

        Raises:
            RuntimeError: If llm_service not available
        """
        llm_service = kwargs.get('llm_service')
        if not llm_service:
            raise RuntimeError("LLM agent requires llm_service")

        # Build prompt
        system_message = self.config.get('system_message', 'You are a helpful assistant.')
        user_message = input_data.get('prompt', input_data.get('message', str(input_data)))

        # Combine into full prompt (simple approach)
        full_prompt = f"{system_message}\n\nUser: {user_message}\nAssistant:"

        # Use ia_modules LLM provider
        response = await llm_service.generate(
            prompt=full_prompt,
            temperature=self.config.get('temperature', 0.7),
            max_tokens=self.config.get('max_tokens', 2048)
        )

        # Extract response text
        response_text = response.get('text', '') if isinstance(response, dict) else str(response)

        return {"response": response_text, "input": input_data}

    # === Validation Wrapper ===

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

    # === Factory Methods ===

    @classmethod
    def from_module(cls, module_path: str, config: Optional[Dict[str, Any]] = None) -> 'FiberAgent':
        """
        Create agent instance from module path.

        Args:
            module_path: Python module path to agent class
            config: Optional configuration dict

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

            return agent_class(config=config)

        except Exception as e:
            logger.error(f"Failed to load agent from {module_path}: {e}")
            raise

    @classmethod
    def from_function(cls, func: Callable, config: Optional[Dict[str, Any]] = None) -> 'FiberAgent':
        """
        Create FUNCTION type agent from a function.

        Args:
            func: Function to wrap (sync or async)
            config: Optional configuration dict

        Returns:
            FiberAgent instance wrapping the function
        """
        agent_config = config or {}
        agent_config['agent_type'] = 'function'

        # Extract function metadata
        if 'name' not in agent_config:
            agent_config['name'] = func.__name__
        if 'description' not in agent_config and func.__doc__:
            agent_config['description'] = func.__doc__.strip()

        return cls(config=agent_config, agent_function=func)

    def __repr__(self) -> str:
        return f"<FiberAgent(name={self.role.name}, type={self.agent_type})>"
