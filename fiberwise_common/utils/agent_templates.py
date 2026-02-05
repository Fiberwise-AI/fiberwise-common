"""
Agent template utilities for FiberWise.

This module provides utilities for generating agent code templates and scaffolding
for hybrid-compatible agents that work with both FiberWise and ia_modules.
"""

from typing import Optional, Dict, Any
import yaml


def create_class_agent_template(
    agent_name: str,
    description: str = "",
    include_imports: bool = True,
    include_docstring: bool = True
) -> str:
    """
    Create a CLASS agent template extending FiberAgent (hybrid-compatible).

    Generates a Python class that:
    - Extends FiberAgent (which extends ia_modules BaseAgent)
    - Uses execute() method (not run_agent_async)
    - Includes proper imports from ia_modules
    - Is compatible with both FiberWise and ia_modules orchestration

    Args:
        agent_name: Name of the agent class to generate
        description: Optional description of what the agent does
        include_imports: Whether to include import statements
        include_docstring: Whether to include docstrings

    Returns:
        String containing CLASS agent template code

    Example:
        >>> code = create_class_agent_template("DataProcessor", "Processes data")
        >>> print(code)  # doctest: +SKIP
    """
    if not agent_name or not agent_name.strip():
        raise ValueError("Agent name cannot be empty")

    # Ensure agent_name is a valid Python identifier
    clean_name = agent_name.strip()
    if not clean_name.isidentifier():
        raise ValueError(f"Agent name '{clean_name}' is not a valid Python identifier")

    # Build imports section
    imports = ""
    if include_imports:
        imports = '''"""
{description}
"""

from typing import Dict, Any
from fiberwise_common.entities.fiber_agent import FiberAgent

'''.format(description=description or f"{clean_name} agent implementation")

    # Build docstring
    class_docstring = ""
    if include_docstring:
        class_docstring = f'''    """
    {description or f"{clean_name} agent implementation."}

    This agent extends FiberAgent (which extends ia_modules BaseAgent).
    Implement the execute() method for agent logic.
    """

'''

    # Build the template
    template = f'''{imports}class {clean_name}(FiberAgent):
{class_docstring}    async def execute(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute the agent with input data.

        Args:
            input_data: Input data for the agent
            **kwargs: Additional context (fiber, llm_service, db, etc.)

        Returns:
            Dict containing agent execution results
        """
        # Access injected services if needed
        # fiber = kwargs.get('fiber')
        # llm_service = kwargs.get('llm_service')
        # db = kwargs.get('db')

        # Implement your agent logic here
        result = {{
            "status": "success",
            "message": "Agent executed successfully",
            "data": input_data
        }}

        return result
'''

    return template


def create_function_agent_template(agent_name: str, description: str = "") -> str:
    """
    Create a FUNCTION agent template (hybrid-compatible).

    Function agents are standalone functions that can be wrapped by FiberAgent.
    They use execute() pattern internally.

    Args:
        agent_name: Name for the agent (used in comments and function name)
        description: Optional description of what the agent does

    Returns:
        String containing a function-based agent template
    """
    if not agent_name or not agent_name.strip():
        raise ValueError("Agent name cannot be empty")

    clean_name = agent_name.strip().lower().replace(" ", "_")

    template = f'''"""
{description or f"{agent_name} agent implementation"}
"""

from typing import Dict, Any


async def execute(input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Execute the agent with input data.

    Args:
        input_data: Input data for the agent
        **kwargs: Additional context (fiber, llm_service, db, etc.)

    Returns:
        Dict containing agent execution results
    """
    # Access injected services if needed
    # fiber = kwargs.get('fiber')
    # llm_service = kwargs.get('llm_service')

    # Implement your agent logic here
    result = {{
        "status": "success",
        "message": "Agent executed successfully",
        "data": input_data
    }}

    return result
'''

    return template


def create_llm_agent_config(
    agent_name: str,
    description: str = "",
    system_message: str = "",
    provider_id: str = "openai-default",
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> Dict[str, Any]:
    """
    Create an LLM agent configuration (no code required).

    LLM agents are config-only agents that use an LLM provider
    without custom Python code.

    Args:
        agent_name: Name of the agent
        description: Description of what the agent does
        system_message: System message/prompt for the LLM
        provider_id: LLM provider ID (e.g., "openai-default", "anthropic-claude")
        temperature: LLM temperature (0.0-1.0)
        max_tokens: Maximum tokens for LLM response

    Returns:
        Dict containing LLM agent configuration
    """
    return {
        "name": agent_name,
        "description": description or f"{agent_name} LLM agent",
        "agent_type": "llm",
        "provider_id": provider_id,
        "config": {
            "system_message": system_message or f"You are {agent_name}, a helpful AI assistant.",
            "temperature": temperature,
            "max_tokens": max_tokens
        }
    }


def create_agent_manifest(
    agent_name: str,
    agent_type: str,
    description: str = "",
    version: str = "1.0.0",
    implementation_file: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create an agent manifest with agent_type field.

    Args:
        agent_name: Name of the agent
        agent_type: Type of agent ("class", "function", or "llm")
        description: Description of what the agent does
        version: Agent version
        implementation_file: Path to implementation file (for class/function agents)
        config: Additional configuration (for LLM agents)

    Returns:
        Dict containing agent manifest
    """
    manifest = {
        "name": agent_name,
        "agent_type": agent_type.lower(),
        "description": description or f"{agent_name} agent",
        "version": version
    }

    if agent_type.lower() in ["class", "function"]:
        if implementation_file:
            manifest["implementation_file"] = implementation_file
        else:
            manifest["implementation_file"] = f"agents/{agent_name.lower()}_agent.py"

    if agent_type.lower() == "llm":
        manifest["config"] = config or {
            "provider_id": "openai-default",
            "system_message": f"You are {agent_name}, a helpful AI assistant.",
            "temperature": 0.7,
            "max_tokens": 2048
        }

    return manifest


def create_agent_manifest_yaml(
    agent_name: str,
    agent_type: str,
    description: str = "",
    version: str = "1.0.0",
    implementation_file: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create an agent manifest in YAML format.

    Args:
        agent_name: Name of the agent
        agent_type: Type of agent ("class", "function", or "llm")
        description: Description of what the agent does
        version: Agent version
        implementation_file: Path to implementation file (for class/function agents)
        config: Additional configuration (for LLM agents)

    Returns:
        String containing agent manifest in YAML format
    """
    manifest = create_agent_manifest(
        agent_name=agent_name,
        agent_type=agent_type,
        description=description,
        version=version,
        implementation_file=implementation_file,
        config=config
    )

    return yaml.dump(manifest, default_flow_style=False, sort_keys=False)


def create_minimal_agent_code(agent_name: str, include_docstring: bool = True) -> str:
    """
    DEPRECATED: Use create_class_agent_template() instead.

    Create minimal agent implementation code with a basic class structure.
    This is kept for backward compatibility but generates hybrid-compatible code.

    Args:
        agent_name: Name of the agent class to generate
        include_docstring: Whether to include a basic docstring in the class

    Returns:
        String containing minimal Python class code for the agent
    """
    # Generate hybrid-compatible template instead
    return create_class_agent_template(
        agent_name=agent_name,
        description=f"{agent_name} agent implementation",
        include_imports=True,
        include_docstring=include_docstring
    )