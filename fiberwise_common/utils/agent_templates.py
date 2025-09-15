"""
Agent template utilities for FiberWise.

This module provides utilities for generating agent code templates and scaffolding.
"""

from typing import Optional


def create_minimal_agent_code(agent_name: str, include_docstring: bool = True) -> str:
    """
    Create minimal agent implementation code with a basic class structure.
    
    This function generates a minimal but functional Python class template for agents.
    The generated code includes:
    - A class with the specified agent name
    - A run() method that can be immediately extended
    - Optional docstring for the class
    
    Args:
        agent_name: Name of the agent class to generate
        include_docstring: Whether to include a basic docstring in the class
        
    Returns:
        String containing minimal Python class code for the agent
        
    Example:
        >>> code = create_minimal_agent_code("MyAgent")
        >>> print(code)  # doctest: +SKIP
        # MyAgent agent implementation
        class MyAgent:    MyAgent agent implementation
            
            def run(self):
                pass
        
        >>> create_minimal_agent_code("TestAgent", include_docstring=False)
        '# TestAgent agent implementation\\nclass TestAgent:\\n    def run(self):\\n        pass'
    """
    if not agent_name or not agent_name.strip():
        raise ValueError("Agent name cannot be empty")
    
    # Ensure agent_name is a valid Python identifier
    clean_name = agent_name.strip()
    if not clean_name.isidentifier():
        raise ValueError(f"Agent name '{clean_name}' is not a valid Python identifier")
    
    # Build the template
    lines = [f"# {clean_name} agent implementation", f"class {clean_name}:"]
    
    if include_docstring:
        lines.extend([f'    """{clean_name} agent implementation."""', ""])
    
    lines.extend(["    def run(self):", "        pass"])
    
    return "\n".join(lines)


def create_function_agent_template(agent_name: str) -> str:
    """
    Create a function-based agent template.
    
    Args:
        agent_name: Name for the agent (used in comments and function name)
        
    Returns:
        String containing a function-based agent template
            return {"status": "success", "data": input_data}
    """
    if not agent_name or not agent_name.strip():
        raise ValueError("Agent name cannot be empty")
    
    clean_name = agent_name.strip().lower()
    
    template = f'''# {clean_name} agent implementation
def run_agent(input_data):
    """
    {clean_name} agent implementation.
    
    Args:
        input_data: Input data for the agent
        
    Returns:
        Agent execution result
    """
    # TODO: Implement agent logic
    return {{"status": "success", "data": input_data}}'''
    
    return template