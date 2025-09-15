"""
Agent utilities for FiberWise.

This module provides utilities and mixins for agent-related functionality
that can be shared across different agent implementations.
"""

from typing import Dict, Any
import os
import importlib.util
import inspect


class MetadataMixin:
    """
    Mixin class providing metadata management functionality for agents.
    
    This mixin provides a standardized way to handle agent metadata
    across different agent implementations, eliminating code duplication.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'metadata'):
            self.metadata: Dict[str, Any] = {}
    
    def set_agent_metadata(self, **metadata) -> None:
        """
        Set agent metadata.
        
        Updates the agent's metadata dictionary with the provided key-value pairs.
        This method merges new metadata with existing metadata, overwriting
        any duplicate keys.
        
        Args:
            **metadata: Arbitrary keyword arguments to be added to metadata
            
        Example:
            >>> agent.set_agent_metadata(version="1.2.0", author="FiberWise Team")
            >>> agent.set_agent_metadata(description="Updated agent", tags=["nlp", "ai"])
        """
        self.metadata.update(metadata)
    
    def get_agent_metadata(self) -> Dict[str, Any]:
        """
        Get agent metadata.
        
        Returns a copy of the agent's metadata dictionary to prevent
        external modification of the internal state.
        
        Returns:
            Dictionary containing the agent's metadata
            
        Example:
            >>> metadata = agent.get_agent_metadata()
            >>> print(metadata["version"])
            "1.2.0"
        """
        return self.metadata.copy()
    
    def clear_agent_metadata(self) -> None:
        """
        Clear all agent metadata.
        
        Removes all key-value pairs from the agent's metadata dictionary.
        """
        self.metadata.clear()
    
    def remove_metadata_key(self, key: str) -> bool:
        """
        Remove a specific key from agent metadata.
        
        Args:
            key: The metadata key to remove
            
        Returns:
            True if the key was found and removed, False otherwise
            
        Example:
            >>> success = agent.remove_metadata_key("deprecated_field")
            >>> print(success)
            True
        """
        if key in self.metadata:
            del self.metadata[key]
            return True
        return False


def extract_agent_metadata(file_path: str) -> Dict[str, str]:
    """
    Extract agent metadata from a file by introspection.
    
    This function analyzes Python agent files to extract metadata including
    name, description, and type. It handles both function-based and class-based
    agents, using various fallback strategies for metadata extraction.
    
    Args:
        file_path: Path to the agent file to analyze
        
    Returns:
        Dictionary containing agent metadata with keys: name, description, type
        
    Example:
        >>> metadata = extract_agent_metadata("/path/to/agent.py")
        >>> print(metadata["type"])  # "function" or "class"
    """
    try:
        # Load the module to inspect
        spec = importlib.util.spec_from_file_location("agent_module", file_path)
        if not spec or not spec.loader:
            return {"name": os.path.abspath(file_path), "description": "", "type": "function"}
        
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        
        # Check for function-based agents first
        if hasattr(agent_module, 'run_agent') and callable(agent_module.run_agent):
            return {
                "name": os.path.abspath(file_path), 
                "description": getattr(agent_module.run_agent, '__doc__', "") or "", 
                "type": "function"
            }
        
        # Look for class-based agents
        for name, obj in inspect.getmembers(agent_module, inspect.isclass):
            if obj.__module__ == agent_module.__name__ and name not in ('Agent', 'FiberAgent'):
                if hasattr(obj, 'run_agent') or hasattr(obj, 'execute'):
                    # Try to get description from the class instance
                    try:
                        instance = obj()
                        # Use full file path + class name for unique identification
                        agent_name = f"{os.path.abspath(file_path)}::{name}"
                        description = ""
                        
                        # Check for _description attribute
                        if hasattr(instance, '_description'):
                            description = instance._description
                        elif hasattr(instance, 'description'):
                            description = instance.description
                        elif obj.__doc__:
                            description = obj.__doc__.strip().split('\n')[0]  # First line of docstring
                        
                        return {"name": agent_name, "description": description, "type": "class"}
                    except Exception:
                        # If we can't instantiate, use class name and docstring
                        description = obj.__doc__.strip().split('\n')[0] if obj.__doc__ else ""
                        agent_name = f"{os.path.abspath(file_path)}::{name}"
                        return {"name": agent_name, "description": description, "type": "class"}
        
        # Fallback to function type if no suitable class found
        return {"name": os.path.abspath(file_path), "description": "", "type": "function"}
        
    except Exception:
        # If anything fails, fallback to function type
        return {"name": os.path.abspath(file_path), "description": "", "type": "function"}