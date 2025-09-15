"""
Configuration management for FiberWise SDK.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import os

class FiberWiseConfig:
    """
    Configuration container for FiberWise SDK with read-only protection.
    """
    
    def __init__(self, initial_values: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with optional initial values.
        
        Args:
            initial_values: Dictionary of initial configuration values
        """
        # Initialize internal storage
        self._config = {}
        
        # Track which keys should be read-only
        self.read_only_keys = []
        
        # Set initial values if provided
        if initial_values:
            self.set_all(initial_values)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            The configuration value or the default
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> 'FiberWiseConfig':
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            
        Returns:
            Self for method chaining
        """
        # Check if this key is read-only
        if key in self.read_only_keys:
            print(f"Warning: Cannot modify read-only config key: {key}")
            return self
            
        self._config[key] = value
        return self
    
    def set_all(self, values: Dict[str, Any]) -> 'FiberWiseConfig':
        """
        Set multiple configuration values at once.
        
        Args:
            values: Dictionary of key-value pairs
            
        Returns:
            Self for method chaining
        """
        # Filter out read-only keys
        filtered_values = {k: v for k, v in values.items() if k not in self.read_only_keys}
        
        # If any keys were filtered, output a warning
        if len(filtered_values) < len(values):
            print("Warning: Some read-only config keys were not updated")
            
        # Update with filtered values
        self._config.update(filtered_values)
        return self
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary of all configuration values
        """
        return dict(self._config)
   
    def reset(self, initial_values: Optional[Dict[str, Any]] = None) -> 'FiberWiseConfig':
        """
        Reset the configuration to initial state or new values.
        
        Args:
            initial_values: New initial values (optional)
            
        Returns:
            Self for method chaining
        """
        self._config = {}
        if initial_values:
            self.set_all(initial_values)
        return self
