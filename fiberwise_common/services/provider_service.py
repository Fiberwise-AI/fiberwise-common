"""
Provider Configuration Service
Manages LLM and AI provider configurations (OpenAI, Anthropic, Google, etc.).
Separated from account configs which are for FiberWise service authentication.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base_service import BaseService
from ..database.query_adapter import QueryAdapter, ParameterStyle
import logging

logger = logging.getLogger(__name__)


class ProviderService(BaseService):
    """
    Manages LLM and AI provider configurations.
    
    This service handles:
    - Provider configuration management (OpenAI, Anthropic, Google, etc.)
    - API key storage and retrieval for providers
    - Default provider configuration
    - Provider-specific settings (model, temperature, max_tokens, etc.)
    """
    
    def __init__(self, db_provider, config_dir: Optional[str] = None):
        super().__init__(db_provider)
        self.query_adapter = QueryAdapter(ParameterStyle.SQLITE)
        self.config_dir = Path(config_dir or os.path.expanduser("~/.fiberwise/providers"))
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    async def add_provider_config(self, name: str, provider_type: str, api_key: str, 
                                 base_url: Optional[str] = None, model: Optional[str] = None,
                                 max_tokens: Optional[int] = 1000, temperature: Optional[float] = 0.7,
                                 user_id: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Add provider configuration.
        
        Args:
            name: Configuration name (unique identifier)
            provider_type: Type of provider (openai, anthropic, google, etc.)
            api_key: API key for the provider
            base_url: Base URL for the provider API
            model: Default model to use
            max_tokens: Maximum tokens for requests
            temperature: Temperature setting for requests
            user_id: Optional user ID
            **kwargs: Additional configuration parameters
        
        Returns:
            Dict containing the created configuration
        """
        import uuid
        current_time = datetime.now().isoformat()
        provider_id = str(uuid.uuid4())
        
        # Get default configuration template from llm_provider_defaults
        default_config = {}
        try:
            # Handle provider type mapping for consistency
            template_provider_type = provider_type
            if provider_type == "google":
                template_provider_type = "gemini"  # Map google -> gemini for template lookup
            
            default_query = "SELECT default_configuration FROM llm_provider_defaults WHERE provider_type = ?"
            default_result = await self.db.fetch_one(default_query, template_provider_type)
            if default_result and default_result['default_configuration']:
                default_config = json.loads(default_result['default_configuration'])
        except Exception as e:
            logger.warning(f"Could not load default config for {provider_type}: {e}")
        
        # Prepare config data by merging defaults with provided values
        config_data = {
            **default_config,
            "api_key": api_key,
            "model": model or default_config.get("default_model"),
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        # Remove None values
        config_data = {k: v for k, v in config_data.items() if v is not None}
        
        query = """
            INSERT INTO llm_providers 
            (provider_id, name, provider_type, api_endpoint, configuration, 
             default_model, created_by, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            await self.db.execute(
                query, provider_id, name, provider_type, base_url, json.dumps(config_data),
                model, user_id, current_time, current_time
            )
            logger.info(f"Added provider configuration: {name} ({provider_type})")
            
            # Return the created config
            return await self.get_provider_config(name)
            
        except Exception as e:
            logger.error(f"Failed to add provider configuration {name}: {e}")
            raise
    
    async def get_provider_config(self, name: str, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get a specific provider configuration by name (with user isolation)."""
        if user_id is not None:
            # Filter by user ownership (system providers + user's own providers)
            query = "SELECT * FROM llm_providers WHERE name = ? AND (is_system = true OR created_by = ?)"
            result = await self.db.fetch_one(query, name, user_id)
        else:
            # Legacy mode without user filtering (for backwards compatibility)
            query = "SELECT * FROM llm_providers WHERE name = ?"
            result = await self.db.fetch_one(query, name)
        
        if result:
            config = dict(result)
            # Parse configuration JSON
            if config.get('configuration'):
                try:
                    config['configuration'] = json.loads(config['configuration'])
                except json.JSONDecodeError:
                    config['configuration'] = {}
            return config
        return None
    
    async def get_provider_configs(self, provider_type: Optional[str] = None, 
                                 user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all provider configurations, optionally filtered by provider type or user.
        
        Args:
            provider_type: Optional provider type filter
            user_id: Optional user ID filter
        
        Returns:
            List of configuration dictionaries
        """
        base_query = "SELECT * FROM llm_providers WHERE 1=1"
        
        try:
            if provider_type and user_id:
                query = base_query + " AND provider_type = ? AND (created_by = ? OR is_system = true) ORDER BY created_at DESC"
                results = await self.db.fetch_all(query, provider_type, user_id)
            elif provider_type:
                query = base_query + " AND provider_type = ? ORDER BY created_at DESC"
                results = await self.db.fetch_all(query, provider_type)
            elif user_id:
                query = base_query + " AND (created_by = ? OR is_system = true) ORDER BY created_at DESC"
                results = await self.db.fetch_all(query, user_id)
            else:
                query = base_query + " ORDER BY created_at DESC"
                results = await self.db.fetch_all(query)
            
            configs = []
            
            for result in results:
                config = dict(result)
                # Parse configuration JSON
                if config.get('configuration'):
                    try:
                        config['configuration'] = json.loads(config['configuration'])
                    except json.JSONDecodeError:
                        config['configuration'] = {}
                configs.append(config)
            
            return configs
            
        except Exception as e:
            logger.error(f"Failed to get provider configurations: {e}")
            raise
    
    async def set_default_provider(self, name: str) -> bool:
        """Set a provider configuration as the default."""
        try:
            # Wrap both operations in a transaction for atomicity
            async with self.db.transaction():
                # First, unset all defaults
                await self.db.execute("UPDATE llm_providers SET is_default = 0")
                
                # Then set the specified one as default
                affected_rows = await self.db.execute(
                    "UPDATE llm_providers SET is_default = 1 WHERE name = ?", name
                )
                
                if affected_rows > 0:
                    logger.info(f"Set provider {name} as default")
                    return True
                else:
                    logger.warning(f"Provider {name} not found")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to set default provider {name}: {e}")
            raise
    
    async def get_default_provider(self, provider_type: Optional[str] = None, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get the default provider configuration, optionally for a specific provider type.
        
        Args:
            provider_type: Optional provider type filter
            user_id: Optional user ID for access control
        
        Returns:
            Default configuration dictionary or None
        """
        try:
            # Build base query with proper boolean handling and user isolation
            base_conditions = "(is_active = 1 OR is_active = true OR is_active = 'true')"
            
            if user_id is not None:
                # Filter by user ownership (system providers + user's own providers)
                base_conditions += " AND (is_system = true OR created_by = ?)"
                
                if provider_type:
                    query = f"SELECT * FROM llm_providers WHERE {base_conditions} AND provider_type = ? ORDER BY is_default DESC, created_at DESC LIMIT 1"
                    result = await self.db.fetch_one(query, user_id, provider_type)
                else:
                    query = f"SELECT * FROM llm_providers WHERE {base_conditions} ORDER BY is_default DESC, created_at DESC LIMIT 1"
                    result = await self.db.fetch_one(query, user_id)
            else:
                # Legacy mode without user filtering
                if provider_type:
                    query = f"SELECT * FROM llm_providers WHERE {base_conditions} AND provider_type = ? ORDER BY is_default DESC, created_at DESC LIMIT 1"
                    result = await self.db.fetch_one(query, provider_type)
                else:
                    query = f"SELECT * FROM llm_providers WHERE {base_conditions} ORDER BY is_default DESC, created_at DESC LIMIT 1"
                    result = await self.db.fetch_one(query)
                
            if result:
                config = dict(result)
                # Parse configuration JSON and extract key fields to top level
                if config.get('configuration'):
                    try:
                        config_json = json.loads(config['configuration'])
                        config['configuration'] = config_json
                        
                        # Extract key fields to top level for compatibility
                        config['api_key'] = config_json.get('api_key')
                        config['provider_type'] = config_json.get('provider_type') or config.get('provider_type')
                        config['model'] = config_json.get('model')
                        config['base_url'] = config_json.get('base_url')
                        config['temperature'] = config_json.get('temperature')
                        
                    except json.JSONDecodeError:
                        config['configuration'] = {}
                return config
            return None
            
        except Exception as e:
            logger.error(f"Failed to get default provider: {e}")
            raise
    
    async def update_provider_config(self, name: str, **kwargs) -> bool:
        """Update a provider configuration."""
        if not kwargs:
            return False
        
        # Separate direct columns from configuration data
        direct_columns = {'provider_type', 'api_endpoint', 'default_model', 'created_by', 'is_active', 'is_default'}
        updates = []
        params = []
        config_updates = {}
        
        for key, value in kwargs.items():
            if key in direct_columns:
                updates.append(f"{key} = ?")
                params.append(value)
            else:
                config_updates[key] = value
        
        # Handle configuration updates
        if config_updates:
            # Get current configuration
            current_config = await self.get_provider_config(name)
            if current_config and current_config.get('configuration'):
                current_config['configuration'].update(config_updates)
                config_data = current_config['configuration']
            else:
                config_data = config_updates
            
            updates.append("configuration = ?")
            params.append(json.dumps(config_data))
        
        # Add updated_at
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        
        # Add name for WHERE clause
        params.append(name)
        
        query = f"UPDATE llm_providers SET {', '.join(updates)} WHERE name = ?"
        
        try:
            result = await self.db.execute(query, params)
            if result.rowcount > 0:
                logger.info(f"Updated provider configuration: {name}")
                return True
            else:
                logger.warning(f"Provider configuration {name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update provider configuration {name}: {e}")
            raise
    
    async def delete_provider_config(self, name: str) -> bool:
        """Delete a provider configuration."""
        try:
            result = await self.db.execute(
                "DELETE FROM llm_providers WHERE name = ?", [name]
            )
            
            if result.rowcount > 0:
                logger.info(f"Deleted provider configuration: {name}")
                return True
            else:
                logger.warning(f"Provider configuration {name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete provider configuration {name}: {e}")
            raise
    
    async def validate_provider_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate provider configuration data.
        
        Args:
            config: Configuration dictionary to validate
        
        Returns:
            Dict with validation results
        """
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ['name', 'provider_type', 'api_key']
        for field in required_fields:
            if not config.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Provider type validation
        valid_providers = ['openai', 'anthropic', 'google', 'local', 'mock']
        if config.get('provider_type') and config['provider_type'] not in valid_providers:
            warnings.append(f"Unknown provider type: {config['provider_type']}")
        
        # Numeric field validation
        if config.get('max_tokens') is not None:
            try:
                max_tokens = int(config['max_tokens'])
                if max_tokens <= 0:
                    errors.append("max_tokens must be a positive integer")
            except (ValueError, TypeError):
                errors.append("max_tokens must be a valid integer")
        
        if config.get('temperature') is not None:
            try:
                temperature = float(config['temperature'])
                if not (0.0 <= temperature <= 2.0):
                    warnings.append("temperature should be between 0.0 and 2.0")
            except (ValueError, TypeError):
                errors.append("temperature must be a valid number")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
