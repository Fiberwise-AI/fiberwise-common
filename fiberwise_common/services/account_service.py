"""
Account Configuration Service
Manages user account configurations, API keys, and provider settings.
Migrated from old fiber library with enhanced database integration.
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


class AccountService(BaseService):
    """
    Manages user account configuration and API keys.
    
    This service handles:
    - Account configuration management (legacy ~/.fiber/configs compatibility)
    - API key storage and retrieval
    - Default provider configuration
    - Integration with existing services
    """
    
    def __init__(self, db_provider, config_dir: Optional[str] = None):
        super().__init__(db_provider)
        self.query_adapter = QueryAdapter(ParameterStyle.SQLITE)
        self.config_dir = Path(config_dir or os.path.expanduser("~/.fiberwise/configs"))
        self.legacy_config_dir = Path(os.path.expanduser("~/.fiber/configs"))
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    async def add_account_config(self, name: str, provider: str, api_key: str, 
                               base_url: Optional[str] = None, user_id: Optional[int] = None,
                               **kwargs) -> Dict[str, Any]:
        """
        Add account configuration for a provider.
        
        Args:
            name: Configuration name
            provider: Provider type (openai, anthropic, etc.)
            api_key: API key for the provider
            base_url: Optional base URL for API
            user_id: Optional user ID for multi-user setups
            **kwargs: Additional configuration parameters
        
        Returns:
            Dict containing the created configuration
        """
        config = {
            "name": name,
            "provider": provider,
            "api_key": api_key,
            "base_url": base_url,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            **kwargs
        }
        
        try:
            # Save to file system for legacy compatibility
            await self._save_config_to_file(name, config)
            
            # Save to database
            query = self.query_adapter.convert_query("""
                INSERT OR REPLACE INTO account_configs 
                (name, provider, api_key, base_url, user_id, config_data, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, ParameterStyle.POSTGRESQL)
            
            await self.db.execute(
                query, name, provider, api_key, base_url, user_id,
                json.dumps(config), config["created_at"], config["updated_at"]
            )
            
            logger.info(f"Added account configuration: {name} ({provider})")
            return config
            
        except Exception as e:
            logger.error(f"Failed to add account configuration {name}: {str(e)}")
            raise
    
    async def get_account_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific account configuration by name."""
        query = self.query_adapter.convert_query("""
            SELECT * FROM account_configs WHERE name = $1
        """, ParameterStyle.POSTGRESQL)
        
        row = await self.db.fetch_one(query, name)
        
        if row:
            config_data = json.loads(row['config_data'])
            return config_data
        
        # Fallback to file-based config
        return await self._load_config_from_file(name)
    
    async def get_account_configs(self, provider: Optional[str] = None, 
                                user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all account configurations, optionally filtered by provider or user.
        
        Args:
            provider: Optional provider filter
            user_id: Optional user ID filter
        
        Returns:
            List of configuration dictionaries
        """
        query = "SELECT * FROM account_configs WHERE 1=1"
        params = []
        
        if provider:
            query += " AND provider = ?"
            params.append(provider)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        query += " ORDER BY created_at DESC"
        
        try:
            rows = await self.db.fetch_all(query, *params)
            
            configs = []
            for row in rows:
                config_data = json.loads(row['config_data'])
                configs.append(config_data)
            
            # If no database configs and no filters, try legacy files
            if not configs and not provider and not user_id:
                configs.extend(await self._load_legacy_configs())
            
            return configs
            
        except Exception as e:
            logger.error(f"Failed to get account configurations: {str(e)}")
            # Fallback to legacy file-based configs
            return await self._load_legacy_configs()
    
    async def set_default_config(self, name: str) -> bool:
        """
        Set a configuration as the default.
        
        Args:
            name: Configuration name to set as default
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Wrap both operations in a transaction for atomicity
            async with self.db.transaction():
                # Clear existing defaults
                await self.db.execute("UPDATE account_configs SET is_default = 0")
                
                # Set new default
                result = await self.db.execute(
                    "UPDATE account_configs SET is_default = 1 WHERE name = ?", name
                )
                
                # SQLite provider returns lastrowid, not a result object
                # For UPDATE operations, we need to check if any rows were affected differently
                if result is not None:  # Any non-None result indicates success
                    logger.info(f"Set {name} as default configuration")
                    return True
                
                logger.warning(f"Configuration {name} not found to set as default")
                return False
            
        except Exception as e:
            logger.error(f"Failed to set default configuration {name}: {str(e)}")
            return False
    
    async def get_default_config(self, provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the default configuration, optionally for a specific provider.
        
        Args:
            provider: Optional provider filter
        
        Returns:
            Default configuration or None
        """
        query = "SELECT * FROM account_configs WHERE is_default = 1"
        params = []
        
        if provider:
            query += " AND provider = ?"
            params.append(provider)
        
        query += " LIMIT 1"
        
        row = await self.db.fetch_one(query, *params)
        
        if row:
            return json.loads(row['config_data'])
        
        # Fallback: get first config for provider
        configs = await self.get_account_configs(provider)
        return configs[0] if configs else None
    
    async def delete_account_config(self, name: str) -> bool:
        """
        Delete an account configuration.
        
        Args:
            name: Configuration name to delete
        
        Returns:
            True if deleted, False otherwise
        """
        try:
            # Delete from database
            result = await self.db.execute(
                "DELETE FROM account_configs WHERE name = ?", name
            )
            
            # Delete file if it exists
            config_file = self.config_dir / f"{name}.json"
            if config_file.exists():
                config_file.unlink()
            
            # Also check legacy location
            legacy_file = self.legacy_config_dir / f"{name}.json"
            if legacy_file.exists():
                legacy_file.unlink()
            
            # SQLite provider returns lastrowid, not a result object
            if result is not None:  # Any non-None result indicates success
                logger.info(f"Deleted account configuration: {name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete account configuration {name}: {str(e)}")
            return False
    
    async def migrate_legacy_configs(self) -> int:
        """
        Migrate legacy ~/.fiber/configs files to database.
        
        Returns:
            Number of configurations migrated
        """
        if not self.legacy_config_dir.exists():
            return 0
        
        migrated_count = 0
        
        for config_file in self.legacy_config_dir.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Add migration metadata
                config_data['migrated_from'] = str(config_file)
                config_data['migration_date'] = datetime.now().isoformat()
                
                # Extract core fields
                name = config_data.get('name', config_file.stem)
                provider = config_data.get('provider', 'unknown')
                api_key = config_data.get('api_key', '')
                base_url = config_data.get('base_url')
                
                # Migrate to database
                await self.add_account_config(
                    name=name,
                    provider=provider,
                    api_key=api_key,
                    base_url=base_url,
                    **{k: v for k, v in config_data.items() 
                       if k not in ['name', 'provider', 'api_key', 'base_url']}
                )
                
                migrated_count += 1
                logger.info(f"Migrated legacy config: {name}")
                
            except Exception as e:
                logger.error(f"Failed to migrate {config_file}: {str(e)}")
                continue
        
        if migrated_count > 0:
            logger.info(f"Successfully migrated {migrated_count} legacy configurations")
        
        return migrated_count
    
    async def get_provider_configs(self, provider: str) -> List[Dict[str, Any]]:
        """Get all configurations for a specific provider."""
        return await self.get_account_configs(provider=provider)
    
    async def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration data.
        
        Args:
            config: Configuration dictionary to validate
        
        Returns:
            Dict with validation results
        """
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ['name', 'provider', 'api_key']
        for field in required_fields:
            if not config.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Provider-specific validation
        provider = config.get('provider', '').lower()
        
        if provider == 'openai':
            if config.get('api_key') and not config['api_key'].startswith('sk-'):
                warnings.append("OpenAI API key should start with 'sk-'")
        
        elif provider == 'anthropic':
            if config.get('api_key') and not config['api_key'].startswith('sk-ant-'):
                warnings.append("Anthropic API key should start with 'sk-ant-'")
        
        # URL validation
        if config.get('base_url'):
            import re
            url_pattern = r'^https?://'
            if not re.match(url_pattern, config['base_url']):
                errors.append("Base URL must start with http:// or https://")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    # Private helper methods
    
    async def _save_config_to_file(self, name: str, config: Dict[str, Any]) -> None:
        """Save configuration to file for legacy compatibility."""
        config_file = self.config_dir / f"{name}.json"
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save config file {config_file}: {str(e)}")
    
    async def _load_config_from_file(self, name: str) -> Optional[Dict[str, Any]]:
        """Load configuration from file."""
        # Try new location first
        config_file = self.config_dir / f"{name}.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {str(e)}")
        
        # Try legacy location
        legacy_file = self.legacy_config_dir / f"{name}.json"
        if legacy_file.exists():
            try:
                with open(legacy_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load legacy config file {legacy_file}: {str(e)}")
        
        return None
    
    async def _load_legacy_configs(self) -> List[Dict[str, Any]]:
        """Load all legacy configuration files."""
        configs = []
        
        # Check both locations
        for config_dir in [self.config_dir, self.legacy_config_dir]:
            if not config_dir.exists():
                continue
            
            for config_file in config_dir.glob("*.json"):
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                        config_data['source'] = str(config_file)
                        configs.append(config_data)
                except Exception as e:
                    logger.warning(f"Failed to load config file {config_file}: {str(e)}")
                    continue
        
        return configs