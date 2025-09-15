"""
Enhanced Configuration class with SDK integration for fiberwise_common
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from ..database.manager import DatabaseManager


class EnhancedConfig:
    """
    Enhanced configuration class that supports both CLI and web contexts
    with SDK service configuration.
    """
    
    def __init__(self, db_manager=None, **kwargs):
        # Database configuration
        self._setup_database_config(db_manager)
        
        # SDK service configuration
        self.app_id = kwargs.get('app_id') or os.getenv('FIBERWISE_APP_ID', 'common-app')
        self.api_key = kwargs.get('api_key') or os.getenv('FIBERWISE_API_KEY', 'common-key')
        self.base_url = kwargs.get('base_url') or os.getenv('FIBERWISE_BASE_URL', 'https://api.fiberwise.ai/api/v1')
        
        # General settings
        self.settings = {
            'activation_timeout': kwargs.get('activation_timeout', 30),
            'default_mode': kwargs.get('default_mode', 'standard'),
            'db_path': str(self.db_path),
            'verbose': kwargs.get('verbose', False),
            **kwargs
        }
    
    def _setup_database_config(self, db_manager=None):
        """Setup database configuration"""
        # Check for environment variable first
        env_db_path = os.getenv('FIBERWISE_DB_PATH')
        
        if env_db_path:
            # Use environment variable path
            db_path = Path(env_db_path)
            # Ensure parent directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite:///{str(db_path)}"
        else:
            # Use default path in user's home
            home_dir = Path.home()
            fiberwise_dir = home_dir / '.fiberwise'
            fiberwise_dir.mkdir(exist_ok=True)
            db_path = fiberwise_dir / 'fiberwise.db'
            database_url = f"sqlite:///{str(db_path)}"
        
        # Store both URL and path for compatibility
        self.DATABASE_URL = database_url
        self.DB_PROVIDER = "sqlite"
        self.db_path = db_path
        
        # Create empty database file if it doesn't exist
        if not db_path.exists():
            db_path.touch()
        
        # Use DatabaseManager instead of direct provider
        if db_manager:
            self.db_manager = db_manager
        else:
            self.db_manager = DatabaseManager(database_url)
        
        # Provide db_provider for backward compatibility
        self.db_provider = self.db_manager.provider
    
    def get_sdk_config(self) -> Dict[str, Any]:
        """Get configuration for SDK services"""
        return {
            'app_id': self.app_id,
            'api_key': self.api_key,
            'base_url': self.base_url,
            'version': '1.0.0'
        }
    
    def update_settings(self, **kwargs):
        """Update configuration settings"""
        self.settings.update(kwargs)
        
        # Update SDK settings if provided
        if 'app_id' in kwargs:
            self.app_id = kwargs['app_id']
        if 'api_key' in kwargs:
            self.api_key = kwargs['api_key']
        if 'base_url' in kwargs:
            self.base_url = kwargs['base_url']


# Legacy Config class for backward compatibility
class Config(EnhancedConfig):
    """Legacy Config class - use EnhancedConfig for new code"""
    pass
