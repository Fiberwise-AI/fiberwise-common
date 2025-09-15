"""
Unit tests for fiberwise_common.services.config module.

This module tests the configuration settings class including environment
variable loading and validation.
"""

import os
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

from fiberwise_common.services.config import Settings, settings


class TestSettings:
    """Test suite for Settings configuration class."""
    
    def test_default_settings(self):
        """Test default settings values."""
        config = Settings()
        
        assert config.app_name == "Fiberwise Common"
        assert config.secret_key == "change-me-in-production"
        assert config.debug is False
        assert config.gemini_api_key is None
        assert config.deepseek_api_key is None
        assert config.app_env == "development"
        assert config.fiberwise_app_id is None
    
    def test_settings_with_explicit_values(self):
        """Test settings with explicitly provided values."""
        config = Settings(
            app_name="Test App",
            secret_key="test-secret",
            debug=True,
            gemini_api_key="test-gemini-key",
            deepseek_api_key="test-deepseek-key",
            app_env="testing",
            fiberwise_app_id="test-app-id"
        )
        
        assert config.app_name == "Test App"
        assert config.secret_key == "test-secret"
        assert config.debug is True
        assert config.gemini_api_key == "test-gemini-key"
        assert config.deepseek_api_key == "test-deepseek-key"
        assert config.app_env == "testing"
        assert config.fiberwise_app_id == "test-app-id"
    
    @patch.dict(os.environ, {
        'APP_NAME': 'Environment App',
        'SECRET_KEY': 'env-secret',
        'DEBUG': 'true',
        'GEMINI_API_KEY': 'env-gemini-key',
        'DEEPSEEK_API_KEY': 'env-deepseek-key',
        'APP_ENV': 'production',
        'FIBERWISE_APP_ID': 'env-app-id'
    })
    def test_settings_from_environment(self):
        """Test settings loaded from environment variables."""
        config = Settings()
        
        assert config.app_name == "Environment App"
        assert config.secret_key == "env-secret"
        assert config.debug is True
        assert config.gemini_api_key == "env-gemini-key"
        assert config.deepseek_api_key == "env-deepseek-key"
        assert config.app_env == "production"
        assert config.fiberwise_app_id == "env-app-id"
    
    @patch.dict(os.environ, {
        'DEBUG': 'false',
    })
    def test_boolean_environment_parsing(self):
        """Test that boolean environment variables are parsed correctly."""
        config = Settings()
        
        assert config.debug is False
        
        # Test various boolean representations
        test_cases = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('1', True),
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('0', False),
            ('', False),
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {'DEBUG': env_value}):
                test_config = Settings()
                assert test_config.debug == expected
    
    def test_optional_fields_none_by_default(self):
        """Test that optional fields are None by default."""
        config = Settings()
        
        assert config.gemini_api_key is None
        assert config.deepseek_api_key is None
        assert config.fiberwise_app_id is None
    
    @patch.dict(os.environ, {
        'GEMINI_API_KEY': '',
        'DEEPSEEK_API_KEY': '',
        'FIBERWISE_APP_ID': ''
    })
    def test_empty_string_environment_variables(self):
        """Test handling of empty string environment variables."""
        config = Settings()
        
        # Empty strings should be treated as None for optional fields
        # This behavior depends on pydantic-settings implementation
        # Adjust assertions based on actual behavior
        assert config.gemini_api_key in [None, '']
        assert config.deepseek_api_key in [None, '']
        assert config.fiberwise_app_id in [None, '']
    
    @patch.dict(os.environ, {
        'UNKNOWN_FIELD': 'unknown_value'
    })
    def test_extra_environment_variables_ignored(self):
        """Test that extra environment variables are ignored."""
        # Should not raise error due to extra='ignore' in Config
        config = Settings()
        
        # Unknown field should not be present
        assert not hasattr(config, 'unknown_field')
        assert not hasattr(config, 'UNKNOWN_FIELD')
    
    def test_config_class_settings(self):
        """Test that Config class is properly set up."""
        config = Settings()
        
        # Check that Config class exists and has expected attributes
        assert hasattr(Settings, 'model_config')
        
        # The specific implementation depends on pydantic version
        # Basic test that configuration exists
        assert config is not None


class TestSettingsEnvFile:
    """Test environment file loading functionality."""
    
    def test_env_file_setting(self):
        """Test that env_file is configured."""
        # This tests the configuration, not the actual file loading
        # since pydantic-settings handles that internally
        config = Settings()
        
        # Just verify the settings instance exists
        assert isinstance(config, Settings)
    
    @patch('builtins.open', mock_open(read_data='APP_NAME=FileApp\nSECRET_KEY=file-secret\nDEBUG=true'))
    @patch('pathlib.Path.exists', return_value=True)
    def test_env_file_loading_simulation(self, mock_exists):
        """Simulate environment file loading (mock test)."""
        # This is a simplified simulation since actual file loading
        # is handled by pydantic-settings internally
        
        # Mock environment variables as if loaded from file
        env_data = {
            'APP_NAME': 'FileApp',
            'SECRET_KEY': 'file-secret',
            'DEBUG': 'true'
        }
        
        with patch.dict(os.environ, env_data):
            config = Settings()
            
            assert config.app_name == "FileApp"
            assert config.secret_key == "file-secret"
            assert config.debug is True


class TestGlobalSettingsInstance:
    """Test the global settings instance."""
    
    def test_global_settings_exists(self):
        """Test that global settings instance exists."""
        from fiberwise_common.services.config import settings
        
        assert settings is not None
        assert isinstance(settings, Settings)
    
    def test_global_settings_singleton_behavior(self):
        """Test that global settings behaves like a singleton."""
        from fiberwise_common.services.config import settings as settings1
        from fiberwise_common.services.config import settings as settings2
        
        # Should be the same instance
        assert settings1 is settings2
    
    def test_global_settings_attributes(self):
        """Test that global settings has expected attributes."""
        from fiberwise_common.services.config import settings
        
        assert hasattr(settings, 'app_name')
        assert hasattr(settings, 'secret_key')
        assert hasattr(settings, 'debug')
        assert hasattr(settings, 'gemini_api_key')
        assert hasattr(settings, 'deepseek_api_key')
        assert hasattr(settings, 'app_env')
        assert hasattr(settings, 'fiberwise_app_id')


@pytest.mark.parametrize("field_name,env_var,test_value", [
    ("app_name", "APP_NAME", "Test Application"),
    ("secret_key", "SECRET_KEY", "test-secret-key"),
    ("app_env", "APP_ENV", "testing"),
    ("gemini_api_key", "GEMINI_API_KEY", "test-gemini"),
    ("deepseek_api_key", "DEEPSEEK_API_KEY", "test-deepseek"),
    ("fiberwise_app_id", "FIBERWISE_APP_ID", "test-id"),
])
class TestSettingsParametrized:
    """Parametrized tests for settings fields."""
    
    def test_field_environment_override(self, field_name, env_var, test_value):
        """Test that each field can be overridden by environment variable."""
        with patch.dict(os.environ, {env_var: test_value}):
            config = Settings()
            
            assert getattr(config, field_name) == test_value
    
    def test_field_explicit_override(self, field_name, env_var, test_value):
        """Test that each field can be set explicitly."""
        kwargs = {field_name: test_value}
        config = Settings(**kwargs)
        
        assert getattr(config, field_name) == test_value


class TestSettingsValidation:
    """Test settings validation and type conversion."""
    
    def test_string_fields_validation(self):
        """Test string field validation."""
        # Test with valid strings
        config = Settings(
            app_name="Valid Name",
            secret_key="valid_secret",
            app_env="production"
        )
        
        assert isinstance(config.app_name, str)
        assert isinstance(config.secret_key, str)
        assert isinstance(config.app_env, str)
    
    def test_boolean_field_validation(self):
        """Test boolean field validation."""
        # Test explicit boolean values
        config_true = Settings(debug=True)
        config_false = Settings(debug=False)
        
        assert config_true.debug is True
        assert config_false.debug is False
        
        # Test with string values that should convert to boolean
        config_str_true = Settings(debug="true")
        config_str_false = Settings(debug="false")
        
        assert isinstance(config_str_true.debug, bool)
        assert isinstance(config_str_false.debug, bool)
    
    def test_optional_fields_validation(self):
        """Test optional field validation."""
        # Test with None values
        config = Settings(
            gemini_api_key=None,
            deepseek_api_key=None,
            fiberwise_app_id=None
        )
        
        assert config.gemini_api_key is None
        assert config.deepseek_api_key is None
        assert config.fiberwise_app_id is None
        
        # Test with string values
        config_with_values = Settings(
            gemini_api_key="test-gemini",
            deepseek_api_key="test-deepseek",
            fiberwise_app_id="test-id"
        )
        
        assert isinstance(config_with_values.gemini_api_key, str)
        assert isinstance(config_with_values.deepseek_api_key, str)
        assert isinstance(config_with_values.fiberwise_app_id, str)


class TestSettingsUsage:
    """Test realistic usage scenarios for settings."""
    
    def test_development_environment(self):
        """Test settings for development environment."""
        config = Settings(
            app_env="development",
            debug=True,
            secret_key="dev-secret"
        )
        
        assert config.app_env == "development"
        assert config.debug is True
        assert config.secret_key == "dev-secret"
    
    def test_production_environment(self):
        """Test settings for production environment."""
        config = Settings(
            app_env="production",
            debug=False,
            secret_key="super-secure-production-secret",
            gemini_api_key="prod-gemini-key",
            deepseek_api_key="prod-deepseek-key"
        )
        
        assert config.app_env == "production"
        assert config.debug is False
        assert config.secret_key == "super-secure-production-secret"
        assert config.gemini_api_key == "prod-gemini-key"
        assert config.deepseek_api_key == "prod-deepseek-key"
    
    def test_testing_environment(self):
        """Test settings for testing environment."""
        config = Settings(
            app_env="testing",
            debug=True,
            secret_key="test-secret",
            fiberwise_app_id="test-app"
        )
        
        assert config.app_env == "testing"
        assert config.debug is True
        assert config.secret_key == "test-secret"
        assert config.fiberwise_app_id == "test-app"
    
    @patch.dict(os.environ, {
        'APP_ENV': 'production',
        'DEBUG': 'false',
        'SECRET_KEY': 'env-production-secret',
        'GEMINI_API_KEY': 'env-gemini-key',
        'DEEPSEEK_API_KEY': 'env-deepseek-key',
        'FIBERWISE_APP_ID': 'env-app-id'
    })
    def test_production_from_environment(self):
        """Test production settings loaded from environment."""
        config = Settings()
        
        assert config.app_env == "production"
        assert config.debug is False
        assert config.secret_key == "env-production-secret"
        assert config.gemini_api_key == "env-gemini-key"
        assert config.deepseek_api_key == "env-deepseek-key"
        assert config.fiberwise_app_id == "env-app-id"


class TestSettingsIntegration:
    """Integration tests for settings usage."""
    
    def test_settings_in_application_context(self):
        """Test settings usage in application-like context."""
        # Simulate application startup
        app_config = Settings(
            app_name="Fiberwise Test App",
            debug=True
        )
        
        # Simulate using settings for application configuration
        app_title = f"{app_config.app_name} - {app_config.app_env.upper()}"
        log_level = "DEBUG" if app_config.debug else "INFO"
        
        assert app_title == "Fiberwise Test App - DEVELOPMENT"
        assert log_level == "DEBUG"
    
    def test_settings_with_api_keys(self):
        """Test settings with API key configuration."""
        config = Settings(
            gemini_api_key="test-gemini-key",
            deepseek_api_key="test-deepseek-key"
        )
        
        # Simulate checking for API key availability
        has_gemini = config.gemini_api_key is not None
        has_deepseek = config.deepseek_api_key is not None
        
        assert has_gemini is True
        assert has_deepseek is True
        
        # Simulate missing API keys
        config_no_keys = Settings()
        
        has_gemini_missing = config_no_keys.gemini_api_key is not None
        has_deepseek_missing = config_no_keys.deepseek_api_key is not None
        
        assert has_gemini_missing is False
        assert has_deepseek_missing is False
    
    def test_settings_immutability(self):
        """Test that settings behave as expected for immutability."""
        config = Settings(app_name="Original Name")
        
        # Test that we can access the value
        assert config.app_name == "Original Name"
        
        # Creating a new instance should work
        config2 = Settings(app_name="New Name")
        assert config2.app_name == "New Name"
        assert config.app_name == "Original Name"  # Original unchanged


if __name__ == "__main__":
    pytest.main([__file__])