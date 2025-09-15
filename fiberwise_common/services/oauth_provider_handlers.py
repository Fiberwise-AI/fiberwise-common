"""
OAuth Provider Handlers for various identity providers.

This module contains provider-specific OAuth handlers that abstract the differences
between OAuth providers (Google, GitHub, Microsoft, etc.). This is pure business
logic with no web framework dependencies.

Moved from fiberwise-core-web to fiberwise-common for proper architectural separation.
"""
import logging
import json
import urllib.parse
from typing import Dict, Any, List, Optional, Type, ClassVar
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class OAuthProviderHandler(ABC):
    """Base abstract class for OAuth provider handlers."""
    
    # Define a class attribute for default scopes.
    # Subclasses will override this.
    DEFAULT_SCOPES: ClassVar[List[str]] = []
    
    # Provider identifier used in URLs and config
    provider_id: ClassVar[str]
    
    # Default configuration that can be overridden
    default_config: ClassVar[Dict[str, Any]] = {
        "scopes": [],
        "additional_params": {}
    }
    
    def __init__(self, provider_config: Dict[str, Any]):
        """
        Initialize provider handler with configuration.
        
        Args:
            provider_config: Configuration dict for this provider
        """
        self.config = provider_config
        
    @property
    def client_id(self) -> str:
        """Get client ID from config."""
        return self.config.get("client_id", "")
    
    @property
    def client_secret(self) -> str:
        """Get client secret from config."""
        return self.config.get("client_secret", "")
    
    @property
    def redirect_uri(self) -> str:
        """Get redirect URI from config."""
        return self.config.get("redirect_uri", "")
    
    @property
    def token_url(self) -> str:
        """Get token URL from config."""
        return self.config.get("token_url", "")
    
    @property
    def authorization_url(self) -> str:
        """Get authorization URL from config."""
        return self.config.get("authorization_url", "")
    
    @property
    def state_secret(self) -> str:
        """Get state secret from config."""
        return self.config.get("state_secret", "")
    
    def get_scopes(self) -> List[str]:
        """
        Get configured scopes, falling back to the provider-specific default scopes.
        This implementation is shared by all subclasses. It relies on the
        DEFAULT_SCOPES class attribute being set in the subclass.
        """
        return self.config.get("scopes", self.DEFAULT_SCOPES)
    
    def get_additional_auth_params(self) -> Dict[str, str]:
        """
        Get additional provider-specific parameters for authorization request.
        Automatically includes refresh token parameters for all OAuth providers.
        Override in subclasses for provider-specific behavior.
        """
        # Default parameters that should be included for ALL OAuth providers
        # to ensure we get refresh tokens for automatic token renewal
        base_params = {
            "access_type": "offline",  # Request refresh token (Google, others)
            "prompt": "consent"        # Force consent screen for fresh permissions
        }
        
        # Allow subclasses to add additional provider-specific parameters
        provider_params = self._get_provider_specific_params()
        
        # Merge base params with provider-specific params
        base_params.update(provider_params)
        
        return base_params
    
    def _get_provider_specific_params(self) -> Dict[str, str]:
        """
        Override this method in subclasses for provider-specific parameters.
        """
        return {}
    
    def prepare_token_request(self, code: str) -> Dict[str, Any]:
        """
        Prepare parameters for token request.
        
        Args:
            code: Authorization code from callback
            
        Returns:
            Dict containing params and headers for token request
        """
        params = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code"
        }
        
        headers = {
            "Accept": "application/json"
        }
        
        return {
            "params": params,
            "headers": headers
        }
    
    def process_token_response(self, response_data: Any, content_type: str) -> Dict[str, Any]:
        """
        Process and normalize token response from provider.
        
        Args:
            response_data: Raw response data (text or JSON)
            content_type: Content type of the response
            
        Returns:
            Normalized token data dictionary
        """
        token_data = {}
        
        # Handle JSON responses
        if 'application/json' in content_type:
            if isinstance(response_data, str):
                token_data = json.loads(response_data)
            else:
                token_data = response_data
        
        # Handle form-encoded responses 
        elif 'application/x-www-form-urlencoded' in content_type or isinstance(response_data, str):
            text_response = response_data if isinstance(response_data, str) else response_data.decode('utf-8')
            for item in text_response.split('&'):
                if '=' in item:
                    key, value = item.split('=', 1)
                    token_data[key] = urllib.parse.unquote(value)
        
        # Normalize field names if needed
        if 'access_token' not in token_data and 'accessToken' in token_data:
            token_data['access_token'] = token_data.pop('accessToken')
            
        if 'refresh_token' not in token_data and 'refreshToken' in token_data:
            token_data['refresh_token'] = token_data.pop('refreshToken')
            
        if 'expires_in' not in token_data and 'expiresIn' in token_data:
            token_data['expires_in'] = token_data.pop('expiresIn')
        
        # Convert expires_in to int if present
        if 'expires_in' in token_data:
            try:
                token_data['expires_in'] = int(token_data['expires_in'])
            except (ValueError, TypeError):
                logger.warning(f"Invalid expires_in value: {token_data.get('expires_in')}")
                token_data['expires_in'] = 3600  # Default to 1 hour
                
        return token_data

    def prepare_refresh_token_request(self, refresh_token: str) -> Dict[str, Any]:
        """
        Prepare parameters for refresh token request.
        
        Args:
            refresh_token: Refresh token to use
            
        Returns:
            Dict containing params and headers for refresh request
        """
        params = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        }
        
        headers = {
            "Accept": "application/json"
        }
        
        return {
            "params": params,
            "headers": headers,
            "url": self.token_url  # Use the same token URL for refresh
        }


class GoogleOAuthHandler(OAuthProviderHandler):
    """Handler for Google OAuth provider."""
    
    provider_id = "google"
    DEFAULT_SCOPES = ["openid", "email", "profile"]
    
    
    def _get_provider_specific_params(self) -> Dict[str, str]:
        """Get additional Google-specific parameters."""
        # Google already gets these from the base class, but can add others here
        return {}


class GitHubOAuthHandler(OAuthProviderHandler):
    """Handler for GitHub OAuth provider."""
    
    provider_id = "github"
    
    DEFAULT_SCOPES = ["user:email", "read:user"]


class MicrosoftOAuthHandler(OAuthProviderHandler):
    """Handler for Microsoft/Azure OAuth provider."""
    
    provider_id = "microsoft"
    DEFAULT_SCOPES = ["User.Read", "offline_access"]
    
    
    def _get_provider_specific_params(self) -> Dict[str, str]:
        """Get additional Microsoft-specific parameters."""
        return {
            "response_mode": "query"
        }


class GenericOAuthHandler(OAuthProviderHandler):
    """Generic handler for other OAuth2 providers."""
    
    DEFAULT_SCOPES = []
    provider_id = "generic"
    
    def __init__(self, provider_config: Dict[str, Any], provider_id: str):
        """
        Initialize with a custom provider ID.
        
        Args:
            provider_config: Provider configuration
            provider_id: Custom provider identifier
        """
        super().__init__(provider_config)
        self._provider_id = provider_id
        
    @property
    def provider_id(self) -> str:
        """Override to return the custom provider ID."""
        return self._provider_id
    
    
    def _get_provider_specific_params(self) -> Dict[str, str]:
        """Get any additional parameters defined in config."""
        return self.config.get("additional_params", {})


# Registry of provider handlers
PROVIDER_HANDLERS = {
    "google": GoogleOAuthHandler,
    "github": GitHubOAuthHandler,
    "microsoft": MicrosoftOAuthHandler
}


def get_provider_handler(provider_id: str, provider_config: Dict[str, Any]) -> OAuthProviderHandler:
    """
    Factory function to get the appropriate provider handler.
    
    Args:
        provider_id: Provider identifier
        provider_config: Provider configuration
        
    Returns:
        An initialized provider handler
    """
    handler_class = PROVIDER_HANDLERS.get(provider_id)
    
    if handler_class:
        return handler_class(provider_config)
    else:
        # Fall back to generic handler for unknown providers
        logger.info(f"Using generic handler for provider: {provider_id}")
        return GenericOAuthHandler(provider_config, provider_id)


def register_provider_handler(provider_id: str, handler_class: Type[OAuthProviderHandler]) -> None:
    """
    Register a new provider handler.
    
    Args:
        provider_id: Provider identifier
        handler_class: Provider handler class
    """
    PROVIDER_HANDLERS[provider_id] = handler_class
    logger.info(f"Registered new provider handler for: {provider_id}")