"""
OAuth Injection Service

Provides common OAuth service injection logic for both functions and agents.
This ensures consistent OAuth setup across different execution contexts.
"""

import json
import logging
from typing import Dict, Any, Optional, List

from fiberwise_common import DatabaseProvider

logger = logging.getLogger(__name__)


class OAuthInjectionService:
    """
    Service to create OAuth credential services for injection into functions and agents.
    
    This centralizes the logic for loading OAuth authenticators and tokens,
    ensuring both functions and agents get the same OAuth capabilities.
    """
    
    def __init__(self, db: DatabaseProvider):
        self.db = db
    
    async def create_oauth_credential_service(
        self, 
        app_id: str, 
        user_id: Optional[int] = None
    ) -> Optional[Any]:
        """
        Create an OAuth credential service with loaded authenticators and tokens.
        
        Args:
            app_id: The application ID
            user_id: Optional user ID to load user-specific tokens
            
        Returns:
            OAuth credential service instance or None if creation fails
        """
        try:
            # Initialize empty collections
            service_providers = {}
            token_storage = {}
            
            # If we have user_id, fetch the providers and tokens
            if user_id:
                try:
                    # Import here to avoid circular dependencies
                    from .oauth_service import OAuthService
                    
                    oauth_service = OAuthService(self.db)
                    
                    # Get providers with their connection status
                    providers = await oauth_service.get_app_authenticators(
                        app_id=str(app_id),
                        user_id=user_id
                    )
                    
                    logger.info(f"Loading {len(providers)} OAuth authenticators for app {app_id}, user {user_id}")
                    
                    # Convert providers to the format expected by OAuthBackendService
                    for provider in providers:
                        logger.info(f"Processing OAuth authenticator from DB: {provider}")
                        provider_id = provider.get('authenticator_id')
                        
                        if not provider_id:
                            logger.warning(f"Skipping provider with no authenticator_id: {provider}")
                            continue
                        
                        # Create provider config
                        token_url = provider.get('token_url', '')
                        authorize_url = provider.get('auth_url', '') or provider.get('authorize_url', '')
                        
                        provider_config = {
                            "id": provider_id,
                            "name": provider.get('name', ''),
                            "provider_type": provider.get('provider_type', 'oauth2'),
                            "api_base_url": provider.get('api_base_url') or '',
                            "authorize_url": authorize_url,
                            "token_url": token_url,
                            "refresh_url": provider.get('refresh_url') or token_url,
                            "client_id": provider.get('client_id'),
                            "client_secret": provider.get('client_secret'),
                            # Add fields that get_provider_info expects
                            "authenticator_id": provider_id,
                            "authenticator_name": provider.get('name', ''),
                            "authenticator_type": provider.get('provider_type', 'oauth2'),
                            "auth_url": authorize_url,
                            "success": True
                        }
                        
                        logger.info(f"Created provider_config with client_id: {provider_config['client_id'] is not None}, client_secret: {provider_config['client_secret'] is not None}")
                        
                        # Add additional provider info
                        for key in ['display_name', 'authenticator_key', 'scopes', 'configuration', 'brand']:
                            if key in provider and provider[key]:
                                provider_config[key] = provider[key]
                        
                        # Add to service providers dict
                        service_providers[provider_id] = provider_config
                        
                        # If provider is connected, fetch the token data
                        if provider.get('is_connected', False):
                            token_data = await self._fetch_token_data(provider_id, user_id, str(app_id))
                            if token_data:
                                # Store token with multiple possible keys for compatibility
                                token_storage[provider_id] = token_data
                                token_storage[f"{provider_id}:{app_id}"] = token_data
                                token_storage[f"{provider_id}:{app_id}:{user_id}"] = token_data
                                logger.debug(f"Loaded token for authenticator {provider_id}")
                            else:
                                logger.warning(f"No token found for connected authenticator {provider_id}")
                    
                    logger.info(f"Loaded {len(service_providers)} authenticators and {len(token_storage)} tokens for user {user_id}")
                    
                except Exception as db_error:
                    logger.error(f"Error loading OAuth authenticators and tokens: {db_error}")
            
            # Create the OAuth backend service
            return await self._create_oauth_backend_service(
                service_providers=service_providers,
                token_storage=token_storage,
                app_id=str(app_id)
            )
            
        except Exception as e:
            logger.error(f"Error creating OAuth credential service: {e}")
            return None
    
    async def _fetch_token_data(
        self, 
        authenticator_id: str, 
        user_id: int, 
        app_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch token data for a specific authenticator, user, and app.
        
        Args:
            authenticator_id: OAuth authenticator ID
            user_id: User ID
            app_id: Application ID
            
        Returns:
            Token data dictionary or None if not found
        """
        try:
            token_query = """
                SELECT 
                    otg.access_token, 
                    otg.refresh_token, 
                    otg.expires_at, 
                    otg.token_type,
                    otg.scopes as scope
                FROM oauth_token_grants otg
                JOIN user_app_oauth_authentications uaoa 
                    ON otg.grant_id = uaoa.grant_id
                WHERE 
                    uaoa.authenticator_id = $1 AND
                    uaoa.user_id = $2 AND
                    uaoa.app_id = $3 AND
                    uaoa.is_active = true AND
                    otg.is_revoked = false AND
                    otg.access_token IS NOT NULL
                ORDER BY otg.created_at DESC
                LIMIT 1
            """
            
            token_row = await self.db.fetch_one(token_query, authenticator_id, user_id, app_id)
            
            if token_row:
                token_data = {
                    "access_token": token_row.get('access_token'),
                    "refresh_token": token_row.get('refresh_token'),
                    "token_type": token_row.get('token_type', 'Bearer'),
                    "scope": token_row.get('scope'),
                    "expires_at": token_row.get('expires_at')
                }
                
                # Convert expires_at to string if it's a datetime
                if token_data["expires_at"] and hasattr(token_data["expires_at"], 'isoformat'):
                    token_data["expires_at"] = token_data["expires_at"].isoformat()
                
                return token_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching token data for authenticator {authenticator_id}: {e}")
            return None
    
    async def _create_oauth_backend_service(
        self,
        service_providers: Dict[str, Dict[str, Any]],
        token_storage: Dict[str, Dict[str, Any]],
        app_id: str
    ) -> Any:
        """
        Create the OAuth backend service and wrap it in an agent-facing service.
        
        Args:
            service_providers: Dictionary of OAuth authenticator configurations
            token_storage: Dictionary of OAuth tokens
            app_id: Application ID
            
        Returns:
            OAuth credential service for injection into agents/functions
        """
        try:
            # Import the SDK components
            from .oauth_service import OAuthBackendService
            from fiberwise_sdk.credential_agent_service import create_oauth_service_provider_agent
            
            # Create the OAuth backend service
            backend_service = OAuthBackendService(
                service_providers=service_providers,
                token_storage=token_storage,
                app_id=app_id
            )
            
            # Create the agent-facing credential service
            credential_service = create_oauth_service_provider_agent(
                service_provider_backend=backend_service,
                app_id=app_id
            )
            
            logger.info(f"Created OAuth credential service with {len(service_providers)} authenticators for app {app_id}")
            return credential_service
            
        except Exception as e:
            logger.error(f"Error creating OAuth backend service: {e}")
            raise


async def create_oauth_credential_service_for_injection(
    db: DatabaseProvider,
    app_id: str,
    user_id: Optional[int] = None
) -> Optional[Any]:
    """
    Convenience function to create OAuth credential service for injection.
    
    This is the main entry point for both function and agent execution contexts.
    
    Args:
        db: Database provider instance
        app_id: Application ID
        user_id: Optional user ID for user-specific tokens
        
    Returns:
        OAuth credential service ready for injection
    """
    service = OAuthInjectionService(db)
    return await service.create_oauth_credential_service(app_id, user_id)