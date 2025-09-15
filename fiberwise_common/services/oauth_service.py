"""
Consolidated OAuth Service

Manages OAuth authenticator registration, authentication flows, token management,
and agent injection. This service consolidates functionality from:
- oauth_service.py (original)
- oauth_backend_service.py 
- oauth_injection_service.py

All OAuth business logic now lives in fiberwise-common for proper separation.
"""

import json
import time
import secrets
import httpx
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

from .base_service import BaseService
from ..database.query_adapter import QueryAdapter, ParameterStyle
from .oauth_provider_handlers import get_provider_handler

logger = logging.getLogger(__name__)


class OAuthService(BaseService):
    """
    Consolidated OAuth service that handles all OAuth operations.
    
    This service handles:
    - OAuth authenticator registration and configuration
    - OAuth session state management  
    - Authorization flow handling
    - Token exchange, storage, and refresh
    - Agent injection and credential services
    - Provider-specific OAuth logic
    """
    
    def __init__(self, db_provider):
        super().__init__(db_provider)
        self.query_adapter = QueryAdapter(ParameterStyle.SQLITE)
    
    # ===== AUTHENTICATOR MANAGEMENT =====
    
    async def register_oauth_authenticator(self, authenticator_name: str, client_id: str,
                                         client_secret: str, redirect_uri: str,
                                         scopes: List[str], authenticator_type: str = "oauth2",
                                         app_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Register a new OAuth authenticator.
        
        Args:
            authenticator_name: Name of the OAuth authenticator (e.g., 'github', 'google')
            client_id: OAuth client ID
            client_secret: OAuth client secret  
            redirect_uri: Redirect URI for OAuth callback
            scopes: List of OAuth scopes to request
            authenticator_type: Type of authenticator (default: oauth2)
            app_id: Optional app ID to associate with
            **kwargs: Additional authenticator-specific configuration
        
        Returns:
            Dict containing the registered authenticator configuration
        """
        authenticator_id = str(uuid4())
        authenticator_config = {
            "authenticator_id": authenticator_id,
            "authenticator_name": authenticator_name,
            "authenticator_type": authenticator_type,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "scopes": scopes,
            "app_id": app_id,
            "configuration": kwargs,
            "is_active": True,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        try:
            query = self.query_adapter.convert_query("""
                INSERT INTO oauth_authenticators
                (authenticator_id, authenticator_name, authenticator_type, client_id, client_secret, 
                 redirect_uri, scopes, app_id, configuration, is_active, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """, ParameterStyle.POSTGRESQL)
            
            await self.db.execute(
                query, authenticator_id, authenticator_name, authenticator_type,
                client_id, client_secret, redirect_uri, json.dumps(scopes),
                app_id, json.dumps(kwargs), True,
                authenticator_config["created_at"], authenticator_config["updated_at"]
            )
            
            logger.info(f"Registered OAuth authenticator: {authenticator_name}")
            return authenticator_config
            
        except Exception as e:
            logger.error(f"Failed to register OAuth authenticator {authenticator_name}: {str(e)}")
            raise
    
    async def get_oauth_authenticator(self, authenticator_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific OAuth authenticator by ID."""
        query = self.query_adapter.convert_query("""
            SELECT * FROM oauth_authenticators WHERE authenticator_id = $1 AND is_active = true
        """, ParameterStyle.POSTGRESQL)
        
        row = await self.db.fetch_one(query, authenticator_id)
        
        if row:
            # Parse scopes and unescape any HTML entities
            import html
            scopes_raw = json.loads(row["scopes"]) if row["scopes"] else []
            scopes_unescaped = [html.unescape(scope) for scope in scopes_raw] if isinstance(scopes_raw, list) else scopes_raw
            
            return {
                "authenticator_id": row["authenticator_id"],
                "authenticator_name": row["authenticator_name"],
                "authenticator_type": row["authenticator_type"], 
                "client_id": row["client_id"],
                "client_secret": row["client_secret"],
                "redirect_uri": row["redirect_uri"],
                "scopes": scopes_unescaped,
                "configuration": json.loads(row["configuration"]) if row["configuration"] else {},
                "app_id": row["app_id"],
                "is_active": row["is_active"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
        
        return None

    async def get_app_authenticators(self, app_id: str, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get OAuth authenticators for a specific app, optionally with connection status.
        
        Args:
            app_id: Application ID
            user_id: Optional user ID to check connection status
            
        Returns:
            List of authenticator dictionaries with connection info
        """
        try:
            if user_id:
                # Include connection status for the user
                query = self.query_adapter.convert_query("""
                SELECT 
                    oa.authenticator_id,
                    oa.authenticator_name,
                    oa.authenticator_type,
                    oa.scopes,
                    oa.authorize_url,
                    oa.token_url,
                    oa.client_id,
                    oa.client_secret,
                    oa.configuration,
                    CASE 
                        WHEN EXISTS (
                            SELECT 1 FROM user_app_oauth_authentications uaoa
                            JOIN oauth_token_grants otg ON uaoa.grant_id = otg.grant_id
                            WHERE uaoa.authenticator_id = oa.authenticator_id 
                            AND uaoa.app_id = $1 AND uaoa.user_id = $2 
                            AND uaoa.is_active = true AND otg.is_revoked = false
                            AND otg.access_token IS NOT NULL AND otg.access_token != 'pending'
                        )
                        THEN true 
                        ELSE false 
                    END as is_connected,
                    CASE 
                        WHEN EXISTS (
                            SELECT 1 FROM user_app_oauth_authentications uaoa
                            JOIN oauth_token_grants otg ON uaoa.grant_id = otg.grant_id
                            WHERE uaoa.authenticator_id = oa.authenticator_id 
                            AND uaoa.app_id = $3 AND uaoa.user_id = $4 
                            AND uaoa.is_active = true AND otg.is_revoked = false
                            AND otg.access_token = 'pending'
                        )
                        THEN 'pending'
                        WHEN EXISTS (
                            SELECT 1 FROM user_app_oauth_authentications uaoa
                            JOIN oauth_token_grants otg ON uaoa.grant_id = otg.grant_id
                            WHERE uaoa.authenticator_id = oa.authenticator_id 
                            AND uaoa.app_id = $5 AND uaoa.user_id = $6 
                            AND uaoa.is_active = true AND otg.is_revoked = false
                            AND otg.access_token IS NOT NULL AND otg.access_token != 'pending'
                        )
                        THEN 'connected'
                        ELSE 'disconnected'
                    END as connection_status
                FROM oauth_authenticators oa
                WHERE oa.app_id = $7 AND oa.is_active = true
                """)
                authenticators = await self.db.fetch_all(query, str(app_id), user_id, str(app_id), user_id, str(app_id), user_id, str(app_id))
            else:
                # Just get the authenticators, no connection status
                query = self.query_adapter.convert_query("""
                SELECT 
                    oa.authenticator_id,
                    oa.authenticator_name,
                    oa.authenticator_type,
                    oa.scopes,
                    oa.authorize_url,
                    oa.token_url,
                    oa.client_id,
                    oa.client_secret,
                    oa.configuration,
                    false as is_connected
                FROM oauth_authenticators oa
                WHERE oa.app_id = $1 AND oa.is_active = true
                """)
                authenticators = await self.db.fetch_all(query, str(app_id))
            
            # Format results
            result = []
            for auth in authenticators:
                auth_dict = dict(auth)
                logger.info(f"Raw database row: {auth_dict}")  # Debug: see what we actually get from DB
                
                # Parse scopes if they're in PostgreSQL array format or JSON string
                scopes = auth_dict.get('scopes', [])
                if isinstance(scopes, str):
                    # Handle JSON array string format: ["openid", "profile", "email"]
                    if scopes.startswith('[') and scopes.endswith(']'):
                        try:
                            scopes = json.loads(scopes)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON scopes: {scopes}")
                            scopes = []
                    # Handle PostgreSQL array format: {"openid","profile","email"}
                    elif scopes.startswith('{') and scopes.endswith('}'):
                        scopes = [scope.strip('"') for scope in scopes[1:-1].split(',') if scope.strip()]
                    # Handle malformed HTML-encoded format
                    elif '&quot;' in scopes:
                        import html
                        unescaped = html.unescape(scopes)
                        if unescaped.startswith('[') and unescaped.endswith(']'):
                            try:
                                scopes = json.loads(unescaped)
                            except json.JSONDecodeError:
                                scopes = []
                        else:
                            scopes = []
                    else:
                        # Single scope as string
                        scopes = [scopes] if scopes else []
                
                # Ensure scopes is always a list
                if not isinstance(scopes, list):
                    scopes = []
                
                # Format authenticator data
                authenticator_data = {
                    "id": str(auth_dict.get('authenticator_id')),  # Map authenticator_id to id
                    "name": auth_dict.get('authenticator_name'),
                    "authenticator_type": auth_dict.get('authenticator_type'),
                    "scopes": scopes,
                    "is_connected": auth_dict.get('is_connected', False),
                    "connection_status": auth_dict.get('connection_status', 'disconnected'),
                    "auth_url": auth_dict.get('authorize_url'),
                    "token_url": auth_dict.get('token_url'),
                    "client_id": auth_dict.get('client_id'),
                    "client_secret": auth_dict.get('client_secret')
                }
                
                # Include configuration if present (sanitized)
                if auth_dict.get('configuration'):
                    config = auth_dict.get('configuration')
                    if isinstance(config, str):
                        try:
                            config = json.loads(config)
                        except json.JSONDecodeError:
                            config = {}
                    
                    # Remove sensitive fields
                    if isinstance(config, dict):
                        safe_config = {
                            k: v for k, v in config.items() 
                            if k not in ['client_secret', 'api_key', 'password', 'secret']
                        }
                        authenticator_data["configuration"] = safe_config
                
                result.append(authenticator_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving authenticators for app {app_id}: {str(e)}")
            return []
    
    # ===== SESSION MANAGEMENT =====
    
    async def create_oauth_session(self, authenticator_id: str, user_id: Optional[int] = None, 
                                 redirect_uri: Optional[str] = None,
                                 scopes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create an OAuth session for authorization flow.
        
        Args:
            authenticator_id: ID of the OAuth authenticator
            user_id: Optional user ID for the session
            redirect_uri: Optional custom redirect URI
            scopes: Optional custom scopes (overrides authenticator defaults)
        
        Returns:
            Dict containing session information including state token
        """
        # Get authenticator configuration
        authenticator = await self.get_oauth_authenticator(authenticator_id)
        if not authenticator:
            raise ValueError(f"OAuth authenticator '{authenticator_id}' not found or inactive")
        
        # Generate session data
        session_id = secrets.token_urlsafe(32)
        state_token = secrets.token_urlsafe(32)
        expires_at = (datetime.now() + timedelta(hours=1)).isoformat()
        
        # Use custom redirect URI or authenticator default
        final_redirect_uri = redirect_uri or authenticator["redirect_uri"]
        
        # Use custom scopes or authenticator defaults
        final_scopes = scopes or authenticator["scopes"]
        
        session_data = {
            "session_id": session_id,
            "authenticator_id": authenticator_id,
            "user_id": user_id,
            "state_token": state_token,
            "redirect_uri": final_redirect_uri,
            "scopes": final_scopes,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at
        }
        
        try:
            query = self.query_adapter.convert_query("""
                INSERT INTO oauth_sessions
                (session_id, authenticator_id, user_id, state_token, redirect_uri, scopes, created_at, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, ParameterStyle.POSTGRESQL)
            
            await self.db.execute(
                query, session_id, authenticator_id, user_id, state_token,
                final_redirect_uri, json.dumps(final_scopes),
                session_data["created_at"], expires_at
            )
            
            logger.info(f"Created OAuth session for authenticator: {authenticator_id}")
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to create OAuth session: {str(e)}")
            raise
    
    async def get_oauth_session(self, state_token: str) -> Optional[Dict[str, Any]]:
        """Get OAuth session by state token."""
        query = self.query_adapter.convert_query("""
            SELECT * FROM oauth_sessions 
            WHERE state_token = $1 AND expires_at > datetime('now')
        """, ParameterStyle.POSTGRESQL)
        
        row = await self.db.fetch_one(query, state_token)
        
        if row:
            return {
                "id": row["id"],
                "session_id": row["session_id"],
                "authenticator_id": row["authenticator_id"],
                "user_id": row["user_id"],
                "state_token": row["state_token"],
                "redirect_uri": row["redirect_uri"],
                "scopes": json.loads(row["scopes"]) if row["scopes"] else [],
                "created_at": row["created_at"],
                "expires_at": row["expires_at"],
                "completed_at": row.get("completed_at")
            }
        
        return None
    
    # ===== TOKEN MANAGEMENT =====
    
    async def exchange_code_for_token(
        self, 
        authenticator_id: str, 
        code: str, 
        app_id: str,
        user_id: int,
        redirect_uri: str
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for access token using provider-specific logic.
        
        Args:
            authenticator_id: ID of the OAuth authenticator
            code: Authorization code from callback
            app_id: Application ID
            user_id: User ID
            redirect_uri: Redirect URI used in authorization
            
        Returns:
            Dict with token data or error
        """
        try:
            # Get authenticator config
            authenticator = await self.get_oauth_authenticator(authenticator_id)
            if not authenticator:
                return {
                    "success": False,
                    "error": f"Authenticator {authenticator_id} not found"
                }
            
            # Get provider-specific handler
            handler = get_provider_handler(
                authenticator["authenticator_type"], 
                authenticator
            )
            
            # Prepare token request using handler
            request_data = handler.prepare_token_request(code)
            
            # Make token exchange request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    authenticator["token_url"],
                    data=request_data["params"],
                    headers=request_data["headers"],
                    timeout=30.0
                )
                response.raise_for_status()
                
                # Process response using handler
                content_type = response.headers.get("content-type", "")
                token_data = handler.process_token_response(response.json(), content_type)
                
                # Store token in database
                await self._store_token(authenticator_id, app_id, user_id, token_data)
                
                return {
                    "success": True,
                    "token_data": token_data
                }
                
        except Exception as e:
            logger.error(f"Token exchange failed for {authenticator_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _store_token(
        self, 
        authenticator_id: str, 
        app_id: str, 
        user_id: int, 
        token_data: Dict[str, Any]
    ) -> None:
        """Store OAuth token in database."""
        try:
            grant_id = str(uuid4())
            access_token = token_data.get("access_token")
            refresh_token = token_data.get("refresh_token")
            token_type = token_data.get("token_type", "Bearer")
            scope = token_data.get("scope")
            
            # Calculate expiration time
            expires_in = token_data.get("expires_in")
            expires_at = None
            if expires_in:
                expires_at = datetime.now() + timedelta(seconds=int(expires_in))
            
            # Store token grant
            token_query = """
                INSERT INTO oauth_token_grants (
                    grant_id, user_id, authenticator_id, access_token, refresh_token, 
                    expires_at, token_type, scopes, created_at, updated_at, is_revoked
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, false)
            """
            
            await self.db.execute(
                token_query,
                grant_id, user_id, authenticator_id, access_token, refresh_token,
                expires_at, token_type, json.dumps(scope.split() if scope else [])
            )
            
            # Create user-app-authenticator association
            auth_query = """
                INSERT INTO user_app_oauth_authentications (
                    user_id, app_id, authenticator_id, grant_id, 
                    auth_status, created_at, updated_at, is_active
                )
                VALUES ($1, $2, $3, $4, 'completed', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, true)
                ON CONFLICT (user_id, app_id, authenticator_id)
                DO UPDATE SET
                    grant_id = EXCLUDED.grant_id,
                    auth_status = EXCLUDED.auth_status,
                    updated_at = CURRENT_TIMESTAMP,
                    is_active = true
            """
            
            await self.db.execute(auth_query, user_id, app_id, authenticator_id, grant_id)
            
            logger.info(f"Successfully stored token for authenticator {authenticator_id}")
            
        except Exception as e:
            logger.error(f"Failed to store token: {e}")
            raise
    
    async def get_user_token(
        self, 
        authenticator_id: str, 
        user_id: int, 
        app_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get stored token for user and authenticator."""
        try:
            query = """
                SELECT 
                    otg.access_token, 
                    otg.refresh_token, 
                    otg.expires_at, 
                    otg.token_type,
                    otg.scopes
                FROM oauth_token_grants otg
                JOIN user_app_oauth_authentications uaoa ON otg.grant_id = uaoa.grant_id
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
            
            token_row = await self.db.fetch_one(query, authenticator_id, user_id, app_id)
            
            if token_row:
                return {
                    "access_token": token_row.get('access_token'),
                    "refresh_token": token_row.get('refresh_token'),
                    "token_type": token_row.get('token_type', 'Bearer'),
                    "scope": token_row.get('scopes'),
                    "expires_at": token_row.get('expires_at')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching token: {e}")
            return None
    
    # ===== AUTHENTICATED REQUESTS =====
    
    async def make_authenticated_request(
        self,
        authenticator_id: str,
        endpoint: str,
        user_id: int,
        app_id: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Any] = None,
        content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make an authenticated request to a service provider's API.
        
        Args:
            authenticator_id: ID of the OAuth authenticator
            endpoint: API endpoint to call
            user_id: User ID
            app_id: Application ID  
            method: HTTP method
            params: Query parameters
            headers: Request headers
            json_data: JSON body
            content: Raw content body
            
        Returns:
            Dict with response data or error
        """
        try:
            # Get token
            token_data = await self.get_user_token(authenticator_id, user_id, app_id)
            if not token_data or not token_data.get("access_token"):
                return {
                    "success": False,
                    "error": "No valid token available"
                }
            
            # Get authenticator config for API base URL
            authenticator = await self.get_oauth_authenticator(authenticator_id)
            if not authenticator:
                return {
                    "success": False,
                    "error": f"Authenticator {authenticator_id} not found"
                }
            
            # Build URL
            api_base_url = authenticator.get("configuration", {}).get("api_base_url", "")
            if endpoint.startswith(("http://", "https://")):
                url = endpoint
            else:
                url = f"{api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            
            # Prepare headers
            req_headers = headers or {}
            req_headers.setdefault("Authorization", f"Bearer {token_data['access_token']}")
            req_headers.setdefault("Accept", "application/json")
            
            # Make request
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=req_headers,
                    params=params,
                    json=json_data,
                    content=content,
                    timeout=60.0,
                )
                
                if response.status_code >= 400:
                    error_data = {}
                    try:
                        error_data = response.json()
                    except Exception:
                        error_data = {"text": response.text}
                    
                    return {
                        "success": False,
                        "error": f"HTTP error {response.status_code}",
                        "status_code": response.status_code,
                        "error_data": error_data
                    }
                
                # Parse response
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        data = {"text": response.text}
                else:
                    data = {"text": response.text}
                
                return {
                    "success": True,
                    "data": data,
                    "status_code": response.status_code
                }
                
        except Exception as e:
            logger.error(f"Error making authenticated request: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # ===== BACKEND SERVICE METHODS =====
    
    async def _get_token(self, service_provider_id: str, app_id: Optional[str] = None, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a token for the specified provider from the database
        
        Args:
            service_provider_id: ID of the OAuth provider
            app_id: Optional app ID context
            user_id: Optional user ID
            
        Returns:
            Token data or None if not found
        """
        try:
            if not app_id or not user_id:
                logger.warning("Cannot get token without app_id and user_id")
                return None
            
            # Query the database for the token
            query = """
                SELECT 
                    access_token,
                    refresh_token,
                    expires_at,
                    token_type,
                    scopes as scope
                FROM oauth_token_grants
                WHERE authenticator_id = $1 AND user_id = $2
            """
            
            token_record = await self.db.fetch_one(query, service_provider_id, user_id)
            
            if not token_record:
                return None
            
            # Check for expiration
            expires_at = token_record.get("expires_at")
            if expires_at and isinstance(expires_at, datetime) and expires_at < datetime.now():
                logger.info(f"Token for provider {service_provider_id} is expired. Attempting refresh.")
                return await self._refresh_token(service_provider_id, token_record['refresh_token'], app_id, user_id)
            
            return dict(token_record)
                
        except Exception as e:
            logger.error(f"Error retrieving token for {service_provider_id}: {str(e)}")
            return None
    
    async def make_provider_request(
        self,
        service_provider_id: str,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Any] = None,
        content: Optional[str] = None,
        app_id: Optional[str] = None,
        user_id: Optional[int] = None,
        on_behalf_of: Optional[str] = None
    ):
        """
        Make an authenticated request to a service provider's API.
        """
        # Use provided context
        if not app_id or not user_id:
            return {
                "status": 401,
                "body": {"error": "app_id and user_id are required."}
            }
        
        # Get the token from the database
        token_data = await self._get_token(service_provider_id, app_id, user_id)
        if not token_data or "access_token" not in token_data:
            return {
                "status": 401,
                "body": {"error": "Authentication token not available or invalid."}
            }
        
        # Get service provider details
        authenticator = await self.get_oauth_authenticator(service_provider_id)
        if not authenticator:
            return {
                "status": 404,
                "body": {"error": "Service provider not found."}
            }
        
        # Build the request URL
        config = authenticator.get("configuration", {})
        api_base_url = config.get("api_base_url", "")
        
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            url = endpoint
        else:
            url = f"{api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Prepare the request headers
        req_headers = headers or {}
        req_headers.setdefault("Authorization", f"Bearer {token_data['access_token']}")
        req_headers.setdefault("Accept", "application/json")
        
        # Make the request
        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=req_headers,
                    params=params,
                    json=json_data,
                    content=content,
                    timeout=60.0,
                )
                
                # Check for HTTP errors
                if response.status_code >= 400:
                    error_data = {}
                    try:
                        error_data = response.json()
                    except Exception:
                        error_data = {"text": response.text}
                    
                    return {
                        "success": False,
                        "error": f"HTTP error {response.status_code}",
                        "status_code": response.status_code,
                        "error_data": error_data
                    }
                
                # Parse the response
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        data = {"text": response.text}
                else:
                    data = {"text": response.text}
                
                return {
                    "success": True,
                    "data": data,
                    "status_code": response.status_code
                }
                
            except httpx.RequestError as e:
                logger.error(f"Request error for provider {service_provider_id}: {str(e)}")
                return {
                    "success": False,
                    "error": f"Request error: {str(e)}"
                }
        
            except Exception as e:
                logger.error(f"Error making request for provider {service_provider_id}: {str(e)}")
                return {
                    "success": False,
                    "error": str(e)
                }
    
    async def _refresh_token(
        self, 
        service_provider_id: str, 
        refresh_token: str,
        app_id: str,
        user_id: int
    ) -> Optional[Dict[str, Any]]:
        """Refresh an expired OAuth token."""
        authenticator = await self.get_oauth_authenticator(service_provider_id)
        if not authenticator:
            logger.error(f"Authenticator {service_provider_id} not found.")
            return None
        
        config = authenticator.get("configuration", {})
        refresh_url = config.get("refresh_url") or config.get("token_url")
        
        if not refresh_url:
            logger.error(f"Authenticator {service_provider_id} does not support token refresh.")
            return None
        
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": authenticator["client_id"],
            "client_secret": authenticator["client_secret"],
            "redirect_uri": authenticator.get("redirect_uri")
        }
        
        # Make the request to refresh the token
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(refresh_url, data=data)
                response.raise_for_status()
                new_token_data = response.json()
                
                # Update the token in the database
                await self._store_token(service_provider_id, app_id, user_id, new_token_data)
                
                return new_token_data
            except httpx.HTTPStatusError as e:
                logger.error(f"Failed to refresh token: {e.response.status_code} - {e.response.text}")
                return None
            except Exception as e:
                logger.error(f"An unexpected error occurred during token refresh: {e}")
                return None

    async def get_authorization_url(
        self,
        service_provider_id: str,
        redirect_uri: str,
        scope: str = "",
        state: str = ""
    ) -> str:
        """
        Generate the authorization URL for the OAuth provider.
        
        Args:
            service_provider_id: The ID of the service provider
            redirect_uri: The URI to redirect to after authorization
            scope: The scopes requested during authorization
            state: An optional state parameter to maintain state between request and callback
            
        Returns:
            The authorization URL as a string
        """
        authenticator = await self.get_oauth_authenticator(service_provider_id)
        if not authenticator:
            raise ValueError("Invalid service provider or missing authorization URL.")
        
        config = authenticator.get("configuration", {})
        authorize_url = config.get("authorize_url")
        if not authorize_url:
            raise ValueError("Provider does not have an authorize_url configured.")

        # Construct the authorization URL
        import urllib.parse
        
        params = {
            "response_type": "code",
            "client_id": authenticator["client_id"],
            "redirect_uri": redirect_uri,
            "scope": scope,  # This should already be properly formatted
            "state": state
        }
        
        # URL encode the query parameters properly
        query_string = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        
        return f"{authorize_url}?{query_string}"

    async def get_provider_info(
        self,
        service_provider_id: str,
        app_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get provider information (without sensitive details)
        
        Args:
            service_provider_id: ID of the OAuth provider (authenticator_id)
            app_id: Optional app ID context
            
        Returns:
            Dict with provider information
        """
        try:
            logger.info(f"get_provider_info called with service_provider_id: {service_provider_id}")
            
            # Query the oauth_authenticators table for provider information
            query = """
                SELECT 
                    authenticator_id,
                    authenticator_name,
                    authenticator_type,
                    scopes,
                    configuration,
                    is_active
                FROM oauth_authenticators 
                WHERE authenticator_id = $1 AND is_active = true
            """
            
            result = await self.db.fetch_one(query, service_provider_id)
            
            if not result:
                return {
                    "success": False,
                    "error": f"Authenticator {service_provider_id} not found or inactive"
                }
            
            # Convert result to dict and sanitize sensitive data
            provider_data = dict(result)
            
            # Remove any sensitive fields that shouldn't be exposed
            sanitized_data = {
                "authenticator_id": str(provider_data.get("authenticator_id")),
                "authenticator_name": provider_data.get("authenticator_name"),
                "authenticator_type": provider_data.get("authenticator_type"),
                "scopes": provider_data.get("scopes"),
                "is_active": provider_data.get("is_active")
            }
            
            # Include configuration if present, but sanitized
            config = provider_data.get("configuration")
            if config:
                if isinstance(config, str):
                    try:
                        config = json.loads(config)
                    except json.JSONDecodeError:
                        config = {}
                
                # Remove sensitive configuration fields
                if isinstance(config, dict):
                    safe_config = {
                        k: v for k, v in config.items() 
                        if k not in ['client_secret', 'api_key', 'password', 'secret']
                    }
                    sanitized_data["configuration"] = safe_config
                    sanitized_data["authorize_url"] = safe_config.get("authorize_url")
                    sanitized_data["token_url"] = safe_config.get("token_url")
            
            return {
                "success": True,
                "provider_id": service_provider_id,
                "provider_type": provider_data.get("authenticator_type", "unknown"),
                **sanitized_data
            }
                
        except Exception as e:
            logger.error(f"Error getting provider info for {service_provider_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def list_available_providers(
        self,
        app_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List all available providers for the given app
        
        Args:
            app_id: Optional app ID context
            
        Returns:
            Dict with list of available providers
        """
        try:
            # Get all active authenticators
            if app_id:
                query = """
                    SELECT 
                        authenticator_id,
                        authenticator_name,
                        authenticator_type,
                        scopes,
                        is_active
                    FROM oauth_authenticators 
                    WHERE is_active = true AND app_id = $1
                    ORDER BY authenticator_name
                """
                providers_result = await self.db.fetch_all(query, app_id)
            else:
                query = """
                    SELECT 
                        authenticator_id,
                        authenticator_name,
                        authenticator_type,
                        scopes,
                        is_active
                    FROM oauth_authenticators 
                    WHERE is_active = true
                    ORDER BY authenticator_name
                """
                providers_result = await self.db.fetch_all(query)
            
            providers = []
            for provider in providers_result:
                provider_dict = dict(provider)
                
                # Create sanitized provider entry
                provider_entry = {
                    "provider_id": str(provider_dict["authenticator_id"]),
                    "authenticator_id": str(provider_dict["authenticator_id"]),
                    "name": provider_dict.get("authenticator_name"),
                    "provider_type": provider_dict.get("authenticator_type"),
                    "authenticator_type": provider_dict.get("authenticator_type"),
                    "is_active": provider_dict.get("is_active", True)
                }
                
                providers.append(provider_entry)
            
            return {
                "success": True,
                "providers": providers
            }
                
        except Exception as e:
            logger.error(f"Error listing available providers: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "providers": []
            }
    
    # ===== UTILITY METHODS =====
    
    async def deactivate_oauth_authenticator(self, authenticator_id: str) -> bool:
        """Deactivate an OAuth authenticator."""
        try:
            updated_at = datetime.now().isoformat()
            
            result = await self.db.execute("""
                UPDATE oauth_authenticators 
                SET is_active = false, updated_at = ?
                WHERE authenticator_id = ?
            """, updated_at, authenticator_id)
            
            if result is not None:
                logger.info(f"Deactivated OAuth authenticator: {authenticator_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to deactivate OAuth authenticator {authenticator_id}: {str(e)}")
            return False
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired OAuth sessions."""
        try:
            result = await self.db.execute("""
                DELETE FROM oauth_sessions 
                WHERE datetime(expires_at) < datetime('now')
            """)
            
            count = result if isinstance(result, int) else 0
            if count > 0:
                logger.info(f"Cleaned up {count} expired OAuth sessions")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {str(e)}")
            return 0


class OAuthBackendService:
    """
    OAuth Backend Service for handling OAuth requests with stored tokens.
    
    This service acts as a backend provider for making authenticated requests
    to OAuth service providers using stored tokens.
    """
    
    def __init__(self, service_providers: Dict[str, Dict[str, Any]] = None,
                 token_storage: Dict[str, Dict[str, Any]] = None,
                 app_id: Optional[str] = None):
        """
        Initialize OAuth backend service.
        
        Args:
            service_providers: Dictionary of OAuth provider configurations
            token_storage: Dictionary of OAuth tokens keyed by provider ID
            app_id: Optional application ID context
        """
        self.service_providers = service_providers or {}
        self.token_storage = token_storage or {}
        self.app_id = app_id
        
    async def make_provider_request(
        self,
        service_provider_id: str,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Any] = None,
        content: Optional[str] = None,
        app_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make an authenticated request to an OAuth service provider.
        
        Args:
            service_provider_id: ID of the OAuth provider
            endpoint: API endpoint URL
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            headers: Additional headers
            json_data: JSON data for request body
            content: String content for request body
            app_id: Optional app ID context
            
        Returns:
            Dict with response data
        """
        try:
            # Get provider config
            provider_config = self.service_providers.get(service_provider_id)
            if not provider_config:
                return {
                    "success": False,
                    "error": f"OAuth provider {service_provider_id} not found"
                }
            
            # Get stored token
            token_data = self.token_storage.get(service_provider_id)
            if not token_data:
                # Try alternative keys
                alt_keys = [
                    f"{service_provider_id}:{self.app_id}",
                    f"{service_provider_id}:{self.app_id}:{app_id}",
                    f"{service_provider_id}:{app_id}" if app_id else None
                ]
                
                for alt_key in alt_keys:
                    if alt_key and alt_key in self.token_storage:
                        token_data = self.token_storage[alt_key]
                        break
                
                if not token_data:
                    return {
                        "success": False,
                        "error": f"No valid token found for provider {service_provider_id}"
                    }
            
            # Prepare authentication header
            access_token = token_data.get("access_token")
            if not access_token:
                return {
                    "success": False,
                    "error": f"No access token found for provider {service_provider_id}"
                }
            
            token_type = token_data.get("token_type", "Bearer")
            auth_header = f"{token_type} {access_token}"
            
            # Prepare headers
            request_headers = headers or {}
            request_headers["Authorization"] = auth_header
            
            # Make the HTTP request
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=method.upper(),
                    url=endpoint,
                    params=params,
                    headers=request_headers,
                    json=json_data,
                    content=content,
                    timeout=30.0
                )
                
                # Handle 401 - try token refresh if we have refresh token
                if response.status_code == 401:
                    refresh_token = token_data.get("refresh_token")
                    if refresh_token and provider_config.get("token_url"):
                        logger.info(f"Attempting token refresh for provider {service_provider_id}")
                        
                        try:
                            # Attempt to refresh the token
                            refreshed_token = await self._refresh_oauth_token(
                                provider_config, refresh_token
                            )
                            
                            if refreshed_token:
                                # Update token storage
                                self.token_storage[service_provider_id] = refreshed_token
                                
                                # Retry request with new token
                                new_auth_header = f"{refreshed_token.get('token_type', 'Bearer')} {refreshed_token['access_token']}"
                                request_headers["Authorization"] = new_auth_header
                                
                                response = await client.request(
                                    method=method.upper(),
                                    url=endpoint,
                                    params=params,
                                    headers=request_headers,
                                    json=json_data,
                                    content=content,
                                    timeout=30.0
                                )
                                
                                logger.info(f"Token refresh successful, retried request returned {response.status_code}")
                        
                        except Exception as refresh_error:
                            logger.error(f"Token refresh failed: {refresh_error}")
                
                # Handle response
                response_data = {
                    "success": True,
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                }
                
                try:
                    response_data["data"] = response.json()
                except:
                    response_data["text"] = response.text
                
                return response_data
                
        except Exception as e:
            logger.error(f"Error making OAuth provider request: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_provider_info(
        self,
        service_provider_id: str,
        app_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get information about an OAuth provider.
        
        Args:
            service_provider_id: ID of the OAuth provider
            app_id: Optional app ID context
            
        Returns:
            Dict with provider information (without sensitive data)
        """
        try:
            provider_config = self.service_providers.get(service_provider_id)
            if not provider_config:
                return {
                    "success": False,
                    "error": f"Provider {service_provider_id} not found"
                }
            
            logger.info(f"Provider config for {service_provider_id}: {provider_config}")
            
            # Infer brand from provider name, API base URL, or OAuth URLs if not explicitly set
            brand = provider_config.get("brand", "")
            if not brand:
                name = provider_config.get("name", "").lower()
                api_base_url = provider_config.get("api_base_url", "").lower()
                auth_url = provider_config.get("auth_url", "").lower()
                token_url = provider_config.get("token_url", "").lower()
                
                if ("google" in name or "gmail" in name or 
                    "googleapis.com" in api_base_url or
                    "accounts.google.com" in auth_url or
                    "oauth2.googleapis.com" in token_url):
                    brand = "google"
                elif ("microsoft" in name or "outlook" in name or 
                      "graph.microsoft.com" in api_base_url or
                      "login.microsoftonline.com" in auth_url):
                    brand = "microsoft"
                elif ("yahoo" in name or 
                      "yahoo.com" in api_base_url or
                      "login.yahoo.com" in auth_url):
                    brand = "yahoo"
            
            logger.info(f"Brand inference result for {service_provider_id}: '{brand}'")
            
            # Return sanitized provider info
            provider_info = {
                "success": True,
                "provider_id": service_provider_id,
                "authenticator_id": provider_config.get("authenticator_id", service_provider_id),
                "name": provider_config.get("name", ""),
                "authenticator_name": provider_config.get("authenticator_name", ""),
                "provider_type": provider_config.get("provider_type", "oauth2"),
                "authenticator_type": provider_config.get("authenticator_type", "oauth2"),
                "auth_url": provider_config.get("auth_url", ""),
                "api_base_url": provider_config.get("api_base_url", ""),
                "scopes": provider_config.get("scopes", []),
                "brand": brand
            }
            
            # Check if token exists
            has_token = (
                service_provider_id in self.token_storage or
                f"{service_provider_id}:{self.app_id}" in self.token_storage or
                f"{service_provider_id}:{app_id}" in self.token_storage if app_id else False
            )
            provider_info["has_token"] = has_token
            
            return provider_info
            
        except Exception as e:
            logger.error(f"Error getting provider info: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_available_providers(
        self,
        app_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List all available OAuth providers.
        
        Args:
            app_id: Optional app ID context
            
        Returns:
            Dict with list of providers
        """
        try:
            providers = []
            for provider_id, provider_config in self.service_providers.items():
                provider_info = await self.get_provider_info(provider_id, app_id)
                if provider_info.get("success"):
                    providers.append(provider_info)
            
            return {
                "success": True,
                "providers": providers
            }
            
        except Exception as e:
            logger.error(f"Error listing providers: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "providers": []
            }
    
    async def _refresh_oauth_token(
        self, 
        provider_config: Dict[str, Any], 
        refresh_token: str
    ) -> Optional[Dict[str, Any]]:
        """
        Refresh an OAuth token using the refresh token.
        
        Args:
            provider_config: OAuth provider configuration
            refresh_token: Refresh token to use
            
        Returns:
            New token data or None if refresh failed
        """
        try:
            token_url = provider_config.get("token_url") or provider_config.get("refresh_url")
            client_id = provider_config.get("client_id")
            client_secret = provider_config.get("client_secret")
            
            logger.info(f"Token refresh config - token_url: {token_url is not None}, client_id: {client_id is not None}, client_secret: {client_secret is not None}")
            
            if not all([token_url, client_id, client_secret]):
                logger.error(f"Missing required OAuth configuration for token refresh - token_url: {'✓' if token_url else '✗'}, client_id: {'✓' if client_id else '✗'}, client_secret: {'✓' if client_secret else '✗'}")
                return None
            
            # Prepare refresh request
            refresh_data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    token_url,
                    data=refresh_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    token_response = response.json()
                    
                    # Build new token data
                    new_token_data = {
                        "access_token": token_response.get("access_token"),
                        "token_type": token_response.get("token_type", "Bearer"),
                        "refresh_token": token_response.get("refresh_token", refresh_token),  # Keep old if not provided
                        "scope": token_response.get("scope")
                    }
                    
                    # Handle expires_in
                    if "expires_in" in token_response:
                        expires_at = datetime.now() + timedelta(seconds=int(token_response["expires_in"]))
                        new_token_data["expires_at"] = expires_at.isoformat()
                    
                    logger.info("Successfully refreshed OAuth token")
                    return new_token_data
                else:
                    logger.error(f"Token refresh failed with status {response.status_code}: {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error refreshing OAuth token: {str(e)}")
            return None