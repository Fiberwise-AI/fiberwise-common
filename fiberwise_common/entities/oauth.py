"""
OAuth and credentials entities and schemas for the Fiberwise platform.
These models handle OAuth provider management and token registration.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, UUID4, root_validator
from uuid import UUID


class OAuthAuthenticatorCreate(BaseModel):
    """Schema for creating OAuth authenticators"""
    name: str = Field(..., description="Name of the OAuth authenticator")
    authenticator_type: str = Field(..., description="Type of OAuth provider (e.g., 'google', 'github', 'oauth2')")
    client_id: str = Field(..., description="OAuth client ID")
    client_secret: str = Field(..., description="OAuth client secret")
    scopes: List[str] = Field(default=[], description="OAuth scopes")
    authorize_url: Optional[str] = Field(None, description="OAuth authorize URL")
    token_url: Optional[str] = Field(None, description="OAuth token URL")
    redirect_uri: Optional[str] = Field(None, description="OAuth redirect URI")
    configuration: Optional[Dict[str, Any]] = Field(default={}, description="Additional configuration")


class OAuthAuthenticatorResponse(BaseModel):
    """Schema for OAuth authenticator responses"""
    id: str = Field(..., description="Authenticator ID")
    name: str = Field(..., description="Name of the authenticator")
    authenticator_type: str = Field(..., description="Type of OAuth provider")
    scopes: List[str] = Field(default=[], description="OAuth scopes")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    is_active: bool = Field(True, description="Whether the authenticator is active")


class OAuthProviderCreate(BaseModel):
    """Model for OAuth provider registration"""
    provider_name: str = Field(..., description="OAuth provider identifier (e.g., 'google', 'github')")
    client_id: str = Field(..., description="OAuth client ID from the provider")
    client_secret: str = Field(..., description="OAuth client secret from the provider")
    redirect_uri: Optional[str] = Field(None, description="Redirect URI for the OAuth flow")
    token_data: Dict[str, Any] = Field(
        ..., 
        description="Provider-specific connection details including URLs and scopes"
    )
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context data")

    class Config:
        extra = "allow"  # Allow extra fields to be flexible with different provider requirements


class RegisterTokenRequest(BaseModel):
    """Model for token registration"""
    provider_name: Optional[str] = Field(None, description="Alternative name for OAuth provider")
    provider_id: Optional[str] = Field(None, description="OAuth provider identifier")
    app_id: Optional[UUID4] = Field(None, description="Optional App ID to associate the token with")
    token_data: Dict[str, Any] = Field(..., description="Dictionary containing token information (access_token, refresh_token, expires_in, etc.)")
    
    @root_validator(pre=True)
    def check_provider_fields(cls, values):
        """Ensure either provider_id or provider_name is present, and map provider_name to provider_id if needed."""
        provider_id = values.get('provider_id')
        provider_name = values.get('provider_name')
        
        if provider_id is None and provider_name is None:
            raise ValueError("Either provider_id or provider_name must be provided")
            
        if provider_id is None and provider_name is not None:
            values['provider_id'] = provider_name
            
        return values


class OAuthProviderResponse(BaseModel):
    """Response model for OAuth provider information"""
    id: str = Field(..., description="Provider ID")
    provider_name: str = Field(..., description="Provider name")
    client_id: str = Field(..., description="OAuth client ID")
    redirect_uri: Optional[str] = Field(None, description="Redirect URI")
    scopes: Optional[List[str]] = Field(default=[], description="Available scopes")
    is_active: bool = Field(True, description="Whether the provider is active")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


class OAuthTokenResponse(BaseModel):
    """Response model for OAuth token information"""
    token_id: str = Field(..., description="Token ID")
    provider_id: str = Field(..., description="Provider ID")
    app_id: Optional[str] = Field(None, description="Associated app ID")
    expires_at: Optional[datetime] = Field(None, description="Token expiration timestamp")
    is_active: bool = Field(True, description="Whether the token is active")
    created_at: datetime = Field(..., description="Creation timestamp")


class OAuthProviderUpdate(BaseModel):
    """Model for updating OAuth provider settings"""
    provider_name: Optional[str] = Field(None, description="Updated provider name")
    client_id: Optional[str] = Field(None, description="Updated client ID")
    client_secret: Optional[str] = Field(None, description="Updated client secret")
    redirect_uri: Optional[str] = Field(None, description="Updated redirect URI")
    token_data: Optional[Dict[str, Any]] = Field(None, description="Updated token data")
    is_active: Optional[bool] = Field(None, description="Updated active status")


class OAuthAuthenticatorUpdate(BaseModel):
    """Schema for updating OAuth authenticators"""
    name: Optional[str] = Field(None, description="Updated name")
    client_id: Optional[str] = Field(None, description="Updated client ID")
    client_secret: Optional[str] = Field(None, description="Updated client secret")
    scopes: Optional[List[str]] = Field(None, description="Updated scopes")
    authorize_url: Optional[str] = Field(None, description="Updated authorize URL")
    token_url: Optional[str] = Field(None, description="Updated token URL")
    redirect_uri: Optional[str] = Field(None, description="Updated redirect URI")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Updated configuration")
    is_active: Optional[bool] = Field(None, description="Updated active status")