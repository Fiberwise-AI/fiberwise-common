"""
Service provider response models that were moved from app_responses.py
for better organization.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ServiceProviderResponse(BaseModel):
    """Response model for OAuth service providers"""
    provider_id: str = Field(..., description="Unique identifier for the provider")
    name: str = Field(..., description="Provider display name")
    provider_type: str = Field(..., description="Type of OAuth provider")
    is_active: bool = Field(default=True, description="Whether the provider is active")
    is_connected: Optional[bool] = Field(None, description="Whether user is connected to this provider")
    scopes: Optional[List[str]] = Field(default_factory=list, description="Requested OAuth scopes")
    status: Optional[str] = Field(None, description="Connection status")
    connected_at: Optional[datetime] = Field(None, description="When the connection was established")


class ServiceProviderRegistrationResponse(BaseModel):
    """Response model for registering OAuth service providers"""
    success: bool = Field(..., description="Whether the registration was successful")
    provider_id: str = Field(..., description="ID of the registered provider")
    app_id: str = Field(..., description="ID of the app the provider was linked to")
    message: str = Field(..., description="Success message")