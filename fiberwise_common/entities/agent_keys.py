"""
Agent API Key entities and schemas for the Fiberwise platform.
These models handle agent authentication and authorization.
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class CreateAgentKeyRequest(BaseModel):
    """Request model for creating a new agent API key"""
    app_id: str = Field(..., description="The app ID this key is associated with")
    agent_id: Optional[str] = Field(None, description="Optional agent ID this key is associated with")
    description: Optional[str] = Field(None, description="Description of what this key is used for")
    scopes: List[str] = Field(default=["data:read"], description="Permission scopes for this key")
    expiration_hours: Optional[int] = Field(None, description="Hours until expiration (null = no expiration)")
    resource_pattern: Optional[str] = Field(None, description="Resource pattern for limiting access")


class AgentKeyResponse(BaseModel):
    """Response model for agent API key information (without sensitive key value)"""
    key_id: str
    key_value: Optional[str] = None  # Only included when key is first created
    app_id: str
    agent_id: Optional[str] = None
    description: Optional[str] = None
    scopes: List[str]
    expiration: Optional[datetime] = None
    resource_pattern: Optional[str] = None
    is_active: bool
    created_at: datetime
    revoked_at: Optional[datetime] = None
    created_by: Optional[int] = None


class CreateAgentKeyResponse(BaseModel):
    """Response model for agent API key creation (includes sensitive key value)"""
    key_id: str
    key_value: str  # Full key value returned only on creation
    app_id: str
    agent_id: Optional[str] = None
    description: Optional[str] = None
    scopes: List[str]
    expiration: Optional[datetime] = None
    resource_pattern: Optional[str] = None


class AgentKeyListItem(BaseModel):
    """Simplified agent key model for list responses"""
    key_id: str
    app_id: str
    agent_id: Optional[str] = None
    description: Optional[str] = None
    scopes: List[str]
    is_active: bool
    created_at: datetime
    expiration: Optional[datetime] = None


class AgentKeyUpdate(BaseModel):
    """Request model for updating agent API key properties"""
    description: Optional[str] = None
    scopes: Optional[List[str]] = None
    is_active: Optional[bool] = None
    expiration: Optional[datetime] = None