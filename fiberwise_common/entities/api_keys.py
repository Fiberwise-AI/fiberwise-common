"""
API Keys schemas for the FiberWise platform.
"""
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class APIKeyCreate(BaseModel):
    """Schema for creating a new API key."""
    name: str
    scopes: List[str] = []
    expires_in_days: Optional[int] = None


class APIKeyCreateResponse(BaseModel):
    """Response schema for created API key (includes the actual key)."""
    id: int
    key: str
    key_prefix: str
    name: str
    scopes: List[str]
    expires_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class APIKeyResponse(BaseModel):
    """Response schema for API key information (without the actual key)."""
    id: int
    user_id: int
    organization_id: Optional[int] = None
    key_prefix: str
    name: str
    scopes: List[str]
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class APIKeyTokenResponse(BaseModel):
    """Schema for API key token validation response."""
    valid: bool
    user_id: Optional[int] = None
    scopes: List[str] = []
    expires_at: Optional[datetime] = None