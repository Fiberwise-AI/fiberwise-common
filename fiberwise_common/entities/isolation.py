"""
Data isolation schemas for user data separation.
"""

from typing import Optional
from pydantic import BaseModel, Field


class DataIsolationConfig(BaseModel):
    """Configuration for data isolation policies"""
    user_isolation: str = Field("enforced", description="User isolation policy (enforced, optional, disabled)")
    auto_user_assignment: bool = Field(True, description="Automatically assign user_id to new records")
    protect_user_id: bool = Field(True, description="Protect user_id field from modification")


class IsolationMetadata(BaseModel):
    """Metadata for data isolation tracking"""
    isolation_level: str = Field(..., description="Current isolation level")
    user_id: Optional[str] = Field(None, description="User ID for isolation")
    created_with_isolation: bool = Field(True, description="Whether record was created with isolation enabled")