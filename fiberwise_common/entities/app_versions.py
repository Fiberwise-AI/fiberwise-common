"""
App Version entities for the FiberWise platform.
"""
from pydantic import BaseModel, UUID4, Field
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime


class AppVersionCreate(BaseModel):
    """Schema for creating app versions."""
    version: str
    manifest_yaml: str
    status: str = "draft"
    changelog: Optional[str] = None
    is_active: bool = False


class AppVersionUpdate(BaseModel):
    """Schema for updating app versions."""
    status: Optional[str] = None
    changelog: Optional[str] = None
    is_active: Optional[bool] = None


class AppVersionRead(BaseModel):
    """Schema for reading app versions."""
    app_version_id: UUID4
    app_id: UUID4
    version: str
    manifest_yaml: str
    status: str
    changelog: Optional[str] = None
    is_active: bool
    created_by: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    deployed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AppVersionsList(BaseModel):
    """Schema for paginated app versions list."""
    versions: List[AppVersionRead]
    count: int
    total: int
    page: int
    pages: int


class AppVersionRollback(BaseModel):
    """Schema for app version rollback request."""
    target_version_id: UUID4
    reason: Optional[str] = None


class AppVersionRollbackResponse(BaseModel):
    """Schema for app version rollback response."""
    success: bool
    message: str
    old_version_id: UUID4
    new_version_id: UUID4
    rollback_timestamp: datetime

    class Config:
        from_attributes = True


class AppVersionComparison(BaseModel):
    """Schema for comparing two app versions."""
    version_a: AppVersionRead
    version_b: AppVersionRead
    manifest_diff: Dict[str, Any]
    summary: Dict[str, Any]

    class Config:
        from_attributes = True


class AppVersionDeployment(BaseModel):
    """Schema for app version deployment."""
    version_id: UUID4
    deployment_strategy: str = "immediate"  # immediate, scheduled, blue_green
    scheduled_at: Optional[datetime] = None
    rollback_on_failure: bool = True
    notification_settings: Optional[Dict[str, Any]] = None


class AppVersionMetrics(BaseModel):
    """Schema for app version metrics and analytics."""
    version_id: UUID4
    deployment_count: int
    rollback_count: int
    active_duration_hours: Optional[float] = None
    error_rate: Optional[float] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    last_deployed: Optional[datetime] = None
    last_rolled_back: Optional[datetime] = None

    class Config:
        from_attributes = True
