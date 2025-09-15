"""
App schemas for the FiberWise platform.
"""
from pydantic import BaseModel, UUID4, Field
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime


class MarketplaceAppRead(BaseModel):
    """Schema for marketplace app data."""
    app_id: UUID
    name: str
    description: Optional[str] = None
    version: str
    marketplace_status: str
    creator_user_id: int
    created_at: datetime
    publisher_name: str
    manifest_yaml: Optional[str] = None
    install_count: int = 0
    install_path: Optional[UUID] = None
    is_installed: bool = False
    category: str = "Uncategorized"
    icon: Optional[str] = None
    screenshots: List[str] = []
    rating: Optional[float] = None
    
    # Featured apps fields
    icon_class: Optional[str] = None
    install_command: Optional[str] = None
    tutorial_url: Optional[str] = None
    source_url: Optional[str] = None
    featured_tags: Optional[List[str]] = []
    featured_stats: Optional[Dict[str, Any]] = {}
    
    class Config:
        from_attributes = True


class AppModelItemCreate(BaseModel):
    """Schema for creating app model items."""
    data: Dict[str, Any]


class AppModelItem(BaseModel):
    """Schema for app model items."""
    item_id: UUID4
    app_id: UUID4
    model_id: UUID4
    data: Dict[str, Any]
    created_by: int 
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class AppModelItemsList(BaseModel):
    """Schema for paginated app model items list."""
    items: List[AppModelItem]
    count: int
    total: int
    page: int
    pages: int


class AppInstallation(BaseModel):
    """Schema for app installation."""
    installation_id: UUID4
    app_id: UUID4
    installed_by_user_id: int
    installed_at: datetime
    installed_version: str
    is_active: bool

    class Config:
        from_attributes = True


# --- Field Schemas ---

class FieldCreate(BaseModel):
    """Schema for creating app model fields."""
    field_column: str = Field(..., pattern=r"^[a-z0-9_]+(?:-[a-z0-9_]+)*$")
    name: str
    data_type: str
    is_required: bool = False
    is_unique: bool = False
    default_value_json: Optional[str] = None
    validations_json: Optional[str] = None
    relation_details_json: Optional[str] = None
    is_primary_key: bool = False
    is_system_field: bool = False


# --- Model Schemas ---

class ModelCreate(BaseModel):
    """Schema for creating app models."""
    model_slug: str = Field(..., pattern=r"^[a-z0-9_]+(?:-[a-z0-9_]+)*$")
    name: str
    description: Optional[str] = None
    is_system_model: bool = False
    fields: List[FieldCreate] = []


class ModelRead(BaseModel):
    """Schema for reading app models."""
    model_id: UUID
    app_id: UUID
    model_slug: str
    name: str
    description: Optional[str] = None
    is_system_model: bool = False
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    
    class Config:
        from_attributes = True


# --- App CRUD Schemas ---

class AppCreate(BaseModel):
    """Schema for creating apps."""
    app_slug: str = Field(..., pattern=r"^[a-z0-9_]+(?:-[a-z0-9_]+)*$")
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    entry_point_url: Optional[str] = None
    marketplace_status: str = "draft"
    models: List[ModelCreate] = []


class AppUpdate(BaseModel):
    """Schema for updating apps."""
    name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    entry_point_url: Optional[str] = None
    marketplace_status: Optional[str] = None
    models: Optional[List[ModelCreate]] = None


class AppRead(BaseModel):
    """Schema for reading apps."""
    app_id: UUID
    app_slug: str
    name: str
    description: Optional[str] = None
    version: str
    entry_point_url: Optional[str] = None
    marketplace_status: str
    creator_user_id: Optional[int] = None
    owner_id: Optional[int] = None
    organization_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    models: List[ModelRead] = []

    class Config:
        from_attributes = True


# --- App Manifest Schemas ---

class AppManifest(BaseModel):
    """Schema for app manifests."""
    name: str
    app_slug: str = Field(..., pattern=r"^[a-z0-9_]+(?:-[a-z0-9_]+)*$")
    version: Optional[str] = "1.0.0"
    description: Optional[str] = None
    entry_point_url: Optional[str] = None
    bundle_path: Optional[str] = None
    icon: Optional[str] = None
    category: Optional[str] = None
    publisher: Optional[str] = None


# --- App Installation Schemas ---

class AppInstallConfig(BaseModel):
    """Schema for app installation configuration."""
    app_id: Optional[UUID] = None  # Optional since app_id comes from URL path
    config: Dict[str, Any] = {}
    install_at_root: Optional[bool] = False
    payment_method: Optional[str] = None


class AppInstallResponse(BaseModel):
    """Schema for app installation response."""
    installation_id: UUID
    app_id: UUID
    status: str
    message: str
    
    class Config:
        from_attributes = True