"""
Pipeline schemas for the FiberWise platform.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class PipelineStatus(str, Enum):
    """Pipeline execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineCreate(BaseModel):
    """Schema for creating a new pipeline."""
    name: str
    description: Optional[str] = None
    definition: Dict[str, Any] = Field(default_factory=dict, description="Pipeline definition with nodes and edges")
    is_active: bool = Field(True, description="Whether the pipeline is active")


class PipelineUpdate(BaseModel):
    """Schema for updating an existing pipeline."""
    name: Optional[str] = None
    description: Optional[str] = None
    definition: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class PipelineResponse(BaseModel):
    """Schema for pipeline response."""
    pipeline_id: str
    name: str
    description: Optional[str] = None
    definition: Dict[str, Any]
    is_active: bool
    created_by: Optional[int] = None
    app_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class PipelineExecuteRequest(BaseModel):
    """Schema for executing a pipeline."""
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for the pipeline")


class PipelineExecuteResponse(BaseModel):
    """Schema for pipeline execution response."""
    execution_id: str
    pipeline_id: str
    status: str
    started_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True