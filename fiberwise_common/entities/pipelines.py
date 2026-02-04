"""
Pipeline schemas for the FiberWise platform.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import re


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
    
    @field_validator('created_at', 'updated_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        """Parse datetime strings with various formats including PostgreSQL format."""
        if v is None:
            return v
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # Handle PostgreSQL format: '2026-02-03 15:02:39.435726+00'
            # Convert to ISO format by replacing space with T and fixing timezone
            v = v.strip()
            # Replace space separator with T for ISO format
            if ' ' in v and 'T' not in v:
                v = v.replace(' ', 'T', 1)
            # Fix timezone format: +00 -> +00:00
            if re.match(r'.*[+-]\d{2}$', v):
                v = v + ':00'
            return datetime.fromisoformat(v)
        return v
    
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