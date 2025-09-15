"""
Activation schemas for agent execution tracking.
"""

from typing import Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class ActivationStatus(str, Enum):
    """Activation status values"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActivationMetadata(BaseModel):
    """Metadata for activations, including provider and model details"""
    provider_id: Optional[uuid.UUID] = None
    provider_type: Optional[str] = None
    model_id: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    additional_params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ActivationCreate(BaseModel):
    """Schema for creating a new activation"""
    activation_id: uuid.UUID
    agent_id: str
    agent_type_id: str
    input_data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    metadata: Optional[ActivationMetadata] = None
    priority: int = 0
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ActivationUpdate(BaseModel):
    """Schema for updating an activation"""
    status: Optional[ActivationStatus] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    notes: Optional[Dict[str, Any]] = None
    metadata: Optional[ActivationMetadata] = None


class Activation(BaseModel):
    """Complete activation schema"""
    activation_id: uuid.UUID
    agent_id: str
    agent_type_id: str
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    notes: Optional[Dict[str, Any]] = None
    metadata: Optional[ActivationMetadata] = None
    created_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    priority: int = 0
    context: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class ActivationResponse(BaseModel):
    """Response model for agent activations"""
    id: uuid.UUID
    agent_id: uuid.UUID
    agent_type_id: str
    status: str
    started_at: Union[str, datetime]
    completed_at: Optional[Union[str, datetime]] = None
    duration_ms: Optional[float] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Optional[Union[Dict[str, Any], str]] = None
    error: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    
    class Config:
        from_attributes = True
        populate_by_name = True
    
    @classmethod
    def from_db_model(cls, db_model: Dict[str, Any]) -> "ActivationResponse":
        """Create an ActivationResponse from a database model or dict"""
        data = dict(db_model)
        if "activation_id" in data and "id" not in data:
            data["id"] = data["activation_id"]
        for date_field in ["started_at", "completed_at", "updated_at"]:
            if date_field in data and isinstance(data[date_field], datetime):
                data[date_field] = data[date_field].isoformat()
        return cls(**data)


class ActivationLogEntry(BaseModel):
    """Log entry for an activation"""
    log_id: str = Field(..., description="Log entry identifier")
    activation_id: str = Field(..., description="Associated activation ID")
    level: str = Field(..., description="Log level (INFO, WARN, ERROR)")
    message: str = Field(..., description="Log message")
    timestamp: datetime = Field(..., description="Log timestamp")