"""
Pydantic models for pipeline realtime events and updates.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class PipelineStatus(str, Enum):
    """Pipeline execution status values."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StepStatus(str, Enum):
    """Pipeline step execution status values."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineExecutionUpdate(BaseModel):
    """Pipeline execution level update message."""
    type: str = Field(default="pipeline_update", description="Message type identifier")
    execution_id: str = Field(..., description="Pipeline execution ID")
    status: PipelineStatus = Field(..., description="Pipeline execution status")
    message: str = Field(..., description="Human readable status message")
    result: Optional[Dict[str, Any]] = Field(None, description="Execution result data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Update timestamp")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class PipelineStepUpdate(BaseModel):
    """Pipeline step level update message."""
    type: str = Field(default="pipeline_update", description="Message type identifier")
    execution_id: str = Field(..., description="Pipeline execution ID")
    step_id: str = Field(..., description="Step identifier")
    status: StepStatus = Field(..., description="Step execution status")
    message: str = Field(..., description="Human readable status message")
    data: Optional[Dict[str, Any]] = Field(None, description="Step execution data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Update timestamp")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class PipelineBroadcastContext(BaseModel):
    """Context information for pipeline broadcasts."""
    app_id: str = Field(..., description="Application ID")
    organization_id: int = Field(..., description="Organization ID")
    execution_id: str = Field(..., description="Pipeline execution ID")


def create_execution_update(
    execution_id: str,
    status: PipelineStatus,
    message: str,
    result: Optional[Dict[str, Any]] = None
) -> PipelineExecutionUpdate:
    """Create a pipeline execution update message."""
    return PipelineExecutionUpdate(
        execution_id=execution_id,
        status=status,
        message=message,
        result=result
    )


def create_step_update(
    execution_id: str,
    step_id: str,
    status: StepStatus,
    message: str,
    data: Optional[Dict[str, Any]] = None
) -> PipelineStepUpdate:
    """Create a pipeline step update message."""
    return PipelineStepUpdate(
        execution_id=execution_id,
        step_id=step_id,
        status=status,
        message=message,
        data=data
    )