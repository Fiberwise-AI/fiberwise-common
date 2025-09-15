"""
Pydantic models for pipeline entities.
"""

from datetime import datetime
from typing import Optional, List, Any
from uuid import UUID
from pydantic import BaseModel, Field


class Pipeline(BaseModel):
    """Pipeline entity model."""
    pipeline_id: UUID = Field(..., description="Pipeline unique identifier")
    pipeline_slug: str = Field(..., description="Pipeline slug/name")
    name: str = Field(..., description="Pipeline display name")
    description: Optional[str] = Field(None, description="Pipeline description")
    file_path: str = Field(..., description="Path to pipeline file")
    definition: dict = Field(..., description="Pipeline definition/structure")
    config: Optional[dict] = Field(None, description="Pipeline configuration")
    app_id: UUID = Field(..., description="Application ID this pipeline belongs to")
    is_active: bool = Field(True, description="Whether pipeline is active")
    created_by: int = Field(..., description="User ID who created the pipeline")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: str
        }


class PipelineExecution(BaseModel):
    """Pipeline execution entity model."""
    execution_id: str = Field(..., description="Execution unique identifier")
    pipeline_id: str = Field(..., description="Pipeline ID being executed")
    status: str = Field(..., description="Execution status")
    input_data: dict = Field(..., description="Input data for execution")
    results: Optional[dict] = Field(None, description="Execution results")
    error: Optional[str] = Field(None, description="Error message if failed")
    priority: int = Field(10, description="Execution priority")
    created_by: int = Field(..., description="User ID who started execution")
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")
    duration_ms: Optional[int] = Field(None, description="Execution duration in milliseconds")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    # Additional fields for human input workflows
    human_input_config: Optional[dict] = Field(None, description="Human input configuration")
    human_input_data: Optional[dict] = Field(None, description="Human input data")
    waiting_step_id: Optional[str] = Field(None, description="Step ID waiting for human input")

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class PipelineExecutionResult(BaseModel):
    """Result of pipeline execution."""
    execution_id: str = Field(..., description="Execution ID")
    pipeline_id: str = Field(..., description="Pipeline ID")
    status: str = Field(..., description="Final execution status")
    input_data: dict = Field(..., description="Original input data")
    output_data: dict = Field(..., description="Final output data")
    step_results: dict = Field(..., description="Results from each step")
    created_by: int = Field(..., description="User who executed")
    started_at: str = Field(..., description="Start timestamp")
    completed_at: str = Field(..., description="Completion timestamp")

    class Config:
        from_attributes = True


class PipelineStep(BaseModel):
    """Pipeline step definition."""
    id: str = Field(..., description="Step identifier")
    step_class: str = Field(..., description="Python class name for step")
    type: str = Field(..., description="Step type")
    parameters: dict = Field(default_factory=dict, description="Step parameters")

    class Config:
        from_attributes = True


class PipelineExecutionContext(BaseModel):
    """Context for pipeline execution."""
    organization_id: int = Field(..., description="Organization ID")
    app_id: UUID = Field(..., description="Application ID")
    created_by: int = Field(..., description="User ID")

    class Config:
        from_attributes = True
        json_encoders = {
            UUID: str
        }