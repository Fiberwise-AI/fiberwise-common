from typing import Dict, List, Optional, Any, Union
from uuid import UUID
from pydantic import BaseModel, Field, UUID4
from datetime import datetime
from enum import Enum


class StepType(str, Enum):
    """Type of workflow step"""
    START = "start"
    AGENT = "agent"
    FUNCTION = "function"
    CONDITION = "condition"
    END = "end"


class WorkflowStep(BaseModel):
    """Schema for a workflow step"""
    id: str = Field(..., description="Unique identifier for the step")
    name: str = Field(..., description="Name of the step")
    type: StepType = Field(..., description="Type of the step")
    next: Optional[str] = Field(None, description="ID of the next step")
    next_on_false: Optional[str] = Field(None, description="ID of the next step if condition is false (for condition steps)")
    condition: Optional[str] = Field(None, description="Condition expression (for condition steps)")
    agent_id: Optional[str] = Field(None, description="Agent ID to execute (for agent steps)")
    function_name: Optional[str] = Field(None, description="Function to execute (for function steps)")
    input_mapping: Optional[Dict[str, str]] = Field(None, description="Mapping of step inputs")
    output_mapping: Optional[Dict[str, str]] = Field(None, description="Mapping of step outputs")


class WorkflowVariable(BaseModel):
    """Schema for a workflow variable definition"""
    type: str = Field(..., description="Data type of the variable")
    description: Optional[str] = Field(None, description="Description of the variable")
    default_value: Optional[Any] = Field(None, description="Default value for the variable")


class WorkflowManifest(BaseModel):
    """Schema for workflow installation manifest"""
    name: str = Field(..., description="Name of the workflow")
    description: Optional[str] = Field(None, description="Description of the workflow")
    version: str = Field(..., description="Version of the workflow")
    trigger_type: str = Field(default="manual", description="Type of trigger that starts the workflow")
    trigger_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration for the trigger")
    steps: List[WorkflowStep] = Field(..., description="Steps in the workflow")
    variables: Dict[str, WorkflowVariable] = Field(default_factory=dict, description="Workflow variables")
    output_variables: List[str] = Field(default_factory=list, description="Variables to include in workflow output")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing the workflow")
    author: Optional[str] = Field(None, description="Author of the workflow")
    license: Optional[str] = Field(None, description="License information")
    workflow_id: Optional[UUID] = Field(None, description="Optional existing workflow ID for updates")
    is_enabled: bool = Field(default=True, description="Whether the workflow is enabled")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Document Processing Workflow",
                "description": "Processes documents through extraction, summarization and classification",
                "version": "1.0.0",
                "trigger_type": "manual",
                "steps": [
                    {
                        "id": "start",
                        "name": "Start",
                        "type": "start",
                        "next": "extract"
                    },
                    {
                        "id": "extract",
                        "name": "Extract Text",
                        "type": "function",
                        "function_name": "text_extractor",
                        "next": "check_length",
                        "input_mapping": {
                            "document_url": "$workflow.input.document_url"
                        },
                        "output_mapping": {
                            "extracted_text": "$results.text"
                        }
                    },
                    {
                        "id": "check_length",
                        "name": "Check Document Length",
                        "type": "condition",
                        "condition": "len(extracted_text) > 1000",
                        "next": "summarize",
                        "next_on_false": "classify"
                    },
                    {
                        "id": "summarize",
                        "name": "Summarize Text",
                        "type": "agent",
                        "agent_id": "text-summarizer",
                        "next": "classify",
                        "input_mapping": {
                            "text": "$workflow.variables.extracted_text"
                        },
                        "output_mapping": {
                            "summary": "$results.summary"
                        }
                    },
                    {
                        "id": "classify",
                        "name": "Classify Document",
                        "type": "agent",
                        "agent_id": "document-classifier",
                        "next": "end",
                        "input_mapping": {
                            "text": "$workflow.variables.extracted_text"
                        },
                        "output_mapping": {
                            "categories": "$results.categories"
                        }
                    },
                    {
                        "id": "end",
                        "name": "End",
                        "type": "end"
                    }
                ],
                "variables": {
                    "extracted_text": {
                        "type": "string",
                        "description": "Extracted text content from document"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Summary of text content"
                    },
                    "categories": {
                        "type": "array",
                        "description": "Document categories"
                    }
                },
                "output_variables": ["summary", "categories"],
                "tags": ["document-processing", "text-analysis"]
            }
        }


class WorkflowBase(BaseModel):
    name: str
    description: Optional[str] = None
    is_enabled: bool = True
    trigger_type: str = "manual"
    trigger_config: Dict[str, Any] = Field(default_factory=dict)
    variables: Dict[str, WorkflowVariable] = Field(default_factory=dict)
    output_variables: List[str] = Field(default_factory=list)


class WorkflowCreate(WorkflowBase):
    steps: List[WorkflowStep] = Field(default_factory=list)


class WorkflowUpdate(WorkflowBase):
    steps: List[WorkflowStep] = Field(default_factory=list)


class WorkflowResponse(WorkflowBase):
    workflow_id: UUID4
    steps: List[WorkflowStep]
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None

    class Config:
        orm_mode = True


class WorkflowExecuteRequest(BaseModel):
    input_data: Dict[str, Any] = Field(default_factory=dict)


class StepExecution(BaseModel):
    step_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class WorkflowExecutionResponse(BaseModel):
    execution_id: UUID4
    workflow_id: UUID4
    status: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    step_results: Dict[str, StepExecution] = Field(default_factory=dict)
    created_by: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class WorkflowExecutionListResponse(BaseModel):
    items: List[WorkflowExecutionResponse]
    total: int