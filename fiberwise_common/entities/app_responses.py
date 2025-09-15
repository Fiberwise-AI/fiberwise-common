"""
Pydantic models for app-related API responses.
These models replace Dict[str, Any] responses in the app routes.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID

from .app import AppRead
from .llm_providers import LLMProvider


class PaginatedAppsResponse(BaseModel):
    """Response model for paginated app lists (both /my-apps and / endpoints)"""
    apps: List[AppRead] = Field(..., description="List of apps")
    total: int = Field(..., description="Total number of apps matching the criteria")
    limit: int = Field(..., description="Number of items requested per page")
    offset: int = Field(..., description="Number of items skipped")


class ModelItemCreateResponse(BaseModel):
    """Response model for creating model items"""
    success: bool = Field(..., description="Whether the operation was successful")
    item_id: str = Field(..., description="Unique identifier of the created item")
    model_id: str = Field(..., description="ID of the model the item was created in")
    model_slug: str = Field(..., description="Slug of the model the item was created in")
    data: Dict[str, Any] = Field(..., description="The item data that was created")
    message: str = Field(..., description="Success message")


class FunctionExecuteResponse(BaseModel):
    """Response model for function execution results"""
    execution_id: Optional[str] = Field(None, description="Unique identifier for this execution")
    status: str = Field(..., description="Execution status (completed, failed, running)")
    result: Optional[Dict[str, Any]] = Field(None, description="Function execution result data")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time: Optional[float] = Field(None, description="Execution time in milliseconds")
    started_at: Optional[datetime] = Field(None, description="When the execution started")
    completed_at: Optional[datetime] = Field(None, description="When the execution completed")


class FunctionStats(BaseModel):
    """Statistics for a function's execution history"""
    function_id: str = Field(..., description="Function identifier")
    function_name: str = Field(..., description="Function name")
    total_executions: int = Field(default=0, description="Total number of executions")
    successful: int = Field(default=0, description="Number of successful executions")
    failed: int = Field(default=0, description="Number of failed executions")
    avg_execution_time: float = Field(default=0.0, description="Average execution time in milliseconds")
    last_execution: Optional[datetime] = Field(None, description="Timestamp of last execution")


class FunctionHistoryItem(BaseModel):
    """Individual function execution history item"""
    execution_id: str = Field(..., description="Unique identifier for this execution")
    function_id: str = Field(..., description="Function identifier")
    function_name: str = Field(..., description="Function name")
    status: str = Field(..., description="Execution status")
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data provided to function")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data from function")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time: Optional[float] = Field(None, description="Execution time in milliseconds")
    started_at: datetime = Field(..., description="When the execution started")
    completed_at: Optional[datetime] = Field(None, description="When the execution completed")


class FunctionHistoryResponse(BaseModel):
    """Response model for function execution history"""
    items: List[FunctionHistoryItem] = Field(default_factory=list, description="List of execution history items")
    total: int = Field(..., description="Total number of executions matching criteria")
    limit: int = Field(..., description="Number of items requested per page")
    offset: int = Field(..., description="Number of items skipped")
    function_stats: Optional[List[FunctionStats]] = Field(default=None, description="Statistical summary by function")


class AppFunctionHistoryResponse(FunctionHistoryResponse):
    """Response model for app-wide function execution history (extends base with stats)"""
    function_stats: List[FunctionStats] = Field(default_factory=list, description="Statistical summary by function")


# Service provider models moved to service_providers.py


class AppStatusResponse(BaseModel):
    """Response model for app runtime status"""
    app_id: str = Field(..., description="App identifier")
    running: bool = Field(..., description="Whether the app is currently running")
    status: str = Field(..., description="Current status (running, stopped, error, etc.)")
    message: str = Field(..., description="Status message or additional information")
    last_updated: Optional[datetime] = Field(None, description="When the status was last updated")
    health_check: Optional[Dict[str, Any]] = Field(None, description="Health check results")