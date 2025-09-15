"""
Function schemas for the FiberWise platform.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class FunctionType(str, Enum):
    """Function type enumeration."""
    UTILITY = "utility"
    TRANSFORM = "transform"
    SUPPORT_AGENT = "support_agent"


class FunctionCreate(BaseModel):
    """Schema for creating a new function."""
    name: str = Field(..., description="Name of the function")
    description: Optional[str] = Field(None, description="Description of the function")
    function_type: FunctionType = Field(FunctionType.UTILITY, description="Type of function")
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="JSON Schema for function inputs")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="JSON Schema for function outputs")
    implementation: Optional[str] = Field(None, description="Python code implementation (for inline functions)")
    is_async: bool = Field(True, description="Whether the function is asynchronous")
    is_system: bool = Field(False, description="Whether this is a system function")


class FunctionUpdate(BaseModel):
    """Schema for updating an existing function."""
    name: Optional[str] = Field(None, description="Name of the function")
    description: Optional[str] = Field(None, description="Description of the function")
    function_type: Optional[FunctionType] = Field(None, description="Type of function")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for function inputs")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for function outputs")
    implementation: Optional[str] = Field(None, description="Python code implementation")
    is_async: Optional[bool] = Field(None, description="Whether the function is asynchronous")


class FunctionResponse(BaseModel):
    """Schema for function response."""
    function_id: str
    name: str
    description: Optional[str] = None
    function_type: FunctionType
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    implementation: Optional[str] = None
    is_async: bool = True
    is_system: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class FunctionExecuteRequest(BaseModel):
    """Schema for executing a function."""
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for the function")


class FunctionExecuteResponse(BaseModel):
    """Schema for function execution response."""
    execution_id: str
    function_id: str
    function_name: Optional[str] = None
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    class Config:
        from_attributes = True