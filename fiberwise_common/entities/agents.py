"""
Agent entities and schemas for the Fiberwise platform.
These models handle agent definitions, activations, and responses.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, UUID4, ConfigDict


class AgentSummaryWeb(BaseModel):
    """Summary model for agent information in web list responses"""
    id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent name")
    version: str = Field(..., description="Agent version")
    description: Optional[str] = Field(None, description="Agent description")
    is_system: bool = Field(..., description="Whether this is a system agent")
    capabilities: List[str] = Field(default=[], description="Agent capabilities")


class AgentListResponse(BaseModel):
    """Response model for agent list endpoints"""
    agents: List[AgentSummaryWeb] = Field(..., description="List of agents")


class AgentActivationRequest(BaseModel):
    """Request model for activating an agent"""
    input_data: Dict[str, Any] = Field(default={}, description="Input data for the agent")
    context_type: Optional[str] = Field(default="general", description="Context type for activation")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata for activation")


class AgentActivationResponse(BaseModel):
    """Response model for agent activation requests"""
    activation_id: str = Field(..., description="Unique activation ID")
    status: str = Field(..., description="Activation status")
    message: str = Field(..., description="Status message or initial response")


class AgentDetailsResponse(BaseModel):
    """Detailed response model for individual agent information"""
    id: str = Field(..., description="Agent ID")
    name: str = Field(..., description="Agent name")
    version: str = Field(..., description="Agent version")
    description: Optional[str] = Field(None, description="Agent description")
    is_system: bool = Field(..., description="Whether this is a system agent")
    is_enabled: bool = Field(..., description="Whether the agent is enabled")
    capabilities: List[str] = Field(default=[], description="Agent capabilities")
    default_config: Dict[str, Any] = Field(default={}, description="Default configuration")
    app_id: Optional[str] = Field(None, description="Associated app ID")
    app_name: Optional[str] = Field(None, description="Associated app name")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class AgentActivationsResponse(BaseModel):
    """Response model for agent activations list"""
    activations: List[Dict[str, Any]] = Field(..., description="List of agent activations")


class AgentCreate(BaseModel):
    """Model for creating a new agent"""
    name: str = Field(..., description="Agent name")
    version: str = Field(default="1.0.0", description="Agent version")
    description: Optional[str] = Field(None, description="Agent description")
    capabilities: List[str] = Field(default=[], description="Agent capabilities")
    default_config: Dict[str, Any] = Field(default={}, description="Default configuration")
    is_enabled: bool = Field(default=True, description="Whether the agent is enabled")
    app_id: Optional[str] = Field(None, description="Associated app ID")


class AgentUpdate(BaseModel):
    """Model for updating agent information"""
    name: Optional[str] = Field(None, description="Updated agent name")
    version: Optional[str] = Field(None, description="Updated agent version")
    description: Optional[str] = Field(None, description="Updated description")
    capabilities: Optional[List[str]] = Field(None, description="Updated capabilities")
    default_config: Optional[Dict[str, Any]] = Field(None, description="Updated default configuration")
    is_enabled: Optional[bool] = Field(None, description="Updated enabled status")


class AgentExecutionContext(BaseModel):
    """Model for agent execution context"""
    agent_id: str = Field(..., description="Agent ID")
    app_id: Optional[str] = Field(None, description="App ID")
    user_id: Optional[int] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    context_type: str = Field(default="general", description="Context type")
    metadata: Dict[str, Any] = Field(default={}, description="Additional context metadata")


class AgentResponse(BaseModel):
    """Base model for agent responses"""
    agent_id: str = Field(..., description="Agent ID")
    activation_id: str = Field(..., description="Activation ID")
    response_data: Dict[str, Any] = Field(..., description="Response data")
    status: str = Field(..., description="Response status")
    timestamp: datetime = Field(..., description="Response timestamp")
    execution_time_ms: Optional[int] = Field(None, description="Execution time in milliseconds")


class AgentCapability(BaseModel):
    """Model for defining agent capabilities"""
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")
    required_permissions: List[str] = Field(default=[], description="Required permissions")
    parameters: Dict[str, Any] = Field(default={}, description="Capability parameters schema")


class AgentMetrics(BaseModel):
    """Model for agent performance metrics"""
    agent_id: str = Field(..., description="Agent ID")
    total_activations: int = Field(..., description="Total number of activations")
    successful_activations: int = Field(..., description="Number of successful activations")
    failed_activations: int = Field(..., description="Number of failed activations")
    average_execution_time_ms: Optional[float] = Field(None, description="Average execution time")
    last_activation: Optional[datetime] = Field(None, description="Last activation timestamp")
    uptime_percentage: Optional[float] = Field(None, description="Uptime percentage")

# Legacy enums and utilities kept for compatibility
from enum import Enum
import uuid
import re
import unicodedata


class AgentType(str, Enum):
    """Types of agents supported by the platform"""
    FUNCTION = "function"
    CLASS = "class"
    LLM_CHAT = "llm_chat"
    DATA_PROCESSOR = "data_processor"
    WORKFLOW = "workflow"
    PIPELINE = "pipeline"


class AgentStatus(str, Enum):
    """Agent status values"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    ARCHIVED = "archived"


def slugify(text):
    """Convert a string to a slug format (lowercase, alphanumeric with hyphens)"""
    text = str(text).lower().strip()
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[\s_]+', '-', text)
    text = re.sub(r'[^\w\-]', '', text)
    text = re.sub(r'^\-+|\-+$', '', text)
    text = re.sub(r'\-+', '-', text)
    return text


class AgentBase(BaseModel):
    """Base schema with common agent fields"""
    name: str = Field(..., description="Name of the agent")
    description: Optional[str] = Field(None, description="Description of the agent")
    agent_type_id: str = Field(..., description="The type ID of the agent")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    is_enabled: bool = Field(True, description="Whether the agent is enabled")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentManifest(BaseModel):
    """Schema for agent installation manifest"""
    name: str = Field(..., description="Name of the agent")
    agent_type_id: str = Field(..., description="Unique identifier for the agent type")
    agent_code: Optional[str] = Field(None, description="Unique code for the agent (defaults to slugified name if not provided)")
    description: Optional[str] = Field(None, description="Description of the agent")
    version: str = Field(..., description="Version of the agent")
    model_provider: Optional[str] = Field(None, description="Provider of the language model (for LLM agents)")
    model_name: Optional[str] = Field(None, description="Name of the language model (for LLM agents)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific configuration")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for agent input validation")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for agent output validation")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing the agent")
    author: Optional[str] = Field(None, description="Author of the agent")
    license: Optional[str] = Field(None, description="License information")
    homepage: Optional[str] = Field(None, description="URL for agent homepage or documentation")
    is_public: bool = Field(default=True, description="Whether the agent is publicly available")
    agent_id: Optional[UUID4] = Field(None, description="Optional existing agent ID for updates")
    implementation: Optional[str] = Field(None, description="Python function implementation code")
    implementation_path: Optional[str] = Field(None, description="Path to implementation file")


class AgentCreate(AgentBase):
    """Schema for creating a new agent"""
    credentials: Optional[Dict[str, Any]] = Field(None, description="Credentials for the agent")


class AgentUpdate(BaseModel):
    """Schema for updating an existing agent"""
    name: Optional[str] = Field(None, description="Name of the agent")
    description: Optional[str] = Field(None, description="Description of the agent")
    agent_type_id: Optional[str] = Field(None, description="The type ID of the agent")
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration")
    is_enabled: Optional[bool] = Field(None, description="Whether the agent is enabled")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    credentials: Optional[Dict[str, Any]] = Field(None, description="Credentials for the agent")


class AgentResponse(AgentBase):
    """Schema for agent response"""
    agent_id: UUID4
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    status: Optional[str] = None
    agent_type_name: Optional[str] = None
    credential_ids: Optional[List[str]] = None
    
    model_config = ConfigDict(from_attributes=True)


class AgentVersionInfo(BaseModel):
    """Agent version information"""
    version_id: str = Field(..., description="Version identifier")
    agent_id: str = Field(..., description="Agent identifier")
    version: str = Field(..., description="Version string")
    file_path: Optional[str] = Field(None, description="Path to agent file")
    checksum: Optional[str] = Field(None, description="File checksum")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="Input schema")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="Output schema")
    is_active: bool = Field(False, description="Whether this version is active")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")