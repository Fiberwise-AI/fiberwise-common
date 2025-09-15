"""
Shared API schemas and models for FiberWise applications.
These schemas are used across CLI, web, and SDK components.
"""

from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from pydantic import BaseModel, Field, UUID4, field_validator, ConfigDict
from enum import Enum
import uuid
import re
import unicodedata


# ===== FIELD METADATA =====

def is_system_field(field_name: str) -> bool:
    """
    Check if a field is the system user_id field.
    
    Args:
        field_name: Name of the field to check
        
    Returns:
        bool: True if field is the system user_id field
    """
    return field_name.lower() == "user_id"


def get_system_user_field() -> str:
    """
    Get the name of the system user_id field.
    
    Returns:
        The field name for user identification
    """
    return "user_id"


# ===== AGENT SCHEMAS =====

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


class AgentSummary(BaseModel):
    """Summary information for an agent"""
    id: str = Field(..., description="Unique agent identifier")
    agent_id: str = Field(..., description="Agent GUID")
    name: str = Field(..., description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    version: str = Field("1.0.0", description="Current version")
    agent_type: AgentType = Field(AgentType.FUNCTION, description="Type of agent")
    status: AgentStatus = Field(AgentStatus.ACTIVE, description="Agent status")
    is_system: bool = Field(False, description="Whether this is a system agent")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    user_id: Optional[str] = Field(None, description="User ID who owns this agent (system field)")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# Define slugify function for agent code generation
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


# ===== ACTIVATION SCHEMAS =====

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

    model_config = ConfigDict(from_attributes=True)


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
    
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
    
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


# ===== USER SCHEMAS =====

class UserRole(str, Enum):
    """User roles in the system"""
    ADMIN = "admin"
    USER = "user"
    AGENT_DEVELOPER = "agent_developer"
    VIEWER = "viewer"


class UserSummary(BaseModel):
    """Summary user information"""
    id: str = Field(..., description="User identifier (system field)")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="Email address")
    role: UserRole = Field(UserRole.USER, description="User role")
    is_active: bool = Field(True, description="Whether user is active")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


class UserResponse(BaseModel):
    """Full user information"""
    id: str = Field(..., description="User identifier (system field)")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="Email address")
    full_name: Optional[str] = Field(None, description="Full name")
    role: UserRole = Field(UserRole.USER, description="User role")
    is_active: bool = Field(True, description="Whether user is active")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Update timestamp")


# ===== API RESPONSE SCHEMAS =====

class ApiResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    error: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class PaginatedResponse(BaseModel):
    """Paginated API response"""
    success: bool = Field(..., description="Whether the request was successful")
    data: List[Dict[str, Any]] = Field(..., description="Response data items")
    pagination: Dict[str, Any] = Field(..., description="Pagination information")
    message: Optional[str] = Field(None, description="Response message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


# ===== APP SCHEMAS =====

class AppStatus(str, Enum):
    """Application status values"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"


class AppSummary(BaseModel):
    """Summary app information"""
    id: str = Field(..., description="App identifier")
    app_id: str = Field(..., description="App GUID")
    name: str = Field(..., description="App name")
    description: Optional[str] = Field(None, description="App description")
    status: AppStatus = Field(AppStatus.ACTIVE, description="App status")
    version: str = Field("1.0.0", description="App version")
    user_id: Optional[str] = Field(None, description="User ID who owns this app (system field)")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


# ===== HEALTH CHECK SCHEMAS =====

class HealthStatus(str, Enum):
    """Health check status values"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: HealthStatus = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    checks: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Individual check results")
    version: Optional[str] = Field(None, description="Application version")
    uptime: Optional[float] = Field(None, description="Uptime in seconds")


# ===== ERROR SCHEMAS =====

class ErrorDetail(BaseModel):
    """Detailed error information"""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class ValidationError(BaseModel):
    """Validation error details"""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    code: str = Field(..., description="Validation error code")
    value: Optional[Any] = Field(None, description="Invalid value")


# ===== DATA ISOLATION SCHEMAS =====

class DataIsolationConfig(BaseModel):
    """Configuration for data isolation policies"""
    user_isolation: str = Field("enforced", description="User isolation policy (enforced, optional, disabled)")
    auto_user_assignment: bool = Field(True, description="Automatically assign user_id to new records")
    protect_user_id: bool = Field(True, description="Protect user_id field from modification")


class IsolationMetadata(BaseModel):
    """Metadata for data isolation tracking"""
    isolation_level: str = Field(..., description="Current isolation level")
    user_id: Optional[str] = Field(None, description="User ID for isolation")
    created_with_isolation: bool = Field(True, description="Whether record was created with isolation enabled")


# ===== EXPORT ALL SCHEMAS =====

__all__ = [
    # Field metadata utilities
    'is_system_field', 'get_system_user_field', 'slugify',
    
    # Agent schemas
    'AgentType', 'AgentStatus', 'AgentBase', 'AgentManifest', 'AgentCreate', 'AgentUpdate',
    'AgentResponse', 'AgentSummary', 'AgentVersionInfo',
    
    # Activation schemas
    'ActivationStatus', 'ActivationMetadata', 'ActivationCreate', 'ActivationUpdate', 
    'Activation', 'ActivationResponse', 'ActivationLogEntry',
    
    # User schemas
    'UserRole', 'UserSummary', 'UserResponse',
    
    # API response schemas
    'ApiResponse', 'PaginatedResponse',
    
    # App schemas
    'AppStatus', 'AppSummary',
    
    # Health schemas
    'HealthStatus', 'HealthCheckResponse',
    
    # Error schemas
    'ErrorDetail', 'ValidationError',
    
    # Data isolation schemas
    'DataIsolationConfig', 'IsolationMetadata'
]
