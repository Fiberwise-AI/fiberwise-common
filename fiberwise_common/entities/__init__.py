# Import FiberAgent from SDK for consistency
try:
    from fiberwise_sdk import FiberAgent
    _FIBER_AGENT_AVAILABLE = True
except ImportError:
    FiberAgent = None
    _FIBER_AGENT_AVAILABLE = False

from .workflows import (
    StepType,
    WorkflowStep,
    WorkflowVariable,
    WorkflowManifest,
    WorkflowBase,
    WorkflowCreate,
    WorkflowUpdate,
    WorkflowResponse,
    WorkflowExecuteRequest,
    StepExecution,
    WorkflowExecutionResponse,
    WorkflowExecutionListResponse,
)

# Import agent schemas
from .agents import (
    AgentType,
    AgentStatus,
    AgentSummaryWeb,
    AgentBase,
    AgentManifest,
    AgentCreate,
    AgentUpdate,
    AgentResponse,
    AgentVersionInfo,
    slugify,
)

# Import FiberAgent base class
from .fiber_agent import FiberAgent, FiberInjectable

# Import activation schemas
from .activations import (
    ActivationStatus,
    ActivationMetadata,
    ActivationCreate,
    ActivationUpdate,
    Activation,
    ActivationResponse,
    ActivationLogEntry,
)

# Import utility functions from utils module
from .utils import (
    is_system_field,
    get_system_user_field,
)

# Import isolation schemas
from .isolation import (
    DataIsolationConfig,
    IsolationMetadata,
)

# Import app schemas
from .app import (
    MarketplaceAppRead,
    AppModelItem,
    AppModelItemsList,
    AppInstallation,
    AppModelItemCreate,
    FieldCreate,
    ModelCreate,
    ModelRead,
    AppCreate,
    AppUpdate,
    AppRead,
    AppManifest,
    AppInstallConfig,
    AppInstallResponse,
)

# Import API key schemas
from .api_keys import (
    APIKeyCreate,
    APIKeyCreateResponse,
    APIKeyResponse,
    APIKeyTokenResponse,
)

# Import function schemas
from .functions import (
    FunctionType,
    FunctionCreate,
    FunctionUpdate,
    FunctionResponse,
    FunctionExecuteRequest,
    FunctionExecuteResponse,
)

# Import pipeline schemas
from .pipelines import (
    PipelineStatus,
    PipelineCreate,
    PipelineUpdate,
    PipelineResponse,
    PipelineExecuteRequest,
    PipelineExecuteResponse,
)

# Import LLM provider schemas
from .llm_providers import (
    LLMProviderConfig,
    LLMProvider,
)

# Import unified manifest schemas
from .unified_manifest import (
    AppManifest,
    AgentManifest,
    PipelineManifest,
    WorkflowManifest,
    FunctionManifest,
    ComponentInstallationResult,
    ComponentUpdateResult,
    ManifestInstallationResponse,
    UnifiedManifest,
)

# Import app response schemas
from .app_responses import (
    PaginatedAppsResponse,
    ModelItemCreateResponse,
    FunctionExecuteResponse,
    FunctionStats,
    FunctionHistoryItem,
    FunctionHistoryResponse,
    AppFunctionHistoryResponse,
    AppStatusResponse,
)

# Import service provider schemas
from .service_providers import (
    ServiceProviderResponse,
    ServiceProviderRegistrationResponse,
)

__all__ = [
    # Workflow schemas
    "StepType",
    "WorkflowStep", 
    "WorkflowVariable",
    "WorkflowManifest",
    "WorkflowBase",
    "WorkflowCreate",
    "WorkflowUpdate", 
    "WorkflowResponse",
    "WorkflowExecuteRequest",
    "StepExecution",
    "WorkflowExecutionResponse",
    "WorkflowExecutionListResponse",
    
    # Agent schemas
    "AgentType",
    "AgentStatus",
    "AgentSummary",
    "AgentBase",
    "AgentManifest",
    "AgentCreate",
    "AgentUpdate",
    "AgentResponse",
    "AgentVersionInfo",
    "slugify",
    
    # Agent base classes
    "FiberAgent",
    "FiberInjectable",
    
    # Activation schemas
    "ActivationStatus",
    "ActivationMetadata",
    "ActivationCreate",
    "ActivationUpdate",
    "Activation",
    "ActivationResponse",
    "ActivationLogEntry",
    
    # Utility functions
    "is_system_field",
    "get_system_user_field",
    
    # Isolation schemas
    "DataIsolationConfig",
    "IsolationMetadata",
    
    # App schemas
    "MarketplaceAppRead",
    "AppModelItem",
    "AppModelItemsList", 
    "AppInstallation",
    "AppModelItemCreate",
    "FieldCreate",
    "ModelCreate", 
    "ModelRead",
    "AppCreate",
    "AppUpdate",
    "AppRead",
    "AppManifest",
    "AppInstallConfig",
    "AppInstallResponse",
    
    # API key schemas
    "APIKeyCreate",
    "APIKeyCreateResponse", 
    "APIKeyResponse",
    "APIKeyTokenResponse",
    
    # Function schemas
    "FunctionType",
    "FunctionCreate",
    "FunctionUpdate",
    "FunctionResponse", 
    "FunctionExecuteRequest",
    "FunctionExecuteResponse",
    
    # Pipeline schemas
    "PipelineStatus",
    "PipelineCreate",
    "PipelineUpdate",
    "PipelineResponse",
    "PipelineExecuteRequest", 
    "PipelineExecuteResponse",
    
    # LLM provider schemas
    "LLMProviderConfig",
    "LLMProvider",
    
    # Unified manifest schemas
    "AppManifest",
    "AgentManifest", 
    "PipelineManifest",
    "WorkflowManifest",
    "FunctionManifest",
    "ComponentInstallationResult",
    "ComponentUpdateResult",
    "ManifestInstallationResponse",
    "UnifiedManifest",
    
    # App response schemas
    "PaginatedAppsResponse",
    "ModelItemCreateResponse", 
    "FunctionExecuteResponse",
    "FunctionStats",
    "FunctionHistoryItem",
    "FunctionHistoryResponse",
    "AppFunctionHistoryResponse",
    "AppStatusResponse",
    
    # Service provider schemas
    "ServiceProviderResponse",
    "ServiceProviderRegistrationResponse",
]