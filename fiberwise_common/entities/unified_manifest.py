from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, ConfigDict
from uuid import UUID
import re
import logging

logger = logging.getLogger(__name__)

class AppManifest(BaseModel):
    """App manifest model - basic version for now"""
    name: str
    app_slug: str
    version: str
    description: Optional[str] = None
    entryPoint: Optional[str] = None
    icon: Optional[str] = None
    category: Optional[str] = None
    publisher: Optional[str] = None
    user_isolation: Optional[str] = Field(default="enforced", description="User isolation policy: enforced, disabled, or optional")
    models: Optional[List[Dict[str, Any]]] = []
    routes: Optional[List[Dict[str, Any]]] = []
    
class AgentManifest(BaseModel):
    """Agent manifest model - basic version for now"""
    name: str
    version: str
    agent_type_id: Optional[str] = None
    description: Optional[str] = None
    implementation_path: Optional[str] = None

class PipelineManifest(BaseModel):
    """Pipeline manifest model"""
    name: str
    version: str
    description: Optional[str] = None
    structure: Optional[dict] = None  # NEW: Graph-based pipeline definition
    implementation_path: Optional[str] = None  # Legacy fallback, optional now
    input_schema: Optional[dict] = None
    output_schema: Optional[dict] = None
    is_async: Optional[bool] = True

class WorkflowManifest(BaseModel):
    """Workflow manifest model - basic version for now"""
    name: str
    version: str
    description: Optional[str] = None

class FunctionManifest(BaseModel):
    """Function manifest model - basic version for now"""
    name: str
    version: str
    description: Optional[str] = None
    implementation_path: Optional[str] = None
    input_schema: Optional[dict] = None
    output_schema: Optional[dict] = None
    tags: Optional[List[str]] = None
    is_async: Optional[bool] = False

class ComponentInstallationResult(BaseModel):
    """Result of installing a component"""
    success: bool
    component_id: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    app_version_id: Optional[str] = None  # For update operations

class ComponentUpdateResult(BaseModel):
    """Result of updating a component"""
    success: bool
    component_id: Optional[str] = None
    old_version: Optional[str] = None
    new_version: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None

class ManifestInstallationResponse(BaseModel):
    """Response from installing a manifest"""
    success: bool
    app_results: List[ComponentInstallationResult] = Field(default_factory=list)
    agent_results: List[ComponentInstallationResult] = Field(default_factory=list)
    pipeline_results: List[ComponentInstallationResult] = Field(default_factory=list)
    workflow_results: List[ComponentInstallationResult] = Field(default_factory=list)
    function_results: List[ComponentInstallationResult] = Field(default_factory=list)
    message: Optional[str] = None
    error: Optional[str] = None

class UnifiedManifest(BaseModel):
    """
    A unified manifest that can contain different entity types
    (apps, agents, pipelines, workflows, functions)
    """
    # Core app information (required)
    app: AppManifest
    
    # Entity collections with empty defaults
    agents: List[AgentManifest] = Field(default_factory=list)
    pipelines: List[PipelineManifest] = Field(default_factory=list)
    workflows: List[WorkflowManifest] = Field(default_factory=list)  
    functions: List[FunctionManifest] = Field(default_factory=list)
    
    # Configuration
    model_config = ConfigDict(extra="ignore")
    
    def has_entities(self) -> bool:
        """Check if manifest contains any entity definitions besides the app"""
        return (
            len(self.agents) > 0 or
            len(self.pipelines) > 0 or
            len(self.workflows) > 0 or
            len(self.functions) > 0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return self.model_dump()
    
    @field_validator('app')
    @classmethod
    def validate_app_version(cls, app):
        """Validate that app has a valid version"""
        if not hasattr(app, 'version') or not app.version:
            raise ValueError("App must have a version specified")
        
        if not cls.is_valid_semantic_version(app.version):
            raise ValueError(f"App version '{app.version}' is not a valid semantic version (e.g., 1.0.0)")
        
        return app
    
    @field_validator('agents')
    @classmethod
    def validate_agent_versions(cls, agents):
        """Validate that all agents have valid versions"""
        for i, agent in enumerate(agents):
            if not hasattr(agent, 'version') or not agent.version:
                raise ValueError(f"Agent at index {i} (name: {getattr(agent, 'name', 'unknown')}) must have a version specified")
            
            if not cls.is_valid_semantic_version(agent.version):
                raise ValueError(f"Agent '{agent.name}' version '{agent.version}' is not a valid semantic version (e.g., 1.0.0)")
        
        return agents
    
    @field_validator('pipelines')
    @classmethod
    def validate_pipeline_versions(cls, pipelines):
        """Validate that all pipelines have valid versions"""
        for i, pipeline in enumerate(pipelines):
            if not hasattr(pipeline, 'version') or not pipeline.version:
                raise ValueError(f"Pipeline at index {i} (name: {getattr(pipeline, 'name', 'unknown')}) must have a version specified")
            
            if not cls.is_valid_semantic_version(pipeline.version):
                raise ValueError(f"Pipeline '{pipeline.name}' version '{pipeline.version}' is not a valid semantic version (e.g., 1.0.0)")
        
        return pipelines
    
    @field_validator('workflows')
    @classmethod
    def validate_workflow_versions(cls, workflows):
        """Validate that all workflows have valid versions"""
        for i, workflow in enumerate(workflows):
            if not hasattr(workflow, 'version') or not workflow.version:
                raise ValueError(f"Workflow at index {i} (name: {getattr(workflow, 'name', 'unknown')}) must have a version specified")
            
            if not cls.is_valid_semantic_version(workflow.version):
                raise ValueError(f"Workflow '{workflow.name}' version '{workflow.version}' is not a valid semantic version (e.g., 1.0.0)")
        
        return workflows
    
    @field_validator('functions')
    @classmethod
    def validate_function_versions(cls, functions):
        """Validate that all functions have valid versions"""
        for i, function in enumerate(functions):
            if not hasattr(function, 'version') or not function.version:
                raise ValueError(f"Function at index {i} (name: {getattr(function, 'name', 'unknown')}) must have a version specified")
            
            if not cls.is_valid_semantic_version(function.version):
                raise ValueError(f"Function '{function.name}' version '{function.version}' is not a valid semantic version (e.g., 1.0.0)")
        
        return functions
    
    @classmethod
    def is_valid_semantic_version(cls, version: str) -> bool:
        """
        Validates that a version string is in semantic versioning format (e.g., 1.0.0)
        See: https://semver.org/
        """
        # Simple regex for basic semver validation (X.Y.Z format)
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
        return bool(re.match(pattern, version))
    
    def validate_all_versions(self) -> Tuple[bool, Optional[str]]:
        """
        Validate all versions in the manifest
        
        Returns:
            Tuple containing (is_valid, error_message)
        """
        try:
            # App version
            if not hasattr(self.app, 'version') or not self.app.version:
                return False, "App must have a version specified"
            
            if not self.is_valid_semantic_version(self.app.version):
                return False, f"App version '{self.app.version}' is not a valid semantic version (e.g., 1.0.0)"
            
            # Agent versions
            for i, agent in enumerate(self.agents):
                if not hasattr(agent, 'version') or not agent.version:
                    return False, f"Agent at index {i} (name: {getattr(agent, 'name', 'unknown')}) must have a version specified"
                
                if not self.is_valid_semantic_version(agent.version):
                    return False, f"Agent '{agent.name}' version '{agent.version}' is not a valid semantic version"
            
            # Pipeline versions
            for i, pipeline in enumerate(self.pipelines):
                if not hasattr(pipeline, 'version') or not pipeline.version:
                    return False, f"Pipeline at index {i} (name: {getattr(pipeline, 'name', 'unknown')}) must have a version specified"
                
                if not self.is_valid_semantic_version(pipeline.version):
                    return False, f"Pipeline '{pipeline.name}' version '{pipeline.version}' is not a valid semantic version"
            
            # Workflow versions
            for i, workflow in enumerate(self.workflows):
                if not hasattr(workflow, 'version') or not workflow.version:
                    return False, f"Workflow at index {i} (name: {getattr(workflow, 'name', 'unknown')}) must have a version specified"
                
                if not self.is_valid_semantic_version(workflow.version):
                    return False, f"Workflow '{workflow.name}' version '{workflow.version}' is not a valid semantic version"
            
            # Function versions
            for i, function in enumerate(self.functions):
                if not hasattr(function, 'version') or not function.version:
                    return False, f"Function at index {i} (name: {getattr(function, 'name', 'unknown')}) must have a version specified"
                
                if not self.is_valid_semantic_version(function.version):
                    return False, f"Function '{function.name}' version '{function.version}' is not a valid semantic version"
            
            # All versions are valid
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating manifest versions: {str(e)}")
            return False, f"Validation error: {str(e)}"
