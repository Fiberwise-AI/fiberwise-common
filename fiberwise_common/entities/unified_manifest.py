from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from uuid import UUID
import re
import logging

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s_-]+', '-', text)
    return text.strip('-')


def manifest_dict_to_unified(manifest_dict: Dict[str, Any]) -> UnifiedManifest:
    """
    Convert a raw manifest dictionary to a UnifiedManifest object.

    This is the main entry point for parsing app manifests in any format
    (YAML, JSON) into the unified structure for the FiberWise platform.

    Supports:
    - Legacy flat format (app_name, app_version at top level)
    - New nested format (app: {...})
    - ia_modules execution engine configuration
    - All entity types: apps, agents, pipelines, workflows, functions
    - Auto-generates slugs from names where not provided

    Args:
        manifest_dict: Raw manifest data from YAML/JSON file

    Returns:
        UnifiedManifest object ready for processing

    Example:
        >>> from fiberwise_common.utils.file_utils import load_manifest
        >>> from fiberwise_common.entities.unified_manifest import manifest_dict_to_unified
        >>> manifest_data = load_manifest(Path("app_manifest.yaml"))
        >>> unified = manifest_dict_to_unified(manifest_data)
        >>> unified.validate_all_versions()
    """
    normalized = _normalize_manifest_dict(manifest_dict)
    return UnifiedManifest(**normalized)


def _normalize_manifest_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize manifest dict to UnifiedManifest format."""
    normalized = dict(data)

    if "app_name" in normalized or "app_version" in normalized:
        app_data = {
            "name": normalized.get("app_name", normalized.get("name", "Unknown")),
            "version": normalized.get("app_version", normalized.get("version", "1.0.0")),
            "description": normalized.get("description", ""),
            "app_slug": normalized.get("app_slug", normalized.get("app_name", "").lower().replace(" ", "-")),
            "entryPoint": normalized.get("entryPoint"),
            "icon": normalized.get("icon"),
            "category": normalized.get("category"),
            "publisher": normalized.get("publisher"),
            "user_isolation": normalized.get("user_isolation", "enforced"),
        }
        for old_field in ["app_name", "app_version"]:
            normalized.pop(old_field, None)
        normalized["app"] = app_data

    app_version = normalized.get("app", {}).get("version", "1.0.0")

    if "models" not in normalized or not isinstance(normalized.get("models"), list):
        normalized["models"] = []

    if "routes" not in normalized or not isinstance(normalized.get("routes"), list):
        normalized["routes"] = []

    for entity_type in ["agents", "pipelines", "workflows", "functions"]:
        if entity_type not in normalized or not isinstance(normalized.get(entity_type), list):
            normalized[entity_type] = []

    for pipeline in normalized.get("pipelines", []):
        if isinstance(pipeline, dict):
            if pipeline.get("execution_engine") is None:
                pipeline["execution_engine"] = "fiber-default"
            if pipeline.get("version") is None:
                pipeline["version"] = app_version
            if pipeline.get("slug") is None:
                pipeline["slug"] = _slugify(pipeline.get("name", ""))
            if "is_active" not in pipeline:
                pipeline["is_active"] = True
            if "trigger_config" not in pipeline:
                pipeline["trigger_config"] = {}
            if "execution_config" not in pipeline:
                pipeline["execution_config"] = {}

    for agent in normalized.get("agents", []):
        if isinstance(agent, dict) and agent.get("version") is None:
            agent["version"] = app_version

    for func in normalized.get("functions", []):
        if isinstance(func, dict):
            if func.get("version") is None:
                func["version"] = app_version
            if "tags" not in func:
                func["tags"] = []
            if "function_type" not in func:
                func["function_type"] = "utility"
            if "implementation" not in func:
                func["implementation"] = None

    for workflow in normalized.get("workflows", []):
        if isinstance(workflow, dict) and workflow.get("version") is None:
            workflow["version"] = app_version

    return normalized

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
    """Pipeline manifest model with support for multiple execution engines."""
    name: str
    version: str
    slug: Optional[str] = None
    description: Optional[str] = None
    structure: Optional[dict] = None
    implementation_path: Optional[str] = None
    execution_engine: str = "fiber-default"
    pipeline_definition: Optional[str] = None
    engine_config: Optional[dict] = None
    input_schema: Optional[dict] = None
    output_schema: Optional[dict] = None
    is_async: Optional[bool] = True

    @model_validator(mode='after')
    def validate_engine_config(self):
        """Validate engine-specific configuration."""
        if self.execution_engine == "ia_modules":
            if not self.structure and not self.pipeline_definition:
                raise ValueError(
                    "ia_modules pipelines require either 'structure' (graph definition) "
                    "or 'pipeline_definition' (path to workflow file)"
                )
        return self

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
