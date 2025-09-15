from datetime import datetime
from http.client import HTTPException
import logging
from uuid import UUID, uuid4
from typing import Dict, List, Any, Tuple, Optional
import yaml
import json  # Add json import for serialization
import os  # Add os import for file extension checks
from pydantic import BaseModel, Field
import uuid

from fiberwise_common import DatabaseProvider
from fiberwise_common.entities import is_system_field, DataIsolationConfig
from fiberwise_common.entities.user import User
from fiberwise_common.entities import UnifiedManifest, AgentManifest, PipelineManifest, WorkflowManifest, FunctionManifest
from fiberwise_common.services.app_service import import_app_from_manifest
from ..utils.agent_templates import create_minimal_agent_code

logger = logging.getLogger(__name__)


def validate_user_access(current_user, resource_user_id: str, operation: str = "access") -> bool:
    """Validate that a user can access a resource based on user isolation rules."""
    if not current_user:
        logger.warning(f"No current user provided for {operation} validation")
        return False
    
    if not resource_user_id:
        logger.warning(f"No resource user_id provided for {operation} validation")
        return False
    
    # Install operations should allow cross-user access for deployment
    # User isolation enforcement happens at the API/query level, not install level
    return True

def auto_assign_user_id(data: dict, current_user) -> dict:
    """Automatically assign user_id to a data record."""
    if current_user:
        if "user_id" not in data or not data["user_id"]:
            data["user_id"] = str(current_user.id)
            logger.debug(f"Auto-assigned user_id {current_user.id}")
    return data


# Define custom exception for agent code conflicts
class AgentCodeConflictError(Exception):
    """Exception raised when there's a conflict with agent_type_id and agent_code combination"""
    def __init__(self, agent_name, agent_code, agent_type_id, message=None):
        self.agent_name = agent_name
        self.agent_code = agent_code
        self.agent_type_id = agent_type_id
        self.message = message or f"An agent with type ID '{agent_type_id}' and code '{agent_code}' already exists"
        super().__init__(self.message)

# Define Pydantic models for component results
class ComponentResult(BaseModel):
    """Base model for component installation results"""
    id: str
    name: str
    status: str  # created, updated, error
    version_id: Optional[str] = None
    version: Optional[str] = None
    error: Optional[str] = None
    
class AppResult(ComponentResult):
    """App installation result model"""
    model_count: Optional[int] = None
    
class AgentResult(ComponentResult):
    """Agent installation result model"""
    implementation_type: Optional[str] = None
    
class PipelineResult(ComponentResult):
    """Pipeline installation result model"""
    pass
    
class WorkflowResult(ComponentResult):
    """Workflow installation result model"""
    pass
    
class FunctionResult(ComponentResult):
    """Function installation result model"""
    app_association: Optional[str] = None
    app_id: Optional[str] = None

class UnifiedManifestResult(BaseModel):
    """Complete result of processing a unified manifest"""
    app: List[AppResult] = Field(default_factory=list)
    agents: List[AgentResult] = Field(default_factory=list)
    pipelines: List[PipelineResult] = Field(default_factory=list)
    workflows: List[WorkflowResult] = Field(default_factory=list)
    functions: List[FunctionResult] = Field(default_factory=list)

# ---------------------------
# Agent Installation Methods
# ---------------------------
async def _create_new_agent(conn, agent_manifest: AgentManifest, agent_slug: str, current_user: User, app_id: Optional[str] = None):
    """
    Helper function to create a new agent
    
    Args:
        conn: Database connection
        agent_manifest: The agent manifest data
        agent_slug: The slug for the agent
        current_user: The authenticated user
        app_id: Optional app ID to associate with the agent (required by DB schema)
    """
    # Generate a new UUID for the agent
    agent_id = uuid4()
    
    # Validate agent_type_id against agent_types table
    agent_type_id = getattr(agent_manifest, 'agent_type_id', None)
    
    # Check if agent_type_id exists in agent_types table
    type_exists = False
    if agent_type_id is not None:
        check_query = """
            SELECT id FROM agent_types WHERE id = $1
        """
        type_exists = await conn.fetch_val(check_query, agent_type_id)
    
    if not type_exists and agent_type_id is not None:
        # If invalid agent_type_id provided, use 'custom' as fallback
        logger.warning(f"Invalid agent_type_id '{agent_type_id}' for agent '{agent_manifest.name}'. Using 'custom' instead.")
        agent_type_id = 'custom'
    elif agent_type_id is None:
        # If no agent_type_id provided, use 'custom' as default
        agent_type_id = 'custom'
        logger.info(f"No agent_type_id provided for agent '{agent_manifest.name}'. Using 'custom'.")
    
    # Get configuration/config with default
    config = getattr(agent_manifest, 'config', getattr(agent_manifest, 'configuration', {}))
    
    # Get is_enabled with default
    is_enabled = getattr(agent_manifest, 'is_enabled', True)
    
    # Get agent_code with default
    agent_code = getattr(agent_manifest, 'agent_code', getattr(agent_manifest, 'code', None))
    
    # Serialize config dictionary to JSON string
    config_json = json.dumps(config)
    
    # Get schemas
    input_schema = json.dumps(getattr(agent_manifest, 'input_schema', {}))
    output_schema = json.dumps(getattr(agent_manifest, 'output_schema', {}))
    
    # If no app_id is provided, the agent will be associated with the user directly
    if not app_id:
        logger.info(f"No app_id provided for agent '{agent_manifest.name}'. It will be a user-level agent.")
    
    # Check if this is a custom agent type - verify by checking if agent_type_id contains "custom"
    has_custom_code = False
    implementation_code = None
    if agent_type_id and "custom" in agent_type_id.lower():
        # Look for implementation code in the manifest - provide None as default
        implementation_code = getattr(agent_manifest, 'implementation', None)
        
        if implementation_code:
            has_custom_code = True
            logger.info(f"Found implementation code for custom agent: {agent_manifest.name}")
    
    
    # Create the agent record with user isolation enforced
    insert_query = """
        INSERT INTO agents (
            agent_id, name, description, agent_type_id, 
            agent_slug, agent_code, is_enabled, config, created_by, updated_at, app_id,
            input_schema, output_schema
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, CURRENT_TIMESTAMP, $10, $11, $12
        ) RETURNING agent_id
    """
    
    # Log user isolation enforcement
    logger.info(f"Creating agent with user isolation enforced - creator: {current_user.id}")
    
    try:
        await conn.fetch_one(
            insert_query,
            str(agent_id),
            agent_manifest.name,
            agent_manifest.description,
            agent_type_id,
            agent_slug,
            agent_code,
            is_enabled,
            config_json,
            current_user.id,
            app_id,
            input_schema,
            output_schema
        )
    except Exception as e:
        error_msg = str(e)
        if "duplicate key value violates unique constraint" in error_msg and "agents_agent_type_id_agent_code_key" in error_msg:
            raise AgentCodeConflictError(
                agent_name=agent_manifest.name,
                agent_code=agent_code,
                agent_type_id=agent_type_id
            )
        raise  # Re-raise the original exception if it's not a code conflict
    
    # If this is a custom agent with implementation code, store it in agent_implementations table
    if agent_type_id == "custom" and implementation_code:
        # Validate required fields before storing implementation
        if not implementation_code:
            raise ValueError(f"Implementation code is required for custom agent: {agent_manifest.name}")
        
        # Get language from manifest - no default
        language = getattr(agent_manifest, 'language', None)
        if not language:
            raise ValueError(f"Language is required for custom agent: {agent_manifest.name}")
        
        # Get version from manifest - no default
        version = getattr(agent_manifest, 'version', None)
        if not version:
            raise ValueError(f"Version is required for custom agent: {agent_manifest.name}")
        
        # Extract additional implementation details from manifest
        entrypoint_file = getattr(agent_manifest, 'entrypoint_file', None)
        implementation_path = getattr(agent_manifest, 'implementation_path', None)
        class_name = getattr(agent_manifest, 'class_name', None)
        
        # Store implementation code in the agent_implementations table
        implementation_query = """
            INSERT INTO agent_implementations (
                agent_id, implementation_code, language, version, is_active, created_by,
                implementation_path, entrypoint_file, class_name
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9
            ) RETURNING implementation_id
        """
        
        implementation_id = await conn.fetch_val(
            implementation_query,
            str(agent_id),
            implementation_code,
            language,
            version,
            True,  # is_active
            current_user.id,
            implementation_path,
            entrypoint_file,
            class_name
        )
        
        logger.info(f"Stored implementation code for custom agent: {agent_manifest.name} with ID {implementation_id}")
    
    logger.info(f"Created new agent '{agent_manifest.name}' with ID {agent_id} for app {app_id}")
    return agent_id

async def _create_agent_version(conn, agent_id, agent_manifest, current_user: User):
    """
    Helper function to create a new version for an agent
    """
    # Generate a new UUID for the version
    version_id = uuid4()
    
    # Get version from manifest or use default
    version = getattr(agent_manifest, 'version', '0.0.1')
    
    # Get configuration/config with default
    config = getattr(agent_manifest, 'config', getattr(agent_manifest, 'configuration', {}))
    
    # Serialize config dictionary to JSON string
    config_json = json.dumps(config)
    
    # Get implementation path from manifest
    implementation_path = getattr(agent_manifest, 'implementation_path', None)
    
    # Get description and schemas
    description = getattr(agent_manifest, 'description', None)
    input_schema = json.dumps(getattr(agent_manifest, 'input_schema', {}))
    output_schema = json.dumps(getattr(agent_manifest, 'output_schema', {}))
    
    # First check if there's already a draft version for this agent with the same version
    check_draft_query = """
        SELECT version_id, status, is_active FROM agent_versions
        WHERE agent_id = $1 AND version = $2 AND status = 'draft' AND is_active = false
        ORDER BY created_at DESC
        LIMIT 1
    """

    existing_version_row = await conn.fetch_one(check_draft_query, str(agent_id), version)
    existing_version_id = existing_version_row['version_id'] if existing_version_row else None

    logger.info(f"Checking for existing draft version for agent {agent_id} version {version}: found={existing_version_id}")

    if existing_version_row:
        logger.info(f"Found existing draft: version_id={existing_version_row['version_id']}, status={existing_version_row['status']}, is_active={existing_version_row['is_active']}")

    if existing_version_id:
        # Update the existing draft version instead of creating a new one
        update_query = """
            UPDATE agent_versions
            SET config = $1, file_path = $2, description = $3,
                input_schema = $4, output_schema = $5, updated_at = CURRENT_TIMESTAMP
            WHERE version_id = $6
            RETURNING version_id
        """

        result = await conn.fetch_val(
            update_query,
            config_json,
            implementation_path,
            description,
            input_schema,
            output_schema,
            str(existing_version_id)
        )

        logger.info(f"Updated existing draft version {version} with ID {existing_version_id} for agent {agent_id}, file_path: {implementation_path}")
        return existing_version_id
    else:
        # Create a new agent version record
        insert_query = """
            INSERT INTO agent_versions (
                version_id, agent_id, version, config, file_path, created_by, created_at,
                description, input_schema, output_schema, status, is_active
            ) VALUES (
                $1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP, $7, $8, $9, 'draft', false
            ) RETURNING version_id
        """

        result = await conn.fetch_val(
            insert_query,
            str(version_id),
            str(agent_id),
            version,
            config_json,
            implementation_path,
            int(current_user.id) if hasattr(current_user, 'id') else 1,
            description,
            input_schema,
            output_schema
        )

        logger.info(f"Created new version {version} with ID {version_id} for agent {agent_id}, file_path: {implementation_path}")
        return version_id

# -----------------------------
# Pipeline Installation Methods
# -----------------------------
async def _create_new_pipeline(conn, pipeline_manifest: PipelineManifest, pipeline_slug: str, current_user: User, app_id: Optional[str] = None, definition: str = None):
    """Helper function to create a new pipeline with structure support"""
    # Generate a new UUID for the pipeline
    pipeline_id = uuid4()
    
    # Get configuration/config with default
    config = getattr(pipeline_manifest, 'config', getattr(pipeline_manifest, 'configuration', {}))
    config_json = json.dumps(config)
    
    # Get file_path from manifest (for backward compatibility)
    file_path = getattr(pipeline_manifest, 'file_path', getattr(pipeline_manifest, 'implementation_path', None))
    
    # Use provided definition or create empty one
    if definition is None:
        definition = json.dumps({})
    
    # Create the pipeline record
    insert_query = """
        INSERT INTO pipelines (
            pipeline_id, name, description, config, created_by, created_at, file_path, pipeline_slug, app_id, definition
        ) VALUES (
            $1, $2, $3, $4, $5, CURRENT_TIMESTAMP, $6, $7, $8, $9
        ) RETURNING pipeline_id
    """
    
    await conn.fetch_one(
        insert_query,
        str(pipeline_id),
        pipeline_manifest.name,
        pipeline_manifest.description,
        config_json,
        current_user.id,
        file_path,
        pipeline_slug,
        app_id,
        definition  # NEW: Store pipeline structure
    )
    
    logger.info(f"Created new pipeline '{pipeline_manifest.name}' with ID {pipeline_id}")
    return pipeline_id

async def _create_pipeline_version(conn, pipeline_id, pipeline_manifest, current_user: User):
    """Helper function to create a new version for a pipeline"""
    # Generate a new UUID for the version
    version_id = uuid4()
    
    # Get version from manifest or use default
    version = getattr(pipeline_manifest, 'version', '0.0.1')
    
    # Get configuration/config with default
    config = getattr(pipeline_manifest, 'config', getattr(pipeline_manifest, 'configuration', {}))
    
    # Serialize config dictionary to JSON string
    config_json = json.dumps(config)

    # Get file_path from manifest
    file_path = getattr(pipeline_manifest, 'file_path', getattr(pipeline_manifest, 'implementation_path', None))
    
    # Serialize manifest to YAML
    manifest_yaml = yaml.dump(pipeline_manifest.dict())
    
    # Create the pipeline version record
    insert_query = """
        INSERT INTO pipeline_versions (
            version_id, pipeline_id, version, description, config, file_path, created_by, created_at, manifest_yaml
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP, $8
        ) RETURNING version_id
    """
    
    await conn.fetch_val(
        insert_query,
        str(version_id),
        str(pipeline_id),
        version,
        pipeline_manifest.description,
        config_json,
        file_path,
        current_user.id,
        manifest_yaml
    )
    
    logger.info(f"Created new version {version} with ID {version_id} for pipeline {pipeline_id}")
    return version_id

# -----------------------------
# Workflow Installation Methods
# -----------------------------
async def _create_new_workflow(conn, workflow_manifest: WorkflowManifest, current_user: User):
    """Helper function to create a new workflow"""
    # Generate a new UUID for the workflow
    workflow_id = uuid4()
    
    # Get configuration/config with default
    config = getattr(workflow_manifest, 'config', getattr(workflow_manifest, 'configuration', {}))
    
    # Serialize config dictionary to JSON string
    config_json = json.dumps(config)
    
    # Create the workflow record
    insert_query = """
        INSERT INTO workflows (
            workflow_id, name, description, config, created_by, created_at
        ) VALUES (
            $1, $2, $3, $4, $5, CURRENT_TIMESTAMP
        ) RETURNING workflow_id
    """
    
    await conn.fetch_one(
        insert_query,
        str(workflow_id),
        workflow_manifest.name,
        workflow_manifest.description,
        config_json,
        current_user.id
    )
    
    logger.info(f"Created new workflow '{workflow_manifest.name}' with ID {workflow_id}")
    return workflow_id

async def _create_workflow_version(conn, workflow_id, workflow_manifest, current_user: User):
    """Helper function to create a new version for a workflow"""
    # Generate a new UUID for the version
    version_id = uuid4()
    
    # Get version from manifest or use default
    version = getattr(workflow_manifest, 'version', '0.0.1')
    
    # Get configuration/config with default
    config = getattr(workflow_manifest, 'config', getattr(workflow_manifest, 'configuration', {}))
    
    # Serialize config dictionary to JSON string
    config_json = json.dumps(config)
    
    # Create the workflow version record
    insert_query = """
        INSERT INTO workflow_versions (
            version_id, workflow_id, version, config, created_by, created_at
        ) VALUES (
            $1, $2, $3, $4, $5, CURRENT_TIMESTAMP
        ) RETURNING version_id
    """
    
    await conn.fetch_one(
        insert_query,
        str(version_id),
        str(workflow_id),
        version,
        config_json,
        current_user.id
    )
    
    logger.info(f"Created new version for workflow {workflow_id}")
    return version_id

# -----------------------------
# Function Installation Methods
# -----------------------------
async def _create_new_function(conn, function_manifest: FunctionManifest, current_user: User):
    """Helper function to create a new function"""
    # Generate a new UUID for the function
    function_id = uuid4()
    
    # Get configuration/config with default
    config = getattr(function_manifest, 'config', getattr(function_manifest, 'configuration', {}))
    
    # Serialize config dictionary to JSON string
    config_json = json.dumps(config)
    
    # Get all the fields from the manifest
    function_type = getattr(function_manifest, 'function_type', 'utility')
    tags = json.dumps(getattr(function_manifest, 'tags', []))
    input_schema = json.dumps(getattr(function_manifest, 'input_schema', {}))
    output_schema = json.dumps(getattr(function_manifest, 'output_schema', {}))
    is_async = 1 if getattr(function_manifest, 'is_async', False) else 0
    implementation = getattr(function_manifest, 'implementation', None)
    
    # Create the function record with all necessary fields
    insert_query = """
        INSERT INTO functions (
            function_id, name, description, function_type, tags, 
            input_schema, output_schema, config, is_async,
            created_by, created_at, implementation
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, CURRENT_TIMESTAMP, $11
        ) RETURNING function_id
    """
    
    await conn.execute(
        insert_query,
        str(function_id),
        function_manifest.name,
        function_manifest.description,
        function_type,
        tags,
        input_schema,
        output_schema,
        config_json,
        is_async,
        current_user.id,
        implementation
    )
    
    logger.info(f"Created new function '{function_manifest.name}' with ID {function_id}")
    return function_id

async def _create_function_version(conn, function_id, function_manifest, current_user: User):
    """Helper function to create a new version for a function"""
    # Generate a new UUID for the version
    version_id = uuid4()
    
    # Get version from manifest or use default
    version = getattr(function_manifest, 'version', '0.0.1')
    
    # Get configuration/config with default
    config = getattr(function_manifest, 'config', getattr(function_manifest, 'configuration', {}))
    
    # Serialize config dictionary to JSON string
    config_json = json.dumps(config)
    
    # Get fields from manifest
    description = getattr(function_manifest, 'description', None)
    implementation = getattr(function_manifest, 'implementation', None)
    input_schema = json.dumps(getattr(function_manifest, 'input_schema', {}))
    output_schema = json.dumps(getattr(function_manifest, 'output_schema', {}))
    
    # Create the function version record
    insert_query = """
        INSERT INTO function_versions (
            version_id, function_id, version, config, created_by, created_at,
            description, implementation, input_schema, output_schema
        ) VALUES (
            $1, $2, $3, $4, $5, CURRENT_TIMESTAMP, $6, $7, $8, $9
        ) RETURNING version_id
    """
    
    await conn.fetch_one(
        insert_query,
        str(version_id),
        str(function_id),
        version,
        config_json,
        current_user.id,
        description,
        implementation,
        input_schema,
        output_schema
    )
    
    logger.info(f"Created new version for function {function_id}")
    return version_id

async def _associate_function_with_app(conn, function_id: str, app_id: str) -> bool:
    """
    Associate a function with an app in the functions_app table.
    Will not create duplicate entries.
    """
    # Check if the association already exists
    check_query = """
        SELECT 1 FROM functions_app WHERE function_id = $1 AND app_id = $2
    """
    exists = await conn.fetch_val(check_query, function_id, app_id)
    
    if exists:
        logger.info(f"Function {function_id} is already associated with app {app_id}")
        return False
    
    # Create the association
    insert_query = """
        INSERT INTO functions_app (function_id, app_id) VALUES ($1, $2)
    """
    await conn.execute(insert_query, function_id, app_id)
    
    logger.info(f"Associated function {function_id} with app {app_id}")
    return True

async def _extract_and_store_routes_from_manifest(conn, app_id: str, app_version_id: str, manifest_yaml: str) -> bool:
    """
    Extract routes from app manifest YAML and store them in the app_routes table.
    
    Args:
        conn: Database connection
        app_id: The app ID
        app_version_id: The app version ID
        manifest_yaml: The YAML manifest content as string
    
    Returns:
        bool: True if routes were processed successfully
    """
    try:
        if not manifest_yaml:
            logger.warning(f"No manifest YAML provided for app {app_id}")
            return False
            
        # Parse the YAML manifest
        try:
            manifest_data = yaml.safe_load(manifest_yaml)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse manifest YAML for app {app_id}: {e}")
            return False
        
        if not isinstance(manifest_data, dict):
            logger.warning(f"Manifest data is not a dictionary for app {app_id}")
            return False
            
        # Extract routes from the manifest
        routes = manifest_data.get('routes', [])
        if not routes:
            logger.info(f"No routes found in manifest for app {app_id}")
            return True  # Not an error, just no routes
        
        # First, delete existing routes for this app version to avoid duplicates
        delete_query = """
            DELETE FROM app_routes 
            WHERE app_id = $1 AND app_version_id = $2
        """
        await conn.execute(delete_query, app_id, app_version_id)
        logger.info(f"Cleared existing routes for app {app_id} version {app_version_id}")
        
        # Process each route
        routes_created = 0
        for idx, route in enumerate(routes):
            if not isinstance(route, dict):
                logger.warning(f"Route {idx} is not a dictionary, skipping")
                continue
                
            # Generate route ID
            route_id = str(uuid.uuid4())
            
            # Extract route data with defaults
            path = route.get('path', '')
            title = route.get('title', route.get('name', 'Untitled'))
            icon = route.get('icon', 'fas fa-file')
            component = route.get('component', '')
            description = route.get('description', '')
            route_type = route.get('type', 'page')
            sort_order = route.get('order', idx)  # Use array index as default order
            
            # Store additional metadata as JSON
            metadata = {
                'original_index': idx,
                'requires_auth': route.get('requires_auth', True),
                'permissions': route.get('permissions', []),
                'layout': route.get('layout', 'default'),
                'meta': route.get('meta', {})
            }
            
            # Insert the route
            insert_query = """
                INSERT INTO app_routes (
                    route_id, app_id, app_version_id, path, title, icon, 
                    component, description, route_type, sort_order, metadata
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                )
            """
            
            await conn.execute(
                insert_query,
                route_id, app_id, app_version_id, path, title, icon,
                component, description, route_type, sort_order, json.dumps(metadata)
            )
            
            routes_created += 1
            logger.debug(f"Created route {path} for app {app_id}")
            
        logger.info(f"Successfully created {routes_created} routes for app {app_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting and storing routes for app {app_id}: {e}")
        return False

# Function processing methods
async def process_app_from_manifest(conn, manifest: UnifiedManifest, current_user: User) -> Dict[str, Any]:
    """Process an app from a manifest"""
    try:
        # No need to create a mock User object, we have the real one
        # Convert FULL manifest to YAML for storage (including agents, functions, etc.)
        full_manifest_dict = manifest.to_dict()
        manifest_yaml = yaml.dump(full_manifest_dict)
        logger.info(f"Stored full manifest with sections: {list(full_manifest_dict.keys())}")

        # Use the app_service import function with our transaction connection
        app_result = await import_app_from_manifest(
            manifest=manifest,
            current_user=current_user,
            connection=conn
        )
        
        # Get the latest app version ID (which will be the one we just created by import_app_from_manifest)
        latest_version_query = """
            SELECT app_version_id, manifest_yaml
            FROM app_versions 
            WHERE app_id = $1
            ORDER BY created_at DESC LIMIT 1
        """
        latest_version = await conn.fetch_one(latest_version_query, str(app_result.app_id))
        version_id = str(latest_version["app_version_id"]) if latest_version else None
        version_manifest_yaml = latest_version["manifest_yaml"] if latest_version else manifest_yaml

        # Extract and store routes from the manifest
        logger.info(f"Extracting routes from manifest for app {app_result.app_id}")
        routes_success = await _extract_and_store_routes_from_manifest(
            conn, 
            str(app_result.app_id), 
            str(version_id), 
            version_manifest_yaml
        )
        
        if not routes_success:
            logger.warning(f"Failed to extract routes for app {app_result.app_id}, but continuing...")
        
        # The version_id is the latest version
        latest_version_id = version_id
        
        # Return in a format consistent with other methods
        return {
            "id": str(app_result.app_id),
            "version_id": str(version_id),
            "latest_version_id": latest_version_id,
            "name": app_result.name,
            "version": app_result.version,
            "model_count": len(getattr(app_result, 'models', [])),
            "status": "created"
        }
    except HTTPException as http_e:
        logger.error(f"HTTP error processing app manifest: {http_e.detail}")
        return {
            "name": manifest.app.name,
            "status": "error",
            "error": http_e.detail
        }
    except Exception as e:
        logger.error(f"Error processing app manifest: {str(e)}")
        return {
            "name": manifest.app.name,
            "status": "error",
            "error": str(e)
        }

async def process_agent_from_manifest(conn, agent_manifest: AgentManifest, current_user: User, app_id: Optional[str] = None) -> Dict[str, Any]:
    """Process an agent from a manifest"""
    try:
        # Generate agent_slug from agent_slug field or derive from name
        agent_slug = getattr(agent_manifest, 'agent_slug', None)
        if not agent_slug:
            # Create agent_slug from name: lowercase, replace spaces with hyphens
            agent_slug = agent_manifest.name.lower().replace(' ', '-')
            # Remove any special characters that aren't URL-friendly
            agent_slug = ''.join(c for c in agent_slug if c.isalnum() or c == '-')
            logger.info(f"Generated agent_slug '{agent_slug}' from agent name")

        # Check if an agent with this slug already exists for this app
        check_query = """
            SELECT agent_id FROM agents 
            WHERE agent_slug = $1 AND app_id = $2
        """
        existing_agent = await conn.fetch_val(check_query, agent_slug, app_id)
        
        if existing_agent:
            # Return error for duplicate agent instead of trying to create
            return {
                "name": agent_manifest.name,
                "status": "error",
                "error": f"An agent with slug '{agent_slug}' already exists for this app",
                "agent_slug": agent_slug
            }
        
        # Only continue with creation if not found
        agent_id = await _create_new_agent(conn, agent_manifest, agent_slug, current_user, app_id)
        # Create initial version
        version_id = await _create_agent_version(conn, agent_id, agent_manifest, current_user)
        
        # If agent is custom type but no implementation provided, add a minimal implementation
        agent_type_id = getattr(agent_manifest, 'agent_type_id', 'custom')
        if agent_type_id and "custom" in agent_type_id.lower():
            if not hasattr(agent_manifest, 'implementation') and not hasattr(agent_manifest, 'implementation_path'):
                # Get language to use for minimal implementation
                language = getattr(agent_manifest, 'language', 'python')
                
                # Create minimal implementation and store it
                minimal_code = create_minimal_agent_code(language)
                implementation_query = """
                    INSERT INTO agent_implementations (
                        agent_id, implementation_code, language, version, is_active, created_by
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6
                    ) RETURNING implementation_id
                """
                
                implementation_id = await conn.fetch_val(
                    implementation_query,
                    str(agent_id),
                    minimal_code,
                    language,
                    agent_manifest.version,
                    True,  # is_active
                    current_user.id
                )
                
                logger.info(f"Created minimal implementation for custom agent: {agent_manifest.name}")

        # Activate the agent version if it has an implementation path
        if hasattr(agent_manifest, 'implementation_path') and agent_manifest.implementation_path:
            activate_query = """
                UPDATE agent_versions
                SET is_active = true, status = 'active'
                WHERE version_id = $1
            """
            await conn.execute(activate_query, str(version_id))
            logger.info(f"Activated agent version {version_id} for agent {agent_manifest.name}")

        return {
            "id": str(agent_id),
            "version_id": str(version_id),
            "name": agent_manifest.name,
            "status": "created"
        }
    except AgentCodeConflictError as e:
        logger.error(f"Agent code conflict: {str(e)}")
        return {
            "name": agent_manifest.name,
            "status": "error",
            "error": e.message,
            "agent_code": e.agent_code,
            "agent_type_id": e.agent_type_id
        }
    except Exception as e:
        logger.error(f"Error processing agent manifest: {str(e)}")
        return {
            "name": agent_manifest.name,
            "status": "error",
            "error": str(e)
        }

async def process_pipeline_from_manifest(conn, pipeline_manifest: PipelineManifest, current_user: User, app_id: Optional[str] = None) -> Dict[str, Any]:
    """Process a pipeline from a manifest with new structure support"""
    try:
        # Generate pipeline_slug from name
        pipeline_slug = pipeline_manifest.name.lower().replace(' ', '-')
        pipeline_slug = ''.join(c for c in pipeline_slug if c.isalnum() or c == '-')

        # NEW: Validate pipeline structure if it exists
        definition = None
        if hasattr(pipeline_manifest, 'structure') and pipeline_manifest.structure:
            structure = pipeline_manifest.structure
            
            # Validate step classes (following agent pattern - no individual file paths)
            for step in structure.get('steps', []):
                step_class = step.get('step_class')
                if not step_class:
                    raise ValueError(f"Step {step.get('id')} missing step_class")
            
            # Store structure in database
            definition = json.dumps(structure)
        elif hasattr(pipeline_manifest, 'implementation_path') and pipeline_manifest.implementation_path:
            # Legacy support - keep existing implementation_path logic
            definition = None
        else:
            raise ValueError("Pipeline manifest missing required 'structure' field")

        # Check if pipeline exists
        pipeline_query = "SELECT pipeline_id FROM pipelines WHERE pipeline_slug = $1 AND app_id = $2"
        existing_pipeline = await conn.fetch_one(pipeline_query, pipeline_slug, app_id)

        logger.info(f"[PIPELINE_VERSION_DEBUG] Checking for existing pipeline: slug={pipeline_slug}, app_id={app_id}")
        logger.info(f"[PIPELINE_VERSION_DEBUG] Existing pipeline result: {existing_pipeline}")

        if existing_pipeline:
            pipeline_id = existing_pipeline["pipeline_id"]
            logger.info(f"[PIPELINE_VERSION_DEBUG] Found existing pipeline with ID {pipeline_id}")

            # Update pipeline definition if we have new structure
            if definition:
                update_query = "UPDATE pipelines SET definition = $1 WHERE pipeline_id = $2"
                await conn.execute(update_query, definition, str(pipeline_id))
                logger.info(f"[PIPELINE_VERSION_DEBUG] Updated pipeline definition for {pipeline_id}")

            # Check if we already have a version for this pipeline from this deployment
            version_check_query = "SELECT COUNT(*) FROM pipeline_versions WHERE pipeline_id = $1 AND version = $2"
            existing_version_count = await conn.fetch_val(version_check_query, str(pipeline_id), getattr(pipeline_manifest, 'version', '0.0.1'))

            if existing_version_count > 0:
                logger.warning(f"[PIPELINE_VERSION_DEBUG] Version {getattr(pipeline_manifest, 'version', '0.0.1')} already exists for pipeline {pipeline_id}, skipping version creation")
                return {
                    "id": str(pipeline_id),
                    "version_id": "existing",
                    "name": pipeline_manifest.name,
                    "status": "version_exists"
                }

            # Create new version
            logger.info(f"[PIPELINE_VERSION_DEBUG] Creating new version for existing pipeline {pipeline_id}")
            version_id = await _create_pipeline_version(conn, pipeline_id, pipeline_manifest, current_user)
            
            return {
                "id": str(pipeline_id),
                "version_id": str(version_id),
                "name": pipeline_manifest.name,
                "status": "updated"
            }
        else:
            logger.info(f"[PIPELINE_VERSION_DEBUG] No existing pipeline found, creating new one")
            # Create new pipeline with structure
            pipeline_id = await _create_new_pipeline(conn, pipeline_manifest, pipeline_slug, current_user, app_id, definition)
            logger.info(f"[PIPELINE_VERSION_DEBUG] Created new pipeline with ID {pipeline_id}")
            # Create initial version
            logger.info(f"[PIPELINE_VERSION_DEBUG] Creating initial version for new pipeline {pipeline_id}")
            version_id = await _create_pipeline_version(conn, pipeline_id, pipeline_manifest, current_user)

            # Note: New graph-based pipelines use bundled step classes, not implementation_path
            # The bundling happens in the upload service via process_pipelines_in_bundle()
            logger.info(f"Pipeline uses graph-based execution with bundled step classes")
            
            return {
                "id": str(pipeline_id),
                "version_id": str(version_id),
                "name": pipeline_manifest.name,
                "status": "created"
            }
    except Exception as e:
        logger.error(f"Error processing pipeline manifest: {str(e)}")
        return {
            "name": pipeline_manifest.name,
            "status": "error",
            "error": str(e)
        }

async def process_workflow_from_manifest(conn, workflow_manifest: WorkflowManifest, current_user: User) -> Dict[str, Any]:
    """Process a workflow from a manifest"""
    try:
        # Check if workflow exists
        workflow_query = "SELECT workflow_id FROM workflows WHERE name = $1"
        existing_workflow = await conn.fetch_one(workflow_query, workflow_manifest.name)
            
        if existing_workflow:
            workflow_id = existing_workflow["workflow_id"]
            logger.info(f"Found existing workflow with ID {workflow_id}")
            # Create new version
            version_id = await _create_workflow_version(conn, workflow_id, workflow_manifest, current_user)
            
            return {
                "id": str(workflow_id),
                "version_id": str(version_id),
                "name": workflow_manifest.name,
                "status": "updated"
            }
        else:
            # Create new workflow
            workflow_id = await _create_new_workflow(conn, workflow_manifest, current_user)
            # Create initial version
            version_id = await _create_workflow_version(conn, workflow_id, workflow_manifest, current_user)
            
            return {
                "id": str(workflow_id),
                "version_id": str(version_id),
                "name": workflow_manifest.name,
                "status": "created"
            }
    except Exception as e:
        logger.error(f"Error processing workflow manifest: {str(e)}")
        return {
            "name": workflow_manifest.name,
            "status": "error",
            "error": str(e)
        }

async def process_function_from_manifest(conn, function_manifest: FunctionManifest, current_user: User, app_id: Optional[str] = None) -> Dict[str, Any]:
    """Process a function from a manifest"""
    try:
        # Check if function exists
        function_query = "SELECT function_id FROM functions WHERE name = $1"
        existing_function = await conn.fetch_one(function_query, function_manifest.name)
            
        if existing_function:
            function_id = existing_function["function_id"]
            logger.info(f"Found existing function with ID {function_id}")
            # Create new version
            version_id = await _create_function_version(conn, function_id, function_manifest, current_user)
            
            # Associate with app if provided
            associated = False
            if app_id:
                associated = await _associate_function_with_app(conn, str(function_id), str(app_id))
            
            result = {
                "id": str(function_id),
                "version_id": str(version_id),
                "name": function_manifest.name,
                "status": "updated"
            }
            # Add association info if app_id was provided
            if app_id:
                result["app_association"] = "created" if associated else "already_exists"
                result["app_id"] = str(app_id)
            
            return result
        else:
            # Create new function
            function_id = await _create_new_function(conn, function_manifest, current_user)
            # Create initial version
            version_id = await _create_function_version(conn, function_id, function_manifest, current_user)
            
            # Associate with app if provided
            if app_id:
                await _associate_function_with_app(conn, str(function_id), str(app_id))
            
            result = {
                "id": str(function_id),
                "version_id": str(version_id),
                "name": function_manifest.name,
                "status": "created"
            }
            # Add association info if app_id was provided
            if app_id:
                result["app_association"] = "created"
                result["app_id"] = str(app_id)
            
            return result
    except Exception as e:
        logger.error(f"Error processing function manifest: {str(e)}")
        return {
            "name": function_manifest.name,
            "status": "error",
            "error": str(e)
        }

async def process_unified_manifest(manifest: UnifiedManifest, current_user: User, db: DatabaseProvider) -> UnifiedManifestResult:
    """
    Process a unified manifest containing multiple entity types
        
    Args:
        manifest: A UnifiedManifest object with apps, agents, pipelines, etc.
        current_user: The authenticated user
    
    Returns:
        UnifiedManifestResult with results for each entity type that was processed
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[ROCKET] Starting process_unified_manifest for user {current_user.id}")
    
    # Debug the manifest structure
    logger.error(f"DEPLOYMENT DEBUG: Manifest structure - hasattr agents: {hasattr(manifest, 'agents')}")
    logger.error(f"DEPLOYMENT DEBUG: Manifest dir: {[attr for attr in dir(manifest) if not attr.startswith('_')]}")
    
    # Initialize the result model
    results = UnifiedManifestResult()
    app_id = None
    
    logger.info("[MEMO] Validating manifest versions...")
    # First validate all versions before processing
    is_valid, error_message = manifest.validate_all_versions()
    if not is_valid:
        logger.warning(f"Manifest validation failed: {error_message}")
        # Don't return early, attempt to process what we can
    
    logger.info("[ARROWS] Starting database operations...")
    # Use proper transaction for all database operations
    try:
        async with db.transaction() as conn:
            # Process app if present
            if manifest.app:
                logger.info(f"[PHONE] Processing app: {manifest.app.name}")
                
                # Check if user already has an app with this slug - use transaction connection
                check_query = "SELECT app_id FROM apps WHERE app_slug = $1 AND creator_user_id = $2"
                logger.info(f"[SEARCH] Checking for existing app with query: {check_query}")
                logger.info(f"[SEARCH] Parameters: app_slug='{manifest.app.app_slug}', user_id={current_user.id}")
                
                try:
                    import asyncio
                    existing_app = await asyncio.wait_for(
                        conn.fetch_one(check_query, manifest.app.app_slug, current_user.id), 
                        timeout=10.0
                    )
                    logger.info(f"[CHECK] Database query completed. Result: {existing_app}")
                except asyncio.TimeoutError:
                    logger.error("[X] Database query timed out after 10 seconds")
                    raise Exception("Database query timed out")
                except Exception as e:
                    logger.error(f"[X] Database query failed: {e}")
                    raise
                
                if existing_app:
                    # App already exists - create new version instead of reusing
                    logger.info(f"[PHONE] App with slug '{manifest.app.app_slug}' already exists, creating new deployment")
                    app_data = await process_app_from_manifest(conn, manifest, current_user)
                    
                    app_result = AppResult(
                        id=app_data.get("id", ""),
                        name=app_data.get("name", manifest.app.name),
                        status=app_data.get("status", "unknown"),
                        version_id=app_data.get("version_id"),
                        version=app_data.get("version"),
                        error=app_data.get("error"),
                        model_count=app_data.get("model_count", 0)
                    )
                    results.app.append(app_result)
                    # Set app_id for agent processing
                    if app_data.get("status") == "created" and app_data.get("id"):
                        app_id = app_data.get("id")
                else:
                    # Process the app since it doesn't exist
                    logger.info("[PHONE] Processing new app...")
                    app_data = await process_app_from_manifest(conn, manifest, current_user)
                    
                    app_result = AppResult(
                        id=app_data.get("id", ""),
                        name=app_data.get("name", manifest.app.name),
                        status=app_data.get("status", "unknown"),
                        version_id=app_data.get("version_id"),
                        version=app_data.get("version"),
                        error=app_data.get("error"),
                        model_count=app_data.get("model_count", 0)
                    )
                    results.app.append(app_result)
                    # Set app_id for agent processing
                    if app_data.get("status") == "created" and app_data.get("id"):
                        app_id = app_data.get("id")
            
            # Process agents if present
            logger.error(f"DEPLOYMENT DEBUG: manifest.agents = {getattr(manifest, 'agents', 'MISSING')}")
            logger.error(f"DEPLOYMENT DEBUG: manifest.agents type = {type(getattr(manifest, 'agents', None))}")
            if hasattr(manifest, 'agents') and manifest.agents:
                logger.error(f"DEPLOYMENT DEBUG: Found {len(manifest.agents)} agents to process")
                
                # Check if we have a valid app_id before processing agents
                if not app_id:
                    logger.error(f"DEPLOYMENT DEBUG: No valid app_id available, skipping agent processing")
                    for agent_manifest in manifest.agents:
                        agent_result = AgentResult(
                            id="",
                            name=agent_manifest.name,
                            status="error",
                            error="Cannot create agent: App creation failed or app_id is missing"
                        )
                        results.agents.append(agent_result)
                else:
                    logger.info(f"[ROBOT] Processing {len(manifest.agents)} agents with app_id: {app_id}...")
                    for agent_manifest in manifest.agents:
                        logger.info(f"[ROBOT] Processing agent: {agent_manifest.name}")
                        agent_data = await process_agent_from_manifest(conn, agent_manifest, current_user, app_id)
                        
                        agent_result = AgentResult(
                            id=agent_data.get("id", ""),
                            name=agent_data.get("name", agent_manifest.name),
                            status=agent_data.get("status", "unknown"),
                            version_id=agent_data.get("version_id"),
                            error=agent_data.get("error"),
                            implementation_type=getattr(agent_manifest, 'agent_type_id', 'unknown')
                        )
                        results.agents.append(agent_result)
            
            # Process pipelines if present
            if manifest.pipelines:
                logger.info(f"[WRENCH] Processing {len(manifest.pipelines)} pipelines...")
                for pipeline_manifest in manifest.pipelines:
                    logger.info(f"[WRENCH] Processing pipeline: {pipeline_manifest.name}")
                    pipeline_data = await process_pipeline_from_manifest(conn, pipeline_manifest, current_user, app_id)
                    
                    pipeline_result = PipelineResult(
                        id=pipeline_data.get("id", ""),
                        name=pipeline_data.get("name", pipeline_manifest.name),
                        status=pipeline_data.get("status", "unknown"),
                        version_id=pipeline_data.get("version_id"),
                        error=pipeline_data.get("error")
                    )
                    results.pipelines.append(pipeline_result)
            
            # Process workflows if present
            if manifest.workflows:
                logger.info(f"[ARROWS] Processing {len(manifest.workflows)} workflows...")
                for workflow_manifest in manifest.workflows:
                    logger.info(f"[ARROWS] Processing workflow: {workflow_manifest.name}")
                    workflow_data = await process_workflow_from_manifest(conn, workflow_manifest, current_user)
                    
                    workflow_result = WorkflowResult(
                        id=workflow_data.get("id", ""),
                        name=workflow_data.get("name", workflow_manifest.name),
                        status=workflow_data.get("status", "unknown"),
                        version_id=workflow_data.get("version_id"),
                        error=workflow_data.get("error")
                    )
                    results.workflows.append(workflow_result)
            
            # Process functions if present (functions will be processed separately below to avoid SQLite concurrency issues)
            if manifest.functions:
                logger.info(f"⚡ Functions will be processed separately: {len(manifest.functions)} functions...")
            
            logger.info("[CHECK] Completed processing manifest components in main transaction")
        
        # Process functions separately with individual transactions to avoid SQLite concurrency issues  
        if manifest.functions:
            logger.info(f"⚡ Processing {len(manifest.functions)} functions with separate transactions...")
            for function_manifest in manifest.functions:
                logger.info(f"⚡ Processing function: {function_manifest.name}")
                try:
                    async with db.transaction() as conn:
                        function_data = await process_function_from_manifest(conn, function_manifest, current_user, app_id)
                    
                    function_result = FunctionResult(
                        id=function_data.get("id", ""),
                        name=function_data.get("name", function_manifest.name),
                        status=function_data.get("status", "unknown"),
                        version_id=function_data.get("version_id"),
                        error=function_data.get("error"),
                        app_association=function_data.get("app_association"),
                        app_id=function_data.get("app_id")
                    )
                    results.functions.append(function_result)
                    
                except Exception as func_error:
                    logger.error(f"Error processing function {function_manifest.name}: {func_error}")
                    error_result = FunctionResult(
                        id="",
                        name=function_manifest.name,
                        status="error",
                        error=str(func_error)
                    )
                    results.functions.append(error_result)
    
    except Exception as e:
        logger.error(f"[X] Error in process_unified_manifest: {e}")
        raise
    
    return results


class InstallService:
    """Service for handling app/component installation operations"""
    
    def __init__(self, db: DatabaseProvider):
        self.db = db
    
    async def process_unified_manifest_service(self, manifest: UnifiedManifest, current_user: User) -> UnifiedManifestResult:
        """Service wrapper for process_unified_manifest function"""
        return await process_unified_manifest(manifest, current_user, self.db)

