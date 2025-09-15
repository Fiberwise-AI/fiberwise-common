"""
Service for handling component updates (apps, agents, pipelines, etc.)
"""
import logging
import json
import os
from uuid import UUID, uuid4
from typing import Dict, List, Any, Optional
import yaml

from fiberwise_common.entities import AgentManifest, PipelineManifest, WorkflowManifest, FunctionManifest

# Import the upload service which now handles agent/function creation  
# Temporary fix for schema dependencies
from typing import Dict as User
# from .app_upload_service import AppUploadService  # Still in web project
class AppUploadService:
    def __init__(self, db, storage_provider):
        self.db = db
        self.storage_provider = storage_provider
from fiberwise_common.services import get_storage_provider

from fiberwise_common import DatabaseProvider

logger = logging.getLogger(__name__)

class UpdateService:
    def __init__(self, db: DatabaseProvider):
        self.db = db
        self.upload_service = AppUploadService(db, get_storage_provider())

    async def update_agent(self, conn, agent_id: str, agent_manifest: AgentManifest, current_user: User, app_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Update an existing agent with a new version
        
        Args:
            conn: Database connection
            agent_id: ID of the agent to update
            agent_manifest: Agent manifest with new version details
            current_user: Current user performing the update
            app_id: Optional app ID the agent belongs to
        
        Returns:
            Update result
        """
        try:
            # Verify agent exists and get current details
            agent_query = """
                SELECT name, agent_type_id, app_id FROM agents
                WHERE agent_id = $1
            """
            agent_record = await conn.fetch_one(agent_query, agent_id)
            
            if not agent_record:
                return {
                    "id": agent_id,
                    "name": agent_manifest.name,
                    "status": "error",
                    "error": f"Agent {agent_id} not found"
                }
            
            # Use app_id from agent record if not explicitly provided
            if not app_id and agent_record["app_id"]:
                app_id = agent_record["app_id"]
            
            # Create new version - using the module import
            version_id = await self.install_service._create_agent_version(conn, agent_id, agent_manifest, current_user)
            
            # Check which implementation type this agent has
            implementation = None
            implementation_type = 'unknown'
            implementation_path = None
            language = getattr(agent_manifest, 'language', 'python')
            
            # Check for inline implementation (code directly in manifest)
            if hasattr(agent_manifest, 'implementation') and agent_manifest.implementation:
                implementation = agent_manifest.implementation
                implementation_type = 'inline'
            
            # Check for file-based implementation using implementation_path
            elif hasattr(agent_manifest, 'implementation_path') and agent_manifest.implementation_path:
                implementation_path = agent_manifest.implementation_path
                implementation_type = 'file'
                
                # Try to determine language from file extension if not explicitly set
                if not language and implementation_path:
                    ext = os.path.splitext(implementation_path)[1].lower()
                    if ext == '.py':
                        language = 'python'
                    elif ext == '.js':
                        language = 'javascript'
                    elif ext == '.rb':
                        language = 'ruby'
                    else:
                        language = 'unknown'
            
            # Handle implementation storage based on type
            if implementation_type != 'unknown':
                # Set has_custom_code flag to true
                await conn.execute(
                    "UPDATE agents SET has_custom_code = true WHERE agent_id = $1",
                    agent_id
                )
                
                # Deactivate all existing implementations
                await conn.execute(
                    "UPDATE agent_implementations SET is_active = false WHERE agent_id = $1",
                    agent_id
                )
                
                # For inline implementation, store it directly
                if implementation_type == 'inline':
                    # Store new implementation code in the agent_implementations table
                    implementation_query = """
                        INSERT INTO agent_implementations (
                            agent_id, implementation_code, implementation_path,
                            entrypoint_file, class_name, language, is_active, created_by
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8
                        ) RETURNING implementation_id
                    """
                    
                    # Get version from manifest
                    version = getattr(agent_manifest, 'version', '1.0.0')
                    
                    implementation_id = await conn.fetch_val(
                        implementation_query,
                        agent_id,
                        implementation,  # Use the processed implementation code for inline type
                        None,            # implementation_path
                        None,            # entrypoint_file
                        None,            # class_name
                        language,
                        True,            # is_active
                        current_user.id
                    )
                    
                    logger.info(f"Stored inline implementation for agent: {agent_manifest.name} with ID {implementation_id}")
                
                # For file and directory types, just store the path info
                elif implementation_type == 'file':
                    implementation_query = """
                        INSERT INTO agent_implementations (
                            agent_id, implementation_code, implementation_path,
                            entrypoint_file, class_name, language, is_active, created_by
                        ) VALUES (
                            $1, NULL, $2, $3, $4, $5, $6, $7
                        ) RETURNING implementation_id
                    """
                    
                    # Get entrypoint file and class name for directory implementations
                    entrypoint_file = getattr(agent_manifest, 'entrypoint_file', None)
                    class_name = getattr(agent_manifest, 'class_name', None)
                    
                    implementation_id = await conn.fetch_val(
                        implementation_query,
                        agent_id,
                        implementation_path,  # implementation_path
                        entrypoint_file,
                        class_name,
                        language,
                        True,  # is_active
                        current_user.id
                    )
                    
                    logger.info(f"Stored file implementation reference for agent: {agent_manifest.name} (language: {language})")
            
            # Mark the version as active
            activate_query = """
                UPDATE agent_versions
                SET is_active = true
                WHERE version_id = $1
            """
            await conn.execute(activate_query, str(version_id))
            
            return {
                "id": agent_id,
                "version_id": str(version_id),
                "name": agent_manifest.name,
                "status": "updated",
                "implementation_type": implementation_type,
                "message": "Agent updated successfully and activated"
            }
        except Exception as e:
            logger.error(f"Error updating agent: {str(e)}")
            return {
                "id": agent_id,
                "name": agent_manifest.name,
                "status": "error",
                "error": str(e)
            }

    # Add similar update functions for other component types
    async def update_pipeline(self, conn, pipeline_id: str, pipeline_manifest: PipelineManifest, current_user: User) -> Dict[str, Any]:
        """Update an existing pipeline with a new version"""
        try:
            # Verify pipeline exists
            pipeline_query = "SELECT name FROM pipelines WHERE pipeline_id = $1"
            pipeline_name = await conn.fetch_val(pipeline_query, pipeline_id)
            
            if not pipeline_name:
                return {
                    "id": pipeline_id,
                    "name": pipeline_manifest.name,
                    "status": "error",
                    "error": f"Pipeline {pipeline_id} not found"
                }
            
            # Create new version
            version_id = await self.install_service._create_pipeline_version(conn, pipeline_id, pipeline_manifest, current_user)

            # Update file_path on the main pipeline record
            file_path = getattr(pipeline_manifest, 'file_path', getattr(pipeline_manifest, 'implementation_path', None))
            if file_path:
                update_query = "UPDATE pipelines SET file_path = $1, updated_at = CURRENT_TIMESTAMP WHERE pipeline_id = $2"
                await conn.execute(update_query, file_path, pipeline_id)
            
            # Mark the version as active
            activate_query = """
                UPDATE pipeline_versions
                SET is_active = true
                WHERE version_id = $1
            """
            await conn.execute(activate_query, str(version_id))
            
            return {
                "id": pipeline_id,
                "version_id": str(version_id),
                "name": pipeline_manifest.name,
                "status": "updated",
                "message": "Pipeline updated successfully and activated"
            }
        except Exception as e:
            logger.error(f"Error updating pipeline: {str(e)}")
            return {
                "id": pipeline_id,
                "name": pipeline_manifest.name,
                "status": "error",
                "error": str(e)
            }

    async def update_workflow(self, conn, workflow_id: str, workflow_manifest: WorkflowManifest, current_user: User) -> Dict[str, Any]:
        """Update an existing workflow with a new version"""
        try:
            # Verify workflow exists
            workflow_query = "SELECT name FROM workflows WHERE workflow_id = $1"
            workflow_name = await conn.fetch_val(workflow_query, workflow_id)
            
            if not workflow_name:
                return {
                    "id": workflow_id,
                    "name": workflow_manifest.name,
                    "status": "error",
                    "error": f"Workflow {workflow_id} not found"
                }
            
            # Create new version
            version_id = await self.install_service._create_workflow_version(conn, workflow_id, workflow_manifest, current_user)
            
            # Mark the version as active
            activate_query = """
                UPDATE workflow_versions
                SET status = 'active'
                WHERE workflow_version_id = $1
            """
            await conn.execute(activate_query, str(version_id))
            
            return {
                "id": workflow_id,
                "version_id": str(version_id),
                "name": workflow_manifest.name,
                "status": "updated",
                "message": "Workflow updated successfully and activated"
            }
        except Exception as e:
            logger.error(f"Error updating workflow: {str(e)}")
            return {
                "id": workflow_id,
                "name": workflow_manifest.name,
                "status": "error",
                "error": str(e)
            }

    async def update_function(self, conn, function_id: str, function_manifest: FunctionManifest, current_user: User) -> Dict[str, Any]:
        """Update an existing function with a new version"""
        try:
            # Verify function exists
            function_query = "SELECT name FROM functions WHERE function_id = $1"
            function_name = await conn.fetch_val(function_query, function_id)
            
            if not function_name:
                return {
                    "id": function_id,
                    "name": function_manifest.name,
                    "status": "error",
                    "error": f"Function {function_id} not found"
                }
            
            # Create new version
            version_id = await self.install_service._create_function_version(conn, function_id, function_manifest, current_user)
            
            # Mark the version as active
            activate_query = """
                UPDATE function_versions
                SET status = 'active'
                WHERE function_version_id = $1
            """
            await conn.execute(activate_query, str(version_id))
            
            return {
                "id": function_id,
                "version_id": str(version_id),
                "name": function_manifest.name,
                "status": "updated",
                "message": "Function updated successfully and activated"
            }
        except Exception as e:
            logger.error(f"Error updating function: {str(e)}")
            return {
                "id": function_id,
                "name": function_manifest.name,
                "status": "error",
                "error": str(e)
            }

    # Check if version has changed
    async def has_version_changed(self, component_type: str, component_id: str, new_version: str) -> bool:
        """
        Check if the version being updated is different from the current version
        
        Args:
            component_type: Type of component (agent, pipeline, workflow, function)
            component_id: ID of the component
            new_version: The new version to compare against
            
        Returns:
            bool: True if version has changed, False otherwise
        """
        try:
            # Determine the correct table and column names based on component type
            if component_type == "agent":
                table = "agent_versions"
                id_column = "agent_id"
            elif component_type == "pipeline":
                table = "pipeline_versions"
                id_column = "pipeline_id"
            elif component_type == "workflow":
                table = "workflow_versions"
                id_column = "workflow_id"
            elif component_type == "function":
                table = "function_versions"
                id_column = "function_id"
            else:
                # For unrecognized types or app, always return True to force update
                return True
            
            # Query the latest version
            query = f"""
                SELECT version 
                FROM {table} 
                WHERE {id_column} = $1
                ORDER BY created_at DESC
                LIMIT 1
            """
            current_version = await self.db.fetch_val(query, component_id)
            
            # If no current version exists, or the version has changed, return True
            if current_version is None or current_version != new_version:
                return True
            
            # Version is the same, no update needed
            return False
        except Exception as e:
            logger.error(f"Error checking version change: {str(e)}")
            # On error, default to True to allow update
            return True

# Standalone functions for backward compatibility
async def update_agent(conn, agent_id: str, agent_manifest: AgentManifest, current_user: User, app_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Standalone function to update an agent.
    
    Args:
        conn: Database connection
        agent_id: ID of the agent to update
        agent_manifest: Agent manifest with new version details
        current_user: Current user performing the update
        app_id: Optional app ID the agent belongs to
    
    Returns:
        Update result
    """
    # Create a temporary service instance with a mock database provider
    # Since we're using a connection, we need to create a minimal wrapper
    class ConnectionWrapper:
        def __init__(self, connection):
            self.connection = connection
        
        async def fetch_val(self, query, *args):
            return await self.connection.fetch_val(query, *args)
        
        async def fetch_one(self, query, *args):
            return await self.connection.fetch_one(query, *args)
        
        async def fetch_all(self, query, *args):
            return await self.connection.fetch_all(query, *args)
        
        async def execute(self, query, *args):
            return await self.connection.execute(query, *args)
    
    db_wrapper = ConnectionWrapper(conn)
    service = UpdateService(db_wrapper)
    return await service.update_agent(conn, agent_id, agent_manifest, current_user, app_id)

async def update_pipeline(conn, pipeline_id: str, pipeline_manifest: PipelineManifest, current_user: User, app_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Standalone function to update a pipeline.
    
    Args:
        conn: Database connection
        pipeline_id: ID of the pipeline to update
        pipeline_manifest: Pipeline manifest with new version details
        current_user: Current user performing the update
        app_id: Optional app ID the pipeline belongs to
    
    Returns:
        Update result
    """
    class ConnectionWrapper:
        def __init__(self, connection):
            self.connection = connection
        
        async def fetch_val(self, query, *args):
            return await self.connection.fetch_val(query, *args)
        
        async def fetch_one(self, query, *args):
            return await self.connection.fetch_one(query, *args)
        
        async def fetch_all(self, query, *args):
            return await self.connection.fetch_all(query, *args)
        
        async def execute(self, query, *args):
            return await self.connection.execute(query, *args)
    
    db_wrapper = ConnectionWrapper(conn)
    service = UpdateService(db_wrapper)
    return await service.update_pipeline(conn, pipeline_id, pipeline_manifest, current_user, app_id)

async def update_workflow(conn, workflow_id: str, workflow_manifest: WorkflowManifest, current_user: User, app_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Standalone function to update a workflow.
    
    Args:
        conn: Database connection
        workflow_id: ID of the workflow to update
        workflow_manifest: Workflow manifest with new version details
        current_user: Current user performing the update
        app_id: Optional app ID the workflow belongs to
    
    Returns:
        Update result
    """
    class ConnectionWrapper:
        def __init__(self, connection):
            self.connection = connection
        
        async def fetch_val(self, query, *args):
            return await self.connection.fetch_val(query, *args)
        
        async def fetch_one(self, query, *args):
            return await self.connection.fetch_one(query, *args)
        
        async def fetch_all(self, query, *args):
            return await self.connection.fetch_all(query, *args)
        
        async def execute(self, query, *args):
            return await self.connection.execute(query, *args)
    
    db_wrapper = ConnectionWrapper(conn)
    service = UpdateService(db_wrapper)
    return await service.update_workflow(conn, workflow_id, workflow_manifest, current_user, app_id)

async def update_function(conn, function_id: str, function_manifest: FunctionManifest, current_user: User, app_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Standalone function to update a function.
    
    Args:
        conn: Database connection
        function_id: ID of the function to update
        function_manifest: Function manifest with new version details
        current_user: Current user performing the update
        app_id: Optional app ID the function belongs to
    
    Returns:
        Update result
    """
    class ConnectionWrapper:
        def __init__(self, connection):
            self.connection = connection
        
        async def fetch_val(self, query, *args):
            return await self.connection.fetch_val(query, *args)
        
        async def fetch_one(self, query, *args):
            return await self.connection.fetch_one(query, *args)
        
        async def fetch_all(self, query, *args):
            return await self.connection.fetch_all(query, *args)
        
        async def execute(self, query, *args):
            return await self.connection.execute(query, *args)
    
    db_wrapper = ConnectionWrapper(conn)
    service = UpdateService(db_wrapper)
    return await service.update_function(conn, function_id, function_manifest, current_user, app_id)

async def has_version_changed(component_type: str, component_id: str, new_version: str, db: DatabaseProvider) -> bool:
    """
    Standalone function to check if version has changed.
    
    Args:
        component_type: Type of component (agent, pipeline, workflow, function)
        component_id: ID of the component
        new_version: The new version to compare against
        db: Database provider
    
    Returns:
        bool: True if version has changed, False otherwise
    """
    service = UpdateService(db)
    return await service.has_version_changed(component_type, component_id, new_version)
