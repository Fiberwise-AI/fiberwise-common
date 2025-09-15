import json
import logging
import os
import shutil
import tempfile
from typing import Any
from fastapi import HTTPException, status
from uuid import UUID, uuid4

import yaml
from fiberwise_common.entities import UnifiedManifest, FunctionManifest

# User schema not available in common - will be passed as parameter when needed
from fiberwise_common.services import StorageProvider
from .agent_version_service import AgentVersionService

# Global settings instance - will be set by calling code
settings = None
from fiberwise_common import DatabaseProvider

# Import consolidated language detection utility
from ..utils.language_utils import detect_language_with_fallback

logger = logging.getLogger(__name__)

# Import consolidated code validation utility
from ..utils.code_validators import validate_code_snippet

def validate_and_process_code(code: str, language: str) -> tuple:
    """Basic code validation - returns (processed_code, warnings)"""
    return validate_code_snippet(code, language)

def validate_directory_implementation(directory: str) -> dict:
    """Validate directory structure for agent implementation"""
    try:
        if not os.path.exists(directory):
            return {'valid': False, 'error': 'Directory does not exist'}
        
        # Look for Python files
        python_files = [f for f in os.listdir(directory) if f.endswith('.py')]
        if python_files:
            return {'valid': True, 'language': 'python', 'files': python_files}
        
        # Look for JavaScript/TypeScript files
        js_files = [f for f in os.listdir(directory) if f.endswith(('.js', '.ts'))]
        if js_files:
            return {'valid': True, 'language': 'javascript', 'files': js_files}
        
        return {'valid': False, 'error': 'No implementation files found'}
    except Exception as e:
        return {'valid': False, 'error': f'Validation error: {str(e)}'}

# Importing consolidated agent template utility
from ..utils.agent_templates import create_minimal_agent_code

class AppUploadService:
    def __init__(self, db: DatabaseProvider, storage_provider: StorageProvider, settings_instance=None):
        self.db = db
        self.storage_provider = storage_provider
        self.agent_version_service = AgentVersionService(db)
        # Use provided settings or global settings
        if settings_instance:
            global settings
            settings = settings_instance
            logger.info(f"[DEBUG] AppUploadService received settings: ENTITY_BUNDLES_DIR={getattr(settings, 'ENTITY_BUNDLES_DIR', 'NOT_SET')}")
        else:
            logger.warning(f"[DEBUG] AppUploadService: No settings_instance provided, global settings is: {settings}")

    async def ensure_directory_exists(self, directory_path: str) -> str:
        """
        Ensure a directory exists using the storage provider
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            Path to the directory
        """
        file_info = await self.storage_provider.get_file_info(directory_path)
        
        if not file_info.get('exists', False):
            if await self.storage_provider.is_local():
                os.makedirs(directory_path, exist_ok=True)
            else:
                marker_file = os.path.join(directory_path, '.keep')
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_path = temp_file.name
                    pass
                try:
                    await self.storage_provider.upload_file(temp_path, marker_file)
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
        
        return directory_path

    async def get_app_bundle_path(self, app_id: UUID, app_version_id: UUID = None) -> str:
        """
        Get the path to an app bundle directory, ensuring it exists in the storage provider
        
        Args:
            app_id: The app ID (UUID)
            app_version_id: Optional app version ID (UUID)
            
        Returns:
            Path to the app bundle directory
        """
        # Use settings-based path instead of hardcoded project root
        app_dir = os.path.join(settings.APP_BUNDLES_DIR, str(app_id))
        if app_version_id:
            app_dir = os.path.join(app_dir, str(app_version_id))
        
        await self.ensure_directory_exists(app_dir)
        
        return app_dir

    async def get_entity_bundle_path(self, app_id: UUID, entity_type: str, entity_id: UUID = None, version_id: UUID = None) -> str:
        """
        Get the path to an entity bundle directory, ensuring it exists in the storage provider
        
        Args:
            app_id: The app ID (UUID)
            entity_type: Type of entity (agent, pipeline, workflow, function)
            entity_id: Optional entity ID (UUID)
            version_id: Optional version ID (UUID)
            
        Returns:
            Path to the entity bundle directory
        """
        valid_types = ["agent", "pipeline", "workflow", "function"]
        if entity_type not in valid_types:
            raise ValueError(f"Invalid entity type: {entity_type}. Must be one of: {', '.join(valid_types)}")
        
        # Use settings-based path instead of cwd
        if settings is None:
            logger.error(f"[DEBUG] get_entity_bundle_path: settings is None! Cannot create entity path for {entity_type}")
            raise ValueError("Settings not initialized - cannot determine entity bundle path")
        
        entity_dir = os.path.join(settings.ENTITY_BUNDLES_DIR, "apps", str(app_id), entity_type)
        logger.info(f"[DEBUG] get_entity_bundle_path: Creating entity dir: {entity_dir}")
        
        if entity_id:
            entity_dir = os.path.join(entity_dir, str(entity_id))
            if version_id:
                entity_dir = os.path.join(entity_dir, str(version_id))
        
        await self.ensure_directory_exists(entity_dir)
        
        return entity_dir

    async def create_agents_from_manifest(self, manifest: UnifiedManifest, app_id: UUID, user_id: int):
        """
        Create agent records from manifest if they don't already exist
        """
        if not manifest.agents:
            return
            
        logger.info(f"Creating {len(manifest.agents)} agents from manifest for app {app_id}")
        
        for agent_manifest in manifest.agents:
            agent_name = agent_manifest.name
            if not agent_name:
                continue
                
            # Get agent_slug from manifest or generate from name
            agent_slug = getattr(agent_manifest, 'agent_slug', None)
            if not agent_slug:
                agent_slug = agent_name.lower().replace(' ', '-')
                agent_slug = ''.join(c for c in agent_slug if c.isalnum() or c == '-')
                logger.info(f"Generated agent_slug '{agent_slug}' from agent name")
            
            # Check if agent already exists for this app (by name OR agent_slug)
            check_query = "SELECT agent_id FROM agents WHERE app_id = ? AND (name = ? OR agent_slug = ?)"
            existing_agent_id = await self.db.fetch_val(check_query, str(app_id), agent_name, agent_slug)
            
            # Create new agent with proper agent_type_id
            if not existing_agent_id:
                agent_id = str(uuid4())
            else:
                agent_id = existing_agent_id
                logger.info(f"Agent {agent_name} (slug: {agent_slug}) already exists for app {app_id}, will update...")
            
            agent_type_id = getattr(agent_manifest, 'agent_type_id', 'custom')
            
            # Verify agent_type_id exists in agent_types table
            type_check_query = "SELECT id FROM agent_types WHERE id = ?"
            type_exists = await self.db.fetch_val(type_check_query, agent_type_id)
            
            if not type_exists:
                logger.warning(f"Agent type '{agent_type_id}' not found, using 'custom'")
                agent_type_id = 'custom'
            
            # Use INSERT OR REPLACE to handle both insert and update cases
            upsert_query = """
                INSERT OR REPLACE INTO agents (
                    agent_id, name, description, agent_type_id, agent_slug,
                    agent_code, is_enabled, config, created_by, updated_at, app_id, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
            """
            
            agent_code = getattr(agent_manifest, 'agent_code', agent_name.lower().replace(' ', '-'))
            description = getattr(agent_manifest, 'description', f"Agent for {agent_name}")
            config = getattr(agent_manifest, 'config', {})
            is_enabled = getattr(agent_manifest, 'is_enabled', True)
            
            await self.db.execute(
                upsert_query,
                agent_id, agent_name, description, agent_type_id, agent_slug,
                agent_code, is_enabled, json.dumps(config), user_id, str(app_id), True
            )
            
            logger.info(f"Created/Updated agent {agent_name} with type {agent_type_id} for app {app_id}")
            
            logger.info(f"Created agent {agent_name} with type {agent_type_id} for app {app_id}")

    async def create_functions_from_manifest(self, manifest: UnifiedManifest, app_id: UUID, user_id: int):
        """
        Create function records from manifest if they don't already exist
        """
        if not manifest.functions:
            return
            
        logger.info(f"Creating {len(manifest.functions)} functions from manifest for app {app_id}")
        
        for function_manifest in manifest.functions:
            function_name = function_manifest.name
            if not function_name:
                continue
                
            # Check if function already exists for this app
            check_query = """
                SELECT f.function_id FROM functions f
                JOIN functions_app fa ON f.function_id = fa.function_id
                WHERE fa.app_id = ? AND f.name = ?
            """
            existing_function = await self.db.fetch_val(check_query, str(app_id), function_name)
            
            if existing_function:
                logger.info(f"Function {function_name} already exists for app {app_id}")
                continue
            
            # Create new function
            function_id = str(uuid4())
            
            # Insert into functions table
            function_insert_query = """
                INSERT INTO functions (
                    function_id, name, description, function_type, implementation, 
                    tags, input_schema, output_schema, is_async, 
                    created_by, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """
            
            description = getattr(function_manifest, 'description', f"Function {function_name}")
            function_type = getattr(function_manifest, 'function_type', 'custom')
            implementation = getattr(function_manifest, 'implementation', None)
            tags = getattr(function_manifest, 'tags', [])
            input_schema = getattr(function_manifest, 'input_schema', {})
            output_schema = getattr(function_manifest, 'output_schema', {})
            is_async = getattr(function_manifest, 'is_async', False)
            
            await self.db.execute(
                function_insert_query,
                function_id, function_name, description, function_type, implementation,
                json.dumps(tags), json.dumps(input_schema), 
                json.dumps(output_schema), is_async, user_id
            )
            
            # Link function to app
            link_query = """
                INSERT INTO functions_app (function_id, app_id, created_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """
            await self.db.execute(link_query, function_id, str(app_id))
            
            logger.info(f"Created function {function_name} for app {app_id}")

    async def create_pipelines_from_manifest(self, manifest: UnifiedManifest, app_id: UUID, user_id: int):
        """
        Create pipeline records from manifest if they don't already exist
        """
        if not manifest.pipelines:
            return
            
        logger.info(f"Creating {len(manifest.pipelines)} pipelines from manifest for app {app_id}")
        
        for pipeline_manifest in manifest.pipelines:
            pipeline_name = pipeline_manifest.name
            if not pipeline_name:
                continue
                
            # Check if pipeline already exists for this app
            check_query = """
                SELECT pipeline_id FROM pipelines
                WHERE app_id = ? AND name = ?
            """
            existing_pipeline = await self.db.fetch_val(check_query, str(app_id), pipeline_name)
            
            if existing_pipeline:
                logger.info(f"Pipeline {pipeline_name} already exists for app {app_id}")
                continue
            
            # Create new pipeline
            pipeline_id = str(uuid4())
            
            # Insert into pipelines table
            pipeline_insert_query = """
                INSERT INTO pipelines (
                    pipeline_id, name, description, slug, version, structure,
                    created_by, app_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """
            
            description = getattr(pipeline_manifest, 'description', f"Pipeline {pipeline_name}")
            slug = getattr(pipeline_manifest, 'slug', pipeline_name.lower().replace(' ', '-'))
            version = getattr(pipeline_manifest, 'version', '1.0.0')
            structure = getattr(pipeline_manifest, 'structure', {})
            
            await self.db.execute(
                pipeline_insert_query,
                pipeline_id, pipeline_name, description, slug, version,
                json.dumps(structure), user_id, str(app_id)
            )
            
            logger.info(f"Created pipeline {pipeline_name} for app {app_id}")

    async def process_app_bundle(self, file_path: str, app_id: UUID, app_version_id: UUID, user_id: int, final_status: str = "published") -> dict:
        """
        Process an uploaded app bundle zip file:
        - Extract the zip file
        - Verify required directories exist
        - Move to the appropriate location in app_bundles directory
        
        Args:
            file_path: Path to the uploaded zip file
            app_id: UUID of the app
            app_version_id: UUID of the app version
            
        Returns:
            dict: Information about the processed bundle
        """
        # VALIDATE INPUTS UPFRONT - FAIL FAST
        if not user_id or not isinstance(user_id, int) or user_id <= 0:
            raise ValueError(f"Invalid user_id: {user_id}. Must be a positive integer.")
        
        if not app_id:
            raise ValueError("app_id is required")
            
        if not app_version_id:
            raise ValueError("app_version_id is required")
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Bundle file not found: {file_path}")
        
        logger.info(f"Processing app bundle for user_id={user_id}, app_id={app_id}")
        
        try:
            extract_dir = await self.get_app_bundle_path(app_id, app_version_id)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                await self.storage_provider.extract_archive(file_path, temp_dir)

                # Check for dist folder first (preferred structure)
                dist_path = os.path.join(temp_dir, "dist")
                source_content_path = dist_path
                use_dist_folder = True
                
                if not os.path.exists(dist_path) or not os.path.isdir(dist_path):
                    # No dist folder found, check if there are files in the root
                    temp_files = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]
                    
                    if not temp_files:
                        logger.error(f"No 'dist' folder or files found in the uploaded zip for app_version_id {app_version_id}")
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="The uploaded zip file must contain a 'dist' folder or files in the root"
                        )
                    
                    # Use root directory as source since there's no dist folder
                    source_content_path = temp_dir
                    use_dist_folder = False
                    logger.info(f"No 'dist' folder found for app_version_id {app_version_id}, using root directory files")

                # Load the current manifest from database (may have HTML encoding issues)
                query = "SELECT manifest_yaml FROM app_versions WHERE app_version_id = ?"
                manifest_yaml = await self.db.fetch_val(query, str(app_version_id))
                logger.info(f"DEBUG: Read from app_versions table - manifest found: {manifest_yaml is not None}")
                
                db_manifest = None
                if manifest_yaml:
                    try:
                        # Try to parse database manifest, but don't fail if it's corrupted
                        db_manifest_dict = yaml.safe_load(manifest_yaml)
                        if db_manifest_dict:
                            db_manifest = UnifiedManifest.model_validate(db_manifest_dict)
                            logger.info(f"DEBUG: Database manifest loaded successfully")
                    except Exception as e:
                        logger.warning(f"Database manifest corrupted/invalid (will use bundle): {e}")
                        db_manifest = None
                
                # Extract paths from database manifest for directory processing
                agent_paths = []
                function_paths = []
                if db_manifest:
                    for agent in db_manifest.agents:
                        if agent.implementation_path:
                            agent_paths.append(agent.implementation_path)
                    
                    for func in db_manifest.functions:
                        if func.implementation_path:
                            function_paths.append(func.implementation_path)
                
                # Determine directories to process based on whether we have a dist folder
                if use_dist_folder:
                    directories_to_process = ["dist"] + agent_paths + function_paths
                else:
                    # For root-level files, we process the entire root content as "dist"
                    directories_to_process = ["."] + agent_paths + function_paths
                
                processed_dirs = []
                
                for dir_path in directories_to_process:
                    if dir_path == ".":
                        # Special case: copy all root files to the dist folder in target
                        source_path = temp_dir
                        target_path = os.path.join(extract_dir, "dist")
                    elif dir_path == "dist" and use_dist_folder:
                        # Copy dist folder contents to bundle root (not dist subfolder)
                        # This prevents double dist/ structure when router adds /dist/ to path
                        # Result: activation-chat/dist/index.js -> bundle_root/index.js (not bundle_root/dist/index.js)
                        source_path = os.path.join(temp_dir, dir_path)
                        target_path = extract_dir  # Copy directly to bundle root
                    else:
                        source_path = os.path.join(temp_dir, dir_path)
                        target_path = os.path.join(extract_dir, dir_path)
                    
                    if os.path.exists(source_path):
                        if await self.storage_provider.file_exists(target_path):
                            await self.storage_provider.delete_file(target_path)
                        
                        parent_dir = os.path.dirname(target_path)
                        await self.ensure_directory_exists(parent_dir)
                        
                        # Handle special cases for copying directory contents
                        if (dir_path == "." and not use_dist_folder) or (dir_path == "dist" and use_dist_folder):
                            # Copy directory contents to target (not the directory itself)
                            # This handles both root files -> dist and dist contents -> bundle root
                            if await self.storage_provider.is_local():
                                os.makedirs(target_path, exist_ok=True)
                                for item in os.listdir(source_path):
                                    src_item = os.path.join(source_path, item)
                                    dst_item = os.path.join(target_path, item)
                                    if os.path.isfile(src_item):
                                        shutil.copy2(src_item, dst_item)
                                    elif os.path.isdir(src_item):
                                        shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                            else:
                                # Cloud storage - upload each file individually
                                for root, dirs, files in os.walk(source_path):
                                    for file in files:
                                        src_file = os.path.join(root, file)
                                        rel_path = os.path.relpath(src_file, source_path)
                                        dest_file = os.path.join(target_path, rel_path)
                                        await self.storage_provider.upload_file(src_file, dest_file)
                        elif os.path.isdir(source_path):
                            if await self.storage_provider.is_local():
                                shutil.copytree(source_path, target_path)
                            else:
                                for root, dirs, files in os.walk(source_path):
                                    for file in files:
                                        src_file = os.path.join(root, file)
                                        rel_path = os.path.relpath(src_file, source_path)
                                        dest_file = os.path.join(target_path, rel_path)
                                        await self.storage_provider.upload_file(src_file, dest_file)
                        else:
                            if await self.storage_provider.is_local():
                                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                                shutil.copy2(source_path, target_path)
                            else:
                                await self.storage_provider.upload_file(source_path, target_path)
                        
                        processed_dirs.append(dir_path)
                        logger.info(f"Processed '{dir_path}' for app_version_id {app_version_id}")
                    else:
                        if dir_path in agent_paths:
                            logger.warning(f"Agent implementation path '{dir_path}' not found in the uploaded zip")

                # Check for manifest in the bundle, but use database manifest as primary source
                # Debug: List what's actually in the temp directory
                try:
                    if os.path.exists(temp_dir):
                        logger.info(f"DEBUG: temp_dir contents: {os.listdir(temp_dir)}")
                    else:
                        logger.error(f"DEBUG: temp_dir does not exist: {temp_dir}")
                    logger.info(f"DEBUG: source_content_path: {source_content_path}")
                    if source_content_path != temp_dir and os.path.exists(source_content_path):
                        logger.info(f"DEBUG: source_content_path contents: {os.listdir(source_content_path)}")
                except Exception as e:
                    logger.error(f"DEBUG: Error accessing temp directories: {e}")
                
                # Try multiple possible locations for the manifest file
                import sys
                possible_manifest_paths = [
                    os.path.normpath(os.path.join(temp_dir, "app_manifest.yaml")),  # Root of extracted bundle
                    os.path.normpath(os.path.join(source_content_path, "app_manifest.yaml")),
                    os.path.normpath(os.path.join(temp_dir, "app_manifest.yml")),
                    os.path.normpath(os.path.join(source_content_path, "app_manifest.yml"))
                ]
                
                # Remove duplicates while preserving order
                seen = set()
                unique_paths = []
                for path in possible_manifest_paths:
                    if path not in seen:
                        seen.add(path)
                        unique_paths.append(path)
                
                # Load bundle manifest using Pydantic validation
                bundle_manifest = None
                manifest_path = None
                
                for path in unique_paths:
                    logger.info(f"DEBUG: Checking for manifest at: {path}")
                    try:
                        if os.path.exists(path):
                            manifest_path = path
                            logger.info(f"DEBUG: Found manifest at: {path}")
                            break
                    except Exception as e:
                        logger.error(f"DEBUG: Error checking path {path}: {e}")
                
                if manifest_path:
                    try:
                        # Read the raw YAML content to preserve original formatting
                        with open(manifest_path, 'r', encoding='utf-8') as f:
                            raw_manifest_yaml = f.read()
                        
                        # Also load and validate the YAML for processing
                        with open(manifest_path, 'r', encoding='utf-8') as f:
                            bundle_manifest_dict = yaml.safe_load(f)
                            if bundle_manifest_dict:
                                # Validate using Pydantic
                                bundle_manifest = UnifiedManifest.model_validate(bundle_manifest_dict)
                                logger.info(f"DEBUG: Bundle manifest loaded and validated successfully")
                                logger.info(f"DEBUG: Bundle contains - agents: {len(bundle_manifest.agents)}, functions: {len(bundle_manifest.functions)}, pipelines: {len(bundle_manifest.pipelines)}")
                                
                                # Update app_versions with raw YAML to preserve original formatting
                                update_manifest_query = """
                                    UPDATE app_versions 
                                    SET manifest_yaml = $1 
                                    WHERE app_version_id = $2
                                """
                                await self.db.execute(update_manifest_query, raw_manifest_yaml, str(app_version_id))
                                logger.info(f"DEBUG: Updated app_version {app_version_id} with raw YAML manifest to preserve formatting")
                            else:
                                logger.warning(f"Bundle manifest is empty: {manifest_path}")
                    except Exception as e:
                        logger.error(f"Error loading/validating bundle manifest: {e}")
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid manifest file: {e}"
                        )
                else:
                    logger.warning(f"No manifest file found in bundle. Tried: {unique_paths}")
                
                # Use bundle manifest as source of truth (it should contain the complete, valid manifest)
                if not bundle_manifest:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No valid manifest found in bundle"
                    )
                
                # The bundle manifest is now our validated, clean manifest
                manifest = bundle_manifest
                logger.info(f"DEBUG: Using bundle manifest as source of truth")
                logger.info(f"DEBUG: Final manifest contains - agents: {len(manifest.agents)}, functions: {len(manifest.functions)}, pipelines: {len(manifest.pipelines)}")
                
                logger.info(f"App bundle processed successfully for app_version_id {app_version_id}, dirs: {', '.join(processed_dirs)}")
                
                # Manifest is already validated and ready to use - no normalization needed with Pydantic
                unified_manifest = manifest
                
                logger.info(f"DEBUG: Unified manifest functions: {len(unified_manifest.functions) if unified_manifest.functions else 0}")
                if unified_manifest.functions:
                    logger.info(f"DEBUG: Function names: {[f.name for f in unified_manifest.functions]}")
                
                # First, create agents from manifest if they don't exist
                await self.create_agents_from_manifest(unified_manifest, app_id, user_id)
                
                # Also create functions from manifest if they don't exist
                await self.create_functions_from_manifest(unified_manifest, app_id, user_id)
                
                # Also create pipelines from manifest if they don't exist
                await self.create_pipelines_from_manifest(unified_manifest, app_id, user_id)
                
                agent_results = await self.process_agents_in_bundle(unified_manifest, app_id, app_version_id, user_id, bundle_source_dir=temp_dir)
                
                function_results = await self.process_functions_in_bundle(unified_manifest, app_id, app_version_id, user_id, bundle_source_dir=temp_dir)
                
                pipeline_results = await self.process_pipelines_in_bundle(unified_manifest, app_id, app_version_id, user_id, bundle_source_dir=temp_dir)

            logger.info(f"Starting entry point processing for app {app_id}")
            try:
                # Update the app's entry_point_url if it's not set and we have an entrypoint in the manifest
                entry_point = None
                
                # Get entry point from validated manifest
                entry_point = unified_manifest.app.entryPoint if unified_manifest.app.entryPoint else None
                    
                logger.info(f"DEBUG: Looking for entry point in manifest. Found: {entry_point}")
                logger.info(f"DEBUG: App entry point: {unified_manifest.app.entryPoint}")
                
                if not entry_point:
                    # Try alternative locations for the entry point from routes
                    routes = getattr(unified_manifest.app, 'routes', [])
                    if routes and len(routes) > 0:
                        first_route = routes[0]
                        if isinstance(first_route, dict):
                            entry_point = first_route.get('component') or first_route.get('entrypoint')
                            logger.info(f"DEBUG: Found entry point from routes: {entry_point}")
                    
                if not entry_point:
                    # Default to index.js if not found
                    entry_point = "index.js"
                    logger.info(f"DEBUG: Using default entry point: {entry_point}")
                
                if entry_point:
                    # Check if the app's entry_point_url is null
                    check_query = "SELECT entry_point_url FROM apps WHERE app_id = ?"
                    current_entry_point_url = await self.db.fetch_val(check_query, str(app_id))
                    logger.info(f"DEBUG: Current entry_point_url in DB: {current_entry_point_url}")
                    
                    if not current_entry_point_url:
                        # Update the app record with the entry point URL
                        update_query = """
                            UPDATE apps 
                            SET entry_point_url = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE app_id = ?
                        """
                        await self.db.execute(update_query, entry_point, str(app_id))
                        logger.info(f"Updated app {app_id} with entry_point_url: {entry_point}")
                    
                    # Also update the app version's entry_point_url
                    version_update_query = """
                        UPDATE app_versions 
                        SET entry_point_url = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE app_version_id = ?
                    """
                    await self.db.execute(version_update_query, entry_point, final_status, str(app_version_id))
                    logger.info(f"Updated app version {app_version_id} with entry_point_url: {entry_point} and status: {final_status}")
                    
            except Exception as e:
                logger.error(f"Error updating entry point: {str(e)}", exc_info=True)
                # Don't raise the exception so bundle processing can complete
            
            logger.info(f"Completed entry point processing for app {app_id}")

            return {
                "app_id": str(app_id),
                "app_version_id": str(app_version_id),
                "dist_path": os.path.join(extract_dir, "dist"),
                "processed_dirs": processed_dirs,
                "agents": agent_results,
                "functions": function_results,
                "pipelines": pipeline_results,
                "success": True
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing app bundle: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process app bundle: {str(e)}"
            )

    async def process_agents_in_bundle(self, manifest: UnifiedManifest, app_id: UUID, app_version_id: UUID, user_id: int, bundle_source_dir: str) -> list:
        """
        Process agent implementation files in an extracted app bundle
        
        Args:
            manifest: The app manifest
            app_id: UUID of the app
            app_version_id: UUID of the app version
            user: The user processing the bundle
            bundle_source_dir: The path to the temporary directory where the bundle was extracted
            
        Returns:
            list: Results of processing each agent
        """
        results = []
        try:
            agents = manifest.agents
            if not agents:
                logger.info(f"No agents found in manifest for app {app_id}")
                return results
        
            for agent in agents:
                agent_name = agent.name
                if not agent_name:
                    continue
                    
                impl_file = agent.implementation_path
                agent_type = getattr(agent, 'agent_type_id', getattr(agent, 'type', 'custom'))
                
                # Debug logging for agent processing
                logger.info(f"Processing agent '{agent_name}':")
                logger.info(f"  - Raw agent_type_id from getattr: {getattr(agent, 'agent_type_id', None)}")
                logger.info(f"  - Raw type from getattr: {getattr(agent, 'type', None)}")
                logger.info(f"  - Final resolved agent_type: {agent_type}")
                logger.info(f"  - implementation_path: {impl_file}")
                logger.info(f"  - Available agent attributes: {[attr for attr in dir(agent) if not attr.startswith('_')]}")
                
                # Get all manifest attributes for debugging
                agent_attrs = {}
                for attr in ['agent_type_id', 'type', 'implementation_path', 'implementation', 'file_path', 'path', 'code']:
                    value = getattr(agent, attr, '<not found>')
                    agent_attrs[attr] = value
                logger.info(f"  - Agent attributes: {agent_attrs}")
                
                # LLM agents don't need implementation files, others do
                if not impl_file and agent_type != 'llm':
                    # Try to auto-detect implementation path from common patterns
                    potential_paths = [
                        f"agents/{agent_name}.py",
                        f"agents/{agent_name.lower()}.py", 
                        f"{agent_name}.py",
                        f"{agent_name.lower()}.py",
                        f"src/agents/{agent_name}.py",
                        f"src/{agent_name}.py"
                    ]
                    
                    logger.info(f"Attempting to auto-detect implementation path for agent '{agent_name}'")
                    logger.info(f"Looking for files: {potential_paths}")
                    
                    # Check if any of these files exist in the bundle
                    detected_path = None
                    bundle_dir = await self.get_app_bundle_path(app_id, app_version_id)
                    
                    for potential_path in potential_paths:
                        full_path = os.path.join(bundle_dir, potential_path)
                        if os.path.exists(full_path):
                            detected_path = potential_path
                            logger.info(f"Auto-detected implementation path: {detected_path}")
                            break
                    
                    if detected_path:
                        impl_file = detected_path
                        logger.info(f"Using auto-detected implementation path: {impl_file}")
                    else:
                        logger.warning(f"Agent {agent_name} is type '{agent_type}' but has no implementation_path and auto-detection failed, skipping")
                        logger.info(f"Bundle directory contents: {os.listdir(bundle_dir) if os.path.exists(bundle_dir) else 'Directory not found'}")
                        continue
                else:
                    logger.info(f"Agent {agent_name} already has implementation_path: {impl_file}")
                    
                if not impl_file and agent_type != 'llm':
                    logger.warning(f"Agent {agent_name} is type '{agent_type}' but has no implementation_path, skipping")
                    continue
                    
                logger.info(f"Found agent {agent_name} with implementation path: {impl_file}")
                
                agent_query = "SELECT agent_id FROM agents WHERE app_id = ? AND name = ?"
                agent_id = await self.db.fetch_val(agent_query, str(app_id), agent_name)
                
                if not agent_id:
                    logger.warning(f"Agent {agent_name} not found in database for app {app_id}")
                    continue
                
                verification_query = "SELECT COUNT(*) FROM agents WHERE agent_id = ? AND app_id = ?"
                agent_belongs_to_app = await self.db.fetch_val(verification_query, str(agent_id), str(app_id))
                
                if not agent_belongs_to_app:
                    logger.error(f"Agent {agent_id} does not belong to app {app_id}")
                    continue
                
                try:
                    agent_version = getattr(agent, 'version', '1.0.0')
                    agent_description = getattr(agent, 'description', f"Version created during app bundle processing")
                    
                    version_id = await self.agent_version_service.create_agent_version(
                        agent_id=agent_id,
                        version=agent_version,
                        description=agent_description,
                        manifest=agent,
                        created_by=user_id,
                        file_path=impl_file
                    )
                    
                    logger.info(f"Created new agent version {agent_version} with ID {version_id} for agent {agent_name}")
                except Exception as ve:
                    logger.error(f"Failed to create agent version: {str(ve)}")
                    continue
                
                # Process implementation file only if agent has one (non-LLM agents)
                if impl_file:
                    # Look for agent files in the app bundle directory, not the temp directory
                    app_bundle_dir = await self.get_app_bundle_path(app_id, app_version_id)
                    file_path_in_app_bundle = os.path.join(app_bundle_dir, impl_file)
                    
                    if not await self.storage_provider.file_exists(file_path_in_app_bundle):
                        logger.warning(f"Implementation file not found in app bundle: {file_path_in_app_bundle}")
                        # Also try looking in the temp directory as fallback
                        temp_file_path = os.path.join(bundle_source_dir, impl_file)
                        if await self.storage_provider.file_exists(temp_file_path):
                            file_path_in_app_bundle = temp_file_path
                            logger.info(f"Found implementation file in temp directory: {temp_file_path}")
                        else:
                            logger.warning(f"Implementation file not found in temp directory either: {temp_file_path}")
                            continue
                    
                    result = await self.process_agent_implementation_file(
                        app_id, 
                        app_version_id, 
                        agent_id, 
                        agent, 
                        file_path_in_app_bundle, 
                        user_id,
                        version_id
                    )
                else:
                    # For LLM agents without implementation files
                    result = {
                        "success": True,
                        "agent_id": str(agent_id),
                        "agent_type": agent_type,
                        "implementation_type": "llm",
                        "message": "LLM agent processed without implementation file"
                    }
                
                await self.agent_version_service.activate_agent_version(version_id)
                
                result["version_id"] = str(version_id)
                result["version"] = agent_version
                
                results.append(result)
                logger.info(f"Processed implementation file for agent {agent_name}")
                
        except Exception as e:
            logger.error(f"Error processing agents in bundle: {str(e)}", exc_info=True)
        
        return results

    async def process_agent_implementation_file(self, app_id: UUID, app_version_id: UUID, agent_id: UUID, agent, file_path: str, 
                                              user_id: int, version_id: UUID) -> dict:
        """
        Process an agent implementation file from an app bundle:
        - Find the file in the app bundle
        - COPY it to the entity bundle directory (not just reference)
        - Store the entity bundle path in the agent_implementations table
        
        Args:
            app_id: UUID of the app
            app_version_id: UUID of the app version
            agent_id: UUID of the agent
            file_path: Full path to the implementation file in the app bundle
            user_id: ID of the user creating the implementation
            version_id: Optional UUID of the agent version
            
        Returns:
            dict: Information about the processed implementation
        """
        try:
            if not await self.storage_provider.file_exists(file_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Implementation file/directory not found: {file_path}"
                )
            
            implementation_code = getattr(agent, 'implementation', None)
            implementation_type = 'unknown'
            
            if implementation_code:
                implementation_type = 'inline'
            elif await self.storage_provider.file_exists(file_path):
                is_directory = os.path.isdir(file_path) if settings.STORAGE_PROVIDER.lower() == 'local' else False
                if not is_directory and settings.STORAGE_PROVIDER.lower() != 'local':
                    file_info = await self.storage_provider.get_file_info(file_path)
                    is_directory = file_info.get('is_directory', False)
                    
                implementation_type = 'directory' if is_directory else 'file'
            
            language = getattr(agent, 'language', 'python')
            
            if implementation_type == 'file':
                detected_language = detect_language_with_fallback(file_path, 'python')
                if detected_language:
                    language = detected_language
            
            entrypoint_file = getattr(agent, 'entrypoint_file', None)
            class_name = getattr(agent, 'class_name', None)
            
            if not entrypoint_file:
                if implementation_type == 'file':
                    entrypoint_file = os.path.basename(file_path)
                elif implementation_type == 'directory':
                    if language == 'python':
                        entrypoint_file = '__init__.py'
                    elif language == 'javascript' or language == 'typescript':
                        entrypoint_file = 'index.js'
                    else:
                        entrypoint_file = 'main.py'
                elif implementation_type == 'inline':
                    if language == 'python':
                        entrypoint_file = 'agent.py'
                    elif language == 'javascript' or language == 'typescript':
                        entrypoint_file = 'agent.js'
                    else:
                        entrypoint_file = f'agent.{language}'
            
            agent_entity_dir = await self.get_entity_bundle_path(app_id, 'agent', agent_id, version_id)
            
            is_directory = os.path.isdir(file_path) if settings.STORAGE_PROVIDER.lower() == 'local' else False
            if not is_directory and settings.STORAGE_PROVIDER.lower() != 'local':
                file_info = await self.storage_provider.get_file_info(file_path)
                is_directory = file_info.get('is_directory', False)
            
            implementation_type = 'directory' if is_directory else 'file'
            
            language = 'python'
            if implementation_type == 'file':
                _, ext = os.path.splitext(file_path)
                language = detect_language_with_fallback(file_path, 'python')
            else:
                init_file = os.path.join(file_path, '__init__.py')
                if await self.storage_provider.file_exists(init_file):
                    language = 'python'
            
            if await self.storage_provider.file_exists(agent_entity_dir):
                await self.storage_provider.delete_file(agent_entity_dir)
            await self.ensure_directory_exists(agent_entity_dir)
            
            if implementation_type == 'directory':
                if await self.storage_provider.is_local():
                    if os.path.exists(file_path):
                        shutil.copytree(file_path, agent_entity_dir, dirs_exist_ok=True)
                else:
                    source_files = await self.storage_provider.list_files(file_path, recursive=True)
                    for src_file in source_files:
                        src_path = os.path.join(file_path, src_file)
                        dest_path = os.path.join(agent_entity_dir, src_file)
                        await self.storage_provider.copy_file(src_path, dest_path)
            else:
                logger.info(f"[DEBUG AGENT COPY] About to copy agent file FROM: {file_path}")
                logger.info(f"[DEBUG AGENT COPY] About to copy agent file TO: {agent_entity_dir}")
                logger.info(f"[DEBUG AGENT COPY] Source file exists: {os.path.exists(file_path)}")
                
                if await self.storage_provider.is_local():
                    dest_file = os.path.join(agent_entity_dir, os.path.basename(file_path))
                    logger.info(f"[DEBUG AGENT COPY] Local copy: {file_path} -> {dest_file}")
                    shutil.copy2(file_path, dest_file)
                    logger.info(f"[DEBUG AGENT COPY] Copy completed, dest file exists: {os.path.exists(dest_file)}")
                else:
                    dest_file = os.path.join(agent_entity_dir, os.path.basename(file_path))
                    logger.info(f"[DEBUG AGENT COPY] Cloud copy: {file_path} -> {dest_file}")
                    await self.storage_provider.copy_file(file_path, dest_file)
                    logger.info(f"[DEBUG AGENT COPY] Cloud copy completed")
            
            rel_entity_path = os.path.relpath(agent_entity_dir, settings.ENTITY_BUNDLES_DIR)
            logger.info(f"Copied agent code from app bundle to entity bundle: {rel_entity_path}")
            logger.info(f"🗂️ Full agent entity bundle path: {agent_entity_dir}")
            
            validation_info = {
                "is_valid": True,
                "has_run_agent": True,
                "warnings": []
            }
            
            if implementation_type == 'directory' and settings.STORAGE_PROVIDER.lower() == 'local':
                validation_info = validate_directory_implementation(agent_entity_dir)
            
            entrypoint_file = getattr(agent, 'entrypoint_file', None)
            class_name = getattr(agent, 'class_name', None)
            
            if not entrypoint_file and implementation_type == 'file':
                entrypoint_file = os.path.basename(file_path)
            
            # File copied successfully - no database insertion needed in upload service
            implementation_id = "file_copied"
            
            return {
                "success": True,
                "implementation_id": str(implementation_id),
                "agent_id": str(agent_id),
                "entity_path": rel_entity_path,
                "language": language,
                "implementation_type": implementation_type,
                "entrypoint_file": entrypoint_file,
                "class_name": class_name,
                "validation": {
                    "has_run_agent": validation_info.get("has_run_agent", False),
                    "warnings": validation_info.get("warnings", [])
                }
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing agent implementation file: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process agent implementation file: {str(e)}"
            )

    async def process_function_implementation_file(self, app_id: UUID, app_version_id: UUID, function_id: UUID, function, file_path: str, 
                                                  user_id: int, version_id: UUID) -> dict:
        """
        Process a function implementation file from an app bundle:
        - Find the file in the app bundle
        - COPY it to the entity bundle directory (not just reference)
        - Store the entity bundle path in the function_implementations table
        
        Args:
            app_id: UUID of the app
            app_version_id: UUID of the app version
            function_id: UUID of the function
            function: Function manifest object
            file_path: Full path to the implementation file in the app bundle
            user_id: ID of the user creating the implementation
            version_id: Optional UUID of the function version
            
        Returns:
            dict: Information about the processed implementation
        """
        try:
            if not await self.storage_provider.file_exists(file_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Function implementation file not found: {file_path}"
                )
            
            # Detect language from file extension
            language = 'python'
            _, ext = os.path.splitext(file_path)
            if ext:
                language = detect_language_with_fallback(file_path, 'python') or 'python'
            
            # Get function entity directory
            function_entity_dir = await self.get_entity_bundle_path(app_id, 'function', function_id, version_id)
            
            # Clean up existing directory if it exists
            if await self.storage_provider.file_exists(function_entity_dir):
                await self.storage_provider.delete_file(function_entity_dir)
            await self.ensure_directory_exists(function_entity_dir)
            
            # Copy the function file to the entity bundle
            if await self.storage_provider.is_local():
                dest_file = os.path.join(function_entity_dir, os.path.basename(file_path))
                shutil.copy2(file_path, dest_file)
            else:
                dest_file = os.path.join(function_entity_dir, os.path.basename(file_path))
                await self.storage_provider.copy_file(file_path, dest_file)
            
            # Calculate relative path for database storage
            rel_entity_path = os.path.relpath(function_entity_dir, settings.ENTITY_BUNDLES_DIR)
            logger.info(f"Copied function code from app bundle to entity bundle: {rel_entity_path}")
            logger.info(f"🗂️ Full function entity bundle path: {function_entity_dir}")
            
            # Generate unique code_id and checksum
            import hashlib
            code_id = str(uuid4())
            checksum = hashlib.md5(rel_entity_path.encode()).hexdigest()
            
            # Store function implementation in function_code table (similar to agent_code)
            query = """
                INSERT INTO function_code (
                    code_id, function_id, name, implementation_type, file_path, 
                    language, is_active, checksum
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, 1, ?
                ) RETURNING code_id
            """
            
            implementation_id = await self.db.fetch_val(
                query,
                code_id,
                str(function_id),
                function.name,  # Use function name for the name field
                "file",  # implementation_type
                rel_entity_path,
                language,
                checksum
            )
            
            logger.info(f"Created function implementation record: {implementation_id}")
            
            return {
                "success": True,
                "implementation_id": str(implementation_id),
                "function_id": str(function_id),
                "entity_path": rel_entity_path,
                "language": language,
                "implementation_type": "file",
                "entrypoint_file": os.path.basename(file_path)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing function implementation file: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process function implementation file: {str(e)}"
            )

    async def process_functions_in_bundle(self, manifest: UnifiedManifest, app_id: UUID, app_version_id: UUID, user_id: int, bundle_source_dir: str) -> list:
        """
        Process function implementation files in an extracted app bundle
        
        Args:
            manifest: The app manifest
            app_id: UUID of the app
            app_version_id: UUID of the app version
            user: The user processing the bundle
            bundle_source_dir: The path to the temporary directory where the bundle was extracted
            
        Returns:
            list: Results of processing each function
        """
        results = []
        try:
            functions = manifest.functions
            if not functions:
                logger.info(f"No functions found in manifest for app {app_id}")
                return results
        
            for function in functions:
                function_name = function.name
                if not function_name:
                    continue
                
                function_query = """
                    SELECT f.function_id FROM functions f
                    JOIN functions_app fa ON f.function_id = fa.function_id
                    WHERE fa.app_id = ? AND f.name = ?
                """
                function_id = await self.db.fetch_val(function_query, str(app_id), function_name)
                
                if not function_id:
                    logger.warning(f"Function {function_name} not found in database for app {app_id}")
                    continue
                
                verification_query = "SELECT COUNT(*) FROM functions_app WHERE function_id = ? AND app_id = ?"
                function_belongs_to_app = await self.db.fetch_val(verification_query, str(function_id), str(app_id))
                
                if not function_belongs_to_app:
                    logger.error(f"Function {function_id} does not belong to app {app_id}")
                    continue
                
                try:
                    result = None
                    
                    # Handle both inline content and file-based implementations
                    if hasattr(function, 'content') and function.content:
                        # Inline function - store content directly in database
                        result = await self.process_inline_function(app_id, function_id, function, user_id)
                        
                    elif hasattr(function, 'implementation_path') and function.implementation_path:
                        # File-based function
                        file_path_in_app_bundle = os.path.join(bundle_source_dir, function.implementation_path)
                        
                        if not await self.storage_provider.file_exists(file_path_in_app_bundle):
                            logger.warning(f"Implementation file not found in app bundle: {file_path_in_app_bundle}")
                            continue
                        
                        result = await self.process_function_implementation_file(
                            app_id, 
                            app_version_id, 
                            function_id, 
                            function, 
                            file_path_in_app_bundle, 
                            user_id,
                            app_version_id  # Use app_version_id as the version_id for entity bundles
                        )
                    
                    elif hasattr(function, 'entrypoint') and function.entrypoint:
                        # Alternative path for entrypoint-based functions
                        file_path_in_app_bundle = os.path.join(bundle_source_dir, function.entrypoint)
                        
                        if not await self.storage_provider.file_exists(file_path_in_app_bundle):
                            logger.warning(f"Entrypoint file not found in app bundle: {file_path_in_app_bundle}")
                            continue
                        
                        result = await self.process_function_implementation_file(
                            app_id, 
                            app_version_id, 
                            function_id, 
                            function, 
                            file_path_in_app_bundle, 
                            user_id,
                            app_version_id  # Use app_version_id as the version_id for entity bundles
                        )
                    else:
                        logger.warning(f"Function {function_name} has neither content, implementation_path, nor entrypoint - skipping")
                        continue
                    
                    if result:
                        result["version"] = getattr(function, 'version', '1.0.0')
                        result["version_id"] = None  # No version created
                        results.append(result)
                        logger.info(f"Processed implementation for function {function_name}")
                
                except Exception as ve:
                    logger.error(f"Failed to process function {function_name}: {str(ve)}")
                    continue
                
        except Exception as e:
            logger.error(f"Error processing functions in bundle: {str(e)}", exc_info=True)
        
        return results

    async def process_pipelines_in_bundle(self, manifest: UnifiedManifest, app_id: UUID, app_version_id: UUID, user_id: int, bundle_source_dir: str) -> list:
        """
        Process pipeline step files in an extracted app bundle
        
        Args:
            manifest: The app manifest
            app_id: UUID of the app
            app_version_id: UUID of the app version
            user_id: The user ID processing the bundle
            bundle_source_dir: The path to the temporary directory where the bundle was extracted
            
        Returns:
            list: Results of processing each pipeline
        """
        results = []
        try:
            logger.info(f"[DEBUG] Starting pipeline processing for app {app_id}")
            pipelines = manifest.pipelines
            logger.info(f"[DEBUG] Found {len(pipelines) if pipelines else 0} pipelines in manifest")
            if not pipelines:
                logger.info(f"No pipelines found in manifest for app {app_id}")
                return results
            
            for pipeline in pipelines:
                pipeline_name = pipeline.name
                logger.info(f"[DEBUG] Processing pipeline: {pipeline_name}")
                if not pipeline_name:
                    logger.warning(f"[DEBUG] Pipeline has no name, skipping")
                    continue
                
                logger.info(f"Processing pipeline bundle: {pipeline_name}")
                
                # Get pipeline ID from database (it was created during deployment)
                pipeline_query = "SELECT pipeline_id FROM pipelines WHERE name = ? AND app_id = ?"
                logger.info(f"[DEBUG] Looking for pipeline '{pipeline_name}' in app {app_id}")
                pipeline_record = await self.db.fetch_one(pipeline_query, pipeline_name, str(app_id))
                
                if not pipeline_record:
                    logger.error(f"Pipeline {pipeline_name} not found in database")
                    continue
                
                pipeline_id = pipeline_record['pipeline_id']
                
                # Check if pipeline version already exists (from install step)
                # First check what versions exist for this pipeline
                all_versions_query = """
                    SELECT version_id, version, file_path, status, is_active, created_at
                    FROM pipeline_versions
                    WHERE pipeline_id = ?
                    ORDER BY created_at DESC
                """
                all_versions = await self.db.fetch_all(all_versions_query, pipeline_id)
                logger.info(f"[UPLOAD_DEBUG] Found {len(all_versions)} existing versions for pipeline {pipeline_id}")
                for v in all_versions:
                    logger.info(f"[UPLOAD_DEBUG] Version: {v['version']}, ID: {v['version_id']}, file_path: {v['file_path']}, status: {v['status']}")

                # Look for existing version - try exact match first, then get most recent
                existing_version_query = """
                    SELECT version_id FROM pipeline_versions
                    WHERE pipeline_id = ? AND version = ?
                """
                existing_version = await self.db.fetch_one(
                    existing_version_query,
                    pipeline_id,
                    pipeline.version
                )

                # Get version from pipeline using same logic as install service
                pipeline_version = getattr(pipeline, 'version', '0.0.1')
                logger.info(f"[UPLOAD_DEBUG] Pipeline version from manifest: '{pipeline_version}'")
                logger.info(f"[UPLOAD_DEBUG] Looking for version '{pipeline_version}' in pipeline {pipeline_id}")

                # Try again with the corrected version
                existing_version = await self.db.fetch_one(
                    existing_version_query,
                    pipeline_id,
                    pipeline_version
                )
                logger.info(f"[UPLOAD_DEBUG] Exact version match result: {existing_version}")

                # If no exact match, get the most recent version (likely from install service)
                if not existing_version and all_versions:
                    existing_version = {'version_id': all_versions[0]['version_id']}
                    logger.info(f"[UPLOAD_DEBUG] Using most recent version: {existing_version['version_id']}")

                if existing_version:
                    # Update existing version with file_path
                    version_id = existing_version['version_id']
                    logger.info(f"Updating existing pipeline version {version_id} with file_path")

                    # Get the main pipeline implementation path (like agents do)
                    pipeline_implementation_path = getattr(pipeline, 'implementation_path', None)
                    if not pipeline_implementation_path:
                        # For pipelines without explicit implementation_path, use the bundle directory
                        pipeline_implementation_path = f"apps/{app_id}/pipeline/{pipeline_id}/{version_id}/"

                    update_query = """
                        UPDATE pipeline_versions
                        SET file_path = ?, status = 'active', is_active = 1, updated_at = CURRENT_TIMESTAMP
                        WHERE version_id = ?
                    """
                    await self.db.execute(
                        update_query,
                        pipeline_implementation_path,
                        version_id
                    )
                    logger.info(f"Updated pipeline version file_path to: {pipeline_implementation_path}")
                    logger.info(f"Updated existing pipeline version {pipeline_version} with ID {version_id}")
                else:
                    # Create new version (fallback case) - include all fields like install service
                    from uuid import uuid4
                    version_id = str(uuid4())

                    logger.info(f"[UPLOAD_DEBUG] No existing version found, creating new version {version_id}")

                    version_query = """
                        INSERT INTO pipeline_versions (
                            version_id, pipeline_id, version, description, config, file_path,
                            status, is_active, created_by, created_at, updated_at, manifest_yaml
                        ) VALUES (?, ?, ?, ?, ?, ?, 'active', 1, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                    """

                    # Get fields from pipeline manifest (same as install service)
                    import yaml
                    import json
                    config = getattr(pipeline, 'config', getattr(pipeline, 'configuration', {}))
                    config_json = json.dumps(config)
                    manifest_yaml = yaml.dump(pipeline.dict()) if hasattr(pipeline, 'dict') else ""

                    # Get the main pipeline implementation path (like agents do)
                    pipeline_implementation_path = getattr(pipeline, 'implementation_path', None)
                    if not pipeline_implementation_path:
                        # For pipelines without explicit implementation_path, use the bundle directory
                        pipeline_implementation_path = f"apps/{app_id}/pipeline/{pipeline_id}/{version_id}/"

                    await self.db.execute(
                        version_query,
                        version_id,
                        pipeline_id,
                        pipeline_version,
                        getattr(pipeline, 'description', ''),
                        config_json,
                        pipeline_implementation_path,
                        user_id,
                        manifest_yaml
                    )
                    logger.info(f"Created pipeline version with file_path: {pipeline_implementation_path}")
                    logger.info(f"[UPLOAD_DEBUG] Created new pipeline version {pipeline_version} with ID {version_id} (all fields populated)")
                
                # Create entity bundle directory for this pipeline
                # Format: apps/{app_id}/pipeline/{pipeline_id}/{version_id}/
                entity_bundles_dir = os.getenv('ENTITY_BUNDLES_DIR', 'local_data/entity_bundles')
                pipeline_bundle_dir = os.path.join(
                    entity_bundles_dir, 
                    'apps', 
                    str(app_id), 
                    'pipeline', 
                    str(pipeline_id),
                    version_id  # Use UUID version_id instead of pipeline.version
                )
                
                # Ensure directory exists
                os.makedirs(pipeline_bundle_dir, exist_ok=True)
                logger.info(f"Created pipeline bundle directory: {pipeline_bundle_dir}")
                
                # Process pipeline files - copy entire pipelines directory structure
                # This includes step files, model classes, and any dependencies
                pipeline_source_dir = os.path.join(bundle_source_dir, 'pipelines')
                logger.info(f"[STEP_FILES_DEBUG] Looking for pipeline files in: {pipeline_source_dir}")
                logger.info(f"[STEP_FILES_DEBUG] Pipeline source directory exists: {os.path.exists(pipeline_source_dir)}")

                if os.path.exists(pipeline_source_dir):
                    # Copy entire pipelines directory to preserve all dependencies
                    import shutil

                    # List what we're copying
                    logger.info(f"[STEP_FILES_DEBUG] Copying entire pipelines directory structure:")
                    for root, dirs, files in os.walk(pipeline_source_dir):
                        level = root.replace(pipeline_source_dir, '').count(os.sep)
                        indent = ' ' * 2 * level
                        rel_path = os.path.relpath(root, pipeline_source_dir)
                        logger.info(f"[STEP_FILES_DEBUG] {indent}📁 {rel_path if rel_path != '.' else 'pipelines'}/" )
                        subindent = ' ' * 2 * (level + 1)
                        for file in files:
                            if file.endswith('.py'):
                                logger.info(f"[STEP_FILES_DEBUG] {subindent}🐍 {file}")
                            else:
                                logger.info(f"[STEP_FILES_DEBUG] {subindent}📄 {file}")

                    # Copy all pipeline files and directories recursively
                    for item in os.listdir(pipeline_source_dir):
                        source_path = os.path.join(pipeline_source_dir, item)
                        dest_path = os.path.join(pipeline_bundle_dir, item)

                        if os.path.isdir(source_path):
                            if os.path.exists(dest_path):
                                shutil.rmtree(dest_path)
                            shutil.copytree(source_path, dest_path)
                            logger.info(f"✅ Copied pipeline directory: {item}/ -> entity bundle")
                        else:
                            shutil.copy2(source_path, dest_path)
                            logger.info(f"✅ Copied pipeline file: {item} -> entity bundle")

                    # Verify step classes are present (if structure is available)
                    if hasattr(pipeline, 'structure') and pipeline.structure:
                        structure = pipeline.structure
                        steps = structure.get('steps', [])
                        logger.info(f"[STEP_FILES_DEBUG] Verifying {len(steps)} step classes in copied files:")

                        for step in steps:
                            step_class = step.get('step_class')
                            if step_class:
                                # Search for the class in copied files
                                class_found = False
                                for root, dirs, files in os.walk(pipeline_bundle_dir):
                                    for file in files:
                                        if file.endswith('.py'):
                                            file_path = os.path.join(root, file)
                                            try:
                                                with open(file_path, 'r', encoding='utf-8') as f:
                                                    if f'class {step_class}' in f.read():
                                                        rel_path = os.path.relpath(file_path, pipeline_bundle_dir)
                                                        logger.info(f"✅ Found {step_class} in {rel_path}")
                                                        class_found = True
                                                        break
                                            except:
                                                continue
                                    if class_found:
                                        break

                                if not class_found:
                                    logger.warning(f"⚠️  Step class {step_class} not found in copied files")
                else:
                    logger.warning(f"⚠️  No pipelines directory found in bundle: {pipeline_source_dir}")

                # Record the successful processing
                results.append({
                    'pipeline_id': pipeline_id,
                    'version_id': version_id,
                    'name': pipeline_name,
                    'bundle_dir': pipeline_bundle_dir,
                    'status': 'bundled'
                })
                logger.info(f"Successfully bundled pipeline: {pipeline_name} with version {version_id}")
                    
        except Exception as e:
            logger.error(f"Error processing pipelines in bundle: {str(e)}", exc_info=True)
        
        return results

    async def _store_pipeline_step_code(self, pipeline_manifest, pipeline_id: str, pipeline_source_dir: str, user_id: int):
        """Store pipeline step implementation code in database (following agent_code pattern)"""
        try:
            if not hasattr(pipeline_manifest, 'structure') or not pipeline_manifest.structure:
                logger.warning(f"Pipeline {pipeline_manifest.name} has no structure - skipping code storage")
                return
            
            structure = pipeline_manifest.structure
            steps = structure.get('steps', []) if isinstance(structure, dict) else []
            
            for step_def in steps:
                step_id = step_def.get('id')
                step_class = step_def.get('step_class')
                
                if not step_id or not step_class:
                    logger.warning(f"Step missing id or step_class: {step_def}")
                    continue
                
                # Find the Python file containing this step class
                step_code = await self._find_step_implementation_code(step_class, pipeline_source_dir)
                
                if step_code:
                    # Store in pipeline_code table
                    insert_query = """
                        INSERT INTO pipeline_code (
                            pipeline_id, step_id, step_class, implementation_code, 
                            language, version, created_by
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7
                        ) ON CONFLICT (pipeline_id, step_id, version) 
                        DO UPDATE SET
                            step_class = EXCLUDED.step_class,
                            implementation_code = EXCLUDED.implementation_code,
                            updated_at = CURRENT_TIMESTAMP
                    """
                    
                    await self.db.execute(
                        insert_query,
                        pipeline_id,
                        step_id,
                        step_class,
                        step_code,
                        'python',
                        pipeline_manifest.version,
                        user_id
                    )
                    
                    logger.info(f"Stored code for pipeline step: {step_id} ({step_class})")
                else:
                    logger.error(f"Could not find implementation code for step class: {step_class}")
                    
        except Exception as e:
            logger.error(f"Error storing pipeline step code: {str(e)}", exc_info=True)

    async def _find_step_implementation_code(self, step_class_name: str, pipeline_source_dir: str) -> str:
        """Find and read the Python file containing the step class implementation"""
        try:
            # Search through all Python files in the pipelines directory
            for root, dirs, files in os.walk(pipeline_source_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                                
                                # Simple check if the class is defined in this file
                                if f"class {step_class_name}" in file_content:
                                    logger.info(f"Found step class {step_class_name} in {file_path}")
                                    return file_content
                        except Exception as e:
                            logger.warning(f"Error reading file {file_path}: {str(e)}")
                            continue
            
            logger.error(f"Step class {step_class_name} not found in any Python file")
            return None
            
        except Exception as e:
            logger.error(f"Error searching for step class {step_class_name}: {str(e)}")
            return None

    async def process_inline_function(self, app_id: UUID, function_id: UUID, function, user_id: int) -> dict:
        """
        Process an inline function by storing its content directly in the function_code table
        
        Args:
            app_id: UUID of the app
            function_id: UUID of the function
            function: Function manifest object with content
            user_id: ID of the user creating the implementation
            
        Returns:
            dict: Information about the processed implementation
        """
        try:
            language = getattr(function, 'language', 'python')
            content = function.content
            
            # Generate unique code_id and checksum
            import hashlib
            code_id = str(uuid4())
            checksum = hashlib.md5(content.encode()).hexdigest()
            
            # Store function implementation in function_code table with content
            query = """
                INSERT INTO function_code (
                    code_id, function_id, name, implementation_type, content,
                    language, is_active, checksum
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, 1, ?
                ) RETURNING code_id
            """
            
            implementation_id = await self.db.fetch_val(
                query,
                code_id,
                str(function_id),
                function.name,
                "content",  # implementation_type for inline
                content,
                language,
                checksum
            )
            
            logger.info(f"Created inline function implementation record: {implementation_id}")
            
            return {
                "success": True,
                "implementation_id": str(implementation_id),
                "function_id": str(function_id),
                "language": language,
                "implementation_type": "content",
                "content_length": len(content)
            }
            
        except Exception as e:
            logger.error(f"Error processing inline function: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process inline function: {str(e)}"
            )

    async def deploy_functions_from_filesystem(self, manifest: UnifiedManifest, app_id: UUID, app_version_id: UUID, user_id: int, app_source_dir: str) -> list:
        """
        Deploy function implementations from a filesystem-based app directory during manifest installation
        
        Args:
            manifest: The app manifest
            app_id: UUID of the app  
            app_version_id: UUID of the app version
            user: The user processing the deployment
            app_source_dir: Path to the source app directory containing function files
            
        Returns:
            list: Results of deploying each function
        """
        results = []
        
        if not manifest.functions:
            logger.info(f"No functions found in manifest for app {app_id}")
            return results
            
        logger.info(f"Deploying {len(manifest.functions)} functions from {app_source_dir} for app {app_id}")
        
        for function in manifest.functions:
            function_name = function.name
            if not function_name:
                continue
                
            # Get function_id from database
            function_query = """
                SELECT f.function_id FROM functions f
                JOIN functions_app fa ON f.function_id = fa.function_id
                WHERE fa.app_id = ? AND f.name = ?
            """
            function_id = await self.db.fetch_val(function_query, str(app_id), function_name)
            
            if not function_id:
                logger.warning(f"Function {function_name} not found in database for app {app_id}")
                continue
                
            try:
                # Look for function file in app source directory
                # Common patterns: functions/{name}.py, {name}.py, etc.
                possible_paths = [
                    os.path.join(app_source_dir, "functions", f"{function_name}.py"),
                    os.path.join(app_source_dir, f"{function_name}.py"),
                    os.path.join(app_source_dir, "functions", function_name, f"{function_name}.py"),
                    os.path.join(app_source_dir, "functions", function_name, "main.py"),
                ]
                
                source_file_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        source_file_path = path
                        break
                        
                if not source_file_path:
                    logger.warning(f"Function implementation file not found for {function_name} in {app_source_dir}")
                    results.append({
                        "success": False,
                        "function_name": function_name,
                        "error": "Implementation file not found"
                    })
                    continue
                    
                # Create entity bundles directory structure
                function_entity_dir = os.path.join(
                    settings.ENTITY_BUNDLES_DIR, 
                    "apps", 
                    str(app_id), 
                    "function", 
                    str(function_id), 
                    str(app_version_id)
                )
                os.makedirs(function_entity_dir, exist_ok=True)
                
                # Copy function file to entity bundles directory
                filename = os.path.basename(source_file_path)
                target_file_path = os.path.join(function_entity_dir, filename)
                shutil.copy2(source_file_path, target_file_path)
                
                # Create function_implementations record
                relative_path = os.path.relpath(function_entity_dir, settings.ENTITY_BUNDLES_DIR)
                
                # Delete any existing implementations for this function
                await self.db.execute("DELETE FROM function_implementations WHERE function_id = ?", str(function_id))
                
                # Create new implementation record
                implementation_id = str(uuid4())
                impl_query = """
                    INSERT INTO function_implementations 
                    (implementation_id, function_id, implementation_path, language, entrypoint_file, function_name, is_active, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """
                await self.db.execute(impl_query, 
                    implementation_id, 
                    str(function_id), 
                    relative_path, 
                    "python", 
                    filename, 
                    "run", 
                    1
                )
                
                logger.info(f"Deployed function {function_name} to {target_file_path}")
                results.append({
                    "success": True,
                    "function_name": function_name,
                    "function_id": str(function_id),
                    "implementation_path": relative_path,
                    "entrypoint_file": filename
                })
                
            except Exception as e:
                logger.error(f"Error deploying function {function_name}: {str(e)}", exc_info=True)
                results.append({
                    "success": False,
                    "function_name": function_name,
                    "error": str(e)
                })
                
        return results

    async def cleanup_old_bundles(self, app_id: UUID, keep_versions: int = 3) -> None:
        """
        Cleanup old app bundle versions to save disk space.
        Keeps the specified number of most recent versions.
        
        Args:
            app_id: UUID of the app
            keep_versions: Number of most recent versions to keep
        """
        try:
            app_dir = await self.get_app_bundle_path(app_id)
            
            if not await self.storage_provider.file_exists(app_dir):
                return
                
            try:
                versions = []
                
                if settings.STORAGE_PROVIDER.lower() == 'local':
                    versions = [d for d in os.listdir(app_dir) if os.path.isdir(os.path.join(app_dir, d))]
                    versions.sort(key=lambda v: os.path.getctime(os.path.join(app_dir, v)), reverse=True)
                else:
                    dir_items = await self.storage_provider.list_files(app_dir)
                    versions = []
                    
                    for item in dir_items:
                        item_path = os.path.join(app_dir, item)
                        item_info = await self.storage_provider.get_file_info(item_path)
                        
                        if item_info.get('is_directory', False):
                            versions.append({
                                'name': item,
                                'modified_at': item_info.get('modified_at')
                            })
                    
                    versions.sort(key=lambda v: v.get('modified_at', 0), reverse=True)
                    versions = [v['name'] for v in versions]
                
                if len(versions) > keep_versions:
                    for old_version in versions[keep_versions:]:
                        old_path = os.path.join(app_dir, old_version)
                        await self.storage_provider.delete_file(old_path)
                        logger.info(f"Cleaned up old app version: {old_version} for app {app_id}")
            except Exception as e:
                logger.warning(f"Error during cleanup of old bundles for app {app_id}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old bundles: {str(e)}", exc_info=True)

    async def check_bundle_exists(self, app_id: UUID, app_version_id: UUID) -> bool:
        """
        Check if a bundle exists for the specified app version
        
        Args:
            app_id: UUID of the app
            app_version_id: UUID of the app version
            
        Returns:
            bool: True if bundle exists, False otherwise
        """
        bundle_path = os.path.join(await self.get_app_bundle_path(app_id, app_version_id), "dist")
        return await self.storage_provider.file_exists(bundle_path)

    async def get_bundle_info(self, app_id: UUID, app_version_id: UUID) -> dict:
        """
        Get information about a specific app bundle
        
        Args:
            app_id: UUID of the app
            app_version_id: UUID of the app version
            
        Returns:
            dict: Information about the bundle
        """
        try:
            bundle_path = await self.get_app_bundle_path(app_id, app_version_id)
            
            if not await self.storage_provider.file_exists(bundle_path):
                return {
                    "exists": False,
                    "app_id": str(app_id),
                    "app_version_id": str(app_version_id),
                }
            
            file_info = await self.storage_provider.get_file_info(bundle_path)
            
            if settings.STORAGE_PROVIDER.lower() == 'local':
                total_size = 0
                file_count = 0
                for root, dirs, files in os.walk(bundle_path):
                    file_count += len(files)
                    total_size += sum(os.path.getsize(os.path.join(root, name)) for name in files)
                
                return {
                    "exists": True,
                    "app_id": str(app_id),
                    "app_version_id": str(app_version_id),
                    "path": bundle_path,
                    "size_bytes": total_size,
                    "file_count": file_count,
                    "created_at": os.path.getctime(bundle_path)
                }
            else:
                return {
                    "exists": True,
                    "app_id": str(app_id),
                    "app_version_id": str(app_version_id),
                    "path": bundle_path,
                    "size_bytes": file_info.get('size_bytes', 0),
                    "created_at": file_info.get('created_at'),
                    "modified_at": file_info.get('modified_at'),
                    "storage_provider": settings.STORAGE_PROVIDER
                }
            
        except Exception as e:
            logger.error(f"Error getting bundle info: {str(e)}", exc_info=True)
            return {
                "exists": False,
                "app_id": str(app_id),
                "app_version_id": str(app_version_id),
                "error": str(e)
            }
        
# Standalone functions for backward compatibility
async def process_app_bundle(file_path: str, app_id: UUID, app_version_id: UUID, user_id: int, db: DatabaseProvider = None, storage_provider: StorageProvider = None, final_status: str = "published", settings_instance=None) -> dict:
    """
    Standalone function to process an app bundle.
    
    Args:
        file_path: Path to the uploaded zip file
        app_id: UUID of the app
        app_version_id: UUID of the app version
        user: User processing the bundle
        db: Database provider instance
        storage_provider: Storage provider instance
        final_status: The final status for the app version ('published' or 'submitted')
        
    Returns:
        dict: Information about the processed bundle
    """
    if db is None:
        raise ValueError("Database provider is required for standalone process_app_bundle function")
    
    if storage_provider is None:
        from fiberwise_common.services import get_storage_provider
        storage_provider = get_storage_provider()
    
    service = AppUploadService(db, storage_provider, settings_instance)
    return await service.process_app_bundle(file_path, app_id, app_version_id, user_id, final_status)

async def cleanup_old_bundles(app_id: UUID, keep_versions: int = 3, db: DatabaseProvider = None, storage_provider: StorageProvider = None) -> None:
    """
    Standalone function to cleanup old app bundles.
    
    Args:
        app_id: UUID of the app
        keep_versions: Number of most recent versions to keep
        db: Database provider instance
        storage_provider: Storage provider instance
    """
    if db is None:
        raise ValueError("Database provider is required for standalone cleanup_old_bundles function")
    
    if storage_provider is None:
        from fiberwise_common.services import get_storage_provider
        storage_provider = get_storage_provider()
    
    service = AppUploadService(db, storage_provider)
    return await service.cleanup_old_bundles(app_id, keep_versions)

async def check_bundle_exists(app_id: UUID, app_version_id: UUID, db: DatabaseProvider = None, storage_provider: StorageProvider = None) -> bool:
    """
    Standalone function to check if a bundle exists.
    
    Args:
        app_id: UUID of the app
        app_version_id: UUID of the app version
        db: Database provider instance
        storage_provider: Storage provider instance
        
    Returns:
        bool: True if bundle exists, False otherwise
    """
    if db is None:
        raise ValueError("Database provider is required for standalone check_bundle_exists function")
    
    if storage_provider is None:
        from fiberwise_common.services import get_storage_provider
        storage_provider = get_storage_provider()
    
    service = AppUploadService(db, storage_provider)
    return await service.check_bundle_exists(app_id, app_version_id)

async def get_bundle_info(app_id: UUID, app_version_id: UUID, db: DatabaseProvider = None, storage_provider: StorageProvider = None) -> dict:
    """
    Standalone function to get bundle information.
    
    Args:
        app_id: UUID of the app
        app_version_id: UUID of the app version
        db: Database provider instance
        storage_provider: Storage provider instance
        
    Returns:
        dict: Information about the bundle
    """
    if db is None:
        raise ValueError("Database provider is required for standalone get_bundle_info function")
    
    if storage_provider is None:
        from fiberwise_common.services import get_storage_provider
        storage_provider = get_storage_provider()
    
    service = AppUploadService(db, storage_provider)
    return await service.get_bundle_info(app_id, app_version_id)

async def get_entity_bundle_path(app_id: UUID, entity_type: str, entity_id: UUID = None, version_id: UUID = None, db: DatabaseProvider = None, storage_provider: StorageProvider = None) -> str:
    """
    Standalone function to get entity bundle path.
    
    Args:
        app_id: The app ID (UUID)
        entity_type: Type of entity (agent, pipeline, workflow, function)
        entity_id: Optional entity ID (UUID)
        version_id: Optional version ID (UUID)
        db: Database provider instance (optional for this function)
        storage_provider: Storage provider instance
        
    Returns:
        Path to the entity bundle directory
    """
    if storage_provider is None:
        from fiberwise_common.services import get_storage_provider
        storage_provider = get_storage_provider()
    
    valid_types = ["agent", "pipeline", "workflow", "function"]
    if entity_type not in valid_types:
        raise ValueError(f"Invalid entity type: {entity_type}. Must be one of: {', '.join(valid_types)}")
    
    # Use settings-based path instead of cwd
    entity_dir = os.path.join(settings.ENTITY_BUNDLES_DIR, "apps", str(app_id), entity_type)
    
    if entity_id:
        entity_dir = os.path.join(entity_dir, str(entity_id))
        if version_id:
            entity_dir = os.path.join(entity_dir, str(version_id))
    
    # Ensure directory exists
    await ensure_directory_exists(entity_dir, db, storage_provider)
    
    logger.info(f"🗂️ Entity bundle path created: {entity_dir}")
    return entity_dir

async def ensure_directory_exists(directory_path: str, db: DatabaseProvider = None, storage_provider: StorageProvider = None) -> str:
    """
    Standalone function to ensure a directory exists.
    
    Args:
        directory_path: Path to the directory
        db: Database provider instance (optional for this function)
        storage_provider: Storage provider instance
        
    Returns:
        Path to the directory
    """
    if storage_provider is None:
        from fiberwise_common.services import get_storage_provider
        storage_provider = get_storage_provider()
    
    import tempfile
    
    file_info = await storage_provider.get_file_info(directory_path)
    
    if not file_info.get('exists', False):
        if await storage_provider.is_local():
            os.makedirs(directory_path, exist_ok=True)
        else:
            marker_file = os.path.join(directory_path, '.keep')
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
                pass
            try:
                await storage_provider.upload_file(temp_path, marker_file)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
    
    return directory_path

async def get_app_bundle_path(app_id: UUID, app_version_id: UUID = None, db: DatabaseProvider = None, storage_provider: StorageProvider = None) -> str:
    """
    Standalone function to get app bundle path.
    
    Args:
        app_id: The app ID (UUID)
        app_version_id: Optional app version ID (UUID)
        db: Database provider instance (optional for this function)
        storage_provider: Storage provider instance
        
    Returns:
        Path to the app bundle directory
    """
    if storage_provider is None:
        from fiberwise_common.services import get_storage_provider
        storage_provider = get_storage_provider()
    
    # Use settings-based path instead of hardcoded project root
    app_dir = os.path.join(settings.APP_BUNDLES_DIR, str(app_id))
    if app_version_id:
        app_dir = os.path.join(app_dir, str(app_version_id))
    
    await ensure_directory_exists(app_dir, db, storage_provider)
    
    return app_dir
    
    return app_dir
