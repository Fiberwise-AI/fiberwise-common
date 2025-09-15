import json
import logging
from typing import Any, Dict, Optional, List
import yaml
from uuid import uuid4, UUID
from pydantic import UUID4

from fastapi import HTTPException, status

from fiberwise_common.entities import UnifiedManifest
from fiberwise_common import DatabaseProvider

logger = logging.getLogger(__name__)

# Minimal stub for AppRead and User (replace with actual imports if available)
class AppRead:
    def __init__(self, app_id, name, version, models=None, **kwargs):
        self.app_id = app_id
        self.name = name
        self.version = version
        self.models = models or []
        # Add any additional fields passed in kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def model_validate(cls, data):
        """Create instance from dict data (pydantic-like interface)"""
        return cls(**data)

class User:
    def __init__(self, id, username=None, **kwargs):
        self.id = id
        self.username = username
        for key, value in kwargs.items():
            setattr(self, key, value)


class AppService:
    """
    Consolidated app service for all app-related operations.
    
    Handles app data access, model management, validation, and permissions.
    This consolidates functionality from the web tier into the common business logic tier.
    """
    
    def __init__(self, db: DatabaseProvider):
        self.db = db

    async def get_app_by_id(self, app_id: str) -> Dict[str, Any]:
        """Get app by ID with validation"""
        query = "SELECT * FROM apps WHERE app_id = $1"
        app = await self.db.fetch_one(query, app_id)
        if not app:
            raise ValueError(f"App with id '{app_id}' not found")
        return dict(app)

    async def get_app_by_slug(self, app_slug: str) -> Dict[str, Any]:
        """Get app by slug with validation"""
        query = "SELECT * FROM apps WHERE app_slug = $1"
        app = await self.db.fetch_one(query, app_slug)
        if not app:
            raise ValueError(f"App with slug '{app_slug}' not found")
        return dict(app)

    async def get_model_by_slug(self, app_id: str, model_slug: str) -> Dict[str, Any]:
        """Get model by slug with validation"""
        query = "SELECT * FROM models WHERE app_id = $1 AND model_slug = $2"
        model = await self.db.fetch_one(query, str(app_id), model_slug)
        if not model:
            # Add debug logging to help troubleshoot
            logger.error(f"Model lookup failed for app_id='{app_id}', model_slug='{model_slug}'")
            
            # Check what models exist for this app
            debug_query = "SELECT model_slug FROM models WHERE app_id = $1"
            existing_models = await self.db.fetch_all(debug_query, str(app_id))
            existing_slugs = [dict(m)['model_slug'] for m in existing_models] if existing_models else []
            logger.error(f"Available models for app '{app_id}': {existing_slugs}")
            
            # Check if app exists at all
            app_check_query = "SELECT app_id, name FROM apps WHERE app_id = $1"
            app_exists = await self.db.fetch_one(app_check_query, str(app_id))
            if app_exists:
                app_info = dict(app_exists)
                logger.error(f"App exists: id='{app_info['app_id']}', name='{app_info['name']}'")
            else:
                logger.error(f"App with id '{app_id}' does not exist in apps table")
            
            raise ValueError(f"Model with slug '{model_slug}' not found in app")
        return dict(model)

    async def get_fields_for_model(self, model_id: str) -> List[Dict[str, Any]]:
        """Get all fields for a model"""
        query = "SELECT * FROM fields WHERE model_id = $1"
        fields = await self.db.fetch_all(query, str(model_id))
        
        # Process each field to parse JSON strings
        processed_fields = []
        for field in fields:
            field_dict = dict(field)
            
            # Parse JSON fields if they are strings
            for json_field in ['default_value_json', 'validations_json', 'relation_details_json']:
                if json_field in field_dict and field_dict[json_field]:
                    try:
                        if isinstance(field_dict[json_field], str):
                            field_dict[json_field] = json.loads(field_dict[json_field])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Failed to parse {json_field} for field {field_dict.get('field_column', 'unknown')}: {field_dict[json_field]}")
                        field_dict[json_field] = {}
            
            processed_fields.append(field_dict)
            
        return processed_fields

    async def validate_data_against_fields(self, data: Dict[str, Any], fields: List[Dict[str, Any]], is_create: bool = False) -> bool:
        """
        Validate data against field definitions
        
        Args:
            data: Dictionary of field values
            fields: List of field definitions
            is_create: If True, skips validation for primary key fields (used for creation operations)
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails with list of errors
        """
        errors = []
        
        # Check required fields
        for field in fields:
            if field["is_required"] and field["field_column"] not in data:
                # Skip required validation for primary keys during creation
                if is_create and field.get("is_primary_key", False):
                    continue
                errors.append(f"Field '{field['field_column']}' is required")
        
        # Check data types and validations
        for field_column, value in data.items():
            field_def = next((f for f in fields if f["field_column"] == field_column), None)
            if not field_def:
                errors.append(f"Field '{field_column}' is not defined in the model")
                continue
            
            # Basic type checking
            try:
                if field_def["data_type"] == "string" and not isinstance(value, str):
                    errors.append(f"Field '{field_column}' must be a string")
                elif field_def["data_type"] == "integer" and not isinstance(value, int):
                    errors.append(f"Field '{field_column}' must be an integer")
                elif field_def["data_type"] == "float" and not isinstance(value, (int, float)):
                    errors.append(f"Field '{field_column}' must be a number")
                elif field_def["data_type"] == "boolean" and not isinstance(value, bool):
                    errors.append(f"Field '{field_column}' must be a boolean")
                
            
                    
                # Apply additional validations from validations_json if present
                if field_def["validations_json"]:
                    try:
                        # Parse validations_json if it's a string
                        if isinstance(field_def["validations_json"], str):
                            validations = json.loads(field_def["validations_json"])
                        else:
                            validations = field_def["validations_json"]
                        
                        # Apply validation rules (example implementation)
                        if isinstance(value, str) and validations.get("max_length") and len(value) > validations["max_length"]:
                            errors.append(f"Field '{field_column}' exceeds maximum length of {validations['max_length']}")
                    except (json.JSONDecodeError, TypeError):
                        # Skip validation if JSON parsing fails
                        logger.warning(f"Invalid validations_json for field {field_column}: {field_def['validations_json']}")
                        
            except Exception as e:
                errors.append(f"Validation error on field '{field_column}': {str(e)}")
        
        if errors:
            # Preserve original error format for route compatibility
            error_details = {"errors": errors}
            raise ValueError(f"Validation failed: {error_details}")
        
        return True

    async def check_app_access(self, app_id: str, user_id: int) -> bool:
        """Check if user has access to the app (creator or has installed it)"""
        # Check if user is the creator
        creator_query = "SELECT creator_user_id FROM apps WHERE app_id = $1"
        creator_id = await self.db.fetch_val(creator_query, str(app_id))
        
        if creator_id == user_id:
            return True
        
        # Check if user has installed the app
        installation_query = """
            SELECT installation_id FROM app_installations 
            WHERE app_id = $1 AND user_id = $2 AND status = 'active'
        """
        installation = await self.db.fetch_val(installation_query, str(app_id), user_id)
        
        if not installation:
            raise ValueError("You don't have access to this app")
        
        return True

    async def fetch_app_data(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive app details with models and fields."""
        try:
            app_query = "SELECT * FROM apps WHERE app_id = $1"
            app = await self.db.fetch_one(app_query, str(app_id))
            if not app: 
                return None
            app_data = dict(app)

            # Fetch models, including is_active status if the column exists
            models_query = "SELECT * FROM models WHERE app_id = $1 ORDER BY model_id"
            models = await self.db.fetch_all(models_query, str(app_id))
            app_models = []
            
            for model in models:
                model_data = dict(model)
                # Fetch fields, including is_active status if the column exists
                fields_query = "SELECT * FROM fields WHERE model_id = $1 ORDER BY field_id"
                fields = await self.db.fetch_all(fields_query, model_data["model_id"])
                
                # Parse JSON fields if stored as strings
                def safe_json_parse(value):
                    """Safely parse JSON, handling empty strings and None values"""
                    if isinstance(value, str) and value.strip():
                        try:
                            return json.loads(value)
                        except json.JSONDecodeError:
                            return None
                    return value if value is not None else None
                
                model_data["fields"] = [
                    {**dict(field),
                     'default_value_json': safe_json_parse(field.get('default_value_json')),
                     'validations_json': safe_json_parse(field.get('validations_json')),
                     'relation_details_json': safe_json_parse(field.get('relation_details_json'))
                     } for field in fields
                ]
                app_models.append(model_data)
            
            app_data["models"] = app_models
            return app_data
            
        except Exception as e:
            logger.error(f"Error fetching app data for {app_id}: {e}", exc_info=True)
            return None

    # ===== APP CREATION AND MANIFEST FUNCTIONS =====

    async def create_new_app(self, app_manifest, user_id: str, connection=None):
        """Create a new application from manifest"""
        app_id = uuid4()
        conn = connection or self.db
        
        insert_query = """
            INSERT INTO apps (
                app_id, app_slug, name, description, version, creator_user_id, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP
            ) RETURNING app_id
        """
        
        created_app_id = await conn.fetch_val(
            insert_query,
            str(app_id),
            app_manifest.app_slug,
            app_manifest.name,
            app_manifest.description,
            app_manifest.version,
            user_id
        )
        
        logger.info(f"Created new app with ID {created_app_id}")
        return created_app_id

    async def create_app_version(self, app_id, app_manifest, user_id, connection=None, raw_manifest_yaml=None):
        """Create a new version for an app from a manifest"""
        conn = connection or self.db
        version_id = uuid4()
        
        # Use raw YAML if provided, otherwise fall back to encoding (for backward compatibility)
        manifest_yaml = None
        if raw_manifest_yaml:
            # Use the raw YAML content to preserve original formatting
            manifest_yaml = raw_manifest_yaml
        else:
            # Fallback: Convert manifest to YAML (may cause encoding issues)
            if hasattr(app_manifest, 'model_dump'):
                manifest_dict = app_manifest.model_dump()
                manifest_yaml = yaml.dump(manifest_dict)
            elif hasattr(app_manifest, 'dict'):
                manifest_dict = app_manifest.dict()
                manifest_yaml = yaml.dump(manifest_dict)
            else:
                manifest_dict = app_manifest
                if isinstance(manifest_dict, str):
                    if manifest_dict.startswith('---') or manifest_dict.strip().startswith('app:'):
                        manifest_yaml = manifest_dict
                    else:
                        try:
                            manifest_dict = json.loads(manifest_dict)
                        except:
                            manifest_dict = {}
                if not manifest_yaml:
                    manifest_yaml = yaml.dump(manifest_dict)
        
        # Extract version and description from manifest object or fallback dict
        if hasattr(app_manifest, 'version'):
            version = app_manifest.version
        elif hasattr(app_manifest, 'dict'):
            version = app_manifest.dict().get('version', '1.0.0')
        elif isinstance(app_manifest, dict):
            version = app_manifest.get('version', '1.0.0')
        else:
            version = '1.0.0'
            
        if hasattr(app_manifest, 'description'):
            description = app_manifest.description
        elif hasattr(app_manifest, 'dict'):
            description = app_manifest.dict().get('description', '')
        elif isinstance(app_manifest, dict):
            description = app_manifest.get('description', '')
        else:
            description = ''
        
        entry_point = None
        if hasattr(app_manifest, 'entryPoint'):
            entry_point = app_manifest.entryPoint
        elif hasattr(app_manifest, 'entry_point_url'):
            entry_point = app_manifest.entry_point_url
        elif hasattr(app_manifest, 'dict'):
            manifest_dict = app_manifest.dict()
            entry_point = manifest_dict.get('entryPoint', manifest_dict.get('entry_point_url', 'index.js'))
        elif isinstance(app_manifest, dict):
            entry_point = app_manifest.get('entryPoint', app_manifest.get('entry_point_url', 'index.js'))
        else:
            entry_point = 'index.js'
        
        insert_query = """
            INSERT INTO app_versions (
                app_version_id, app_id, version, description, manifest_yaml, status, 
                entry_point_url, created_by
            ) VALUES (
                $1, $2, $3, $4, $5, 'draft', $6, $7
            ) RETURNING app_version_id
        """
        
        version_id = await conn.fetch_val(
            insert_query,
            str(version_id),
            str(app_id),
            version,
            description,
            manifest_yaml,
            entry_point,
            user_id
        )
        
        logger.info(f"Created new version {version} with ID {version_id} for app {app_id}")
        return UUID(version_id) if isinstance(version_id, str) else version_id


# ===== MISSING FUNCTION FOR MANIFEST IMPORT =====

async def _process_models_from_manifest(db, app_id: str, models_list):
    """Process and create models from manifest data"""
    import uuid
    
    logger.error(f"DEPLOYMENT DEBUG: _process_models_from_manifest called with app_id={app_id}, models_count={len(models_list)}")
    
    for i, model in enumerate(models_list):
        logger.info(f"Processing model {i+1}/{len(models_list)}: {getattr(model, 'name', 'Unknown')}")
        
        # Extract model information (handle both dict and object formats)
        model_slug = getattr(model, 'model_slug', None) or (model.get('model_slug') if isinstance(model, dict) else None)
        model_name = getattr(model, 'name', None) or (model.get('name') if isinstance(model, dict) else None)
        model_description = getattr(model, 'description', None) or (model.get('description') if isinstance(model, dict) else None)
        
        # Generate defaults if missing
        if not model_slug and model_name:
            model_slug = model_name.lower().replace(' ', '_').replace('-', '_')
            logger.warning(f"Generated model_slug '{model_slug}' from model name '{model_name}'")
        elif not model_slug:
            model_slug = f"model_{i+1}"
            logger.warning(f"Generated default model_slug '{model_slug}'")
        
        if not model_name:
            model_name = f"Model {i+1}"
            logger.warning(f"Generated default model_name '{model_name}'")
        
        # Check if model already exists
        check_query = "SELECT model_id FROM models WHERE app_id = $1 AND model_slug = $2"
        existing_model = await db.fetch_one(check_query, app_id, model_slug)
        
        if existing_model:
            logger.info(f"Model '{model_slug}' already exists, skipping creation")
            model_id = existing_model['model_id']
        else:
            # Create new model
            model_id = str(uuid.uuid4())
            create_query = """
                INSERT INTO models (model_id, app_id, model_slug, name, description, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """
            await db.execute(create_query, model_id, app_id, model_slug, model_name, model_description)
            logger.info(f"Created model '{model_name}' with slug '{model_slug}' and ID {model_id}")
        
        # Process model fields
        model_fields = getattr(model, 'fields', None) or (model.get('fields', []) if isinstance(model, dict) else [])
        for j, field in enumerate(model_fields):
            logger.debug(f"Processing field {j+1}/{len(model_fields)} for model {model_slug}")
            
            # Extract field information
            field_column = (getattr(field, 'field_column', None) or 
                          (field.get('field_column') if isinstance(field, dict) else None))
            field_name = (getattr(field, 'name', None) or 
                         (field.get('name') if isinstance(field, dict) else None))
            data_type = (getattr(field, 'type', None) or 
                        (field.get('type') if isinstance(field, dict) else None))
            is_required = (getattr(field, 'required', False) or 
                          (field.get('required', False) if isinstance(field, dict) else False))
            is_primary_key = (getattr(field, 'is_primary_key', False) or 
                             (field.get('is_primary_key', False) if isinstance(field, dict) else False))
            default_value = (getattr(field, 'default', None) or 
                           (field.get('default') if isinstance(field, dict) else None))
            
            if not field_column or not field_name or not data_type:
                logger.warning(f"Skipping incomplete field: column={field_column}, name={field_name}, type={data_type}")
                continue
            
            # Check if field already exists
            field_check_query = "SELECT field_id FROM fields WHERE model_id = $1 AND field_column = $2"
            existing_field = await db.fetch_one(field_check_query, model_id, field_column)
            
            if existing_field:
                logger.debug(f"Field '{field_column}' already exists for model {model_slug}")
                continue
            
            # Create field
            field_id = str(uuid.uuid4())
            default_json = json.dumps(default_value) if default_value is not None else None
            
            field_create_query = """
                INSERT INTO fields (
                    field_id, model_id, field_column, name, is_primary_key, 
                    data_type, is_required, is_unique, default_value_json, 
                    validations_json, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, CURRENT_TIMESTAMP)
            """
            
            await db.execute(
                field_create_query, 
                field_id, model_id, field_column, field_name, is_primary_key,
                data_type, is_required, False, default_json, '{}' 
            )
            logger.debug(f"Created field '{field_name}' ({field_column}) for model {model_slug}")


async def import_app_from_manifest(manifest: UnifiedManifest, current_user: User, connection=None) -> AppRead:
    """
    Import an app from a unified manifest.
    
    This function creates a new app and its initial version from a manifest.
    Used by the install service to process app manifests.
    
    Args:
        manifest: UnifiedManifest containing the app definition
        current_user: The user creating the app
        connection: Optional database connection (uses default if not provided)
        
    Returns:
        AppRead object with the created app details
    """
    if not manifest.app:
        raise ValueError("Manifest does not contain an app definition")
    
    app_manifest = manifest.app
    
    # Use provided connection or default database
    db = connection if connection is not None else None
    if db is None:
        # This would need to be injected or accessed differently
        # For now, we'll assume the connection is always provided
        raise ValueError("Database connection is required")
    
    # Create the app
    app_service = AppService(db)
    app_id = await app_service.create_new_app(app_manifest, str(current_user.id), connection)
    
    # Create the initial version
    version_id = await app_service.create_app_version(app_id, app_manifest, str(current_user.id), connection)
    
    # Process models if they exist in the manifest
    if hasattr(app_manifest, 'models') and app_manifest.models:
        logger.error(f"DEPLOYMENT DEBUG: Creating {len(app_manifest.models)} models for app {app_id}")
        logger.error(f"DEPLOYMENT DEBUG: Models to create: {[getattr(m, 'name', 'Unknown') for m in app_manifest.models]}")
        await _process_models_from_manifest(db, str(app_id), app_manifest.models)
        logger.error(f"DEPLOYMENT DEBUG: Model creation completed for app {app_id}")
    else:
        logger.error(f"DEPLOYMENT DEBUG: No models found in manifest for app {app_id}")
        logger.error(f"DEPLOYMENT DEBUG: app_manifest.models = {getattr(app_manifest, 'models', 'MISSING')}")
    
    # Return AppRead object with the created app details
    return AppRead(
        app_id=str(app_id),
        name=app_manifest.name,
        version=getattr(app_manifest, 'version', '1.0.0'),
        models=getattr(app_manifest, 'models', [])
    )


# ===== STANDALONE FUNCTIONS FOR BACKWARDS COMPATIBILITY =====

async def validate_app_access(db: DatabaseProvider, app_id: str, user: User, organization_id: int) -> bool:
    """Standalone function to validate that the app exists and user has access within an organization"""
    try:
        if not all([app_id, user, organization_id]):
            return False

        # Check if user is an active member of the organization
        member_query = """
            SELECT 1 FROM organization_members 
            WHERE user_id = $1 AND organization_id = $2 AND status = 'active'
        """
        is_member = await db.fetch_val(member_query, user.id, organization_id)
        if not is_member:
            logger.warning(f"User {user.id} is not an active member of organization {organization_id}")
            return False

        # Check if the app is available to the organization
        creator_query = """
            SELECT 1 FROM apps a
            JOIN organization_members om ON a.creator_user_id = om.user_id
            WHERE a.app_id = $1 AND om.organization_id = $2
        """
        creator_in_org = await db.fetch_val(creator_query, app_id, organization_id)
        if creator_in_org:
            return True

        install_query = """
            SELECT 1 FROM app_installations ai
            JOIN organization_members om ON ai.user_id = om.user_id
            WHERE ai.app_id = $1 AND om.organization_id = $2 AND ai.status = 'active'
            LIMIT 1
        """
        installed_in_org = await db.fetch_val(install_query, app_id, organization_id)
        if installed_in_org:
            return True
        
        logger.warning(f"App {app_id} is not available to organization {organization_id}")
        return False
        
    except Exception as e:
        logger.error(f"Error validating app access for org {organization_id}: {str(e)}")
        return False

