"""
Model Operations Service - Handles dynamic data model operations
Extracted from apps.py import-manifest route for reuse in updates
"""
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID, uuid4

from fastapi import HTTPException, status
from fiberwise_common.entities import UnifiedManifest
from fiberwise_common import DatabaseProvider
from .base_service import BaseService

logger = logging.getLogger(__name__)

class ModelOperationsService(BaseService):
    def __init__(self, db: DatabaseProvider):
        super().__init__(db)

    def get_model_attr(self, model: Any, attr: str) -> Any:
        """
        Safely retrieves an attribute from a model object or a key from a dictionary.

        Args:
            model: The object or dictionary to retrieve the value from.
            attr: The name of the attribute or key.

        Returns:
            The value of the attribute/key, or None if it does not exist.
        """
        if isinstance(model, dict):
            return model.get(attr)
        return getattr(model, attr, None)

    async def get_current_models(self, app_id: UUID) -> List[Dict[str, Any]]:
        """
        Get current models for an app from the database
        
        Args:
            app_id: UUID of the app
            
        Returns:
            List of model dictionaries with fields
        """
        try:
            # Get models (no is_active column in schema)
            models_query = "SELECT * FROM models WHERE app_id = $1 ORDER BY model_id"
            models = await self.db.fetch_all(models_query, str(app_id))
            
            logger.info(f"Found {len(models)} existing models for app {app_id}")
            
            result = []
            for model in models:
                model_data = dict(model)
                logger.info(f"Processing existing model: {model_data.get('model_slug')} - {model_data.get('name')}")
                
                # Get fields for this model (no is_active column in schema)
                fields_query = "SELECT * FROM fields WHERE model_id = $1 ORDER BY field_id"
                fields = await self.db.fetch_all(fields_query, model_data["model_id"])
                
                model_data["fields"] = [dict(field) for field in fields]
                result.append(model_data)
                
            return result
        except Exception as e:
            logger.error(f"Error getting current models for app {app_id}: {str(e)}", exc_info=True)
            return []

    async def compare_models(self, current_models: List[Dict], new_manifest_models: List) -> Dict[str, Any]:
        """
        Compare current models with new manifest models
        Focus on safe changes only: new models, new optional fields
        
        Args:
            current_models: Current models from database
            new_manifest_models: New models from manifest (can be dicts or objects)
            
        Returns:
            Dictionary with changes detected
        """
        try:
            # Create lookup maps
            current_model_slugs = {model["model_slug"] for model in current_models}
            logger.info(f"Current model slugs in database: {current_model_slugs}")
            
            # Handle both dict and object formats for manifest models
            new_model_slugs = {self.get_model_attr(model, 'model_slug') for model in new_manifest_models}
            logger.info(f"New model slugs from manifest: {new_model_slugs}")
            
            # Find new models (safe to add)
            new_models = []
            for manifest_model in new_manifest_models:
                model_slug = self.get_model_attr(manifest_model, 'model_slug')
                if model_slug and model_slug not in current_model_slugs:
                    logger.info(f"Found new model to add: {model_slug}")
                    new_models.append(manifest_model)
                else:
                    logger.info(f"Model {model_slug} already exists, skipping")
            
            # Find new fields in existing models (safe if optional)
            new_fields = []
            for manifest_model in new_manifest_models:
                model_slug = self.get_model_attr(manifest_model, 'model_slug')
                if model_slug and model_slug in current_model_slugs:
                    # Find the current model
                    current_model = next(
                        (m for m in current_models if m["model_slug"] == model_slug), 
                        None
                    )
                    if current_model:
                        current_field_columns = {field["field_column"] for field in current_model["fields"]}
                        
                        manifest_fields = self.get_model_attr(manifest_model, 'fields') or []
                        for manifest_field in manifest_fields:
                            field_column = self.get_model_attr(manifest_field, 'field_column')
                            is_required = self.get_model_attr(manifest_field, 'is_required')
                            
                            if field_column and field_column not in current_field_columns:
                                # Only add if it's optional (safe change)
                                if not is_required:
                                    new_fields.append({
                                        "model_slug": model_slug,
                                        "field": manifest_field
                                    })
                                else:
                                    logger.warning(f"Skipping required field addition: {field_column} in {model_slug}")
            
            return {
                "new_models": new_models,
                "new_fields": new_fields,
                "has_changes": len(new_models) > 0 or len(new_fields) > 0
            }
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}", exc_info=True)
            return {"new_models": [], "new_fields": [], "has_changes": False}

    async def create_models_from_manifest(self, app_id: UUID, models: List, connection=None) -> Dict[str, Any]:
        """
        Create new models and fields from manifest
        Extracted and adapted from apps.py import-manifest route
        
        Args:
            app_id: UUID of the app
            models: List of model definitions from manifest (can be dicts or objects)
            connection: Required database connection (transaction)
            
        Returns:
            Dictionary with creation results
        """
        if connection is None:
            raise ValueError("Database connection is required for model creation")
        
        conn = connection  # Use the provided transaction connection
        
        try:
            created_models = []
            model_slug_to_id = {}
            
            # Create models
            for model_in in models:
                model_slug = self.get_model_attr(model_in, 'model_slug')
                model_name = self.get_model_attr(model_in, 'name')
                model_description = self.get_model_attr(model_in, 'description')
                
                # Generate UUID for the model
                model_id = str(uuid4())
                
                model_query = """
                    INSERT INTO models (model_id, app_id, model_slug, name, description, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
                    RETURNING model_id, created_at, updated_at
                """
                logger.info(f"Creating model: {model_slug} for app {app_id} with ID: {model_id}")
                model_result = await conn.fetch_one(
                    model_query,
                    model_id, str(app_id), model_slug, model_name, model_description
                )
                
                if not model_result:
                    logger.error(f"Failed to create model {model_slug} - no result returned")
                    continue
                    
                model_id = model_result["model_id"]
                model_created_at = model_result["created_at"]
                model_updated_at = model_result["updated_at"]
                logger.info(f"Created model {model_slug} with ID: {model_id}")
                
                # Store the model ID for relation resolution
                model_slug_to_id[model_slug] = model_id
                
                created_fields = []
                
                # Process all fields for this model
                model_fields = self.get_model_attr(model_in, 'fields') or []
                for field_in in model_fields:
                    # Extract basic field attributes that exist in the schema
                    field_column = self.get_model_attr(field_in, 'field_column')
                    field_name = self.get_model_attr(field_in, 'name')
                    data_type = self.get_model_attr(field_in, 'data_type') or self.get_model_attr(field_in, 'type')
                    is_required = self.get_model_attr(field_in, 'is_required') or self.get_model_attr(field_in, 'required') or False
                    is_unique = self.get_model_attr(field_in, 'is_unique') or False
                    is_primary_key = self.get_model_attr(field_in, 'is_primary_key') or False
                    
                    # Generate UUID for the field
                    field_id = str(uuid4())
                    
                    # Only insert fields with columns that exist in the schema
                    logger.info(f"Creating field {field_column} for model {model_slug} (model_id: {model_id})")
                    field_query = """
                        INSERT INTO fields (
                            field_id, model_id, field_column, name, data_type, is_required, is_unique, is_primary_key
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        RETURNING field_id
                    """
                    
                    field_result = await conn.fetch_one(
                        field_query,
                        field_id, model_id, field_column, field_name, data_type,
                        is_required, is_unique, is_primary_key
                    )
                    returned_field_id = field_result["field_id"]

                    # Construct FieldRead compatible dict with only available columns
                    created_fields.append({
                        "field_id": returned_field_id, 
                        "model_id": model_id,
                        "field_column": field_column, 
                        "name": field_name, 
                        "data_type": data_type,
                        "is_required": is_required, 
                        "is_unique": is_unique,
                        "is_primary_key": is_primary_key
                    })

                # Construct ModelRead compatible dict and store
                created_models.append({
                    "model_id": model_id, 
                    "app_id": str(app_id),
                    "model_slug": model_slug, 
                    "name": model_name,
                    "description": model_description,
                    "created_at": model_created_at,
                    "updated_at": model_updated_at,
                    "fields": created_fields
                })
                
                logger.info(f"Created model {model_slug} with {len(created_fields)} fields")

            return {
                "success": True,
                "created_models": created_models,
                "model_count": len(created_models),
                "total_fields": sum(len(m["fields"]) for m in created_models)
            }
            
        except Exception as e:
            logger.error(f"Error creating models: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create models: {str(e)}"
            )

    async def add_fields_to_existing_models(self, app_id: UUID, new_fields: List[Dict], connection=None) -> Dict[str, Any]:
        """
        Add new fields to existing models
        
        Args:
            app_id: UUID of the app
            new_fields: List of new field definitions with model_slug
            connection: Required database connection (transaction)
            
        Returns:
            Dictionary with addition results
        """
        if connection is None:
            raise ValueError("Database connection is required for field addition")
        
        conn = connection  # Use the provided transaction connection
        
        try:
            added_fields = []
            
            for field_info in new_fields:
                model_slug = field_info["model_slug"]
                field_def = field_info["field"]
                
                # Get model ID
                model_query = "SELECT model_id FROM models WHERE app_id = $1 AND model_slug = $2"
                model_id = await conn.fetch_val(model_query, str(app_id), model_slug)
                
                if not model_id:
                    logger.warning(f"Model {model_slug} not found for app {app_id}")
                    continue
                
                # Generate UUID for the field
                field_id = str(uuid4())
                
                # Add field
                # Extract basic field attributes that exist in the schema
                field_column = self.get_model_attr(field_def, 'field_column')
                field_name = self.get_model_attr(field_def, 'name')
                data_type = self.get_model_attr(field_def, 'data_type') or self.get_model_attr(field_def, 'type')
                is_required = self.get_model_attr(field_def, 'is_required') or self.get_model_attr(field_def, 'required') or False
                is_unique = self.get_model_attr(field_def, 'is_unique') or False
                is_primary_key = self.get_model_attr(field_def, 'is_primary_key') or False
                
                # Generate UUID for the field
                field_id = str(uuid4())
                
                field_query = """
                    INSERT INTO fields (
                        field_id, model_id, field_column, name, data_type, is_required, is_unique, is_primary_key
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING field_id
                """
                
                field_result = await conn.fetch_one(
                    field_query,
                    field_id, model_id, field_column, field_name, data_type,
                    is_required, is_unique, is_primary_key
                )
                
                added_fields.append({
                    "model_slug": model_slug,
                    "field_column": field_column,
                    "field_id": field_result["field_id"]
                })
                
                logger.info(f"Added field {field_column} to model {model_slug}")
            
            return {
                "success": True,
                "added_fields": added_fields,
                "field_count": len(added_fields)
            }
            
        except Exception as e:
            logger.error(f"Error adding fields: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to add fields: {str(e)}"
            )

    async def process_model_updates(self, app_id: UUID, new_models: List, connection=None) -> Dict[str, Any]:
        """
        Process model updates during app update
        Only handles safe changes: new models and new optional fields
        
        Args:
            app_id: UUID of the app
            new_models: List of model definitions from new manifest
            connection: Optional database connection
            
        Returns:
            Dictionary with update results
        """
        try:
            # Get current models
            current_models = await self.get_current_models(app_id)
            
            # Compare and find changes
            changes = await self.compare_models(current_models, new_models)
            
            if not changes["has_changes"]:
                return {
                    "success": True,
                    "message": "No model changes detected",
                    "new_models": 0,
                    "new_fields": 0
                }
            
            # Process new models
            created_result = {"created_models": [], "model_count": 0}
            if changes["new_models"]:
                created_result = await self.create_models_from_manifest(
                    app_id, changes["new_models"], connection
                )
            
            # Process new fields
            added_result = {"added_fields": [], "field_count": 0}
            if changes["new_fields"]:
                added_result = await self.add_fields_to_existing_models(
                    app_id, changes["new_fields"], connection
                )
            
            return {
                "success": True,
                "message": f"Added {created_result['model_count']} new models and {added_result['field_count']} new fields",
                "new_models": created_result["model_count"],
                "new_fields": added_result["field_count"],
                "created_models": created_result["created_models"],
                "added_fields": added_result["added_fields"]
            }
            
        except Exception as e:
            logger.error(f"Error processing model updates: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Model update failed: {str(e)}",
                "new_models": 0,
                "new_fields": 0
            }
