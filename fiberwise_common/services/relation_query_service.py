"""
Service for optimizing queries with JSON relations - Generic Implementation
"""
import json
import logging
from typing import Dict, List, Any, Optional, Union
from uuid import UUID

from fiberwise_common import DatabaseProvider


logger = logging.getLogger(__name__)

async def get_related_entities(
    app_id: Union[str, UUID],
    source_model_slug: str,
    target_model_slug: str, 
    relation_field: str,
    target_entity_id: str,
    page: int = 1,
    limit: int = 20,
    db: Optional[DatabaseProvider] = None
) -> Dict[str, Any]:
    """
    Get all entities from one model that reference a specific entity via a relation field.
    
    Args:
        app_id: The app UUID
        source_model_slug: The model containing the relation field
        target_model_slug: The model being referenced
        relation_field: The field name storing the relation
        target_entity_id: ID of the target entity being referenced
        page: Page number for pagination
        limit: Items per page
        db: Database provider instance
        
    Returns:
        Dict with items and pagination info
    """
    if db is None:
        logger.error("get_related_entities called without database provider")
        return {
            "items": [],
            "count": 0,
            "total": 0,
            "page": page,
            "pages": 0
        }
    
    app_id_str = str(app_id)
    offset = (page - 1) * limit
    
    # First try to get the view name from metadata
    view_query = """
        SELECT 'rel_view_' || substring(md5($1::text || '_' || $2 || '_' || $3 || '_' || $4), 1, 20) as view_name
    """
    view_result = await db.fetch_one(view_query, app_id_str, source_model_slug, target_model_slug, relation_field)
    view_name = view_result['view_name'] if view_result else None
    
    # Check if the view actually exists
    view_exists = False
    if view_name:
        check_query = "SELECT to_regclass($1::text) IS NOT NULL"
        view_exists = await db.fetch_val(check_query, view_name)
    
    if view_exists:
        # Use the optimized view
        logger.debug(f"Using optimized view {view_name} for relation query")
        items_query = f"""
            SELECT * FROM {view_name}
            WHERE related_id = $1
            ORDER BY updated_at DESC
            LIMIT $2 OFFSET $3
        """
        count_query = f"""
            SELECT COUNT(*) FROM {view_name}
            WHERE related_id = $1
        """
        
        # Execute queries using the view
        items = await db.fetch_all(items_query, target_entity_id, limit, offset)
        total = await db.fetch_val(count_query, target_entity_id)
    else:
        # Fallback to direct JSONB query with dynamic path
        logger.debug(f"Using direct JSONB query for relation: {source_model_slug}.{relation_field} -> {target_model_slug}")
        
        # Get model ID for the source model
        model_id_query = """
            SELECT model_id FROM models
            WHERE app_id = $1 AND model_slug = $2
        """
        model_id = await db.fetch_val(model_id_query, app_id_str, source_model_slug)
        if not model_id:
            logger.error(f"Model not found: {source_model_slug} in app {app_id_str}")
            return {
                "items": [],
                "count": 0,
                "total": 0,
                "page": page,
                "pages": 0
            }
        
        # Direct query using JSONB operator
        items_query = """
            SELECT i.* FROM app_model_items i
            WHERE i.model_id = $1
            AND i.data->>$2 = $3
            ORDER BY i.updated_at DESC
            LIMIT $4 OFFSET $5
        """
        count_query = """
            SELECT COUNT(*) FROM app_model_items i
            WHERE i.model_id = $1
            AND i.data->>$2 = $3
        """
        
        # Execute queries using direct JSONB filtering
        items = await db.fetch_all(items_query, model_id, relation_field, target_entity_id, limit, offset)
        total = await db.fetch_val(count_query, model_id, relation_field, target_entity_id)
        
        # Create the relation view for future calls if both models exist
        target_exists_query = """
            SELECT 1 FROM models
            WHERE app_id = $1 AND model_slug = $2
            LIMIT 1
        """
        target_exists = await db.fetch_val(target_exists_query, app_id_str, target_model_slug)
        
        if target_exists:
            try:
                logger.info(f"Creating missing relation view for {source_model_slug}.{relation_field} -> {target_model_slug}")
                await db.execute(
                    "SELECT create_relation_view($1, $2, $3, $4)",
                    app_id_str, source_model_slug, target_model_slug, relation_field
                )
            except Exception as e:
                logger.warning(f"Failed to create relation view: {str(e)}")
    
    # Process items - parse JSON data
    result_items = []
    for item in items:
        item_dict = dict(item)
        if isinstance(item_dict.get("data"), str):
            try:
                item_dict["data"] = json.loads(item_dict["data"])
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in data field for item {item_dict.get('item_id')}")
                item_dict["data"] = {}
        result_items.append(item_dict)
    
    return {
        "items": result_items,
        "count": len(result_items),
        "total": total,
        "page": page,
        "pages": (total + limit - 1) // limit if limit > 0 else 0
    }

async def find_entities_by_properties(
    app_id: Union[str, UUID],
    model_slug: str,
    properties: Dict[str, Any],
    page: int = 1,
    limit: int = 20,
    db: Optional[DatabaseProvider] = None
) -> Dict[str, Any]:
    """
    Generic function to find entities by their properties (field values)
    
    Args:
        app_id: The app UUID
        model_slug: The model to query
        properties: Dict of field:value conditions to filter on
        page: Page number
        limit: Items per page
        db: Database provider instance
        
    Returns:
        Dict with items and pagination info
    """
    if db is None:
        logger.error("find_entities_by_properties called without database provider")
        return {
            "items": [],
            "count": 0,
            "total": 0,
            "page": page,
            "pages": 0
        }
    
    app_id_str = str(app_id)
    offset = (page - 1) * limit
    
    # Validate model exists
    model_query = "SELECT model_id FROM models WHERE app_id = $1 AND model_slug = $2"
    model_id = await db.fetch_val(model_query, app_id_str, model_slug)
    if not model_id:
        logger.error(f"Model not found: {model_slug} in app {app_id_str}")
        return {
            "items": [],
            "count": 0,
            "total": 0,
            "page": page,
            "pages": 0
        }
    
    # Build the query dynamically based on filter conditions
    if not properties:
        # Simple query without filters
        items_query = """
            SELECT * FROM app_model_items
            WHERE model_id = $1
            ORDER BY updated_at DESC
            LIMIT $2 OFFSET $3
        """
        count_query = "SELECT COUNT(*) FROM app_model_items WHERE model_id = $1"
        
        items = await db.fetch_all(items_query, model_id, limit, offset)
        total = await db.fetch_val(count_query, model_id)
    else:
        # Build query with filters
        where_clauses = []
        query_params = [model_id]  # Start with model_id
        
        for idx, (field, value) in enumerate(properties.items(), start=1):
            where_clauses.append(f"data->>'{field}' = ${idx + 1}")
            query_params.append(str(value))  # Convert value to string for JSONB comparison
        
        items_query = f"""
            SELECT * FROM app_model_items
            WHERE model_id = $1
            AND {' AND '.join(where_clauses)}
            ORDER BY updated_at DESC
            LIMIT ${len(query_params) + 1} OFFSET ${len(query_params) + 2}
        """
        
        count_query = f"""
            SELECT COUNT(*) FROM app_model_items
            WHERE model_id = $1
            AND {' AND '.join(where_clauses)}
        """
        
        # Add pagination parameters
        items_params = query_params.copy()
        items_params.extend([limit, offset])
        
        # Execute queries
        items = await db.fetch_all(items_query, *items_params)
        total = await db.fetch_val(count_query, *query_params)
    
    # Process items - parse JSON data
    result_items = []
    for item in items:
        item_dict = dict(item)
        if isinstance(item_dict.get("data"), str):
            try:
                item_dict["data"] = json.loads(item_dict["data"])
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in data field for item {item_dict.get('item_id')}")
                item_dict["data"] = {}
        result_items.append(item_dict)
    
    return {
        "items": result_items,
        "count": len(result_items),
        "total": total,
        "page": page,
        "pages": (total + limit - 1) // limit if limit > 0 else 0
    }

