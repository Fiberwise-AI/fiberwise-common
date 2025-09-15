"""
User Isolation Service

Handles user data isolation and username resolution for dynamic app routes.
Provides secure-by-default behavior with manifest-driven configuration.
"""

import json
import logging
from typing import Any, Dict, List, Optional
import yaml

from fiberwise_common import DatabaseProvider

logger = logging.getLogger(__name__)
import logging
from typing import Any, Dict, List, Optional
import yaml

from fiberwise_common import DatabaseProvider

logger = logging.getLogger(__name__)


class UserIsolationService:
    """Service for handling user data isolation and username resolution"""
    
    def __init__(self, db: DatabaseProvider):
        self.db = db
    
    async def get_app_user_isolation_setting(self, app_id: str) -> str:
        """
        Get the user_isolation setting from the app's manifest.
        Returns 'enforced' as default if not specified (secure by default).
        
        Developers must explicitly set 'disabled' to allow shared data.
        
        Future Enhancement: Could be extended to support per-model isolation settings
        by checking model-specific isolation rules in addition to app-level settings.
        
        Args:
            app_id: The application ID
            
        Returns:
            str: 'enforced', 'disabled', or 'optional'
        """
        try:
            # Get the latest app version with manifest
            query = """
                SELECT manifest_yaml FROM app_versions 
                WHERE app_id = $1 
                ORDER BY created_at DESC 
                LIMIT 1
            """
            result = await self.db.fetch_one(query, app_id)
            
            if not result or not result["manifest_yaml"]:
                logger.warning(f"No manifest found for app {app_id}, defaulting to 'enforced' isolation (secure by default)")
                return "enforced"
            
            # Debug: Log the manifest data
            logger.info(f"Raw manifest for app {app_id}: {result['manifest_yaml'][:200]}...")
            
            # Parse manifest YAML
            manifest_data = yaml.safe_load(result["manifest_yaml"])
            if not manifest_data or not isinstance(manifest_data, dict):
                logger.warning(f"Invalid manifest data for app {app_id}, defaulting to 'enforced' isolation")
                return "enforced"
            
            # Debug: Log the parsed structure
            logger.info(f"Parsed manifest structure for app {app_id}: app section = {manifest_data.get('app', {})}")
            
            # Get user_isolation from app section
            app_section = manifest_data.get("app", {})
            user_isolation = app_section.get("user_isolation", "enforced")  # Secure by default
            
            # Validate the setting
            valid_settings = {"enforced", "disabled", "optional"}
            if user_isolation not in valid_settings:
                logger.warning(f"Invalid user_isolation setting '{user_isolation}' for app {app_id}, defaulting to 'enforced'")
                user_isolation = "enforced"
            
            logger.info(f"App {app_id} has user_isolation: {user_isolation}")
            return user_isolation
            
        except Exception as e:
            logger.error(f"Error getting user isolation setting for app {app_id}: {e}")
            return "enforced"  # Fail safe to enforced (secure by default)
    
    async def resolve_usernames_in_data(self, data: Dict[str, Any], fields: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve user_id fields to include username information.
        Adds username fields like user_id_username for display purposes.
        
        Args:
            data: The data dictionary containing user_id fields
            fields: List of field definitions from the model
            
        Returns:
            Dict with resolved username fields added
        """
        resolved_data = data.copy()
        
        try:
            # Find all user_id fields that need username resolution
            user_id_fields = []
            for field in fields:
                field_column = field.get("field_column", "")
                if (field_column.endswith("_id") and 
                    ("user" in field_column.lower() or field.get("is_system_field", False)) and
                    field_column in data and data[field_column]):
                    user_id_fields.append(field_column)
            
            if user_id_fields:
                logger.debug(f"Resolving usernames for fields: {user_id_fields}")
            
            # Resolve usernames for each user_id field
            for field_column in user_id_fields:
                user_id = data[field_column]
                if user_id:
                    # Query for username using uuid (since we store UUIDs in user_id fields)
                    user_query = "SELECT username, display_name FROM users WHERE uuid = $1"
                    user_result = await self.db.fetch_one(user_query, str(user_id))
                    
                    if user_result:
                        user_data = dict(user_result)
                        # Add both username and display_name for flexibility
                        resolved_data[f"{field_column}_username"] = user_data.get("username", "Unknown User")
                        resolved_data[f"{field_column}_display_name"] = user_data.get("display_name", user_data.get("username", "Unknown User"))
                        logger.debug(f"Resolved {field_column}={user_id} to username={user_data.get('username')}")
                    else:
                        resolved_data[f"{field_column}_username"] = "Unknown User"
                        resolved_data[f"{field_column}_display_name"] = "Unknown User"
                        logger.warning(f"Could not resolve username for user_id: {user_id}")
                        
        except Exception as e:
            logger.error(f"Error resolving usernames: {e}")
            # Don't fail the whole request if username resolution fails
        
        return resolved_data
    
    def build_user_isolation_query(self, base_query: str, user_isolation: str, current_user_id: str, 
                                 where_clause_exists: bool = True) -> tuple[str, list]:
        """
        Build a query with user isolation filtering applied.
        
        Args:
            base_query: The base SQL query
            user_isolation: The isolation setting ('enforced', 'disabled', 'optional')
            current_user_id: The current user's ID
            where_clause_exists: Whether the base query already has a WHERE clause
            
        Returns:
            Tuple of (modified_query, additional_params)
        """
        if user_isolation == "enforced":
            # Add user isolation filter
            if where_clause_exists:
                # Query already has WHERE, add AND condition
                isolation_query = base_query.replace(
                    "WHERE", "WHERE", 1  # Only replace first WHERE
                ) + " AND data->>'user_id' = $"
            else:
                # Add WHERE clause
                isolation_query = base_query + " WHERE data->>'user_id' = $"
            
            return isolation_query, [current_user_id]
        else:
            # No isolation
            return base_query, []
    
    def get_isolation_info(self, user_isolation: str) -> Dict[str, Any]:
        """
        Get information about the current isolation setting for API responses.
        
        Args:
            user_isolation: The isolation setting
            
        Returns:
            Dict with isolation information for debugging/transparency
        """
        return {
            "user_isolation": user_isolation,
            "isolation_active": user_isolation == "enforced",
            "shared_data": user_isolation in ("disabled", "optional"),
            "secure_by_default": True  # Our implementation defaults to enforced
        }


# Utility function for backwards compatibility
async def get_user_isolation_service(db: DatabaseProvider) -> UserIsolationService:
    """Factory function to create UserIsolationService instance"""
    return UserIsolationService(db)
