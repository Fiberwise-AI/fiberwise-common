"""
Execution Key Service

This module provides services for managing execution API keys that are used for
temporary authentication during function execution and agent operations.
"""
import logging
import json
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class ExecutionKeyService:
    """Service for managing execution API keys"""
    
    def __init__(self, db_connection):
        """
        Initialize the ExecutionKeyService

        Args:
            db_connection: Database connection object
        """
        self.db = db_connection
    
    async def create_execution_key(
        self,
        app_id: str,
        organization_id: int,
        executor_type_id: str,
        executor_id: str,
        created_by: Optional[int] = None,
        scopes: Optional[List[str]] = None,
        expiration_minutes: int = 60,
        resource_pattern: str = '*',
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new execution API key
        
        Args:
            app_id: Application ID
            organization_id: Organization ID
            executor_type_id: Type of executor (function, agent, etc.)
            executor_id: ID of the specific executor
            created_by: User ID who created the key
            scopes: List of permission scopes
            expiration_minutes: Minutes until expiration (default 60)
            resource_pattern: Pattern for allowed resources
            metadata: Additional metadata
            
        Returns:
            Dictionary with key details or None if creation failed
        """
        try:
            # Generate key components
            key_id = f"exec_{secrets.token_urlsafe(16)}"
            key_value = f"exec_{secrets.token_urlsafe(32)}"
            
            # Calculate expiration
            expiration = datetime.now(timezone.utc) + timedelta(minutes=expiration_minutes)
            expiration_str = expiration.isoformat()
            
            # Prepare data
            scopes_json = json.dumps(scopes or [])
            metadata_json = json.dumps(metadata or {})
            
            # Insert into database using query adapter
            query = """
                INSERT INTO execution_api_keys (
                    key_id, key_value, app_id, organization_id, executor_type_id, executor_id,
                    created_by, scopes, expiration, resource_pattern, metadata
                )
                VALUES (:key_id, :key_value, :app_id, :organization_id, :executor_type_id, :executor_id,
                    :created_by, :scopes, :expiration, :resource_pattern, :metadata)
            """

            params = {"key_id": key_id, "key_value": key_value, "app_id": app_id,
                      "organization_id": organization_id, "executor_type_id": executor_type_id,
                      "executor_id": executor_id, "created_by": created_by, "scopes": scopes_json,
                      "expiration": expiration_str, "resource_pattern": resource_pattern,
                      "metadata": metadata_json}

            logger.info(f"Executing insert query with key_id: {key_id}")

            # Execute the insert
            insert_result = await self.db.execute(query, params)
            logger.info(f"Insert result: {insert_result}")
            
            # Fetch the created record
            fetch_query = "SELECT key_id, key_value, expiration FROM execution_api_keys WHERE key_id = :key_id"

            logger.info(f"Fetching created record with query: {fetch_query}")
            result = await self.db.fetch_one(fetch_query, {"key_id": key_id})
            logger.info(f"Fetch result: {result}")
            
            if result:
                logger.info(f"Created execution API key {key_id} for {executor_type_id}:{executor_id}")
                
                # Handle both tuple and dictionary results
                if isinstance(result, dict):
                    key_id_result = result['key_id']
                    key_value_result = result['key_value']
                    expiration_result = result['expiration']
                else:
                    # Tuple format
                    key_id_result = result[0]
                    key_value_result = result[1]
                    expiration_result = result[2]
                
                return {
                    'key_id': key_id_result,
                    'key_value': key_value_result,
                    'expiration': expiration_result,
                    'app_id': app_id,
                    'organization_id': organization_id,
                    'executor_type_id': executor_type_id,
                    'executor_id': executor_id,
                    'scopes': scopes or [],
                    'resource_pattern': resource_pattern,
                    'metadata': metadata or {}
                }
            else:
                logger.error(f"Failed to create execution API key for {executor_type_id}:{executor_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating execution API key: {e}")
            return None
    
    async def validate_execution_key(self, key_value: str) -> Optional[Dict[str, Any]]:
        """
        Validate an execution API key
        
        Args:
            key_value: The key value to validate
            
        Returns:
            Dictionary with key information if valid, None otherwise
        """
        try:
            query = """
                SELECT
                    key_id, app_id, organization_id, executor_type_id, executor_id,
                    created_by, scopes, expiration, resource_pattern,
                    metadata, created_at
                FROM execution_api_keys
                WHERE key_value = :key_value
                    AND is_revoked = 0
            """

            result = await self.db.fetch_one(query, {"key_value": key_value})
            
            if not result:
                logger.warning(f"Execution API key not found or revoked")
                return None
            
            # Handle both tuple and dictionary results
            if isinstance(result, dict):
                key_id = result['key_id']
                app_id = result['app_id']
                organization_id = result['organization_id']
                executor_type_id = result['executor_type_id']
                executor_id = result['executor_id']
                created_by = result['created_by']
                scopes_json = result['scopes']
                expiration_str = result['expiration']
                resource_pattern = result['resource_pattern']
                metadata_json = result['metadata']
                created_at = result['created_at']
            else:
                # Tuple format
                (key_id, app_id, organization_id, executor_type_id, executor_id, created_by, 
                 scopes_json, expiration_str, resource_pattern, metadata_json, created_at) = result
            
            # Check expiration
            if expiration_str:
                try:
                    expiration = datetime.fromisoformat(expiration_str.replace('Z', '+00:00'))
                    if datetime.now(timezone.utc) > expiration:
                        logger.warning(f"Execution API key {key_id} has expired")
                        return None
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid expiration format for key {key_id}: {e}")
                    return None
            
            # Parse JSON fields
            try:
                scopes = json.loads(scopes_json) if scopes_json else []
                if isinstance(scopes, str):
                    scopes = [scopes]
            except (json.JSONDecodeError, TypeError):
                scopes = []
            
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            
            key_info = {
                'key_id': key_id,
                'app_id': app_id,
                'organization_id': organization_id,
                'executor_type_id': executor_type_id,
                'executor_id': executor_id,
                'created_by': created_by,
                'scopes': scopes,
                'expiration': expiration_str,
                'resource_pattern': resource_pattern,
                'metadata': metadata,
                'created_at': created_at
            }
            
            logger.info(f"Successfully validated execution API key {key_id}")
            return key_info
            
        except Exception as e:
            logger.error(f"Error validating execution API key: {e}")
            return None
    
    async def revoke_execution_key(self, key_id: str) -> bool:
        """
        Revoke an execution API key
        
        Args:
            key_id: ID of the key to revoke
            
        Returns:
            True if revoked successfully, False otherwise
        """
        try:
            query = """
                UPDATE execution_api_keys
                SET is_revoked = 1, updated_at = CURRENT_TIMESTAMP
                WHERE key_id = :key_id
            """

            await self.db.execute(query, {"key_id": key_id})

            # Check if the update was successful by fetching the record
            check_query = "SELECT key_id FROM execution_api_keys WHERE key_id = :key_id AND is_revoked = 1"
            result = await self.db.fetch_one(check_query, {"key_id": key_id})
            
            success = result is not None
            if success:
                logger.info(f"Revoked execution API key {key_id}")
            else:
                logger.warning(f"Failed to revoke execution API key {key_id}: Key not found")
                
            return success
        except Exception as e:
            logger.error(f"Error revoking execution API key: {e}")
            return False
    
    async def cleanup_expired_keys(self) -> int:
        """
        Clean up expired execution keys
        
        Returns:
            Number of keys cleaned up
        """
        try:
            now = datetime.now(timezone.utc).isoformat()

            query = """
                UPDATE execution_api_keys
                SET is_revoked = 1, updated_at = CURRENT_TIMESTAMP
                WHERE expiration < :now AND is_revoked = 0
            """

            await self.db.execute(query, {"now": now})

            # Count how many were updated
            count_query = "SELECT COUNT(*) FROM execution_api_keys WHERE expiration < :now AND is_revoked = 1"
            count_result = await self.db.fetch_one(count_query, {"now": now})
            if count_result:
                count = count_result[0] if isinstance(count_result, (list, tuple)) else list(count_result.values())[0]
            else:
                count = 0
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired execution API keys")
            
            return count
        except Exception as e:
            logger.error(f"Error cleaning up expired execution keys: {e}")
            return 0
    
    async def get_execution_keys(
        self,
        app_id: Optional[str] = None,
        executor_type_id: Optional[str] = None,
        executor_id: Optional[str] = None,
        created_by: Optional[int] = None,
        include_revoked: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get execution keys with optional filters
        
        Args:
            app_id: Filter by application ID
            executor_type_id: Filter by executor type
            executor_id: Filter by executor ID
            created_by: Filter by creator user ID
            include_revoked: Whether to include revoked keys
            
        Returns:
            List of key information dictionaries
        """
        try:
            conditions = []
            params = {}

            if app_id:
                conditions.append("app_id = :app_id")
                params["app_id"] = app_id

            if executor_type_id:
                conditions.append("executor_type_id = :executor_type_id")
                params["executor_type_id"] = executor_type_id

            if executor_id:
                conditions.append("executor_id = :executor_id")
                params["executor_id"] = executor_id

            if created_by:
                conditions.append("created_by = :created_by")
                params["created_by"] = created_by

            if not include_revoked:
                conditions.append("is_revoked = 0")

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

            query = f"""
                SELECT
                    key_id, app_id, organization_id, executor_type_id, executor_id,
                    created_by, scopes, expiration, resource_pattern,
                    metadata, is_revoked, created_at, updated_at
                FROM execution_api_keys
                {where_clause}
                ORDER BY created_at DESC
            """

            results = await self.db.fetch_all(query, params)
            
            keys = []
            for result in results:
                # Handle both tuple and dictionary results
                if isinstance(result, dict):
                    key_id = result['key_id']
                    app_id = result['app_id']
                    organization_id = result['organization_id']
                    executor_type_id = result['executor_type_id']
                    executor_id = result['executor_id']
                    created_by = result['created_by']
                    scopes_json = result['scopes']
                    expiration_str = result['expiration']
                    resource_pattern = result['resource_pattern']
                    metadata_json = result['metadata']
                    is_revoked = result['is_revoked']
                    created_at = result['created_at']
                    updated_at = result['updated_at']
                else:
                    # Tuple format
                    (key_id, app_id, organization_id, executor_type_id, executor_id, created_by,
                     scopes_json, expiration_str, resource_pattern, metadata_json,
                     is_revoked, created_at, updated_at) = result
                
                # Parse JSON fields
                try:
                    scopes = json.loads(scopes_json) if scopes_json else []
                    if isinstance(scopes, str):
                        scopes = [scopes]
                except (json.JSONDecodeError, TypeError):
                    scopes = []
                
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                
                keys.append({
                    'key_id': key_id,
                    'app_id': app_id,
                    'organization_id': organization_id,
                    'executor_type_id': executor_type_id,
                    'executor_id': executor_id,
                    'created_by': created_by,
                    'scopes': scopes,
                    'expiration': expiration_str,
                    'resource_pattern': resource_pattern,
                    'metadata': metadata,
                    'is_revoked': bool(is_revoked),
                    'created_at': created_at,
                    'updated_at': updated_at
                })
            
            return keys
            
        except Exception as e:
            logger.error(f"Error getting execution keys: {e}")
            return []
