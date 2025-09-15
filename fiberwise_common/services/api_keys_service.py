"""
API Keys Service - migrated from fiberwise-core-web/api/services
Manages API key generation, validation, and authentication.
Provides secure API key operations for both web and CLI contexts.
"""

import hashlib
import uuid
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any

from .base_service import BaseService

# Note: This service provides core API key functionality for both web and CLI contexts.
# Web-specific features (FastAPI integration, HTTP middleware) should remain in 
# the web layer and use this service as a dependency.

# Simple data classes for common use (replacing web-specific Pydantic models)
class APIKeyData:
    """Simple data class for API key information."""
    def __init__(self, name: str, scopes: List[str] = None, expires_in_days: Optional[int] = None):
        self.name = name
        self.scopes = scopes or []
        self.expires_in_days = expires_in_days

class APIKeyResponse:
    """Response data for created API keys."""
    def __init__(self, id: int, key: str, key_prefix: str, name: str, scopes: List[str], 
                 expires_at: Optional[datetime] = None, created_at: Optional[datetime] = None):
        self.id = id
        self.key = key
        self.key_prefix = key_prefix
        self.name = name
        self.scopes = scopes
        self.expires_at = expires_at
        self.created_at = created_at

class APIKeyInfo:
    """Information about an API key (without the actual key value)."""
    def __init__(self, id: int, user_id: int, organization_id: Optional[int], 
                 key_prefix: str, name: str, scopes: List[str], 
                 expires_at: Optional[datetime] = None, last_used_at: Optional[datetime] = None,
                 created_at: Optional[datetime] = None):
        self.id = id
        self.user_id = user_id
        self.organization_id = organization_id
        self.key_prefix = key_prefix
        self.name = name
        self.scopes = scopes
        self.expires_at = expires_at
        self.last_used_at = last_used_at
        self.created_at = created_at

class ApiKeyService(BaseService):
    def __init__(self, db_provider):
        """Initialize API Keys Service with database provider."""
        super().__init__(db_provider)

    async def get_keys_for_user(self, user_id: int) -> List[APIKeyInfo]:
        """Retrieve all API keys for a specific user."""
        query = """
            SELECT id, user_id, organization_id, key_prefix, name, scopes, expires_at, last_used_at, created_at
            FROM api_keys 
            WHERE user_id = $1
        """
        keys = await self.db.fetch_all(query, user_id)
        
        # Process keys to handle JSON scopes and add user data
        processed_keys = []
        for key in keys:
            key_dict = dict(key)
            if isinstance(key_dict.get("scopes"), str):
                try:
                    key_dict["scopes"] = json.loads(key_dict["scopes"])
                except (json.JSONDecodeError, TypeError):
                    key_dict["scopes"] = []
            processed_keys.append(APIKeyInfo(**key_dict))
            
        return processed_keys

    async def create_api_key(self, user_id: int, key_data: APIKeyData):
        """Generate a new API key and store its hash in the database"""
        raw_key = str(uuid.uuid4())
        key_prefix = raw_key[:8]
        hashed_key = self._hash_key(raw_key)
        
        expires_at = None
        if key_data.expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=key_data.expires_in_days)

        # Parse scopes if passed as a string
        if isinstance(key_data.scopes, str):
            try:
                key_data.scopes = json.loads(key_data.scopes)
            except json.JSONDecodeError:
                raise ValueError("Invalid format for scopes. Expected a JSON array.")

        # Convert scopes list to JSON string to store in database
        scopes_json = json.dumps(key_data.scopes)

        # Wrap database operation in transaction
        async with self.db.transaction():
            query = """
                INSERT INTO api_keys 
                (user_id, organization_id, key_prefix, key_hash, name, scopes, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id, created_at
            """
            
            params = (
                user_id,
                None,  # organization_id - can be added later if needed
                key_prefix,
                hashed_key,
                key_data.name,
                scopes_json,  # Now a JSON string that PostgreSQL can handle
                expires_at
            )
            
            # Use fetch_one instead of fetch to get a single record as a mapping
            result = await self.db.fetch_one(query, *params)
            if not result:
                raise RuntimeError("Failed to create API key")
        
        # Convert the scopes back from JSON string to Python list
        result_dict = dict(result)
        
        # Debug: log what we got from database
        print(f"Database result for API key: {result_dict}")
        
        # Create APIKeyResponse with only the expected fields
        try:
            db_key = APIKeyResponse(
                id=result_dict["id"],
                key=raw_key,
                key_prefix=key_prefix,
                name=key_data.name,
                scopes=key_data.scopes,
                expires_at=expires_at,
                created_at=result_dict.get("created_at")
            )
        except Exception as e:
            print(f"Error creating APIKeyResponse: {e}")
            print(f"Result dict keys: {list(result_dict.keys())}")
            raise
        return raw_key, db_key

    async def update_api_key(self, key_id: int, user_id: int, update_data: Dict[str, Any]) -> Optional[APIKeyInfo]:
        """Update an API key's name or scopes."""
        async with self.db.transaction():
            # Check if key exists and belongs to the user
            query = "SELECT * FROM api_keys WHERE id = $1 AND user_id = $2"
            key = await self.db.fetch_one(query, key_id, user_id)
            
            if not key:
                return None
            
            # Build update query dynamically
            update_fields = []
            params = []
            param_index = 1
            
            if "name" in update_data:
                update_fields.append(f"name = ${param_index}")
                params.append(update_data["name"])
                param_index += 1
            
            if "scopes" in update_data:
                update_fields.append(f"scopes = ${param_index}")
                params.append(json.dumps(update_data["scopes"]))
                param_index += 1
            
            if not update_fields:
                return APIKeyInfo(**dict(key))

            params.append(key_id)
            
            update_query = f"""
                UPDATE api_keys 
                SET {', '.join(update_fields)}, updated_at = NOW()
                WHERE id = ${param_index}
                RETURNING *
            """
            
            updated_key = await self.db.fetch_one(update_query, *params)
            if updated_key:
                key_dict = dict(updated_key)
            if isinstance(key_dict.get("scopes"), str):
                try:
                    key_dict["scopes"] = json.loads(key_dict["scopes"])
                except (json.JSONDecodeError, TypeError):
                    key_dict["scopes"] = []
            return APIKeyInfo(**key_dict)
        return None

    async def delete_api_key(self, key_id: int, user_id: int) -> bool:
        """Delete an API key."""
        async with self.db.transaction():
            query = "DELETE FROM api_keys WHERE id = $1 AND user_id = $2 RETURNING id"
            result = await self.db.fetch_val(query, key_id, user_id)
            return result is not None

    async def validate_api_key(self, api_key: str) -> Optional[APIKeyInfo]:
        """Validate an API key and return its associated data"""
        if not api_key:
            return None

        hashed_key = self._hash_key(api_key)
        print(f"DEBUG VALIDATE: Looking for hashed key: {hashed_key[:20]}...")
        query = """
            SELECT id, user_id, organization_id, key_prefix, name, scopes, expires_at, last_used_at, created_at
            FROM api_keys
            WHERE key_hash = $1
        """
        result = await self.db.fetch_one(query, hashed_key)
        print(f"DEBUG VALIDATE: Query result: {result is not None}")
        if not result:
            return None

        # Parse scopes from JSON string if necessary
        result_dict = dict(result)
        if isinstance(result_dict["scopes"], str):
            try:
                result_dict["scopes"] = json.loads(result_dict["scopes"])
            except json.JSONDecodeError:
                result_dict["scopes"] = []

        # Check if key is expired
        expires_at = result_dict.get("expires_at")
        if expires_at:
            # Parse expires_at if it's a string
            if isinstance(expires_at, str):
                try:
                    # Handle different datetime formats from SQLite
                    if expires_at.endswith('Z'):
                        expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                    elif '+' in expires_at or expires_at.endswith('00:00'):
                        expires_at = datetime.fromisoformat(expires_at)
                    else:
                        # Assume UTC if no timezone info
                        expires_at = datetime.fromisoformat(expires_at).replace(tzinfo=timezone.utc)
                except ValueError:
                    # If parsing fails, assume it's not expired
                    expires_at = None
            elif hasattr(expires_at, 'replace') and expires_at.tzinfo is None:
                # If it's a datetime object without timezone, assume UTC
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            
            if expires_at and expires_at < datetime.now(timezone.utc):
                return None

        # Update last used timestamp (optional - might want to make this async)
        try:
            await self.db.execute(
                "UPDATE api_keys SET last_used_at = $1 WHERE id = $2",
                datetime.now(timezone.utc), result_dict["id"]
            )
        except Exception:
            # Don't fail validation if we can't update timestamp
            pass

        return APIKeyInfo(**result_dict)

    def _hash_key(self, key: str) -> str:
        """Hash the API key for secure storage"""
        return hashlib.sha256(key.encode()).hexdigest()

    async def create_pipeline_execution_key(self, app_id: str, pipeline_id: str, created_by: int, organization_id: int) -> Optional[str]:
        """
        Create a temporary execution key for pipeline execution.
        This key is single-use and tied to the execution.
        """
        try:
            import secrets

            # Generate a secure execution key
            execution_key = f"exec_{secrets.token_urlsafe(32)}"
            key_id = str(uuid.uuid4())

            # Set expiration to 1 hour from now
            expiration = datetime.now() + timedelta(hours=1)

            # Insert into execution_api_keys table
            insert_query = """
                INSERT INTO execution_api_keys (
                    key_id, app_id, organization_id, key_value, executor_type_id, executor_id,
                    created_by, expiration
                ) VALUES ($1, $2, $3, $4, 'pipeline', $5, $6, $7)
            """

            await self.db.execute(
                insert_query,
                key_id, app_id, organization_id, execution_key, pipeline_id, created_by, expiration
            )

            return execution_key

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to create pipeline execution key: {e}")
            return None

    async def create_agent_api_key(self, app_id: str, agent_id: str, created_by: int, organization_id: int) -> Optional[str]:
        """
        Get or create an API key for agent activation.
        Reuses existing active keys or creates new ones.
        """
        try:
            # First, try to get an existing active API key for this agent
            query = """
                SELECT key_value FROM agent_api_keys
                WHERE app_id = $1 AND agent_id = $2 AND is_active = 1 AND is_revoked = 0
                ORDER BY created_at DESC LIMIT 1
            """

            existing_key = await self.db.fetch_val(query, app_id, agent_id)
            if existing_key:
                return existing_key

            # No existing key found, create a new one
            import secrets

            # Generate a secure API key with agent_ prefix (compatible with auth middleware)
            api_key = f"agent_{secrets.token_urlsafe(32)}"

            # Insert the new API key
            insert_query = """
                INSERT INTO agent_api_keys (
                    key_id, app_id, agent_id, organization_id, key_value, is_active, is_revoked,
                    created_by, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, 1, 0, $6, NOW(), NOW())
            """

            key_id = str(uuid.uuid4())

            await self.db.execute(
                insert_query, key_id, app_id, agent_id, organization_id, api_key, created_by
            )

            return api_key

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to get or create agent API key: {e}")
            return None


# Dependency functions for web framework integration
async def get_api_key(api_key: str, db) -> Optional[APIKeyInfo]:
    """Dependency to validate API key and return key data"""
    auth = ApiKeyService(db)
    key_data = await auth.validate_api_key(api_key)
    return key_data


async def get_user_from_api_key(api_key: str, db) -> Optional[Dict[str, Any]]:
    """Dependency to get user from API key authentication"""
    auth = ApiKeyService(db)
    key_data = await auth.validate_api_key(api_key)
    
    if not key_data:
        return None
    
    # Return basic user info - web layer can adapt this to their User model
    return {
        "id": key_data.user_id,
        "user_id": key_data.user_id,
        "organization_id": key_data.organization_id
    }


async def log_api_key_usage(api_key_id: int, db, endpoint: str = "unknown", method: str = "unknown"):
    """Log API key usage for audit and tracking"""
    try:
        # Use transaction wrapper for consistency
        async with db.transaction():
            query = """
                INSERT INTO api_key_usage 
                (api_key_id, endpoint, method, timestamp)
                VALUES ($1, $2, $3, $4)
            """
            await db.execute(query, api_key_id, endpoint, method, datetime.utcnow())
            
            # Also update last_used_at on the key
            await db.execute(
                "UPDATE api_keys SET last_used_at = $1 WHERE id = $2",
                datetime.utcnow(), api_key_id
            )
    except Exception as e:
        # Don't fail the main operation if logging fails
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to log API key usage: {e}")


def require_scopes(required_scopes: List[str]):
    """
    Dependency factory to enforce API key scope requirements.
    Returns a function that can be used with dependency injection systems.
    Web frameworks should adapt this to their specific dependency system.
    """
    async def scope_validator(key_data: APIKeyInfo):
        if not key_data or not key_data.scopes:
            raise ValueError("API key has no scopes")
        
        if not any(scope in key_data.scopes for scope in required_scopes):
            raise PermissionError(f"API key missing required scopes: {required_scopes}")
        
        return key_data
    
    return scope_validator