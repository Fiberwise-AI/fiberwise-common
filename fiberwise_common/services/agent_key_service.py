"""
Service for managing agent API keys for programmatic access to the platform.
"""

import uuid
import secrets
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

from .base_service import BaseService
from ..database.query_adapter import QueryAdapter, ParameterStyle

logger = logging.getLogger(__name__)

class AgentKeyService(BaseService):
    """Service for managing agent API keys"""
    
    def __init__(self, db_provider):
        super().__init__(db_provider)
        self.query_adapter = QueryAdapter(ParameterStyle.SQLITE)

    async def create_agent_key(
        self,
        app_id: str,
        agent_id: str,
        description: str,
        scopes: List[str],
        expiration_hours: int,
        resource_pattern: str,
        created_by: int,
        metadata: Dict[str, Any],
        organization_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Create a new agent API key
        
        Args:
            app_id: The app ID this key is associated with
            agent_id: Agent ID this key is associated with
            description: Description for this key
            scopes: List of permission scopes (e.g., ['data:read', 'data:write'])
            expiration_hours: Number of hours until key expires
            resource_pattern: Resource pattern for limiting access (e.g., 'apps/{app_id}/*')
            created_by: User ID of the key creator
            metadata: Additional metadata to store with the key
            organization_id: Organization ID (auto-resolved from user if not provided)
            
        Returns:
            The generated API key value if successful, None otherwise
        """
        try:
            # Generate a unique key
            key_id = str(uuid.uuid4())
            key_value = f"agent_{secrets.token_urlsafe(32)}"
            
            # Set expiration
            expiration = datetime.now(timezone.utc) + timedelta(hours=expiration_hours)
            
            # Auto-resolve organization_id from user if not provided
            if not organization_id and created_by:
                org_query = """
                    SELECT organization_id FROM organization_members 
                    WHERE user_id = $1 AND status = 'active'
                    ORDER BY role = 'owner' DESC, role = 'admin' DESC 
                    LIMIT 1
                """
                org_result = await self._fetch_one(org_query, (created_by,))
                if org_result:
                    organization_id = org_result['organization_id']
                    logger.info(f"🏢 Auto-resolved organization {organization_id} for user {created_by}")
            
            # Convert UUIDs to strings in metadata to ensure serializability
            if metadata:
                serializable_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, uuid.UUID):
                        serializable_metadata[key] = str(value)
                    else:
                        serializable_metadata[key] = value
                metadata = serializable_metadata
                
            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata) if metadata else '{}'
            
            # Ensure app_id and agent_id are strings
            app_id = str(app_id) if app_id else None
            agent_id = str(agent_id) if agent_id else None
            
            # Create query - handle both SQLite and PostgreSQL syntax
            if hasattr(self.db, 'db_type') and self.db.db_type == 'postgresql':
                query = """
                    INSERT INTO agent_api_keys (
                        key_id, key_value, app_id, agent_id, description, 
                        scopes, expiration, resource_pattern, created_by, metadata, organization_id
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                    ) RETURNING key_value
                """
                result = await self.db.fetch_val(
                    query, 
                    key_id, key_value, app_id, agent_id, description,
                    scopes, expiration, resource_pattern, created_by, metadata_json, organization_id
                )
            else:
                # SQLite syntax
                query = """
                    INSERT INTO agent_api_keys (
                        key_id, key_value, app_id, agent_id, description, 
                        scopes, expiration, resource_pattern, created_by, metadata, organization_id
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """
                await self._execute_query(
                    query, 
                    (key_id, key_value, app_id, agent_id, description,
                     json.dumps(scopes), expiration.isoformat(), resource_pattern, created_by, metadata_json, organization_id)
                )
                result = key_value
            
            logger.info(f"Created agent API key {key_id} for app {app_id}, agent {agent_id}, org {organization_id}")
            return result
        except Exception as e:
            logger.error(f"Error creating agent API key: {e}")
            return None
    
    async def validate_agent_key(self, agent_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate an agent API key and return its metadata if valid
        
        Args:
            agent_key: The agent API key to validate (including 'agent_' prefix)
            
        Returns:
            Dictionary with key metadata if valid, None otherwise
        """
        if not agent_key or not agent_key.startswith("agent_"):
            logger.warning("Invalid agent key format (must start with 'agent_')")
            return None
            
        try:
            # Handle both SQLite and PostgreSQL syntax
            if hasattr(self.db, 'db_type') and self.db.db_type == 'postgresql':
                query = """
                    SELECT 
                        key_id, app_id, agent_id, organization_id, scopes, expiration,
                        resource_pattern, created_by
                    FROM agent_api_keys
                    WHERE key_value = $1 AND is_active = true AND 
                        (expiration IS NULL OR expiration > NOW())
                """
                result = await self._fetch_one(query, (agent_key,))
            else:
                # SQLite syntax
                query = """
                    SELECT 
                        key_id, app_id, agent_id, organization_id, scopes, expiration,
                        resource_pattern, created_by
                    FROM agent_api_keys
                    WHERE key_value = ? AND is_active = 1 AND 
                        (expiration IS NULL OR datetime(expiration) > datetime('now'))
                """
                result = await self._fetch_one(query, (agent_key,))
            
            if not result:
                logger.warning(f"Agent API key not found or inactive/expired: {agent_key[:10]}...")
                return None
                
            # Convert to dictionary with string keys
            key_info = dict(result)
            
            # Parse scopes from JSON if it's a string
            if isinstance(key_info.get('scopes'), str):
                try:
                    key_info['scopes'] = json.loads(key_info['scopes'])
                except json.JSONDecodeError:
                    # Handle PostgreSQL array format {item1,item2}
                    if key_info['scopes'].startswith('{') and key_info['scopes'].endswith('}'):
                        items = key_info['scopes'][1:-1].split(',')
                        key_info['scopes'] = [item.strip() for item in items if item.strip()]
                    else:
                        key_info['scopes'] = [key_info['scopes']]
            elif not isinstance(key_info.get('scopes'), list):
                key_info['scopes'] = []
                
            logger.info(f"Successfully validated agent API key {key_info['key_id']}")
            return key_info
        except Exception as e:
            logger.error(f"Error validating agent API key: {e}")
            return None
    
    async def revoke_agent_key(self, key_id: str) -> bool:
        """
        Revoke an agent API key
        
        Args:
            key_id: ID of the key to revoke
            
        Returns:
            True if revoked successfully, False otherwise
        """
        try:
            # Handle both SQLite and PostgreSQL syntax
            if hasattr(self.db, 'db_type') and self.db.db_type == 'postgresql':
                query = """
                    UPDATE agent_api_keys
                    SET is_active = false, revoked_at = NOW()
                    WHERE key_id = $1
                    RETURNING key_id
                """
                result = await self.db.fetch_val(query, key_id)
            else:
                # SQLite syntax
                query = """
                    UPDATE agent_api_keys
                    SET is_active = 0, revoked_at = datetime('now')
                    WHERE key_id = ?
                """
                await self._execute_query(query, (key_id,))
                result = key_id
            
            success = result is not None
            if success:
                logger.info(f"Revoked agent API key {key_id}")
            else:
                logger.warning(f"Failed to revoke agent API key {key_id}: Key not found")
                
            return success
        except Exception as e:
            logger.error(f"Error revoking agent API key: {e}")
            return False
    
    async def get_agent_keys(
        self,
        app_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        created_by: Optional[int] = None,
        include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get agent API keys with optional filtering
        
        Args:
            app_id: Optional app ID to filter by
            agent_id: Optional agent ID to filter by
            created_by: Optional user ID of the creator to filter by
            include_inactive: Whether to include inactive or revoked keys
            
        Returns:
            List of key records matching the filter criteria
        """
        try:
            filters = []
            params = []
            
            # Build WHERE clause with filters
            if app_id:
                filters.append("app_id = ?")
                params.append(app_id)
                
            if agent_id:
                filters.append("agent_id = ?")
                params.append(agent_id)
                
            if created_by:
                filters.append("created_by = ?")
                params.append(created_by)
                
            if not include_inactive:
                filters.append("is_active = 1" if not (hasattr(self.db, 'db_type') and self.db.db_type == 'postgresql') else "is_active = true")
                
            # Construct query
            query = """
                SELECT 
                    key_id, app_id, agent_id, organization_id, description, scopes,
                    expiration, resource_pattern, is_active, 
                    created_at, revoked_at, created_by
                FROM agent_api_keys
            """
            
            if filters:
                query += " WHERE " + " AND ".join(filters)
                
            query += " ORDER BY created_at DESC"
            
            # Execute query
            results = await self._fetch_all(query, tuple(params))
            
            # Convert to list of dictionaries and parse JSON fields
            keys = []
            for row in results:
                key_dict = dict(row)
                # Parse scopes from JSON if needed
                if isinstance(key_dict.get('scopes'), str):
                    try:
                        key_dict['scopes'] = json.loads(key_dict['scopes'])
                    except json.JSONDecodeError:
                        key_dict['scopes'] = []
                keys.append(key_dict)
            
            logger.info(f"Retrieved {len(keys)} agent API keys")
            return keys
        except Exception as e:
            logger.error(f"Error getting agent API keys: {e}")
            return []

    async def create_app_agent_key(self, app_id: str, agent_id: str, created_by: int = 1) -> Optional[str]:
        """
        Create a standard agent key for an app/agent pair with standard scopes.
        
        Args:
            app_id: The app ID
            agent_id: The agent ID  
            created_by: User ID creating the key (defaults to 1)
            
        Returns:
            The generated API key value if successful, None otherwise
        """
        standard_scopes = [
            'data:read',
            'data:write', 
            'activations:read',
            'agents:read'
        ]
        
        return await self.create_agent_key(
            app_id=app_id,
            agent_id=agent_id,
            description=f"Standard agent key for {agent_id}",
            scopes=standard_scopes,
            expiration_hours=8760,  # 1 year
            resource_pattern=f"apps/{app_id}/*",
            created_by=created_by,
            metadata={
                'auto_generated': True,
                'key_type': 'standard_agent_key'
            }
        )
