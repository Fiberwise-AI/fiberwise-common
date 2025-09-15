"""
Agent Version Service - Functions for managing agent versions
"""
import logging
from uuid import UUID, uuid4
import yaml
from fiberwise_common import DatabaseProvider
from fiberwise_common.entities import AgentManifest
from .base_service import BaseService

logger = logging.getLogger(__name__)

class AgentVersionService(BaseService):
    def __init__(self, db: DatabaseProvider):
        super().__init__(db)

    async def create_agent_version(self, agent_id: UUID, version: str, description: str = None, manifest=None, created_by=None, file_path: str = None) -> UUID:
        """
        Create a new agent version
        
        Args:
            agent_id: UUID of the agent
            version: Version string
            description: Optional description
            manifest: Optional manifest data
            created_by: ID of the user creating the version
            file_path: Optional file path for the agent implementation
            
        Returns:
            UUID of the new agent version
        """
        # Generate a new UUID for the version
        version_id = uuid4()
        
        # Prepare manifest YAML if provided
        manifest_yaml = None
        if manifest:
            if hasattr(manifest, 'model_dump'):
                manifest_dict = manifest.model_dump()
            elif hasattr(manifest, 'dict'):
                manifest_dict = manifest.dict()
            else:
                manifest_dict = dict(manifest)
            manifest_yaml = yaml.dump(manifest_dict)
        
        # First check if there's already ANY version for this agent with the same version number
        check_version_query = """
            SELECT version_id FROM agent_versions
            WHERE agent_id = $1 AND version = $2
            ORDER BY created_at DESC
            LIMIT 1
        """

        existing_version_id = await self.db.fetch_val(check_version_query, str(agent_id), version)

        logger.info(f"AgentVersionService: Checking for existing draft version for agent {agent_id} version {version}: found={existing_version_id}")

        if existing_version_id:
            # Return the existing version ID without creating a new one
            logger.info(f"AgentVersionService: Reusing existing version {version} with ID {existing_version_id} for agent {agent_id}")
            return existing_version_id
        else:
            # Insert version record with file_path if provided
            insert_query = """
                INSERT INTO agent_versions (
                    version_id, agent_id, version, file_path,
                    status, is_active, created_by
                ) VALUES (
                    $1, $2, $3, $4, 'draft', false, $5
                ) RETURNING version_id
            """

            version_id = await self.db.fetch_val(
                insert_query,
                str(version_id),
                str(agent_id),
                version,
                file_path,
                str(created_by) if created_by else None
            )
        
        logger.info(f"Created new version {version} with ID {version_id} for agent {agent_id}")
        return version_id

    async def get_latest_agent_version(self, agent_id: UUID) -> dict:
        """
        Get the latest version for an agent
        
        Args:
            agent_id: UUID of the agent
            
        Returns:
            Dict with version info or None if not found
        """
        query = """
            SELECT version_id, version, status, is_active, created_at
            FROM agent_versions
            WHERE agent_id = $1
            ORDER BY created_at DESC
            LIMIT 1
        """
        version_record = await self._fetch_one(query, (str(agent_id),))
        
        if version_record:
            return dict(version_record)
        return None

    async def activate_agent_version(self, version_id: UUID) -> bool:
        """
        Activate an agent version
        
        Args:
            version_id: UUID of the version to activate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First get the agent_id for this version
            agent_query = "SELECT agent_id FROM agent_versions WHERE version_id = $1"
            agent_id = await self.db.fetch_val(agent_query, str(version_id))
            
            if not agent_id:
                logger.error(f"No agent found for version {version_id}")
                return False
            
            # Deactivate all versions for this agent
            deactivate_query = """
                UPDATE agent_versions
                SET is_active = false
                WHERE agent_id = $1
            """
            await self._execute_query(deactivate_query, (str(agent_id),))
            
            # Activate the specified version
            activate_query = """
                UPDATE agent_versions
                SET is_active = true, status = 'active'
                WHERE version_id = $1
            """
            await self._execute_query(activate_query, (str(version_id),))
            
            logger.info(f"Activated agent version {version_id}")
            return True
        except Exception as e:
            logger.error(f"Error activating agent version {version_id}: {str(e)}")
            return False
