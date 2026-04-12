from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime

from fiberwise_common import DatabaseProvider

logger = logging.getLogger(__name__)


class AppAgentService:
    """
    Service for managing AI agents within an app context.
    Provides filtered queries for agents belonging to a specific app.
    """
    def __init__(self, db: DatabaseProvider):
        self.db = db

    async def get_agents_by_app(
        self,
        app_id: str,
        user_id: int,
        agent_type_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get all agents for a specific app with filtering"""
        try:
            query_parts = ["""
                SELECT agent_id, app_id, name, description, agent_type_id, config,
                       metadata, is_active, is_enabled, created_by, created_at, updated_at
                FROM agents WHERE app_id = :app_id
            """]
            params = {"app_id": app_id}

            if agent_type_id:
                query_parts.append("AND agent_type_id = :agent_type_id")
                params["agent_type_id"] = agent_type_id

            if status:
                params["is_active"] = status.lower() == 'enabled'
                query_parts.append("AND is_active = :is_active")

            query_parts.append("ORDER BY name ASC")
            query_parts.append("LIMIT :limit OFFSET :offset")
            params["limit"] = limit
            params["offset"] = offset

            query = " ".join(query_parts)
            agent_records = await self.db.fetch_all(query, params)

            result = []
            for record in agent_records:
                agent_dict = dict(record)

                if agent_dict.get('created_by') is not None:
                    agent_dict['created_by'] = str(agent_dict['created_by'])

                # Parse JSON text fields, ensure result is always a dict
                for field in ['config', 'metadata']:
                    val = agent_dict.get(field)
                    if isinstance(val, str):
                        try:
                            val = json.loads(val)
                        except (json.JSONDecodeError, TypeError):
                            val = {}
                    agent_dict[field] = val if isinstance(val, dict) else {}

                # Parse timestamp strings to datetime objects
                for field in ['created_at', 'updated_at']:
                    val = agent_dict.get(field)
                    if isinstance(val, str):
                        try:
                            agent_dict[field] = datetime.fromisoformat(val)
                        except (ValueError, TypeError):
                            pass

                # Frontend (app-agents-tab) expects `id` — keep agent_id as
                # well for any callers that read the raw column name.
                if 'agent_id' in agent_dict and 'id' not in agent_dict:
                    agent_dict['id'] = str(agent_dict['agent_id'])

                result.append(agent_dict)

            return result
        except Exception as e:
            logger.error(f"Error retrieving agents: {str(e)}")
            raise
