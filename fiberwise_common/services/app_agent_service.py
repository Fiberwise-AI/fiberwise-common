from typing import Dict, List, Any, Optional
import logging
import uuid
import json
from datetime import datetime

from ..schemas import AgentResponse

from fiberwise_common import DatabaseProvider

logger = logging.getLogger(__name__)

class AppAgentService:
    """
    Service for managing AI agents and their activations with app.
    Provides methods for registering, configuring, and activating agents.
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
    ) -> List[AgentResponse]:
        """Get all agents for a specific app with filtering"""
        try:
            # Build query with proper parameters
            query_parts = ["SELECT * FROM agents WHERE app_id = $1"]
            params = [app_id]
            param_idx = 2
            
            # Add optional filters
            if agent_type_id:
                query_parts.append(f"AND agent_type_id = ${param_idx}")
                params.append(agent_type_id)
                param_idx += 1
                
            if status:
                is_active = status.lower() == 'enabled'
                query_parts.append(f"AND is_active = ${param_idx}")
                params.append(is_active)
                param_idx += 1
                
            # Add order and pagination
            query_parts.append("ORDER BY name ASC")
            query_parts.append(f"LIMIT ${param_idx} OFFSET ${param_idx + 1}")
            params.extend([limit, offset])
            
            # Execute query
            query = " ".join(query_parts)
            agent_records = await self.db.fetch_all(query, *params)
            
            # Convert to response models, ensuring created_by is a string
            agents = []
            for record in agent_records:
                agent_dict = dict(record)
                
                # Convert created_by to string if it's not already
                if agent_dict.get('created_by') is not None:
                    agent_dict['created_by'] = str(agent_dict['created_by'])
                    
                # Handle JSON fields
                for field in ['configuration', 'parameters', 'config', 'metadata']:
                    if field in agent_dict and isinstance(agent_dict[field], str):
                        try:
                            agent_dict[field] = json.loads(agent_dict[field])
                        except (json.JSONDecodeError, TypeError):
                            # If parsing fails, keep it as a string or set to default
                            agent_dict[field] = {} if agent_dict[field] else None
                
                agents.append(AgentResponse(**agent_dict))
                
            return agents
        except Exception as e:
            logger.error(f"Error retrieving agents: {str(e)}")
            raise
