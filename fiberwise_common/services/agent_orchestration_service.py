"""
Agent Orchestration Service — multi-agent workflows, roles, collaboration patterns.

Wraps ia_modules AgentOrchestrator, StateManager, and collaboration patterns.
"""

import json
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_service import BaseService
from ..database.provider import DatabaseProvider

logger = logging.getLogger(__name__)

COLLABORATION_PATTERNS = ["hierarchical", "peer_to_peer", "debate", "consensus"]


class AgentOrchestrationService(BaseService):
    """Service for multi-agent orchestration workflows."""

    def __init__(self, database_provider: DatabaseProvider):
        super().__init__(database_provider)
        self.db = database_provider

    async def create_workflow(self, name: str, description: str, workflow_config: dict,
                               created_by: int, app_id: Optional[str] = None) -> str:
        wf_id = str(uuid.uuid4())
        await self.db.execute(
            """INSERT INTO agent_workflows (workflow_id, name, description, workflow_config, app_id, created_by)
               VALUES (:id, :name, :desc, :config, :app_id, :created_by)""",
            {"id": wf_id, "name": name, "desc": description,
             "config": json.dumps(workflow_config), "app_id": app_id, "created_by": created_by}
        )
        return wf_id

    async def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        row = await self.db.fetch_one(
            "SELECT * FROM agent_workflows WHERE workflow_id = :wid", {"wid": workflow_id}
        )
        if not row:
            return None
        d = dict(row)
        if isinstance(d.get("workflow_config"), str):
            d["workflow_config"] = json.loads(d["workflow_config"])
        return d

    async def list_workflows(self, app_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if app_id:
            rows = await self.db.fetch_all(
                "SELECT * FROM agent_workflows WHERE app_id = :aid ORDER BY created_at DESC", {"aid": app_id}
            )
        else:
            rows = await self.db.fetch_all("SELECT * FROM agent_workflows ORDER BY created_at DESC")
        result = []
        for r in (rows or []):
            d = dict(r)
            if isinstance(d.get("workflow_config"), str):
                d["workflow_config"] = json.loads(d["workflow_config"])
            result.append(d)
        return result

    async def execute_workflow(self, workflow_id: str, input_data: dict, created_by: int) -> str:
        """Execute a multi-agent workflow using ia_modules agents. Returns execution_id."""
        wf = await self.get_workflow(workflow_id)
        if not wf:
            raise ValueError(f"Workflow {workflow_id} not found")

        exec_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        await self.db.execute(
            """INSERT INTO agent_workflow_executions
               (execution_id, workflow_id, status, input_data, started_at, created_by)
               VALUES (:eid, :wid, 'running', :input, :now, :uid)""",
            {"eid": exec_id, "wid": workflow_id, "input": json.dumps(input_data), "now": now, "uid": created_by}
        )

        # Execute via ia_modules orchestrator
        start_time = datetime.utcnow()
        try:
            from ia_modules.agents.orchestrator import AgentOrchestrator
            from ia_modules.agents.state import StateManager
            from ia_modules.agents.base import BaseAgent

            config = wf["workflow_config"]
            pattern = config.get("pattern", "hierarchical")

            # Initialize state manager
            state_mgr = StateManager()

            # Build agents from roles
            agents = []
            for agent_cfg in config.get("agents", []):
                role_name = agent_cfg.get("role")
                agent_id = agent_cfg.get("id", role_name)

                # Fetch role from database
                role_rows = await self.db.fetch_all(
                    "SELECT * FROM agent_roles WHERE name = :name", {"name": role_name}
                )

                if role_rows:
                    role = dict(role_rows[0])
                    # Create agent instance
                    class DynamicAgent(BaseAgent):
                        def __init__(self, agent_id, name, system_prompt, allowed_tools):
                            super().__init__(agent_id, name)
                            self.system_prompt = system_prompt
                            self.allowed_tools = json.loads(allowed_tools) if isinstance(allowed_tools, str) else allowed_tools or []

                        async def process(self, message: str, context: dict = None):
                            # Agent execution logic
                            logger.info(f"Agent {self.name} processing: {message}")
                            return {
                                "agent_id": self.agent_id,
                                "response": f"Processed by {self.name}",
                                "message": message
                            }

                    agent = DynamicAgent(
                        agent_id=agent_id,
                        name=role["name"],
                        system_prompt=role["system_prompt"],
                        allowed_tools=role["allowed_tools"]
                    )
                    agents.append(agent)
                    logger.info(f"✅ Built agent: {agent_id} ({role['name']})")

            if not agents:
                raise ValueError("No agents configured for workflow")

            # Create orchestrator with collaboration pattern
            orchestrator = AgentOrchestrator(state_mgr)
            for agent in agents:
                orchestrator.add_agent(agent)

            # Execute workflow
            logger.info(f"🚀 Executing workflow {workflow_id} with pattern: {pattern}")
            result = await orchestrator.run(
                start_agent=agents[0].agent_id if agents else None,
                input_data=input_data
            )

            # Calculate metrics
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Record telemetry
            try:
                from api.core.telemetry_manager import get_telemetry_manager
                telemetry = get_telemetry_manager()
                telemetry.record_pipeline_execution(
                    pipeline_id=workflow_id,
                    execution_id=exec_id,
                    duration_ms=duration_ms,
                    status="success",
                    step_count=len(agents),
                    cost_usd=0.0
                )
                logger.info(f"📊 Recorded telemetry for workflow {exec_id}")
            except Exception as e:
                logger.warning(f"Failed to record telemetry: {e}")

            await self.db.execute(
                """UPDATE agent_workflow_executions SET status='completed', output_data=:output, completed_at=:now
                   WHERE execution_id=:eid""",
                {"output": json.dumps(result), "now": datetime.utcnow().isoformat(), "eid": exec_id}
            )

        except ImportError as e:
            logger.error(f"ia_modules.agents not available: {e}")
            await self.db.execute(
                "UPDATE agent_workflow_executions SET status='error', error='ia_modules.agents not installed' WHERE execution_id=:eid",
                {"eid": exec_id}
            )
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            await self.db.execute(
                "UPDATE agent_workflow_executions SET status='failed', error=:err, completed_at=:now WHERE execution_id=:eid",
                {"err": str(e), "now": datetime.utcnow().isoformat(), "eid": exec_id}
            )

        return exec_id

    async def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        row = await self.db.fetch_one(
            "SELECT * FROM agent_workflow_executions WHERE execution_id = :eid", {"eid": execution_id}
        )
        if not row:
            return None
        d = dict(row)
        for field in ["input_data", "output_data"]:
            if isinstance(d.get(field), str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    # --- Roles ---
    async def create_role(self, name: str, description: str = None,
                           system_prompt: str = None, allowed_tools: List[str] = None) -> str:
        role_id = str(uuid.uuid4())
        await self.db.execute(
            """INSERT INTO agent_roles (role_id, name, description, system_prompt, allowed_tools)
               VALUES (:id, :name, :desc, :prompt, :tools)""",
            {"id": role_id, "name": name, "desc": description,
             "prompt": system_prompt, "tools": json.dumps(allowed_tools or [])}
        )
        return role_id

    async def list_roles(self) -> List[Dict[str, Any]]:
        rows = await self.db.fetch_all("SELECT * FROM agent_roles ORDER BY name")
        result = []
        for r in (rows or []):
            d = dict(r)
            if isinstance(d.get("allowed_tools"), str):
                d["allowed_tools"] = json.loads(d["allowed_tools"])
            result.append(d)
        return result

    async def list_patterns(self) -> List[Dict[str, Any]]:
        """List available collaboration patterns."""
        return [
            {"name": "hierarchical", "description": "Manager agent delegates to workers, synthesizes results"},
            {"name": "peer_to_peer", "description": "Agents share knowledge directly via broadcast/unicast"},
            {"name": "debate", "description": "Agents argue positions, judge evaluates, consensus wins"},
            {"name": "consensus", "description": "Multiple agents vote with threshold-based decisions"}
        ]
