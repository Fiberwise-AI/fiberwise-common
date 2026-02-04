"""
Checkpoint Service — pipeline execution state snapshots and recovery.

Wraps ia_modules checkpoint backends for the FiberWise platform.
"""

import json
import logging
from typing import Dict, Any, Optional, List

from .base_service import BaseService
from ..database.provider import DatabaseProvider

logger = logging.getLogger(__name__)


class CheckpointService(BaseService):
    """Service for pipeline checkpoint management."""

    def __init__(self, database_provider: DatabaseProvider):
        super().__init__(database_provider)
        self.db = database_provider

    async def list_checkpoints(self, execution_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a pipeline execution."""
        rows = await self.db.fetch_all(
            """SELECT checkpoint_id, thread_id, pipeline_id, step_id, step_index, step_name,
                      timestamp, status, parent_checkpoint_id, metadata
               FROM pipeline_checkpoints WHERE pipeline_id = :eid
               ORDER BY step_index ASC""",
            {"eid": execution_id}
        )
        return [self._row_to_dict(r) for r in rows] if rows else []

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific checkpoint."""
        row = await self.db.fetch_one(
            "SELECT * FROM pipeline_checkpoints WHERE checkpoint_id = :cid",
            {"cid": checkpoint_id}
        )
        if not row:
            return None
        return self._row_to_dict(row, include_state=True)

    async def get_checkpoint_state(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get the state data from a checkpoint."""
        row = await self.db.fetch_one(
            "SELECT state FROM pipeline_checkpoints WHERE checkpoint_id = :cid",
            {"cid": checkpoint_id}
        )
        if not row:
            return None
        state = row["state"]
        if isinstance(state, str):
            return json.loads(state)
        return state

    async def resume_from_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Resume pipeline execution from a checkpoint. Returns info for the caller to re-execute."""
        cp = await self.get_checkpoint(checkpoint_id)
        if not cp:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        state = await self.get_checkpoint_state(checkpoint_id)
        return {
            "checkpoint_id": checkpoint_id,
            "pipeline_id": cp.get("pipeline_id"),
            "resume_from_step": cp.get("step_id"),
            "step_index": cp.get("step_index"),
            "state": state
        }

    def _row_to_dict(self, row, include_state=False) -> Dict[str, Any]:
        d = dict(row)
        if "metadata" in d and isinstance(d["metadata"], str):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = {}
        if not include_state and "state" in d:
            d.pop("state", None)
        elif include_state and "state" in d and isinstance(d["state"], str):
            try:
                d["state"] = json.loads(d["state"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d
