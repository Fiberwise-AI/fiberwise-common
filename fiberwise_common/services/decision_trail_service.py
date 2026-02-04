"""
Decision Trail Service — execution audit trail with decision nodes and edges.

Wraps ia_modules DecisionTrailBuilder for the FiberWise platform.
"""

import json
import logging
from typing import Dict, Any, Optional, List

from .base_service import BaseService
from ..database.provider import DatabaseProvider

logger = logging.getLogger(__name__)


class DecisionTrailService(BaseService):
    """Service for managing decision audit trails."""

    def __init__(self, database_provider: DatabaseProvider):
        super().__init__(database_provider)
        self.db = database_provider

    async def get_decision_trail(self, execution_id: str) -> Dict[str, Any]:
        """Get complete decision trail for an execution."""
        nodes = await self.db.fetch_all(
            "SELECT * FROM decision_trail_nodes WHERE execution_id = :eid ORDER BY created_at",
            {"eid": execution_id}
        )
        edges = await self.db.fetch_all(
            "SELECT * FROM decision_trail_edges WHERE execution_id = :eid",
            {"eid": execution_id}
        )
        node_list = [self._parse_node(r) for r in nodes] if nodes else []
        edge_list = [dict(r) for r in edges] if edges else []

        decision_nodes = [n for n in node_list if n.get("decision_type") == "decision"]
        return {
            "execution_id": execution_id,
            "nodes": node_list,
            "edges": edge_list,
            "statistics": {
                "total_nodes": len(node_list),
                "decision_points": len(decision_nodes),
                "total_edges": len(edge_list),
                "average_confidence": sum(n.get("confidence", 0) for n in decision_nodes) / max(len(decision_nodes), 1)
            }
        }

    async def get_decision_node(self, execution_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        row = await self.db.fetch_one(
            "SELECT * FROM decision_trail_nodes WHERE node_id = :nid AND execution_id = :eid",
            {"nid": node_id, "eid": execution_id}
        )
        return self._parse_node(row) if row else None

    async def get_execution_path(self, execution_id: str) -> List[Dict[str, Any]]:
        """Get ordered decision path."""
        nodes = await self.db.fetch_all(
            "SELECT * FROM decision_trail_nodes WHERE execution_id = :eid ORDER BY created_at",
            {"eid": execution_id}
        )
        return [
            {"step": idx + 1, "node_id": n["node_id"], "decision_type": n.get("decision_type"),
             "decision": n.get("decision"), "created_at": n.get("created_at")}
            for idx, n in enumerate(nodes or [])
        ]

    async def export_trail(self, execution_id: str, fmt: str = "json") -> Any:
        trail = await self.get_decision_trail(execution_id)
        if fmt == "json":
            return trail
        elif fmt == "mermaid":
            lines = ["graph LR"]
            for n in trail["nodes"]:
                lines.append(f'  {n["node_id"]}["{n.get("decision", n["node_id"])}"]')
            for e in trail["edges"]:
                label = e.get("label", "")
                if label:
                    lines.append(f'  {e["from_node"]} -->|{label}| {e["to_node"]}')
                else:
                    lines.append(f'  {e["from_node"]} --> {e["to_node"]}')
            return "\n".join(lines)
        return trail

    def _parse_node(self, row) -> Dict[str, Any]:
        d = dict(row)
        for field in ["evidence", "alternatives"]:
            if field in d and isinstance(d[field], str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    d[field] = []
        return d
