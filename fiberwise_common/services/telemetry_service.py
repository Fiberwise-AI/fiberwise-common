"""
Telemetry Service — execution spans, metrics, and timeline.

Wraps ia_modules telemetry (OpenTelemetry/Prometheus) and provides
DB-backed historical metrics for the FiberWise platform.
"""

import json
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from .base_service import BaseService
from ..database.provider import DatabaseProvider

logger = logging.getLogger(__name__)


class TelemetryService(BaseService):
    """Service for pipeline execution telemetry and metrics."""

    def __init__(self, database_provider: DatabaseProvider, tracer=None):
        super().__init__(database_provider)
        self.db = database_provider
        self.tracer = tracer  # ia_modules SimpleTracer or OpenTelemetryTracer

    async def record_step_metrics(self, execution_id: str, pipeline_id: str, step_id: str, metrics: Dict[str, Any]):
        """Record metrics for a pipeline step execution."""
        for name, value in metrics.items():
            await self.db.execute(
                """INSERT INTO execution_metrics (id, execution_id, pipeline_id, step_id, metric_name, metric_value)
                   VALUES (:id, :execution_id, :pipeline_id, :step_id, :name, :value)""",
                {"id": str(uuid.uuid4()), "execution_id": execution_id, "pipeline_id": pipeline_id,
                 "step_id": step_id, "name": name, "value": float(value)}
            )

    async def get_execution_metrics(self, execution_id: str) -> Dict[str, Any]:
        """Get aggregated metrics for an execution."""
        rows = await self.db.fetch_all(
            "SELECT * FROM execution_metrics WHERE execution_id = :eid ORDER BY recorded_at",
            {"eid": execution_id}
        )
        if not rows:
            return {"total_metrics": 0, "metrics": []}

        metrics = [dict(r) for r in rows]
        return {
            "execution_id": execution_id,
            "total_metrics": len(metrics),
            "metrics": metrics
        }

    async def get_execution_spans(self, execution_id: str) -> List[Dict[str, Any]]:
        """Get telemetry spans from ia_modules tracer for an execution."""
        if not self.tracer:
            return []
        try:
            all_spans = self.tracer.get_spans()
            return [
                self._span_to_dict(s) for s in all_spans
                if getattr(s, 'attributes', {}).get('execution_id') == execution_id
                or (isinstance(s, dict) and s.get('attributes', {}).get('execution_id') == execution_id)
            ]
        except Exception as e:
            logger.error(f"Error getting spans: {e}")
            return []

    async def get_span_timeline(self, execution_id: str) -> List[Dict[str, Any]]:
        """Get spans formatted for timeline visualization."""
        spans = await self.get_execution_spans(execution_id)
        return [
            {
                "span_id": s.get("span_id"),
                "parent_id": s.get("parent_id"),
                "name": s.get("name"),
                "start_time": s.get("start_time"),
                "end_time": s.get("end_time"),
                "duration_ms": s.get("duration_ms", 0),
                "status": s.get("status", "ok"),
                "attributes": s.get("attributes", {})
            }
            for s in spans
        ]

    async def get_pipeline_metrics(self, pipeline_id: str) -> Dict[str, Any]:
        """Get aggregated metrics for a pipeline across all executions."""
        rows = await self.db.fetch_all(
            """SELECT metric_name, COUNT(*) as count, AVG(metric_value) as avg_val,
                      MIN(metric_value) as min_val, MAX(metric_value) as max_val
               FROM execution_metrics WHERE pipeline_id = :pid
               GROUP BY metric_name""",
            {"pid": pipeline_id}
        )
        return {
            "pipeline_id": pipeline_id,
            "metrics": [dict(r) for r in rows] if rows else []
        }

    def _span_to_dict(self, span) -> Dict[str, Any]:
        if isinstance(span, dict):
            return span
        return {
            "span_id": getattr(span, "span_id", None),
            "parent_id": getattr(span, "parent_id", None),
            "name": getattr(span, "name", "unknown"),
            "start_time": self._format_ts(getattr(span, "start_time", None)),
            "end_time": self._format_ts(getattr(span, "end_time", None)),
            "duration_ms": getattr(span, "duration_ms", None),
            "attributes": getattr(span, "attributes", {}),
            "status": getattr(span, "status", "ok")
        }

    def _format_ts(self, ts):
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts.isoformat()
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        return str(ts)
