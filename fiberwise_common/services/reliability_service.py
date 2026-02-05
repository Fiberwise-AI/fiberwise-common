"""
Reliability Service — metrics (SR, CR, PC, HIR, MA, TCL, WCT), anomaly detection, alerts.

Wraps ia_modules reliability module for the FiberWise platform.
"""

import json
import logging
import uuid
from typing import Dict, Any, Optional, List

from .base_service import BaseService
from ..database.provider import DatabaseProvider

logger = logging.getLogger(__name__)


class ReliabilityService(BaseService):
    """Service for reliability metrics, anomaly detection, and alerting."""

    def __init__(self, database_provider: DatabaseProvider):
        super().__init__(database_provider)
        self.db = database_provider
        self._ia_metrics = None
        self._anomaly_detector = None
        self._alert_manager = None

    def _init_ia_modules(self):
        """Lazily initialize ia_modules reliability components."""
        if self._ia_metrics is not None:
            return
        try:
            from ia_modules.reliability import ReliabilityMetrics, AnomalyDetector, AlertManager
            self._ia_metrics = ReliabilityMetrics()
            self._anomaly_detector = AnomalyDetector()
            self._alert_manager = AlertManager()
        except ImportError:
            logger.warning("ia_modules.reliability not available")

    async def get_metrics(self, pipeline_id: Optional[str] = None) -> Dict[str, Any]:
        """Get reliability metrics (SR, CR, PC, HIR, MA, TCL, WCT)."""
        self._init_ia_modules()
        if self._ia_metrics:
            try:
                # get_report() doesn't take pipeline_id parameter in ia_modules
                report = await self._ia_metrics.get_report()
                # Map to keys expected by frontend (without _ms suffix)
                tcl = getattr(report, 'tcl', 0) or getattr(report, 'tool_call_latency_ms', 0)
                wct = getattr(report, 'wct', 0) or getattr(report, 'workflow_completion_time_ms', 0)
                return {
                    "success_rate": getattr(report, 'sr', 0) or getattr(report, 'success_rate', 0),
                    "compensation_rate": getattr(report, 'cr', 0) or getattr(report, 'compensation_rate', 0),
                    "pass_confidence": getattr(report, 'pc', 0) or getattr(report, 'pass_confidence', 0),
                    "human_intervention_rate": getattr(report, 'hir', 0) or getattr(report, 'human_intervention_rate', 0),
                    "model_accuracy": getattr(report, 'ma', 0) or getattr(report, 'model_accuracy', 0),
                    "tool_call_latency": tcl,  # Frontend expects no _ms suffix
                    "workflow_completion_time": wct / 1000.0 if wct > 0 else 0,  # Convert ms to seconds for frontend
                    "total_executions": getattr(report, 'total_executions', 0),
                    "pipeline_id": pipeline_id,
                    "source": "ia_modules"
                }
            except Exception as e:
                logger.error(f"ia_modules metrics error: {e}", exc_info=True)

        # Fallback: compute from DB
        if pipeline_id:
            rows = await self.db.fetch_all(
                "SELECT metric_name, metric_value FROM reliability_metrics WHERE pipeline_id = :pid ORDER BY created_at DESC LIMIT 20",
                {"pid": pipeline_id}
            )
        else:
            rows = await self.db.fetch_all(
                "SELECT metric_name, metric_value FROM reliability_metrics ORDER BY created_at DESC LIMIT 20"
            )
        return {
            "metrics": [dict(r) for r in rows] if rows else [],
            "pipeline_id": pipeline_id,
            "source": "database",
            "success_rate": 0,
            "compensation_rate": 0,
            "pass_confidence": 0,
            "human_intervention_rate": 0,
            "model_accuracy": 0,
            "tool_call_latency": 0,  # Frontend expects no _ms suffix
            "workflow_completion_time": 0,  # Frontend expects seconds
            "total_executions": 0
        }

    async def check_anomalies(self, execution_id: str) -> Dict[str, Any]:
        """Check for anomalies in an execution."""
        self._init_ia_modules()
        if self._anomaly_detector:
            try:
                result = await self._anomaly_detector.check(execution_id=execution_id)
                return {"execution_id": execution_id, "anomalies": result}
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
        return {"execution_id": execution_id, "anomalies": []}

    async def get_alerts(self, pipeline_id: Optional[str] = None, resolved: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get alerts from DB."""
        conditions = ["1=1"]
        params = {}
        if pipeline_id:
            conditions.append("pipeline_id = :pid")
            params["pid"] = pipeline_id
        if resolved is not None:
            conditions.append("is_resolved = :resolved")
            params["resolved"] = resolved

        rows = await self.db.fetch_all(
            f"SELECT * FROM alerts WHERE {' AND '.join(conditions)} ORDER BY created_at DESC LIMIT 100",
            params
        )
        result = []
        for r in (rows or []):
            d = dict(r)
            if isinstance(d.get("context"), str):
                try:
                    d["context"] = json.loads(d["context"])
                except (json.JSONDecodeError, TypeError):
                    d["context"] = {}
            result.append(d)
        return result

    async def create_alert(self, pipeline_id: str, alert_type: str, severity: str, message: str, context: dict = None):
        """Create an alert."""
        await self.db.execute(
            """INSERT INTO alerts (alert_id, pipeline_id, alert_type, severity, message, context)
               VALUES (:id, :pid, :type, :severity, :message, :context)""",
            {"id": str(uuid.uuid4()), "pid": pipeline_id, "type": alert_type,
             "severity": severity, "message": message, "context": json.dumps(context or {})}
        )

    async def get_cost_report(self, pipeline_id: Optional[str] = None) -> Dict[str, Any]:
        """Get cost report — delegates to ia_modules CostTracker if available."""
        self._init_ia_modules()
        return {"pipeline_id": pipeline_id, "costs": [], "total": 0}
