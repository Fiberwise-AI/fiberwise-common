"""
Pipeline Scheduler Service — cron and interval-based pipeline execution.

Wraps ia_modules scheduler with DB-backed job persistence.
"""

import json
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_service import BaseService
from ..database.provider import DatabaseProvider

logger = logging.getLogger(__name__)


class SchedulerService(BaseService):
    """Service for scheduling pipeline executions via cron or interval triggers."""

    def __init__(self, database_provider: DatabaseProvider):
        super().__init__(database_provider)
        self.db = database_provider
        self._scheduler = None
        self._job_functions = {}

    async def _ensure_scheduler(self):
        """Lazily initialize ia_modules scheduler."""
        if self._scheduler is None:
            try:
                from ia_modules.scheduler import Scheduler
                self._scheduler = Scheduler()
                await self._scheduler.start()
                logger.info("ia_modules scheduler started")
            except ImportError:
                logger.warning("ia_modules.scheduler not available, scheduler disabled")

    async def cleanup(self):
        if self._scheduler:
            await self._scheduler.stop()

    async def create_job(
        self,
        job_name: str,
        pipeline_id: str,
        created_by: int,
        app_id: Optional[str] = None,
        cron_expression: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        input_data: Dict[str, Any] = None,
        enabled: bool = True
    ) -> str:
        """Create a scheduled job and persist to DB."""
        if not cron_expression and not interval_seconds:
            raise ValueError("Either cron_expression or interval_seconds required")

        job_id = str(uuid.uuid4())
        await self.db.execute(
            """INSERT INTO scheduled_jobs
               (job_id, job_name, pipeline_id, app_id, cron_expression, interval_seconds,
                input_data, is_enabled, created_by)
               VALUES (:job_id, :job_name, :pipeline_id, :app_id, :cron_expression,
                        :interval_seconds, :input_data, :is_enabled, :created_by)""",
            {
                "job_id": job_id, "job_name": job_name, "pipeline_id": pipeline_id,
                "app_id": app_id, "cron_expression": cron_expression,
                "interval_seconds": interval_seconds,
                "input_data": json.dumps(input_data or {}),
                "is_enabled": enabled, "created_by": created_by
            }
        )

        if enabled:
            await self._register_job_in_scheduler(job_id, pipeline_id, cron_expression, interval_seconds, input_data or {})

        logger.info(f"Created scheduled job {job_id}: {job_name}")
        return job_id

    async def _register_job_in_scheduler(self, job_id, pipeline_id, cron_expression, interval_seconds, input_data):
        """Register job with ia_modules scheduler runtime."""
        await self._ensure_scheduler()
        if not self._scheduler:
            return

        try:
            from ia_modules.scheduler import CronTrigger, IntervalTrigger

            trigger = CronTrigger(cron_expression) if cron_expression else IntervalTrigger(interval_seconds)

            async def job_fn():
                await self._execute_scheduled_job(job_id, pipeline_id, input_data)

            self._job_functions[job_id] = job_fn
            await self._scheduler.schedule_job(job_id=job_id, trigger=trigger, func=job_fn, enabled=True)
        except Exception as e:
            logger.error(f"Failed to register job {job_id} in scheduler: {e}")

    async def _execute_scheduled_job(self, job_id: str, pipeline_id: str, input_data: dict):
        """Execute pipeline and record execution history."""
        exec_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        await self.db.execute(
            """INSERT INTO scheduled_job_executions (id, job_id, execution_id, triggered_by, status, started_at)
               VALUES (:id, :job_id, :exec_id, 'scheduler', 'running', :now)""",
            {"id": str(uuid.uuid4()), "job_id": job_id, "exec_id": exec_id, "now": now}
        )
        await self.db.execute(
            "UPDATE scheduled_jobs SET last_run_at = :now, updated_at = :now WHERE job_id = :job_id",
            {"now": now, "job_id": job_id}
        )
        logger.info(f"Scheduled job {job_id} triggered pipeline {pipeline_id}")

    async def list_jobs(self, pipeline_id: Optional[str] = None, enabled_only: bool = False) -> List[Dict[str, Any]]:
        conditions = ["1=1"]
        params = {}
        if pipeline_id:
            conditions.append("pipeline_id = :pipeline_id")
            params["pipeline_id"] = pipeline_id
        if enabled_only:
            conditions.append("is_enabled = TRUE")

        rows = await self.db.fetch_all(
            f"SELECT * FROM scheduled_jobs WHERE {' AND '.join(conditions)} ORDER BY created_at DESC", params
        )
        return [dict(r) for r in rows] if rows else []

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        row = await self.db.fetch_one("SELECT * FROM scheduled_jobs WHERE job_id = :job_id", {"job_id": job_id})
        return dict(row) if row else None

    async def update_job(self, job_id: str, **kwargs) -> bool:
        job = await self.get_job(job_id)
        if not job:
            return False

        sets = ["updated_at = CURRENT_TIMESTAMP"]
        params = {"job_id": job_id}
        for key in ["job_name", "cron_expression", "interval_seconds", "is_enabled"]:
            if key in kwargs and kwargs[key] is not None:
                sets.append(f"{key} = :{key}")
                params[key] = kwargs[key]
        if "input_data" in kwargs and kwargs["input_data"] is not None:
            sets.append("input_data = :input_data")
            params["input_data"] = json.dumps(kwargs["input_data"])

        await self.db.execute(f"UPDATE scheduled_jobs SET {', '.join(sets)} WHERE job_id = :job_id", params)
        return True

    async def delete_job(self, job_id: str) -> bool:
        job = await self.get_job(job_id)
        if not job:
            return False
        if self._scheduler and job_id in self._job_functions:
            await self._scheduler.remove_job(job_id)
            del self._job_functions[job_id]
        await self.db.execute("DELETE FROM scheduled_jobs WHERE job_id = :job_id", {"job_id": job_id})
        return True

    async def run_job_now(self, job_id: str) -> str:
        """Manually trigger a job execution."""
        job = await self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        input_data = json.loads(job["input_data"]) if isinstance(job["input_data"], str) else job.get("input_data", {})
        await self._execute_scheduled_job(job_id, job["pipeline_id"], input_data)
        return job_id

    async def get_job_history(self, job_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        rows = await self.db.fetch_all(
            "SELECT * FROM scheduled_job_executions WHERE job_id = :job_id ORDER BY created_at DESC LIMIT :limit",
            {"job_id": job_id, "limit": limit}
        )
        return [dict(r) for r in rows] if rows else []
