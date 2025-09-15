"""
Local worker implementation using database polling.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

from .base import WorkerProvider, WorkerConfig, WorkerJob, WorkerType
from ..activation import ActivationProcessor

logger = logging.getLogger(__name__)


class LocalWorker(WorkerProvider):
    """Local worker that polls database for queued activations."""
    
    def __init__(
        self, 
        config: WorkerConfig,
        db_provider,
        llm_provider = None,
        storage_provider = None,
        base_url: Optional[str] = None
    ):
        super().__init__(config, db_provider, llm_provider, storage_provider)
        
        # Create processor - it will handle all service creation per-activation
        self.processor = ActivationProcessor(db_provider, context="worker", base_url=base_url)
        self.active_jobs: Dict[str, WorkerJob] = {}
        self._websocket_broadcaster = None
        
        # Create a unique instance ID for tracking
        import uuid
        self.instance_id = str(uuid.uuid4())[:8]
        logger.info(f"🏭 CLEAN LOCAL WORKER [{self.instance_id}] - No global services, per-activation creation only (base_url: {base_url})")
        
    def set_websocket_broadcaster(self, broadcaster):
        """
        Set the WebSocket broadcaster for sending real-time notifications.
        
        Args:
            broadcaster: Function that takes (activation_id, status, app_id) and broadcasts updates
        """
        self._websocket_broadcaster = broadcaster
        
        # Set the callback in the processor
        if self._websocket_broadcaster:
            self.processor.set_notification_callback(self._websocket_broadcaster)
            logger.info(f"🔗 WebSocket broadcaster registered with ActivationProcessor on LocalWorker [{self.instance_id}]")
        else:
            self.processor.set_notification_callback(None)
            logger.info(f"🔌 WebSocket broadcaster cleared from ActivationProcessor on LocalWorker [{self.instance_id}]")
        
    async def start(self) -> None:
        """Start the local worker polling loop."""
        if self.running:
            logger.warning("Worker is already running")
            return
            
        logger.info(f"🚀 CLEAN WORKER [{self.instance_id}] - Starting with pure per-activation service creation")
            
        self.running = True
        logger.info(f"✅ CLEAN WORKER [{self.instance_id}] - Polling every {self.config.poll_interval}s (max {self.config.max_concurrent_jobs} concurrent jobs)")
        
        try:
            while self.running:
                await self._process_cycle()
                await asyncio.sleep(self.config.poll_interval)
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
        finally:
            self.running = False
            logger.info("Local worker stopped")
    
    async def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info("Stopping local worker...")
        self.running = False
        
        # Wait for active jobs to complete (with timeout)
        if self.active_jobs:
            logger.info(f"Waiting for {len(self.active_jobs)} active jobs to complete")
            timeout = 30  # 30 seconds timeout
            start_time = time.time()
            
            while self.active_jobs and (time.time() - start_time) < timeout:
                await asyncio.sleep(1)
            
            if self.active_jobs:
                logger.warning(f"Forcibly stopping with {len(self.active_jobs)} active jobs")
    
    async def submit_activation(self, activation_id: str, priority: int = 1) -> bool:
        """For local worker, this is a no-op since we poll the database."""
        # Local worker doesn't need explicit submission - it polls the database
        return True
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current worker status."""
        return {
            'healthy': self.running,
            'worker_type': self.config.worker_type.value,
            'running': self.running,
            'active_jobs': len(self.active_jobs),
            'max_concurrent_jobs': self.config.max_concurrent_jobs,
            'poll_interval': self.config.poll_interval,
            'job_details': {
                job_id: {
                    'activation_id': job.activation_id,
                    'started_at': job.started_at,
                    'duration': job.duration,
                    'retry_count': job.retry_count
                }
                for job_id, job in self.active_jobs.items()
            }
        }
    
    async def _process_cycle(self) -> None:
        """Process one cycle of work item polling and processing."""
        try:
            # Check if we can process more jobs
            if len(self.active_jobs) >= self.config.max_concurrent_jobs:
                return
            
            # Get next work item from database (agents or functions)
            work_item = await self.processor.get_next_work_item()
            
            if work_item:
                # Create job based on work type
                work_type = work_item.get('work_type', 'agent')
                if work_type == 'function':
                    job_id = work_item.get('execution_id')
                else:
                    job_id = work_item.get('activation_id')
                
                job = WorkerJob(
                    activation_id=job_id,  # Use generic ID field
                    activation_data=work_item
                )
                
                # Process work item in background
                asyncio.create_task(self._process_job(job))
                
        except Exception as e:
            logger.error(f"Error in process cycle: {e}", exc_info=True)
    
    async def _process_job(self, job: WorkerJob) -> None:
        """Process a single work item (activation or function execution)."""
        job.started_at = time.time()
        self.active_jobs[job.activation_id] = job
        
        try:
            work_type = job.activation_data.get('work_type', 'agent')
            job_id = job.activation_id
            
            logger.info(f"Processing {work_type} {job_id} on LocalWorker [{self.instance_id}]")
            
            # Use the unified work item processor
            await self.processor.process_work_item(job.activation_data)
            
            job.completed_at = time.time()
            logger.info(f"Completed {work_type} {job_id} in {job.duration:.2f}s")
            
        except Exception as e:
            job.error = str(e)
            job.completed_at = time.time()
            logger.error(f"Failed to process {work_type} {job_id}: {e}", exc_info=True)
            
            # Handle retries
            if job.retry_count < self.config.retry_attempts:
                job.retry_count += 1
                logger.info(f"Retrying {work_type} {job_id} (attempt {job.retry_count})")
                
                # Reset job state for retry
                job.started_at = None
                job.completed_at = None
                job.error = None
                
                # Wait before retry
                await asyncio.sleep(self.config.retry_delay)
                
                # Retry the job
                await self._process_job(job)
                return
                
        finally:
            # Remove from active jobs
            self.active_jobs.pop(job.activation_id, None)
