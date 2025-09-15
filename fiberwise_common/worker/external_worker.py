"""
External worker implementation for webhook-based processing.
"""

import logging
import asyncio
from typing import Dict, Any
import aiohttp

from .base import WorkerProvider, WorkerConfig

logger = logging.getLogger(__name__)


class ExternalWorker(WorkerProvider):
    """
    External worker that sends activations to external webhook endpoints.
    Useful for integrating with external job processing systems.
    """
    
    def __init__(
        self, 
        config: WorkerConfig,
        db_provider,
        llm_provider = None,
        storage_provider = None
    ):
        super().__init__(config, db_provider, llm_provider, storage_provider)
        self.session: aiohttp.ClientSession = None
        
    async def start(self) -> None:
        """Start the external worker."""
        if self.running:
            logger.warning("External worker is already running")
            return
            
        if not self.config.webhook_url:
            raise ValueError("webhook_url is required for external worker")
            
        self.running = True
        self.session = aiohttp.ClientSession()
        logger.info(f"External worker started, webhook: {self.config.webhook_url}")
    
    async def stop(self) -> None:
        """Stop the external worker."""
        self.running = False
        if self.session:
            await self.session.close()
        logger.info("External worker stopped")
    
    async def submit_activation(self, activation_id: str, priority: int = 1) -> bool:
        """Submit activation to external webhook."""
        if not self.running or not self.session:
            return False
            
        try:
            # Get activation data from database
            activation = await self.db_provider.fetch_one(
                "SELECT * FROM agent_activations WHERE activation_id = ?",
                activation_id
            )
            
            if not activation:
                logger.error(f"Activation {activation_id} not found")
                return False
            
            # Prepare webhook payload
            payload = {
                'activation_id': activation_id,
                'activation_data': dict(activation),
                'priority': priority,
                'callback_url': f"{self.config.webhook_url}/callback"  # For completion notification
            }
            
            # Add API key if configured
            headers = {}
            if self.config.api_key:
                headers['Authorization'] = f'Bearer {self.config.api_key}'
                headers['X-API-Key'] = self.config.api_key
            
            # Send to webhook
            async with self.session.post(
                self.config.webhook_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    logger.info(f"Successfully submitted activation {activation_id} to external worker")
                    return True
                else:
                    logger.error(f"External worker rejected activation {activation_id}: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to submit activation {activation_id} to external worker: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get external worker status."""
        return {
            'healthy': self.running and self.session is not None,
            'worker_type': self.config.worker_type.value,
            'running': self.running,
            'webhook_url': self.config.webhook_url,
            'has_api_key': bool(self.config.api_key)
        }
