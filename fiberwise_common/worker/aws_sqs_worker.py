"""
AWS SQS worker implementation for distributed processing.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from .base import WorkerProvider, WorkerConfig, WorkerJob

logger = logging.getLogger(__name__)


class AWSSQSWorker(WorkerProvider):
    """
    AWS SQS worker that uses Amazon Simple Queue Service for distributed processing.
    Supports both sending activations to SQS and processing them from SQS.
    """
    
    def __init__(
        self, 
        config: WorkerConfig,
        db_provider,
        llm_provider = None,
        storage_provider = None
    ):
        super().__init__(config, db_provider, llm_provider, storage_provider)
        
        self.sqs_client: Optional[boto3.client] = None
        self.queue_url: str = config.broker_url
        self.active_jobs: Dict[str, WorkerJob] = {}
        self._poll_task: Optional[asyncio.Task] = None
        
        if not self.queue_url:
            raise ValueError("broker_url (SQS queue URL) is required for AWS SQS worker")
        
        # Extract region from queue URL if not set in environment
        self.region = self._extract_region_from_queue_url(self.queue_url)
        
        # Initialize the activation processor
        from ..activation import ActivationProcessor
        self.processor = ActivationProcessor(db_provider, context="aws_sqs_worker")
        
        logger.info(f"AWS SQS Worker initialized for queue: {self.queue_url}")
    
    def _extract_region_from_queue_url(self, queue_url: str) -> str:
        """Extract AWS region from SQS queue URL."""
        try:
            # SQS URL format: https://sqs.region.amazonaws.com/account-id/queue-name
            parts = queue_url.split('.')
            if len(parts) >= 3 and 'sqs' in parts[0]:
                return parts[1]  # Extract region
        except Exception:
            pass
        return 'us-east-1'  # Default fallback
    
    async def start(self) -> None:
        """Start the AWS SQS worker."""
        if self.running:
            logger.warning("AWS SQS worker is already running")
            return
        
        try:
            # Initialize SQS client
            self.sqs_client = boto3.client(
                'sqs',
                region_name=self.region
            )
            
            # Test connection by getting queue attributes
            await self._test_queue_connection()
            
            self.running = True
            
            # Setup services for activation processing
            await self._setup_services()
            
            # Start polling for messages
            self._poll_task = asyncio.create_task(self._poll_messages())
            
            logger.info(f"AWS SQS worker started successfully")
            logger.info(f"Queue URL: {self.queue_url}")
            logger.info(f"Region: {self.region}")
            logger.info(f"Max concurrent jobs: {self.config.max_concurrent_jobs}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Configure AWS credentials via environment variables, IAM role, or AWS credentials file.")
            raise
        except ClientError as e:
            logger.error(f"AWS SQS client error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to start AWS SQS worker: {e}")
            raise
    
    async def _test_queue_connection(self) -> None:
        """Test connection to SQS queue."""
        try:
            # Run in thread pool since boto3 is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.sqs_client.get_queue_attributes(
                    QueueUrl=self.queue_url,
                    AttributeNames=['QueueArn', 'VisibilityTimeout']
                )
            )
            logger.info(f"Successfully connected to SQS queue: {response['Attributes'].get('QueueArn', 'Unknown')}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AWS.SimpleQueueService.NonExistentQueue':
                raise ValueError(f"SQS queue does not exist: {self.queue_url}")
            elif error_code == 'AccessDenied':
                raise ValueError(f"Access denied to SQS queue: {self.queue_url}. Check IAM permissions.")
            else:
                raise ValueError(f"SQS connection failed: {e}")
    
    async def _setup_services(self) -> None:
        """Setup services for activation processing."""
        try:
            # Initialize service registry
            from ..services.service_registry import ServiceRegistry
            
            # Create service registry instance
            service_registry = ServiceRegistry(self.db_provider)
            
            # Setup service injection
            services = {
                'fiber': service_registry.get_service('fiber_app'),
                'llm_service': service_registry.get_service('llm_service'),
                'storage': service_registry.get_service('storage'),
                'oauth_service': service_registry.get_service('oauth_service')
            }
            
            self.processor.inject_services(**services)
            
            logger.info("✅ AWS SQS worker services configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup services: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the AWS SQS worker gracefully."""
        logger.info("Stopping AWS SQS worker...")
        self.running = False
        
        # Cancel polling task
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        
        # Wait for active jobs to complete (with timeout)
        if self.active_jobs:
            logger.info(f"Waiting for {len(self.active_jobs)} active jobs to complete")
            timeout = 30  # 30 seconds timeout
            start_time = time.time()
            
            while self.active_jobs and (time.time() - start_time) < timeout:
                await asyncio.sleep(1)
            
            if self.active_jobs:
                logger.warning(f"Forcibly stopping with {len(self.active_jobs)} active jobs")
        
        logger.info("AWS SQS worker stopped")
    
    async def submit_activation(self, activation_id: str, priority: int = 1) -> bool:
        """Submit activation to SQS queue."""
        if not self.running or not self.sqs_client:
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
            
            # Prepare SQS message
            message_body = {
                'activation_id': activation_id,
                'activation_data': dict(activation),
                'priority': priority,
                'timestamp': time.time(),
                'worker_type': 'aws_sqs',
                'retry_count': 0
            }
            
            # Send message to SQS
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.sqs_client.send_message(
                    QueueUrl=self.queue_url,
                    MessageBody=json.dumps(message_body),
                    MessageAttributes={
                        'priority': {
                            'StringValue': str(priority),
                            'DataType': 'Number'
                        },
                        'activation_id': {
                            'StringValue': activation_id,
                            'DataType': 'String'
                        }
                    }
                )
            )
            
            logger.info(f"Successfully submitted activation {activation_id} to SQS (MessageId: {response['MessageId']})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit activation {activation_id} to SQS: {e}")
            return False
    
    async def _poll_messages(self) -> None:
        """Poll SQS for messages and process them."""
        logger.info("Starting SQS message polling...")
        
        while self.running:
            try:
                # Check if we can process more jobs
                if len(self.active_jobs) >= self.config.max_concurrent_jobs:
                    await asyncio.sleep(1)
                    continue
                
                # Calculate how many messages to receive
                available_slots = self.config.max_concurrent_jobs - len(self.active_jobs)
                max_messages = min(available_slots, 10)  # SQS limit is 10
                
                # Receive messages from SQS
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.sqs_client.receive_message(
                        QueueUrl=self.queue_url,
                        MaxNumberOfMessages=max_messages,
                        WaitTimeSeconds=20,  # Long polling
                        VisibilityTimeout=self.config.timeout,
                        MessageAttributeNames=['All']
                    )
                )
                
                messages = response.get('Messages', [])
                if not messages:
                    continue
                
                logger.info(f"Received {len(messages)} messages from SQS")
                
                # Process each message
                for message in messages:
                    try:
                        message_body = json.loads(message['Body'])
                        activation_id = message_body.get('activation_id')
                        
                        if not activation_id:
                            logger.error("Message missing activation_id")
                            await self._delete_message(message['ReceiptHandle'])
                            continue
                        
                        # Create and process job
                        job = WorkerJob(activation_id, message_body)
                        job.sqs_receipt_handle = message['ReceiptHandle']  # Store for deletion
                        
                        # Process in background
                        asyncio.create_task(self._process_sqs_job(job))
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in SQS message: {e}")
                        await self._delete_message(message['ReceiptHandle'])
                    except Exception as e:
                        logger.error(f"Error processing SQS message: {e}")
                        await self._delete_message(message['ReceiptHandle'])
                
            except Exception as e:
                if self.running:  # Only log errors if we're still running
                    logger.error(f"Error polling SQS messages: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _process_sqs_job(self, job: WorkerJob) -> None:
        """Process a single SQS job."""
        job.started_at = time.time()
        self.active_jobs[job.activation_id] = job
        
        try:
            activation_id = job.activation_id
            logger.info(f"Processing activation {activation_id} from SQS")
            
            # Get work item from activation data
            work_item = job.activation_data.get('activation_data', {})
            work_item['work_type'] = 'agent'  # SQS jobs are typically agent activations
            
            # Process the activation
            await self.processor.process_work_item(work_item)
            
            job.completed_at = time.time()
            logger.info(f"Completed activation {activation_id} in {job.duration:.2f}s")
            
            # Delete message from SQS (successful processing)
            await self._delete_message(job.sqs_receipt_handle)
            
        except Exception as e:
            job.error = str(e)
            job.completed_at = time.time()
            logger.error(f"Failed to process activation {job.activation_id}: {e}", exc_info=True)
            
            # Handle retries
            retry_count = job.activation_data.get('retry_count', 0)
            if retry_count < self.config.retry_attempts:
                logger.info(f"Activation {job.activation_id} will be retried (attempt {retry_count + 1})")
                # Don't delete message - let it become visible again for retry
                # SQS will automatically retry based on visibility timeout
            else:
                logger.error(f"Activation {job.activation_id} exceeded max retries, deleting from queue")
                await self._delete_message(job.sqs_receipt_handle)
            
        finally:
            # Remove from active jobs
            self.active_jobs.pop(job.activation_id, None)
    
    async def _delete_message(self, receipt_handle: str) -> None:
        """Delete message from SQS queue."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.sqs_client.delete_message(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=receipt_handle
                )
            )
        except Exception as e:
            logger.error(f"Failed to delete SQS message: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current AWS SQS worker status."""
        status = {
            'healthy': self.running and self.sqs_client is not None,
            'worker_type': self.config.worker_type.value,
            'running': self.running,
            'queue_url': self.queue_url,
            'region': self.region,
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
        
        # Add queue statistics if possible
        if self.running and self.sqs_client:
            try:
                loop = asyncio.get_event_loop()
                queue_attrs = await loop.run_in_executor(
                    None,
                    lambda: self.sqs_client.get_queue_attributes(
                        QueueUrl=self.queue_url,
                        AttributeNames=[
                            'ApproximateNumberOfMessages',
                            'ApproximateNumberOfMessagesNotVisible',
                            'ApproximateNumberOfMessagesDelayed'
                        ]
                    )
                )
                
                status['queue_stats'] = {
                    'messages_available': int(queue_attrs['Attributes'].get('ApproximateNumberOfMessages', 0)),
                    'messages_in_flight': int(queue_attrs['Attributes'].get('ApproximateNumberOfMessagesNotVisible', 0)),
                    'messages_delayed': int(queue_attrs['Attributes'].get('ApproximateNumberOfMessagesDelayed', 0))
                }
            except Exception as e:
                status['queue_stats_error'] = str(e)
        
        return status
