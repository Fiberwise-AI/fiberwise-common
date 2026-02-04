"""
ia_modules Execution Engine for FiberWise Platform.

Executes pipelines using the ia_modules framework (pipeline runner).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from .base import BaseExecutionEngine

logger = logging.getLogger(__name__)


class IAModulesExecutionEngine(BaseExecutionEngine):
    """Execution engine that delegates to ia_modules pipeline runner."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.engine_name = "ia_modules"

    def load_pipeline_definition(self, file_path: str) -> Dict[str, Any]:
        """Load pipeline definition from JSON or YAML file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {file_path}")

        if path.suffix == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            import yaml
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .yaml")

    def validate_pipeline(self, definition: Dict[str, Any]) -> bool:
        """Validate ia_modules pipeline structure."""
        if 'steps' not in definition:
            raise ValueError("Pipeline missing required 'steps' field")

        steps = definition['steps']
        if not steps:
            raise ValueError("Pipeline must have at least one step")

        for step in steps:
            if 'id' not in step:
                raise ValueError("Step missing 'id' field")
            if 'step_class' not in step:
                raise ValueError(f"Step {step.get('id', '?')} missing 'step_class' field")

        return True

    async def execute_pipeline(
        self,
        pipeline_definition: Dict[str, Any],
        input_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute pipeline using ia_modules runner.

        Context keys:
            db_provider: DatabaseProvider instance
            connection_manager: ConnectionManager for WebSocket updates
            organization_id: int
            user_id: int
            app_id: str
            execution_id: str
            working_directory: str (path to app source for step module imports)
        """
        try:
            from ia_modules.pipeline.runner import create_pipeline_from_json
            from ia_modules.pipeline.services import ServiceRegistry
            from ia_modules.pipeline.core import ExecutionContext

            context = context or {}

            self.validate_pipeline(pipeline_definition)

            # Build ia_modules ServiceRegistry with FiberWise services
            services = ServiceRegistry()

            if context.get('db_provider'):
                services.register('database', context['db_provider'])

            # Register FiberApp from context (created by PipelineService)
            if context.get('fiber_app'):
                services.register('fiber', context['fiber_app'])
                logger.info("Registered FiberApp for ia_modules steps")

            if context.get('connection_manager'):
                services.register('websocket_manager', context['connection_manager'])
            if context.get('user_id'):
                services.register('websocket_user_id', context['user_id'])
            if context.get('execution_id'):
                services.register('websocket_execution_id', context['execution_id'])

            # Set working directory for step module imports
            working_dir = context.get('working_directory')
            if working_dir:
                import sys
                if working_dir not in sys.path:
                    sys.path.insert(0, working_dir)

            # Create execution context
            execution_context = ExecutionContext(
                execution_id=context.get('execution_id', ''),
                pipeline_id=pipeline_definition.get('name', 'unknown'),
                user_id=str(context.get('user_id', 'unknown'))
            )

            # Create and run pipeline
            pipeline_name = pipeline_definition.get('name', 'unknown')
            logger.info(f"Executing ia_modules pipeline: {pipeline_name}")

            pipeline = create_pipeline_from_json(pipeline_definition, services)
            result = await pipeline.run(input_data or {}, execution_context=execution_context)

            return {
                "success": True,
                "result": result,
                "engine": self.engine_name,
                "pipeline_name": pipeline_name
            }

        except Exception as e:
            logger.error(f"ia_modules execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "engine": self.engine_name
            }
