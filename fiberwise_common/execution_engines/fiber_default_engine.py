"""
Fiber-Default Execution Engine (Legacy)

Marker class for the built-in FiberWise pipeline executor.
Actual execution is handled by PipelineService._execute_structured_pipeline().
"""

import logging
from typing import Dict, Any
from .base import BaseExecutionEngine

logger = logging.getLogger(__name__)


class FiberDefaultExecutionEngine(BaseExecutionEngine):
    """
    Marker engine for legacy FiberWise pipeline execution.
    The actual execution remains in PipelineService._execute_structured_pipeline().
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.engine_name = "fiber-default"

    def load_pipeline_definition(self, file_path: str) -> Dict[str, Any]:
        raise NotImplementedError("fiber-default engine uses database-stored definitions")

    async def execute_pipeline(
        self,
        pipeline_definition: Dict[str, Any],
        input_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "fiber-default execution is handled by PipelineService._execute_structured_pipeline()"
        )
