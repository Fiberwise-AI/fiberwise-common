"""
Base execution engine interface for FiberWise.

All execution engines must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseExecutionEngine(ABC):
    """Base class for all pipeline execution engines."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.engine_name = self.__class__.__name__

    @abstractmethod
    async def execute_pipeline(
        self,
        pipeline_definition: Dict[str, Any],
        input_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a pipeline.

        Args:
            pipeline_definition: Pipeline configuration dict
            input_data: Input parameters
            context: Execution context (db, services, user info, etc.)

        Returns:
            Dict with 'success', 'result', 'error' keys
        """
        pass

    @abstractmethod
    def load_pipeline_definition(self, file_path: str) -> Dict[str, Any]:
        """
        Load pipeline definition from file.

        Args:
            file_path: Path to pipeline file (JSON or YAML)

        Returns:
            Pipeline definition dict
        """
        pass

    def validate_pipeline(self, definition: Dict[str, Any]) -> bool:
        """
        Validate pipeline definition. Override in subclasses for engine-specific validation.

        Returns:
            True if valid, raises ValueError if not
        """
        return True
