"""
Execution Engines — pluggable pipeline execution backends.

Usage:
    from fiberwise_common.execution_engines import ExecutionEngineRegistry

    engine = ExecutionEngineRegistry.get_engine("ia_modules")
    result = await engine.execute_pipeline(definition, input_data, context)
"""

from .base import BaseExecutionEngine
from .registry import ExecutionEngineRegistry

__all__ = ["BaseExecutionEngine", "ExecutionEngineRegistry"]
