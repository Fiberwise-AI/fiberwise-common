"""
Execution Engine Registry — manages available pipeline execution engines.
"""

import logging
from typing import Dict, Type, Any
from .base import BaseExecutionEngine

logger = logging.getLogger(__name__)


class ExecutionEngineRegistry:
    """Registry for pipeline execution engines."""

    _engines: Dict[str, Type[BaseExecutionEngine]] = {}

    @classmethod
    def register(cls, name: str, engine_class: Type[BaseExecutionEngine]):
        if not issubclass(engine_class, BaseExecutionEngine):
            raise ValueError(f"Engine must inherit from BaseExecutionEngine")
        cls._engines[name] = engine_class
        logger.info(f"Registered execution engine: {name}")

    @classmethod
    def get_engine(cls, name: str, config: Dict[str, Any] = None) -> BaseExecutionEngine:
        if name not in cls._engines:
            available = list(cls._engines.keys())
            raise ValueError(f"Unknown engine '{name}'. Available: {available}")
        return cls._engines[name](config)

    @classmethod
    def list_engines(cls) -> list:
        return list(cls._engines.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._engines


def _register_builtin_engines():
    """Register built-in execution engines on module load."""
    try:
        from .fiber_default_engine import FiberDefaultExecutionEngine
        ExecutionEngineRegistry.register("fiber-default", FiberDefaultExecutionEngine)
    except ImportError as e:
        logger.warning(f"Could not register fiber-default engine: {e}")

    try:
        from .ia_modules_engine import IAModulesExecutionEngine
        ExecutionEngineRegistry.register("ia_modules", IAModulesExecutionEngine)
    except ImportError as e:
        logger.warning(f"Could not register ia_modules engine: {e}")


_register_builtin_engines()
