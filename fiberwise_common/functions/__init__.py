"""
Function processing components for FiberWise.

This package contains function processing logic that can be used
by both CLI (for instant execution) and web (for worker-based processing).
"""

from .function_processor import FunctionProcessor

__all__ = ['FunctionProcessor']
