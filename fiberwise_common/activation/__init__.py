"""
Activation processing components for Fiberwise.

This package contains common activation processing logic that can be used
by both CLI (for instant execution) and web (for worker-based processing).
"""

from .activation_processor import ActivationProcessor

__all__ = ['ActivationProcessor']