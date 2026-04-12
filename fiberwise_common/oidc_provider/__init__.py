"""OIDC Provider — re-exports from ia_modules.agents.auth.

The auth implementation lives in ia_modules (core library).
This module re-exports for backwards compatibility with existing
fiberwise-common imports.
"""

from ia_modules.agents.auth import (
    IDPAdapter,
    ClientCredentials,
    MiniOIDCAdapter,
    KeycloakAdapter,
    get_adapter,
    get_default_permissions,
    DEFAULT_A2A_PERMISSIONS,
)

__all__ = [
    "IDPAdapter",
    "ClientCredentials",
    "MiniOIDCAdapter",
    "KeycloakAdapter",
    "get_adapter",
    "get_default_permissions",
    "DEFAULT_A2A_PERMISSIONS",
]
