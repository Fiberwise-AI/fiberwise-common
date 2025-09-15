"""Fiberwise common utilities and models."""

from .database.base import DatabaseProvider
from .database.providers import SQLiteProvider, DuckDBProvider, create_database_provider
from .database.factory import get_database_provider
from .database.manager import DatabaseManager
# from .services.local_service import LocalService  # Commented out to avoid circular imports
from .constants import *
from .utils import *
from .utils.file_utils import calculate_file_checksum
from .entities.config import Config, EnhancedConfig
from .entities.fiber_agent import FiberAgent, FiberInjectable

# Services are available via fiberwise_common.services
# Business entities are available via fiberwise_common.entities

__version__ = "0.1.0"
__all__ = [
    "DatabaseProvider",
    "SQLiteProvider", 
    "DuckDBProvider",
    "create_database_provider",
    "get_database_provider",
    "DatabaseManager",
    # "LocalService",  # Commented out
    "Config",
    "EnhancedConfig",
    "FiberAgent",
    "FiberInjectable",
    "validate_input",
    "safe_json_loads", 
    "safe_json_dumps",
    "extract_agent_metadata",
    "calculate_file_checksum",
    "ensure_directory_exists",
    "CLI_APP_UUID",
    "CLI_APP_SLUG",
    "get_cli_app_name",
    "get_cli_app_description",
    "get_cli_user_email",
    "__version__"
]
