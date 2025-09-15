""""
Services package for FiberWise common functionality.
"""

from typing import Optional
from .config import settings
from .base_service import BaseService, ServiceError, ValidationError, NotFoundError, AuthorizationError, service_registry
from .agent_service import AgentService
from .user_service import UserService
from .account_service import AccountService
from .provider_service import ProviderService
from .oauth_service import OAuthService, OAuthBackendService
from .oauth_injection_service import OAuthInjectionService, create_oauth_credential_service_for_injection
from .user_context_service import FiberLocalContextService
from .service_registry import ServiceRegistry, Injectable, create_default_registry, set_global_registry, get_global_registry, inject_services
from .api_keys_service import ApiKeyService, get_api_key, get_user_from_api_key, log_api_key_usage, require_scopes
from .execution_key_service import ExecutionKeyService
from .scoped_storage_provider import ScopedStorageProvider
from .agent_storage_provider import AgentStorageProvider
from .storage_provider import StorageProvider, LocalStorageProvider, get_storage_provider
from .llm_provider_service import LLMProviderService
from .service_factory import ServiceFactory, ServiceContainer, create_service_factory, get_service_container, set_service_container
from .cli_utils import get_default_config_name, load_config, save_config
from .fiber_app_manager import FiberAppManager, AppOperationResult
from .agent_key_service import AgentKeyService
from .organization_service import OrganizationService
from .pipeline_service import PipelineService
from .connection_manager import ConnectionManager
from .model_operations_service import ModelOperationsService
from .function_service import FunctionService
from .install_service import InstallService
from .update_service import UpdateService, update_agent, update_pipeline, update_workflow, update_function, has_version_changed
from .install_service import process_unified_manifest, UnifiedManifestResult
from .app_service import AppService
from .app_agent_service import AppAgentService
from .app_upload_service import AppUploadService, process_app_bundle, cleanup_old_bundles, get_app_bundle_path, get_entity_bundle_path, ensure_directory_exists
from .agent_version_service import AgentVersionService
from .app_version_service import AppVersionService
# oauth_backend_service consolidated into oauth_service
from .email_service import EmailService
from .user_isolation_service import UserIsolationService, get_user_isolation_service
from .app_migration_service import AppMigrationService, validate_app_migration, MigrationRisk, MigrationIssue
from .security import get_password_hash, verify_password

class AuthService:
    """Authentication service utilities."""
    
    @staticmethod
    def validate_token(token: str) -> Optional[dict]:
        """Validate JWT token."""
        # Implementation would go here
        return {"user_id": "123"} if token else None

auth_service = AuthService()

__all__ = [
    'BaseService',
    'ServiceError', 
    'ValidationError',
    'NotFoundError',
    'AuthorizationError',
    'service_registry',
    'AgentService',
    'UserService',
    'AccountService',
    'ProviderService',
    'OAuthService',
    'OAuthInjectionService',
    'create_oauth_credential_service_for_injection',
    'FiberLocalContextService',
    'ServiceRegistry',
    'Injectable',
    'create_default_registry',
    'set_global_registry',
    'get_global_registry',
    'inject_services',
    'ApiKeyService',
    'ExecutionKeyService',
    'get_api_key',
    'get_user_from_api_key', 
    'log_api_key_usage',
    'require_scopes',
    'ScopedStorageProvider',
    'AgentStorageProvider', 
    'StorageProvider',
    'LocalStorageProvider',
    'get_storage_provider',
    'LLMProviderService',
    'ServiceFactory',
    'ServiceContainer',
    'create_service_factory',
    'get_service_container',
    'set_service_container',
    'auth_service',
    'settings',
    'get_default_config_name',
    'load_config',
    'save_config',
    'FiberAppManager',
    'AppOperationResult',
    'AgentKeyService',
    'OrganizationService',
    'PipelineService',
    'ConnectionManager',
    'ModelOperationsService',
    'FunctionService',
    'InstallService',
    'UpdateService',
    'process_unified_manifest',
    'UnifiedManifestResult',
    'update_agent',
    'update_pipeline', 
    'update_workflow',
    'update_function',
    'has_version_changed',
    'AppAgentService',
    'AppService',
    'AppUploadService',
    'process_app_bundle',
    'cleanup_old_bundles',
    'get_app_bundle_path',
    'get_entity_bundle_path',
    'ensure_directory_exists',
    'AgentVersionService',
    'AppVersionService',
    'OAuthBackendService',
    'EmailService',
    'PipelineService',
    'WorkflowService',
    'UserIsolationService',
    'get_user_isolation_service',
    'AppMigrationService',
    'validate_app_migration',
    'MigrationRisk',
    'MigrationIssue',
    'get_password_hash',
    'verify_password'
]
