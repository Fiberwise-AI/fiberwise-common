"""
LLM Provider Factory - Tiny shim to create ia_modules providers from FiberWise DB config.

Creates ia_modules LLMProviderService instances directly from FiberWise database configuration.
No adapter, no wrapping, no translation - just a simple factory pattern.
"""

import json
import logging
from typing import Optional, Dict, Any

from ia_modules.pipeline.llm_provider_service import LLMProviderService

logger = logging.getLogger(__name__)


class LLMProviderFactory:
    """
    Creates ia_modules LLM providers from FiberWise DB config.

    Tiny shim - just reads DB and instantiates ia_modules LLMProviderService.
    No bridging, no wrapping, no translation.
    """

    @classmethod
    async def create_from_db(
        cls,
        db,
        provider_id: str,
        user_id: Optional[int] = None
    ) -> LLMProviderService:
        """
        Create ia_modules LLM provider from DB config.

        Args:
            db: Database provider (BaseDbProvider)
            provider_id: Provider ID from llm_providers table
            user_id: User ID for user-scoped providers (optional)

        Returns:
            ia_modules LLMProviderService instance

        Raises:
            ValueError: If provider not found or invalid configuration
        """
        # Query DB for provider config with user scoping
        if user_id:
            query = """
                SELECT * FROM llm_providers
                WHERE provider_id = :pid
                AND is_active = true
                AND (is_system = true OR created_by = :uid)
            """
            params = {"pid": provider_id, "uid": user_id}
        else:
            query = """
                SELECT * FROM llm_providers
                WHERE provider_id = :pid
                AND is_active = true
                AND is_system = true
            """
            params = {"pid": provider_id}

        config_row = await db.fetch_one(query, params)

        if not config_row:
            raise ValueError(
                f"Provider '{provider_id}' not found or not accessible "
                f"{'for user ' + str(user_id) if user_id else '(system only)'}"
            )

        # Parse configuration JSON
        configuration = config_row.get('configuration', '{}')
        if isinstance(configuration, str):
            try:
                config = json.loads(configuration)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in provider {provider_id} configuration: {e}")
                config = {}
        else:
            config = configuration

        # Extract provider details
        provider_type = config_row.get('provider_type', '')
        model = config.get('default_model') or config.get('model')
        api_key = config.get('api_key')
        base_url = config.get('base_url')

        if not model:
            raise ValueError(f"Provider '{provider_id}' has no default_model configured")

        # Create ia_modules LLMProviderService
        service = LLMProviderService()

        # Register the provider with ia_modules service
        service.register_provider(
            name=provider_id,
            model=model,
            api_key=api_key,
            base_url=base_url,
            is_default=True
        )

        logger.info(f"Created ia_modules provider '{provider_id}' (type: {provider_type}, model: {model})")

        return service

    @classmethod
    async def create_default(
        cls,
        db,
        user_id: Optional[int] = None
    ) -> LLMProviderService:
        """
        Create ia_modules provider for the default LLM provider.

        Args:
            db: Database provider
            user_id: User ID for user-scoped default (optional)

        Returns:
            ia_modules LLMProviderService instance

        Raises:
            ValueError: If no default provider found
        """
        # Find default provider
        if user_id:
            query = """
                SELECT provider_id FROM llm_providers
                WHERE is_active = true
                AND is_default = true
                AND created_by = :uid
            """
            params = {"uid": user_id}
        else:
            query = """
                SELECT provider_id FROM llm_providers
                WHERE is_active = true
                AND is_default = true
                AND is_system = true
            """
            params = {}

        default_row = await db.fetch_one(query, params)

        if not default_row:
            raise ValueError(
                f"No default LLM provider found "
                f"{'for user ' + str(user_id) if user_id else '(system)'}"
            )

        provider_id = default_row['provider_id']
        return await cls.create_from_db(db, provider_id, user_id)
