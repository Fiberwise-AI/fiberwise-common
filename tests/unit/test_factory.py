"""
Unit tests for fiberwise_common.database.factory module.
"""
import pytest
from unittest.mock import MagicMock, patch

from fiberwise_common.database.factory import get_database_provider
from fiberwise_common.database.provider import NexusQLProvider


class TestGetDatabaseProvider:
    """Test the factory function."""

    @patch("fiberwise_common.database.factory.create_database_provider")
    def test_returns_provider(self, mock_create):
        mock_create.return_value = MagicMock(spec=NexusQLProvider)
        settings = MagicMock()
        settings.DATABASE_URL = "sqlite:///app.db"

        provider = get_database_provider(settings)

        mock_create.assert_called_once_with("sqlite:///app.db")
        assert provider is mock_create.return_value

    def test_none_settings_raises(self):
        with pytest.raises(ValueError, match="Settings instance is required"):
            get_database_provider(None)

    def test_missing_database_url_raises(self):
        settings = MagicMock(spec=[])  # no attributes
        with pytest.raises(ValueError, match="DATABASE_URL"):
            get_database_provider(settings)
