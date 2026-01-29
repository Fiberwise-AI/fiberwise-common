"""
Unit tests for fiberwise_common.database.provider module.

Tests NexusQLProvider construction/properties (fiberwise-common code only).
NexusQL's own CRUD behavior is tested by the nexusql package itself.
"""
import pytest
from unittest.mock import patch

from fiberwise_common.database.provider import (
    NexusQLProvider,
    DatabaseProvider,
    create_database_provider,
)


class TestNexusQLProviderProperties:
    """Test NexusQLProvider construction and properties."""

    def test_provider_type_postgresql(self):
        p = NexusQLProvider.__new__(NexusQLProvider)
        p.database_url = "postgresql://localhost/test"
        assert p.provider == "postgresql"

    def test_provider_type_postgres_scheme(self):
        p = NexusQLProvider.__new__(NexusQLProvider)
        p.database_url = "postgres://localhost/test"
        assert p.provider == "postgresql"

    def test_provider_type_mysql(self):
        p = NexusQLProvider.__new__(NexusQLProvider)
        p.database_url = "mysql://localhost/test"
        assert p.provider == "mysql"

    def test_provider_type_mssql(self):
        p = NexusQLProvider.__new__(NexusQLProvider)
        p.database_url = "mssql://localhost/test"
        assert p.provider == "mssql"

    def test_provider_type_sqlite(self):
        p = NexusQLProvider.__new__(NexusQLProvider)
        p.database_url = "sqlite:///test.db"
        assert p.provider == "sqlite"

    def test_provider_type_unknown_defaults_sqlite(self):
        p = NexusQLProvider.__new__(NexusQLProvider)
        p.database_url = "unknown://localhost/test"
        assert p.provider == "sqlite"

    def test_database_provider_alias(self):
        assert DatabaseProvider is NexusQLProvider

    def test_create_database_provider_returns_instance(self):
        with patch("fiberwise_common.database.provider.NexusDB"):
            provider = create_database_provider("sqlite:///test.db")
            assert isinstance(provider, NexusQLProvider)
            assert provider.database_url == "sqlite:///test.db"
