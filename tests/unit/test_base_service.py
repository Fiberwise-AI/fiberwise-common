"""
Unit tests for fiberwise_common.services.base_service module.

Tests the BaseService class, ServiceRegistry, and service exception hierarchy.
"""
import asyncio
import pytest
from typing import Any, List, Dict, Optional
from unittest.mock import Mock, AsyncMock

from fiberwise_common.services.base_service import (
    BaseService,
    ServiceError,
    ValidationError,
    NotFoundError,
    AuthorizationError,
    ServiceRegistry,
    service_registry,
)
from fiberwise_common.database.provider import NexusQLProvider


class ConcreteService(BaseService):
    """Concrete implementation of BaseService for testing."""

    def __init__(self, db_provider, **kwargs):
        super().__init__(db_provider, **kwargs)
        self.test_calls = []

    async def test_method(self, value: str) -> str:
        self.test_calls.append(value)
        return f"processed_{value}"


class TestBaseServiceInit:
    """Test BaseService initialization."""

    @pytest.fixture
    def mock_db(self):
        mock = Mock(spec=NexusQLProvider)
        mock.fetch_all = AsyncMock(return_value=[])
        mock.fetch_one = AsyncMock(return_value=None)
        mock.execute = AsyncMock()
        mock.execute_many = AsyncMock()
        return mock

    def test_initialization(self, mock_db):
        service = ConcreteService(mock_db)
        assert service.db is mock_db
        assert service.logger.name == "ConcreteService"

    def test_custom_logger_name(self, mock_db):
        service = ConcreteService(mock_db, logger_name="custom.logger")
        assert service.logger.name == "custom.logger"

    def test_has_database_methods(self, mock_db):
        service = ConcreteService(mock_db)
        assert hasattr(service, "_fetch_all")
        assert hasattr(service, "_fetch_one")
        assert hasattr(service, "_execute")
        assert hasattr(service, "_execute_query")
        assert hasattr(service, "_execute_many")


class TestBaseServiceDatabaseMethods:
    """Test BaseService database method delegation."""

    @pytest.fixture
    def mock_db(self):
        mock = Mock(spec=NexusQLProvider)
        mock.fetch_all = AsyncMock(return_value=[])
        mock.fetch_one = AsyncMock(return_value=None)
        mock.execute = AsyncMock()
        mock.execute_many = AsyncMock()
        return mock

    @pytest.fixture
    def service(self, mock_db):
        return ConcreteService(mock_db)

    @pytest.mark.asyncio
    async def test_fetch_all_delegates(self, service, mock_db):
        mock_db.fetch_all.return_value = [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
        ]
        result = await service._fetch_all(
            "SELECT * FROM users WHERE active = :active", {"active": True}
        )
        mock_db.fetch_all.assert_called_once_with(
            "SELECT * FROM users WHERE active = :active", {"active": True}
        )
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_fetch_all_empty(self, service, mock_db):
        mock_db.fetch_all.return_value = []
        result = await service._fetch_all("SELECT * FROM empty_table")
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_all_no_params(self, service, mock_db):
        mock_db.fetch_all.return_value = [{"id": 1}]
        await service._fetch_all("SELECT * FROM users")
        mock_db.fetch_all.assert_called_once_with("SELECT * FROM users", None)

    @pytest.mark.asyncio
    async def test_fetch_one_returns_dict(self, service, mock_db):
        mock_db.fetch_one.return_value = {"id": 1, "name": "alice"}
        result = await service._fetch_one(
            "SELECT * FROM users WHERE id = :id", {"id": 1}
        )
        assert result == {"id": 1, "name": "alice"}

    @pytest.mark.asyncio
    async def test_fetch_one_returns_none(self, service, mock_db):
        mock_db.fetch_one.return_value = None
        result = await service._fetch_one(
            "SELECT * FROM users WHERE id = :id", {"id": 999}
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_delegates(self, service, mock_db):
        await service._execute(
            "INSERT INTO users (name) VALUES (:name)", {"name": "charlie"}
        )
        mock_db.execute.assert_called_once_with(
            "INSERT INTO users (name) VALUES (:name)", {"name": "charlie"}
        )

    @pytest.mark.asyncio
    async def test_execute_query_delegates(self, service, mock_db):
        await service._execute_query(
            "UPDATE users SET active = :active WHERE id = :id",
            {"active": False, "id": 5},
        )
        mock_db.execute.assert_called_once_with(
            "UPDATE users SET active = :active WHERE id = :id",
            {"active": False, "id": 5},
        )

    @pytest.mark.asyncio
    async def test_execute_many_delegates(self, service, mock_db):
        data = [{"name": "alice"}, {"name": "bob"}]
        await service._execute_many("INSERT INTO users (name) VALUES (:name)", data)
        mock_db.execute_many.assert_called_once_with(
            "INSERT INTO users (name) VALUES (:name)", data
        )


class TestBaseServiceErrorHandling:
    """Test that BaseService wraps database errors in ServiceError."""

    @pytest.fixture
    def failing_db(self):
        mock = Mock(spec=NexusQLProvider)
        mock.fetch_all = AsyncMock(side_effect=ConnectionError("DB down"))
        mock.fetch_one = AsyncMock(side_effect=ConnectionError("DB down"))
        mock.execute = AsyncMock(side_effect=ConnectionError("DB down"))
        mock.execute_many = AsyncMock(side_effect=ConnectionError("DB down"))
        return mock

    @pytest.fixture
    def service(self, failing_db):
        return ConcreteService(failing_db)

    @pytest.mark.asyncio
    async def test_fetch_all_raises_service_error(self, service):
        with pytest.raises(ServiceError, match="Database fetch failed"):
            await service._fetch_all("SELECT 1")

    @pytest.mark.asyncio
    async def test_fetch_one_raises_service_error(self, service):
        with pytest.raises(ServiceError, match="Database fetch failed"):
            await service._fetch_one("SELECT 1")

    @pytest.mark.asyncio
    async def test_execute_raises_service_error(self, service):
        with pytest.raises(ServiceError, match="Database execute failed"):
            await service._execute("INSERT INTO t VALUES (1)")

    @pytest.mark.asyncio
    async def test_execute_query_raises_service_error(self, service):
        with pytest.raises(ServiceError, match="Database operation failed"):
            await service._execute_query("DROP TABLE t")

    @pytest.mark.asyncio
    async def test_execute_many_raises_service_error(self, service):
        with pytest.raises(ServiceError, match="Database execute many failed"):
            await service._execute_many("INSERT INTO t VALUES (:val)", [{"val": "x"}])

    @pytest.mark.asyncio
    async def test_error_chains_original(self, service):
        with pytest.raises(ServiceError) as exc_info:
            await service._fetch_all("SELECT 1")
        assert isinstance(exc_info.value.__cause__, ConnectionError)


class TestBaseServiceConcurrency:
    """Test concurrent operations through BaseService."""

    @pytest.fixture
    def mock_db(self):
        mock = Mock(spec=NexusQLProvider)
        mock.fetch_all = AsyncMock(return_value=[{"id": 1}])
        mock.fetch_one = AsyncMock(return_value={"id": 1})
        mock.execute = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_concurrent_fetch_all(self, mock_db):
        service = ConcreteService(mock_db)
        results = await asyncio.gather(
            service._fetch_all("SELECT * FROM t1"),
            service._fetch_all("SELECT * FROM t2"),
            service._fetch_all("SELECT * FROM t3"),
        )
        assert len(results) == 3
        assert mock_db.fetch_all.call_count == 3

    def test_independent_instances(self, mock_db):
        s1 = ConcreteService(mock_db)
        s2 = ConcreteService(mock_db)
        s1.test_calls.append("a")
        s2.test_calls.append("b")
        assert s1.test_calls == ["a"]
        assert s2.test_calls == ["b"]


class TestServiceExceptions:
    """Test the exception hierarchy."""

    def test_service_error_is_exception(self):
        assert issubclass(ServiceError, Exception)

    def test_validation_error_is_service_error(self):
        assert issubclass(ValidationError, ServiceError)

    def test_not_found_error_is_service_error(self):
        assert issubclass(NotFoundError, ServiceError)

    def test_authorization_error_is_service_error(self):
        assert issubclass(AuthorizationError, ServiceError)

    def test_service_error_message(self):
        err = ServiceError("something broke")
        assert str(err) == "something broke"

    def test_validation_error_caught_as_service_error(self):
        with pytest.raises(ServiceError):
            raise ValidationError("bad input")

    def test_not_found_error_caught_as_service_error(self):
        with pytest.raises(ServiceError):
            raise NotFoundError("user 42 not found")


class TestServiceRegistry:
    """Test the ServiceRegistry dependency injection container."""

    @pytest.fixture
    def registry(self):
        return ServiceRegistry()

    def test_register_and_get(self, registry):
        service = Mock()
        registry.register("my_service", service)
        assert registry.get("my_service") is service

    def test_get_nonexistent_raises(self, registry):
        with pytest.raises(ServiceError, match="Service not found"):
            registry.get("nonexistent")

    def test_get_all(self, registry):
        s1, s2 = Mock(), Mock()
        registry.register("s1", s1)
        registry.register("s2", s2)
        all_services = registry.get_all()
        assert all_services == {"s1": s1, "s2": s2}

    def test_get_all_returns_copy(self, registry):
        registry.register("s", Mock())
        copy = registry.get_all()
        copy["extra"] = Mock()
        assert "extra" not in registry.get_all()

    def test_clear(self, registry):
        registry.register("s", Mock())
        registry.clear()
        with pytest.raises(ServiceError):
            registry.get("s")

    def test_overwrite_registration(self, registry):
        s1, s2 = Mock(), Mock()
        registry.register("s", s1)
        registry.register("s", s2)
        assert registry.get("s") is s2

    def test_global_registry_exists(self):
        assert isinstance(service_registry, ServiceRegistry)
