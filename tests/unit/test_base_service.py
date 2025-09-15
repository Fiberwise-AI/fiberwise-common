"""
Unit tests for fiberwise_common.services.base_service module.

Tests the BaseService class that other services inherit from.
"""
import pytest
from typing import Any, List, Dict, Optional
from unittest.mock import Mock, AsyncMock, patch

from fiberwise_common.services.base_service import BaseService
from fiberwise_common.database.base import DatabaseProvider


class ConcreteService(BaseService):
    """Concrete implementation of BaseService for testing."""
    
    def __init__(self, db_provider: DatabaseProvider):
        super().__init__(db_provider)
        self.test_calls = []
    
    async def test_method(self, value: str) -> str:
        """Test method that uses database operations."""
        self.test_calls.append(value)
        return f"processed_{value}"


class TestBaseService:
    """Test BaseService functionality."""

    @pytest.fixture
    def mock_database_provider(self) -> Mock:
        """Create mock database provider."""
        db_mock = Mock(spec=DatabaseProvider)
        db_mock.fetch_all = AsyncMock(return_value=[])
        db_mock.fetch_one = AsyncMock(return_value=None)
        db_mock.execute = AsyncMock()
        db_mock.execute_many = AsyncMock()
        db_mock.begin_transaction = AsyncMock()
        db_mock.commit_transaction = AsyncMock()
        db_mock.rollback_transaction = AsyncMock()
        return db_mock

    @pytest.fixture
    def base_service(self, mock_database_provider: Mock) -> ConcreteService:
        """Create concrete service instance for testing."""
        return ConcreteService(mock_database_provider)

    def test_base_service_initialization(self, mock_database_provider: Mock):
        """Test BaseService initialization."""
        service = ConcreteService(mock_database_provider)
        
        assert service.db == mock_database_provider
        assert hasattr(service, '_fetch_all')
        assert hasattr(service, '_fetch_one')
        assert hasattr(service, '_execute')
        assert hasattr(service, '_execute_many')

    def test_base_service_cannot_instantiate_directly(self):
        """Test that BaseService cannot be instantiated directly."""
        mock_db = Mock(spec=DatabaseProvider)
        
        # BaseService is not abstract, but typically wouldn't be instantiated directly
        # This test verifies our concrete implementation works
        service = ConcreteService(mock_db)
        assert isinstance(service, BaseService)

    @pytest.mark.asyncio
    async def test_fetch_all_delegation(self, base_service: ConcreteService, mock_database_provider: Mock):
        """Test that _fetch_all delegates to database provider."""
        mock_database_provider.fetch_all.return_value = [("row1",), ("row2",)]
        
        result = await base_service._fetch_all("SELECT * FROM test", ("param1",))
        
        mock_database_provider.fetch_all.assert_called_once_with("SELECT * FROM test", ("param1",))
        assert result == [("row1",), ("row2",)]

    @pytest.mark.asyncio
    async def test_fetch_one_delegation(self, base_service: ConcreteService, mock_database_provider: Mock):
        """Test that _fetch_one delegates to database provider."""
        mock_database_provider.fetch_one.return_value = ("single_row",)
        
        result = await base_service._fetch_one("SELECT * FROM test WHERE id = ?", (1,))
        
        mock_database_provider.fetch_one.assert_called_once_with("SELECT * FROM test WHERE id = ?", (1,))
        assert result == ("single_row",)

    @pytest.mark.asyncio
    async def test_execute_delegation(self, base_service: ConcreteService, mock_database_provider: Mock):
        """Test that _execute delegates to database provider."""
        await base_service._execute("INSERT INTO test VALUES (?)", ("value1",))
        
        mock_database_provider.execute.assert_called_once_with("INSERT INTO test VALUES (?)", ("value1",))

    @pytest.mark.asyncio
    async def test_execute_many_delegation(self, base_service: ConcreteService, mock_database_provider: Mock):
        """Test that _execute_many delegates to database provider."""
        data = [("value1",), ("value2",), ("value3",)]
        
        await base_service._execute_many("INSERT INTO test VALUES (?)", data)
        
        mock_database_provider.execute_many.assert_called_once_with("INSERT INTO test VALUES (?)", data)

    @pytest.mark.asyncio
    async def test_database_methods_with_no_params(self, base_service: ConcreteService, mock_database_provider: Mock):
        """Test database methods called without parameters."""
        await base_service._fetch_all("SELECT * FROM test")
        await base_service._fetch_one("SELECT COUNT(*) FROM test")
        await base_service._execute("UPDATE test SET active = 1")
        
        mock_database_provider.fetch_all.assert_called_once_with("SELECT * FROM test", None)
        mock_database_provider.fetch_one.assert_called_once_with("SELECT COUNT(*) FROM test", None)
        mock_database_provider.execute.assert_called_once_with("UPDATE test SET active = 1", None)

    @pytest.mark.asyncio
    async def test_service_inheritance_functionality(self, base_service: ConcreteService):
        """Test that derived service maintains its own functionality."""
        result = await base_service.test_method("test_input")
        
        assert result == "processed_test_input"
        assert "test_input" in base_service.test_calls

    @pytest.mark.asyncio
    async def test_database_error_propagation(self, base_service: ConcreteService, mock_database_provider: Mock):
        """Test that database errors are properly propagated."""
        mock_database_provider.fetch_all.side_effect = Exception("Database error")
        
        with pytest.raises(Exception, match="Database error"):
            await base_service._fetch_all("SELECT * FROM test")

    @pytest.mark.parametrize("method_name,db_method", [
        ("_fetch_all", "fetch_all"),
        ("_fetch_one", "fetch_one"),
        ("_execute", "execute"),
        ("_execute_many", "execute_many")
    ])
    @pytest.mark.asyncio
    async def test_all_database_methods_parametrized(self, base_service: ConcreteService, 
                                                   mock_database_provider: Mock, method_name: str, db_method: str):
        """Test all database methods with parametrized approach."""
        service_method = getattr(base_service, method_name)
        db_mock_method = getattr(mock_database_provider, db_method)
        
        # Call the service method
        if method_name == "_execute_many":
            await service_method("SQL", [("data",)])
            db_mock_method.assert_called_once_with("SQL", [("data",)])
        else:
            await service_method("SQL", ("param",))
            db_mock_method.assert_called_once_with("SQL", ("param",))


class TestBaseServiceAdvanced:
    """Advanced tests for BaseService."""

    @pytest.fixture
    def failing_db_provider(self) -> Mock:
        """Create database provider that fails operations."""
        db_mock = Mock(spec=DatabaseProvider)
        db_mock.fetch_all = AsyncMock(side_effect=ConnectionError("DB Connection failed"))
        db_mock.fetch_one = AsyncMock(side_effect=ConnectionError("DB Connection failed"))
        db_mock.execute = AsyncMock(side_effect=ConnectionError("DB Connection failed"))
        db_mock.execute_many = AsyncMock(side_effect=ConnectionError("DB Connection failed"))
        return db_mock

    @pytest.mark.asyncio
    async def test_service_with_failing_database(self, failing_db_provider: Mock):
        """Test service behavior when database operations fail."""
        service = ConcreteService(failing_db_provider)
        
        with pytest.raises(ServiceError):
            await service._fetch_all("SELECT * FROM test")
        
        with pytest.raises(ServiceError):
            await service._fetch_one("SELECT * FROM test")
        
        with pytest.raises(ServiceError):
            await service._execute("INSERT INTO test VALUES (1)")

    @pytest.mark.asyncio
    async def test_concurrent_service_operations(self, mock_database_provider: Mock):
        """Test concurrent operations through BaseService."""
        import asyncio
        
        service = ConcreteService(mock_database_provider)
        mock_database_provider.fetch_all.return_value = [("concurrent_result",)]
        
        # Execute multiple operations concurrently
        tasks = [
            service._fetch_all("SELECT * FROM table1"),
            service._fetch_all("SELECT * FROM table2"),
            service._fetch_all("SELECT * FROM table3")
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert mock_database_provider.fetch_all.call_count == 3

    def test_multiple_service_instances(self, mock_database_provider: Mock):
        """Test that multiple service instances work independently."""
        service1 = ConcreteService(mock_database_provider)
        service2 = ConcreteService(mock_database_provider)
        
        assert service1.db == service2.db  # Same database provider
        assert service1.test_calls != service2.test_calls  # Different instance data
        
        service1.test_calls.append("test1")
        service2.test_calls.append("test2")
        
        assert "test1" in service1.test_calls
        assert "test1" not in service2.test_calls
        assert "test2" in service2.test_calls
        assert "test2" not in service1.test_calls

    @pytest.mark.asyncio
    async def test_service_with_complex_data_types(self, mock_database_provider: Mock):
        """Test service with complex parameter types."""
        service = ConcreteService(mock_database_provider)
        
        # Test with various parameter types
        complex_params = {
            "dict_param": {"key": "value", "nested": {"data": 123}},
            "list_param": [1, 2, 3, "string"],
            "tuple_param": (1, "two", 3.0),
            "none_param": None
        }
        
        for param_name, param_value in complex_params.items():
            await service._execute("INSERT INTO test VALUES (?)", (param_value,))
            mock_database_provider.execute.assert_called_with("INSERT INTO test VALUES (?)", (param_value,))

    @pytest.mark.asyncio
    async def test_service_method_chaining_simulation(self, base_service: ConcreteService, mock_database_provider: Mock):
        """Test simulation of method chaining through multiple database operations."""
        mock_database_provider.fetch_one.return_value = (1, "test_user")
        mock_database_provider.fetch_all.return_value = [(1, "post1"), (2, "post2")]
        
        # Simulate a workflow that uses multiple database operations
        user = await base_service._fetch_one("SELECT id, name FROM users WHERE name = ?", ("test_user",))
        assert user is not None
        
        user_posts = await base_service._fetch_all("SELECT id, title FROM posts WHERE user_id = ?", (user[0],))
        assert len(user_posts) == 2
        
        await base_service._execute("UPDATE users SET last_accessed = NOW() WHERE id = ?", (user[0],))
        
        # Verify all database calls were made
        assert mock_database_provider.fetch_one.call_count == 1
        assert mock_database_provider.fetch_all.call_count == 1
        assert mock_database_provider.execute.call_count == 1


class TestServiceIntegration:
    """Integration-style tests for BaseService with more realistic scenarios."""

    class UserService(BaseService):
        """Example user service extending BaseService."""
        
        async def create_user(self, username: str, email: str) -> int:
            """Create a new user."""
            await self._execute(
                "INSERT INTO users (username, email) VALUES (?, ?)",
                (username, email)
            )
            result = await self._fetch_one("SELECT last_insert_rowid()")
            return result[0] if result else 0
        
        async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
            """Get user by ID."""
            result = await self._fetch_one(
                "SELECT id, username, email FROM users WHERE id = ?",
                (user_id,)
            )
            if result:
                return {"id": result[0], "username": result[1], "email": result[2]}
            return None
        
        async def list_users(self) -> List[Dict[str, Any]]:
            """List all users."""
            results = await self._fetch_all("SELECT id, username, email FROM users")
            return [
                {"id": row[0], "username": row[1], "email": row[2]} 
                for row in results
            ]

    @pytest.fixture
    def user_service(self, mock_database_provider: Mock) -> UserService:
        """Create user service for testing."""
        return self.UserService(mock_database_provider)

    @pytest.mark.asyncio
    async def test_user_service_create_user(self, user_service: UserService, mock_database_provider: Mock):
        """Test creating user through service."""
        mock_database_provider.fetch_one.return_value = (123,)  # Simulated last_insert_rowid
        
        user_id = await user_service.create_user("testuser", "test@example.com")
        
        assert user_id == 123
        mock_database_provider.execute.assert_called_once_with(
            "INSERT INTO users (username, email) VALUES (?, ?)",
            ("testuser", "test@example.com")
        )
        mock_database_provider.fetch_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_user_service_get_user(self, user_service: UserService, mock_database_provider: Mock):
        """Test getting user through service."""
        mock_database_provider.fetch_one.return_value = (1, "testuser", "test@example.com")
        
        user = await user_service.get_user(1)
        
        assert user is not None
        assert user["id"] == 1
        assert user["username"] == "testuser"
        assert user["email"] == "test@example.com"
        
        mock_database_provider.fetch_one.assert_called_once_with(
            "SELECT id, username, email FROM users WHERE id = ?",
            (1,)
        )

    @pytest.mark.asyncio
    async def test_user_service_get_nonexistent_user(self, user_service: UserService, mock_database_provider: Mock):
        """Test getting nonexistent user."""
        mock_database_provider.fetch_one.return_value = None
        
        user = await user_service.get_user(999)
        
        assert user is None

    @pytest.mark.asyncio
    async def test_user_service_list_users(self, user_service: UserService, mock_database_provider: Mock):
        """Test listing users through service."""
        mock_database_provider.fetch_all.return_value = [
            (1, "user1", "user1@example.com"),
            (2, "user2", "user2@example.com"),
            (3, "user3", "user3@example.com")
        ]
        
        users = await user_service.list_users()
        
        assert len(users) == 3
        assert users[0]["username"] == "user1"
        assert users[2]["email"] == "user3@example.com"
        
        mock_database_provider.fetch_all.assert_called_once_with("SELECT id, username, email FROM users")

    @pytest.mark.asyncio
    async def test_service_workflow_integration(self, user_service: UserService, mock_database_provider: Mock):
        """Test complete service workflow."""
        # Setup mock responses for workflow
        mock_database_provider.fetch_one.side_effect = [
            (1,),  # last_insert_rowid for create_user
            (1, "newuser", "new@example.com")  # get_user response
        ]
        
        # Execute workflow: create user then get user
        user_id = await user_service.create_user("newuser", "new@example.com")
        created_user = await user_service.get_user(user_id)
        
        assert user_id == 1
        assert created_user is not None
        assert created_user["username"] == "newuser"
        
        # Verify database interaction sequence
        assert mock_database_provider.execute.call_count == 1
        assert mock_database_provider.fetch_one.call_count == 2