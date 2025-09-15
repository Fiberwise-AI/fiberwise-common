"""
Unit tests for fiberwise_common.database.base module.

This module tests the abstract DatabaseProvider base class and its
query adapter functionality.
"""

import pytest
from unittest.mock import Mock, patch
from abc import ABC, abstractmethod

from fiberwise_common.database.base import DatabaseProvider
from fiberwise_common.database.query_adapter import ParameterStyle


class ConcreteDatabaseProvider(DatabaseProvider):
    """Concrete implementation for testing the abstract base class."""
    
    @property
    def provider(self) -> str:
        return "test_provider"
    
    async def connect(self) -> None:
        pass
    
    async def disconnect(self) -> None:
        pass
    
    async def transaction(self):
        yield None
    
    async def fetch_one(self, query: str, *args):
        return None
    
    async def fetch_all(self, query: str, *args):
        return []
    
    async def fetch_val(self, query: str, *args):
        return None
    
    async def execute(self, query: str, *args):
        return None
    
    async def file_exists(self, path: str) -> bool:
        return False
    
    async def migrate(self, migration_files) -> bool:
        return True


class TestDatabaseProvider:
    """Test suite for DatabaseProvider abstract base class."""
    
    def test_provider_is_abstract(self):
        """Test that DatabaseProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DatabaseProvider()
    
    def test_concrete_implementation_instantiation(self):
        """Test that concrete implementation can be instantiated."""
        provider = ConcreteDatabaseProvider()
        assert isinstance(provider, DatabaseProvider)
        assert provider.provider == "test_provider"
    
    def test_query_adapter_lazy_initialization(self):
        """Test that query adapter is created lazily."""
        provider = ConcreteDatabaseProvider()
        
        # Initially None
        assert provider._query_adapter is None
        
        # Gets created on first access
        adapter = provider.query_adapter
        assert adapter is not None
        
        # Same instance on subsequent access
        adapter2 = provider.query_adapter
        assert adapter is adapter2
    
    @patch('fiberwise_common.database.base.create_query_adapter')
    def test_query_adapter_creation_with_provider_type(self, mock_create_adapter):
        """Test that query adapter is created with correct provider type."""
        mock_adapter = Mock()
        mock_create_adapter.return_value = mock_adapter
        
        provider = ConcreteDatabaseProvider()
        adapter = provider.query_adapter
        
        mock_create_adapter.assert_called_once_with("test_provider")
        assert adapter is mock_adapter
    
    def test_adapt_query_delegates_to_adapter(self):
        """Test that adapt_query method delegates to query adapter."""
        provider = ConcreteDatabaseProvider()
        
        # Mock the query adapter
        mock_adapter = Mock()
        mock_adapter.adapt_query_and_params.return_value = ("adapted_query", ["adapted_params"])
        provider._query_adapter = mock_adapter
        
        # Test delegation
        result = provider.adapt_query("SELECT * FROM table", ["param1", "param2"])
        
        mock_adapter.adapt_query_and_params.assert_called_once_with(
            "SELECT * FROM table", 
            ["param1", "param2"], 
            ParameterStyle.POSTGRESQL
        )
        assert result == ("adapted_query", ["adapted_params"])
    
    def test_adapt_query_with_custom_source_style(self):
        """Test adapt_query with custom source parameter style."""
        provider = ConcreteDatabaseProvider()
        
        # Mock the query adapter
        mock_adapter = Mock()
        mock_adapter.adapt_query_and_params.return_value = ("adapted", "params")
        provider._query_adapter = mock_adapter
        
        # Test with custom source style
        provider.adapt_query("SELECT ?", ["param"], ParameterStyle.SQLITE)
        
        mock_adapter.adapt_query_and_params.assert_called_once_with(
            "SELECT ?", 
            ["param"], 
            ParameterStyle.SQLITE
        )
    
    def test_adapt_query_with_none_params(self):
        """Test adapt_query with None parameters."""
        provider = ConcreteDatabaseProvider()
        
        # Mock the query adapter
        mock_adapter = Mock()
        mock_adapter.adapt_query_and_params.return_value = ("query", None)
        provider._query_adapter = mock_adapter
        
        # Test with None params
        result = provider.adapt_query("SELECT 1", None)
        
        mock_adapter.adapt_query_and_params.assert_called_once_with(
            "SELECT 1", 
            None, 
            ParameterStyle.POSTGRESQL
        )
        assert result == ("query", None)
    
    def test_all_abstract_methods_defined(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = {
            'provider',
            'connect', 
            'disconnect',
            'transaction',
            'fetch_one',
            'fetch_all', 
            'fetch_val',
            'execute',
            'file_exists',
            'migrate'
        }
        
        # Check that DatabaseProvider has these abstract methods
        db_abstract_methods = {name for name, method in DatabaseProvider.__dict__.items()
                             if getattr(method, '__isabstractmethod__', False)}
        
        assert abstract_methods.issubset(db_abstract_methods)
    
    def test_concrete_implementation_has_all_methods(self):
        """Test that concrete implementation has all required methods."""
        provider = ConcreteDatabaseProvider()
        
        # Check all abstract methods are implemented
        assert hasattr(provider, 'provider')
        assert hasattr(provider, 'connect')
        assert hasattr(provider, 'disconnect')
        assert hasattr(provider, 'transaction')
        assert hasattr(provider, 'fetch_one')
        assert hasattr(provider, 'fetch_all')
        assert hasattr(provider, 'fetch_val')
        assert hasattr(provider, 'execute')
        assert hasattr(provider, 'file_exists')
        assert hasattr(provider, 'migrate')
        
        # Check that provider property is not a method
        assert isinstance(provider.provider, str)
        
        # Check that other methods are callable
        assert callable(provider.connect)
        assert callable(provider.disconnect)
        assert callable(provider.fetch_one)
        assert callable(provider.fetch_all)
        assert callable(provider.fetch_val)
        assert callable(provider.execute)
        assert callable(provider.file_exists)
        assert callable(provider.migrate)


class TestDatabaseProviderMethodSignatures:
    """Test method signatures of abstract methods."""
    
    def test_fetch_one_signature(self):
        """Test fetch_one method signature."""
        provider = ConcreteDatabaseProvider()
        
        # Should accept query and variable args
        import inspect
        sig = inspect.signature(provider.fetch_one)
        params = list(sig.parameters.keys())
        
        assert 'query' in params
        assert 'args' in params
        
        # Should have *args parameter
        args_param = sig.parameters['args']
        assert args_param.kind == inspect.Parameter.VAR_POSITIONAL
    
    def test_fetch_all_signature(self):
        """Test fetch_all method signature."""
        provider = ConcreteDatabaseProvider()
        
        import inspect
        sig = inspect.signature(provider.fetch_all)
        params = list(sig.parameters.keys())
        
        assert 'query' in params
        assert 'args' in params
        
        args_param = sig.parameters['args']
        assert args_param.kind == inspect.Parameter.VAR_POSITIONAL
    
    def test_fetch_val_signature(self):
        """Test fetch_val method signature."""
        provider = ConcreteDatabaseProvider()
        
        import inspect
        sig = inspect.signature(provider.fetch_val)
        params = list(sig.parameters.keys())
        
        assert 'query' in params
        assert 'args' in params
        
        args_param = sig.parameters['args']
        assert args_param.kind == inspect.Parameter.VAR_POSITIONAL
    
    def test_execute_signature(self):
        """Test execute method signature."""
        provider = ConcreteDatabaseProvider()
        
        import inspect
        sig = inspect.signature(provider.execute)
        params = list(sig.parameters.keys())
        
        assert 'query' in params
        assert 'args' in params
        
        args_param = sig.parameters['args']
        assert args_param.kind == inspect.Parameter.VAR_POSITIONAL


class TestDatabaseProviderInheritance:
    """Test inheritance behavior of DatabaseProvider."""
    
    def test_subclass_must_implement_all_methods(self):
        """Test that subclass must implement all abstract methods."""
        
        # This should fail because not all methods are implemented
        with pytest.raises(TypeError):
            class IncompleteProvider(DatabaseProvider):
                @property
                def provider(self):
                    return "incomplete"
                
                async def connect(self):
                    pass
                
                # Missing other required methods
            
            IncompleteProvider()
    
    def test_can_override_query_adapter_behavior(self):
        """Test that subclasses can override query adapter behavior."""
        
        class CustomProvider(ConcreteDatabaseProvider):
            @property
            def provider(self) -> str:
                return "custom_provider"
        
        provider = CustomProvider()
        assert provider.provider == "custom_provider"
        
        # Should create adapter for custom provider type
        with patch('fiberwise_common.database.base.create_query_adapter') as mock_create:
            mock_create.return_value = Mock()
            _ = provider.query_adapter
            mock_create.assert_called_once_with("custom_provider")
    
    def test_can_add_additional_methods(self):
        """Test that subclasses can add additional methods."""
        
        class ExtendedProvider(ConcreteDatabaseProvider):
            def custom_method(self):
                return "custom_result"
            
            async def bulk_insert(self, table, records):
                return len(records)
        
        provider = ExtendedProvider()
        
        # Base functionality works
        assert provider.provider == "test_provider"
        
        # Additional methods work
        assert provider.custom_method() == "custom_result"
        
        # Check that new async method exists
        assert callable(provider.bulk_insert)


class FailingConcreteProvider(DatabaseProvider):
    """Provider that simulates failures for testing error handling."""
    
    @property
    def provider(self) -> str:
        return "failing_provider"
    
    async def connect(self) -> None:
        raise RuntimeError("Connection failed")
    
    async def disconnect(self) -> None:
        raise RuntimeError("Disconnect failed")
    
    async def transaction(self):
        raise RuntimeError("Transaction failed")
        yield None  # Never reached
    
    async def fetch_one(self, query: str, *args):
        raise RuntimeError("Fetch one failed")
    
    async def fetch_all(self, query: str, *args):
        raise RuntimeError("Fetch all failed")
    
    async def fetch_val(self, query: str, *args):
        raise RuntimeError("Fetch val failed")
    
    async def execute(self, query: str, *args):
        raise RuntimeError("Execute failed")
    
    async def file_exists(self, path: str) -> bool:
        raise RuntimeError("File exists check failed")
    
    async def migrate(self, migration_files) -> bool:
        raise RuntimeError("Migration failed")


class TestDatabaseProviderErrorHandling:
    """Test error handling in database provider methods."""
    
    @pytest.mark.asyncio
    async def test_connect_error_propagation(self):
        """Test that connect errors are properly propagated."""
        provider = FailingConcreteProvider()
        
        with pytest.raises(RuntimeError, match="Connection failed"):
            await provider.connect()
    
    @pytest.mark.asyncio
    async def test_disconnect_error_propagation(self):
        """Test that disconnect errors are properly propagated."""
        provider = FailingConcreteProvider()
        
        with pytest.raises(RuntimeError, match="Disconnect failed"):
            await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_fetch_one_error_propagation(self):
        """Test that fetch_one errors are properly propagated."""
        provider = FailingConcreteProvider()
        
        with pytest.raises(RuntimeError, match="Fetch one failed"):
            await provider.fetch_one("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_fetch_all_error_propagation(self):
        """Test that fetch_all errors are properly propagated."""
        provider = FailingConcreteProvider()
        
        with pytest.raises(RuntimeError, match="Fetch all failed"):
            await provider.fetch_all("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_fetch_val_error_propagation(self):
        """Test that fetch_val errors are properly propagated."""
        provider = FailingConcreteProvider()
        
        with pytest.raises(RuntimeError, match="Fetch val failed"):
            await provider.fetch_val("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_execute_error_propagation(self):
        """Test that execute errors are properly propagated."""
        provider = FailingConcreteProvider()
        
        with pytest.raises(RuntimeError, match="Execute failed"):
            await provider.execute("INSERT INTO table VALUES (1)")
    
    @pytest.mark.asyncio
    async def test_file_exists_error_propagation(self):
        """Test that file_exists errors are properly propagated."""
        provider = FailingConcreteProvider()
        
        with pytest.raises(RuntimeError, match="File exists check failed"):
            await provider.file_exists("/path/to/file")
    
    @pytest.mark.asyncio
    async def test_migrate_error_propagation(self):
        """Test that migrate errors are properly propagated."""
        provider = FailingConcreteProvider()
        
        with pytest.raises(RuntimeError, match="Migration failed"):
            await provider.migrate(["migration1.sql"])
    
    @pytest.mark.asyncio
    async def test_transaction_error_propagation(self):
        """Test that transaction errors are properly propagated."""
        provider = FailingConcreteProvider()
        
        with pytest.raises(RuntimeError, match="Transaction failed"):
            async with provider.transaction():
                pass


if __name__ == "__main__":
    pytest.main([__file__])