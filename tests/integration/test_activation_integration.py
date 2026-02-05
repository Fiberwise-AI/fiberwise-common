"""
Integration tests for Phase 1 activation flow.

Tests end-to-end activation flow from agent creation through execution,
including backward compatibility with existing agents.
"""
import json
import pytest
import pytest_asyncio
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from fiberwise_common.database.provider import NexusQLProvider
from fiberwise_common.entities.fiber_agent import FiberAgent
from fiberwise_common.services.service_registry import ServiceRegistry
from fiberwise_common.services.llm_service_factory import LLMServiceFactory


# ============================================================================
# Test Agent Implementations
# ============================================================================

class TestClassAgent(FiberAgent):
    """Test CLASS-style agent for integration testing."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._description = "Test CLASS agent"
        self._version = "1.0.0"

    def get_dependencies(self) -> List[str]:
        return ['llm_service']

    async def run_agent_async(self, input_data: Any, **kwargs) -> Any:
        """Execute agent with LLM service."""
        prompt = input_data.get('prompt', 'default prompt')

        if hasattr(self, 'llm_service'):
            # Use LLM service
            response = await self.llm_service.generate_completion(
                prompt=prompt,
                model='gpt-4'
            )
            return {
                'result': response.get('text', ''),
                'agent_type': 'class',
                'used_llm': True
            }

        return {
            'result': f'Processed: {prompt}',
            'agent_type': 'class',
            'used_llm': False
        }


class TestFunctionAgent(FiberAgent):
    """Test function-style agent (legacy)."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._description = "Test function agent"
        self._version = "1.0.0"

    def run_agent(self, input_data: Any, **kwargs) -> Any:
        """Simple function execution."""
        value = input_data.get('value', 0)
        return {
            'result': value * 2,
            'agent_type': 'function'
        }


class LegacyAgent(FiberAgent):
    """Legacy agent using old naming conventions."""

    async def run_agent_async(self, input_data: Any, **kwargs) -> Any:
        """Legacy execution method."""
        return {
            'result': 'legacy output',
            'agent_type': 'legacy'
        }


# ============================================================================
# Database Setup
# ============================================================================

AGENTS_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    app_id TEXT NOT NULL,
    name TEXT NOT NULL,
    agent_type_id TEXT NOT NULL,
    description TEXT,
    configuration TEXT,
    file_path TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

AGENT_ACTIVATIONS_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS agent_activations (
    activation_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    app_id TEXT NOT NULL,
    organization_id TEXT NOT NULL,
    created_by INTEGER,
    input_data TEXT,
    output_data TEXT,
    status TEXT DEFAULT 'pending',
    error_message TEXT,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);
"""

LLM_PROVIDERS_TABLE_SQLITE = """
CREATE TABLE IF NOT EXISTS llm_providers (
    provider_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider_type TEXT NOT NULL,
    api_endpoint TEXT NOT NULL,
    configuration TEXT,
    is_active INTEGER DEFAULT 1,
    is_default INTEGER DEFAULT 0,
    is_system INTEGER DEFAULT 0,
    created_by INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


async def setup_test_database(db: NexusQLProvider):
    """Set up test database schema."""
    await db.execute(AGENTS_TABLE_SQLITE)
    await db.execute(AGENT_ACTIVATIONS_TABLE_SQLITE)
    await db.execute(LLM_PROVIDERS_TABLE_SQLITE)


async def insert_test_agent(
    db: NexusQLProvider,
    agent_id: str = "test-agent-1",
    agent_type: str = "class",
    **overrides
) -> Dict[str, Any]:
    """Insert a test agent into the database."""
    agent_data = {
        "agent_id": agent_id,
        "app_id": "test-app",
        "name": f"Test Agent {agent_id}",
        "agent_type_id": agent_type,
        "description": "Test agent for integration tests",
        "configuration": json.dumps({"test": True}),
        "file_path": "/test/agents/test_agent.py",
        "is_active": 1
    }
    agent_data.update(overrides)

    await db.execute("""
        INSERT INTO agents
        (agent_id, app_id, name, agent_type_id, description, configuration, file_path, is_active)
        VALUES
        (:agent_id, :app_id, :name, :agent_type_id, :description, :configuration, :file_path, :is_active)
    """, agent_data)

    return agent_data


async def insert_test_llm_provider(
    db: NexusQLProvider,
    provider_id: str = "test-openai",
    **overrides
) -> Dict[str, Any]:
    """Insert a test LLM provider into the database."""
    provider_data = {
        "provider_id": provider_id,
        "name": "Test OpenAI",
        "provider_type": "openai",
        "api_endpoint": "https://api.openai.com/v1",
        "configuration": json.dumps({
            "api_key": "sk-test-key",
            "default_model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2048
        }),
        "is_active": 1,
        "is_default": 1,
        "is_system": 1
    }
    provider_data.update(overrides)

    await db.execute("""
        INSERT INTO llm_providers
        (provider_id, name, provider_type, api_endpoint, configuration,
         is_active, is_default, is_system)
        VALUES
        (:provider_id, :name, :provider_type, :api_endpoint, :configuration,
         :is_active, :is_default, :is_system)
    """, provider_data)

    return provider_data


# ============================================================================
# Fixtures
# ============================================================================

@pytest_asyncio.fixture
async def sqlite_db(tmp_path: Path):
    """Create SQLite test database."""
    db_path = tmp_path / "test.db"
    db = NexusQLProvider(f"sqlite:///{db_path}")

    await setup_test_database(db)

    yield db

    # Cleanup
    await db.disconnect()


@pytest.fixture
def mock_activation_processor():
    """Mock activation processor for testing."""
    processor = Mock()
    processor.process_activation = AsyncMock()
    return processor


@pytest.fixture
def mock_llm_service():
    """Mock LLM service with realistic responses."""
    mock_service = AsyncMock()
    mock_service.generate_completion = AsyncMock(return_value={
        'status': 'completed',
        'text': 'Test LLM response',
        'model': 'gpt-4',
        'provider': 'openai',
        'finish_reason': 'stop'
    })
    return mock_service


# ============================================================================
# Test End-to-End Activation Flow
# ============================================================================

class TestActivationFlowIntegration:
    """Test complete activation flow from start to finish."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_class_agent_activation_flow(self, sqlite_db, mock_llm_service):
        """Test full CLASS agent activation flow."""
        # 1. Insert agent into database
        agent_data = await insert_test_agent(
            sqlite_db,
            agent_id="class-agent-1",
            agent_type="class"
        )

        # 2. Create agent instance
        agent = TestClassAgent(json.loads(agent_data['configuration']))
        agent.agent_id = agent_data['agent_id']
        agent.agent_type = agent_data['agent_type_id']

        # 3. Inject LLM service
        agent.inject_services({'llm_service': mock_llm_service})

        # 4. Execute agent
        result = await agent.execute({'prompt': 'test prompt'})

        # 5. Verify results
        assert result['agent_type'] == 'class'
        assert result['used_llm'] is True
        assert result['result'] == 'Test LLM response'

        # Verify LLM service was called
        mock_llm_service.generate_completion.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_function_agent_activation_flow(self, sqlite_db):
        """Test full function-style agent activation flow."""
        # 1. Insert agent into database
        agent_data = await insert_test_agent(
            sqlite_db,
            agent_id="function-agent-1",
            agent_type="function"
        )

        # 2. Create agent instance
        agent = TestFunctionAgent(json.loads(agent_data['configuration']))
        agent.agent_id = agent_data['agent_id']
        agent.agent_type = agent_data['agent_type_id']

        # 3. Execute agent (no services needed)
        result = await agent.execute({'value': 5})

        # 4. Verify results
        assert result['result'] == 10
        assert result['agent_type'] == 'function'

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_activation_with_service_registry(self, sqlite_db, mock_llm_service):
        """Test activation using ServiceRegistry for dependency injection."""
        # 1. Create service registry
        service_registry = ServiceRegistry(sqlite_db)
        service_registry.register_service('llm_service', mock_llm_service)

        # 2. Create agent
        agent = TestClassAgent()
        agent.agent_id = "test-agent"

        # 3. Inject dependencies from registry
        dependencies = agent.get_dependencies()
        services = service_registry.inject_dependencies(agent, dependencies)
        agent.inject_services(services)

        # 4. Execute agent
        result = await agent.execute({'prompt': 'test'})

        # 5. Verify
        assert result['used_llm'] is True
        assert agent.has_service('llm_service')


# ============================================================================
# Test Backward Compatibility
# ============================================================================

class TestBackwardCompatibility:
    """Test backward compatibility with existing agents."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_legacy_agent_works(self, sqlite_db):
        """Test legacy agents still work with new system."""
        # 1. Insert legacy agent
        agent_data = await insert_test_agent(
            sqlite_db,
            agent_id="legacy-agent-1",
            agent_type="llm_chat"  # Old type name
        )

        # 2. Create legacy agent
        agent = LegacyAgent()
        agent.agent_id = agent_data['agent_id']

        # 3. Execute
        result = await agent.execute({'test': 'data'})

        # 4. Verify
        assert result['result'] == 'legacy output'
        assert result['agent_type'] == 'legacy'

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_agent_type_migration(self, sqlite_db):
        """Test that old agent_type values can be migrated."""
        # Insert agents with old type names
        await insert_test_agent(sqlite_db, "agent-1", agent_type="function")
        await insert_test_agent(sqlite_db, "agent-2", agent_type="llm_chat")
        await insert_test_agent(sqlite_db, "agent-3", agent_type="class")

        # Simulate migration
        migration_map = {
            "function": "class",
            "llm_chat": "llm",
            "class": "class"
        }

        for old_type, new_type in migration_map.items():
            await sqlite_db.execute(
                "UPDATE agents SET agent_type_id = :new_type WHERE agent_type_id = :old_type",
                {"old_type": old_type, "new_type": new_type}
            )

        # Verify migration
        agents = await sqlite_db.fetch_all("SELECT agent_id, agent_type_id FROM agents")

        for agent in agents:
            assert agent['agent_type_id'] in ['class', 'llm']


# ============================================================================
# Test Error Scenarios
# ============================================================================

class TestErrorScenarios:
    """Test error handling in activation flow."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_agent_execution_error(self, sqlite_db):
        """Test handling of agent execution errors."""
        class ErrorAgent(FiberAgent):
            async def run_agent_async(self, input_data: Any, **kwargs) -> Any:
                raise ValueError("Test error")

        agent = ErrorAgent()
        agent.agent_id = "error-agent"

        # Execute with validation to catch error
        result = await agent.execute_with_validation({})

        assert result['success'] is False
        assert 'error' in result
        assert 'Test error' in result['error']

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_missing_llm_service(self, sqlite_db):
        """Test agent handles missing LLM service gracefully."""
        agent = TestClassAgent()
        agent.agent_id = "test-agent"

        # Execute without injecting LLM service
        result = await agent.execute({'prompt': 'test'})

        # Should fall back to non-LLM execution
        assert result['used_llm'] is False
        assert 'Processed:' in result['result']

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_llm_service_error(self, sqlite_db):
        """Test handling of LLM service errors."""
        # Create LLM service that raises errors
        error_service = AsyncMock()
        error_service.generate_completion = AsyncMock(
            side_effect=Exception("LLM API Error")
        )

        agent = TestClassAgent()
        agent.inject_services({'llm_service': error_service})

        # Execute and expect error to propagate
        with pytest.raises(Exception) as exc_info:
            await agent.execute({'prompt': 'test'})

        assert 'LLM API Error' in str(exc_info.value)


# ============================================================================
# Test Agent Lifecycle
# ============================================================================

class TestAgentLifecycle:
    """Test complete agent lifecycle from creation to execution."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_lifecycle(self, sqlite_db, mock_llm_service):
        """Test complete agent lifecycle."""
        # 1. Create agent in database
        agent_data = await insert_test_agent(
            sqlite_db,
            agent_id="lifecycle-agent",
            agent_type="class"
        )

        # 2. Load agent from database
        db_agent = await sqlite_db.fetch_one(
            "SELECT * FROM agents WHERE agent_id = :agent_id",
            {"agent_id": "lifecycle-agent"}
        )
        assert db_agent is not None

        # 3. Create agent instance with config
        config = json.loads(db_agent['configuration'])
        agent = TestClassAgent(config)
        agent.agent_id = db_agent['agent_id']
        agent.agent_type = db_agent['agent_type_id']

        # 4. Set up services
        agent.inject_services({'llm_service': mock_llm_service})

        # 5. Execute agent
        result = await agent.execute_with_validation({'prompt': 'test'})

        # 6. Verify complete flow
        assert result['success'] is True
        assert 'output' in result
        assert result['output']['used_llm'] is True
        assert 'execution_time' in result
        assert result['input_validation']['valid'] is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multiple_activations_same_agent(self, sqlite_db, mock_llm_service):
        """Test multiple activations of the same agent."""
        # Create agent
        await insert_test_agent(sqlite_db, "multi-agent", agent_type="class")

        agent = TestClassAgent()
        agent.agent_id = "multi-agent"
        agent.inject_services({'llm_service': mock_llm_service})

        # Execute multiple times
        results = []
        for i in range(3):
            result = await agent.execute({'prompt': f'test {i}'})
            results.append(result)

        # Verify all executions succeeded
        assert len(results) == 3
        for result in results:
            assert result['used_llm'] is True
            assert result['agent_type'] == 'class'

        # Verify LLM service was called 3 times
        assert mock_llm_service.generate_completion.call_count == 3


# ============================================================================
# Test Agent Types
# ============================================================================

class TestAgentTypes:
    """Test different agent type scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_class_agent_type(self, sqlite_db):
        """Test CLASS agent type."""
        await insert_test_agent(sqlite_db, "class-agent", agent_type="class")

        agent = TestClassAgent()
        agent.agent_id = "class-agent"
        agent.agent_type = "class"

        result = await agent.execute({'prompt': 'test'})
        assert result['agent_type'] == 'class'

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_function_agent_type(self, sqlite_db):
        """Test function-style agent type."""
        await insert_test_agent(sqlite_db, "func-agent", agent_type="function")

        agent = TestFunctionAgent()
        agent.agent_id = "func-agent"
        agent.agent_type = "function"

        result = await agent.execute({'value': 3})
        assert result['agent_type'] == 'function'

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_llm_agent_type(self, sqlite_db, mock_llm_service):
        """Test LLM agent type (pure LLM execution)."""
        await insert_test_agent(sqlite_db, "llm-agent", agent_type="llm")

        # LLM agents would use configuration, not code
        # This simulates that behavior
        agent = TestClassAgent()
        agent.agent_id = "llm-agent"
        agent.agent_type = "llm"
        agent.inject_services({'llm_service': mock_llm_service})

        result = await agent.execute({'prompt': 'test'})
        assert result['used_llm'] is True


# ============================================================================
# Test Service Injection Patterns
# ============================================================================

class TestServiceInjection:
    """Test various service injection patterns."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_inject_before_execution(self, mock_llm_service):
        """Test injecting services before execution."""
        agent = TestClassAgent()

        # Inject before execution
        agent.inject_services({'llm_service': mock_llm_service})

        result = await agent.execute({'prompt': 'test'})
        assert result['used_llm'] is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multiple_service_injection(self, mock_llm_service):
        """Test injecting multiple services."""
        agent = TestClassAgent()

        mock_app = Mock()
        mock_oauth = Mock()

        # Inject multiple services
        agent.inject_services({
            'llm_service': mock_llm_service,
            'fiber_app': mock_app,
            'oauth_service': mock_oauth
        })

        assert agent.has_service('llm_service')
        assert agent.has_service('fiber_app')
        assert agent.has_service('oauth_service')

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_service_registry_injection(self, sqlite_db, mock_llm_service):
        """Test injection via ServiceRegistry."""
        registry = ServiceRegistry(sqlite_db)
        registry.register_service('llm_service', mock_llm_service)

        agent = TestClassAgent()

        # Get services from registry
        dependencies = agent.get_dependencies()
        services = registry.inject_dependencies(agent, dependencies)
        agent.inject_services(services)

        assert agent.has_service('llm_service')


# ============================================================================
# Test Database Operations
# ============================================================================

class TestDatabaseOperations:
    """Test database operations during activation flow."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_load_agent_from_db(self, sqlite_db):
        """Test loading agent configuration from database."""
        # Insert agent
        await insert_test_agent(
            sqlite_db,
            "db-agent",
            agent_type="class",
            configuration=json.dumps({"custom_param": "value"})
        )

        # Load from DB
        agent_row = await sqlite_db.fetch_one(
            "SELECT * FROM agents WHERE agent_id = :id",
            {"id": "db-agent"}
        )

        assert agent_row is not None
        assert agent_row['agent_id'] == "db-agent"

        config = json.loads(agent_row['configuration'])
        assert config['custom_param'] == "value"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_query_agents_by_type(self, sqlite_db):
        """Test querying agents by type."""
        # Insert multiple agents
        await insert_test_agent(sqlite_db, "agent-1", agent_type="class")
        await insert_test_agent(sqlite_db, "agent-2", agent_type="class")
        await insert_test_agent(sqlite_db, "agent-3", agent_type="llm")

        # Query CLASS agents
        class_agents = await sqlite_db.fetch_all(
            "SELECT * FROM agents WHERE agent_type_id = :type",
            {"type": "class"}
        )

        assert len(class_agents) == 2

        # Query LLM agents
        llm_agents = await sqlite_db.fetch_all(
            "SELECT * FROM agents WHERE agent_type_id = :type",
            {"type": "llm"}
        )

        assert len(llm_agents) == 1


# ============================================================================
# Test Performance
# ============================================================================

class TestPerformance:
    """Test performance characteristics of activation flow."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_concurrent_activations(self, sqlite_db, mock_llm_service):
        """Test multiple concurrent agent activations."""
        import asyncio

        # Create multiple agents
        agents = []
        for i in range(5):
            await insert_test_agent(sqlite_db, f"concurrent-agent-{i}", agent_type="class")

            agent = TestClassAgent()
            agent.agent_id = f"concurrent-agent-{i}"
            agent.inject_services({'llm_service': mock_llm_service})
            agents.append(agent)

        # Execute concurrently
        tasks = [
            agent.execute({'prompt': f'test {i}'})
            for i, agent in enumerate(agents)
        ]

        results = await asyncio.gather(*tasks)

        # Verify all succeeded
        assert len(results) == 5
        for result in results:
            assert result['used_llm'] is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_execution_time_tracking(self, sqlite_db):
        """Test that execution time is tracked."""
        agent = TestFunctionAgent()
        agent.agent_id = "time-test"

        result = await agent.execute_with_validation({'value': 5})

        assert 'execution_time' in result
        assert result['execution_time'] >= 0
        assert isinstance(result['execution_time'], float)
