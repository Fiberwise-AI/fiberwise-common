"""
Unit tests for FiberAgent class covering Phase 1 integration.

Tests CLASS agent execution, LLM agent execution, schema validation,
service injection, and backward compatibility.
"""
import pytest
import pytest_asyncio
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from fiberwise_common.entities.fiber_agent import FiberAgent, FiberInjectable


# ============================================================================
# Test Agent Implementations
# ============================================================================

class ClassStyleAgent(FiberAgent):
    """Test agent using class-based implementation."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._description = "Class-based test agent"
        self._version = "1.0.0"

    def get_dependencies(self) -> List[str]:
        """Declare that this agent needs LLM service."""
        return ['llm_service']

    def run_agent(self, input_data: Any, **kwargs) -> Any:
        """Synchronous execution."""
        prompt = input_data.get('prompt', '')
        return {
            'result': f"Processed: {prompt}",
            'agent_type': 'class'
        }

    async def run_agent_async(self, input_data: Any, **kwargs) -> Any:
        """Async execution."""
        prompt = input_data.get('prompt', '')

        # Access injected services if available
        if hasattr(self, 'llm_service'):
            llm_response = await self.llm_service.generate_completion(
                prompt=prompt,
                model='gpt-4'
            )
            return {
                'result': llm_response.get('text', ''),
                'agent_type': 'class',
                'used_llm': True
            }

        return {
            'result': f"Processed: {prompt}",
            'agent_type': 'class',
            'used_llm': False
        }


class FunctionStyleAgent(FiberAgent):
    """Test agent using function-based implementation (legacy style)."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._description = "Function-based test agent"
        self._version = "1.0.0"

    def run_agent(self, input_data: Any, **kwargs) -> Any:
        """Simple function-style execution."""
        value = input_data.get('value', 0)
        return {
            'result': value * 2,
            'agent_type': 'function'
        }


class AgentWithSchemas(FiberAgent):
    """Agent with input/output schema validation."""

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "email": {"type": "string"}
            },
            "required": ["name", "age"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "success": {"type": "boolean"}
            },
            "required": ["message", "success"]
        }

    def run_agent(self, input_data: Any, **kwargs) -> Any:
        return {
            'message': f"Hello {input_data.get('name')}",
            'success': True
        }


class MultiServiceAgent(FiberAgent):
    """Agent that uses multiple injected services."""

    def get_dependencies(self) -> List[str]:
        return ['llm_service', 'fiber_app', 'oauth_service']

    def run_agent(self, input_data: Any, **kwargs) -> Any:
        services_available = []
        if hasattr(self, 'llm_service'):
            services_available.append('llm_service')
        if hasattr(self, 'fiber_app'):
            services_available.append('fiber_app')
        if hasattr(self, 'oauth_service'):
            services_available.append('oauth_service')

        return {
            'services': services_available,
            'count': len(services_available)
        }


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    mock_service = AsyncMock()
    mock_service.generate_completion = AsyncMock(return_value={
        'status': 'completed',
        'text': 'LLM generated response',
        'model': 'gpt-4',
        'provider': 'openai'
    })
    return mock_service


@pytest.fixture
def mock_fiber_app():
    """Mock FiberApp service for testing."""
    mock_app = Mock()
    mock_app.app_id = "test-app-123"
    mock_app.name = "Test App"
    return mock_app


@pytest.fixture
def mock_oauth_service():
    """Mock OAuth service for testing."""
    mock_oauth = Mock()
    mock_oauth.get_token = Mock(return_value="test-token")
    return mock_oauth


# ============================================================================
# Test FiberAgent Base Functionality
# ============================================================================

class TestFiberAgentBase:
    """Test FiberAgent base class functionality."""

    def test_agent_initialization(self):
        """Test basic agent initialization."""
        config = {"setting": "value"}
        agent = ClassStyleAgent(config)

        assert agent.config == config
        assert agent._description == "Class-based test agent"
        assert agent._version == "1.0.0"
        assert isinstance(agent.metadata, dict)

    def test_agent_initialization_without_config(self):
        """Test agent can be initialized without config."""
        agent = ClassStyleAgent()
        assert agent.config == {}
        assert isinstance(agent.metadata, dict)

    def test_agent_info(self):
        """Test get_agent_info returns complete information."""
        agent = ClassStyleAgent()
        agent.agent_id = "agent-123"
        agent.agent_type = "class"

        info = agent.get_agent_info()

        assert info['agent_id'] == "agent-123"
        assert info['agent_type'] == "class"
        assert info['version'] == "1.0.0"
        assert info['description'] == "Class-based test agent"
        assert 'dependencies' in info
        assert 'input_schema' in info
        assert 'output_schema' in info
        assert info['class_name'] == 'ClassStyleAgent'

    def test_agent_properties(self):
        """Test agent description and version properties."""
        agent = ClassStyleAgent()
        assert agent.description == "Class-based test agent"
        assert agent.version == "1.0.0"


# ============================================================================
# Test CLASS Agent Execution (Both Styles)
# ============================================================================

class TestClassAgentExecution:
    """Test CLASS agent execution for both class and function styles."""

    def test_class_style_sync_execution(self):
        """Test class-style agent with sync run_agent."""
        agent = ClassStyleAgent()
        result = agent.run_agent({'prompt': 'test input'})

        assert result['result'] == 'Processed: test input'
        assert result['agent_type'] == 'class'

    @pytest.mark.asyncio
    async def test_class_style_async_execution(self):
        """Test class-style agent with async run_agent_async."""
        agent = ClassStyleAgent()
        result = await agent.run_agent_async({'prompt': 'test input'})

        assert result['result'] == 'Processed: test input'
        assert result['agent_type'] == 'class'
        assert result['used_llm'] is False

    @pytest.mark.asyncio
    async def test_class_style_with_llm_service(self, mock_llm_service):
        """Test class-style agent using injected LLM service."""
        agent = ClassStyleAgent()

        # Inject LLM service
        agent.inject_services({'llm_service': mock_llm_service})

        result = await agent.run_agent_async({'prompt': 'test prompt'})

        assert result['used_llm'] is True
        assert result['result'] == 'LLM generated response'
        assert result['agent_type'] == 'class'

        # Verify LLM service was called
        mock_llm_service.generate_completion.assert_called_once()

    def test_function_style_execution(self):
        """Test function-style agent (legacy)."""
        agent = FunctionStyleAgent()
        result = agent.run_agent({'value': 5})

        assert result['result'] == 10
        assert result['agent_type'] == 'function'

    @pytest.mark.asyncio
    async def test_execute_method(self):
        """Test execute() method wrapper."""
        agent = ClassStyleAgent()
        result = await agent.execute({'prompt': 'test'})

        assert isinstance(result, dict)
        assert 'result' in result
        assert result['agent_type'] == 'class'

    @pytest.mark.asyncio
    async def test_execute_method_returns_dict(self):
        """Test execute() always returns a dict."""
        # Agent that returns a non-dict value
        class SimpleAgent(FiberAgent):
            def run_agent(self, input_data: Any, **kwargs) -> Any:
                return "just a string"

        agent = SimpleAgent()
        result = await agent.execute({})

        assert isinstance(result, dict)
        assert result['result'] == "just a string"


# ============================================================================
# Test Schema Validation
# ============================================================================

class TestSchemaValidation:
    """Test input and output schema validation."""

    def test_input_validation_success(self):
        """Test valid input passes validation."""
        agent = AgentWithSchemas()

        validation = agent.validate_input({
            'name': 'John Doe',
            'age': 30,
            'email': 'john@example.com'
        })

        assert validation['valid'] is True
        assert len(validation['errors']) == 0

    def test_input_validation_missing_required(self):
        """Test missing required field fails validation."""
        agent = AgentWithSchemas()

        validation = agent.validate_input({
            'name': 'John Doe'
            # Missing 'age' field
        })

        assert validation['valid'] is False
        assert any('age' in error for error in validation['errors'])

    def test_input_validation_wrong_type(self):
        """Test wrong field type fails validation."""
        agent = AgentWithSchemas()

        validation = agent.validate_input({
            'name': 'John Doe',
            'age': 'thirty'  # Should be number
        })

        assert validation['valid'] is False
        assert any('age' in error and 'number' in error for error in validation['errors'])

    def test_output_validation_success(self):
        """Test valid output passes validation."""
        agent = AgentWithSchemas()

        validation = agent.validate_output({
            'message': 'Hello World',
            'success': True
        })

        assert validation['valid'] is True
        assert len(validation['errors']) == 0

    def test_output_validation_missing_required(self):
        """Test missing required output field fails validation."""
        agent = AgentWithSchemas()

        validation = agent.validate_output({
            'message': 'Hello World'
            # Missing 'success' field
        })

        assert validation['valid'] is False
        assert any('success' in error for error in validation['errors'])

    def test_default_schemas(self):
        """Test agents have default schemas if not overridden."""
        agent = ClassStyleAgent()

        input_schema = agent.get_input_schema()
        output_schema = agent.get_output_schema()

        assert input_schema['type'] == 'object'
        assert output_schema['type'] == 'object'


# ============================================================================
# Test Service Injection
# ============================================================================

class TestServiceInjection:
    """Test FiberWise service injection pattern."""

    def test_get_dependencies(self):
        """Test agent can declare dependencies."""
        agent = ClassStyleAgent()
        dependencies = agent.get_dependencies()

        assert 'llm_service' in dependencies

    def test_inject_single_service(self, mock_llm_service):
        """Test injecting a single service."""
        agent = ClassStyleAgent()
        agent.inject_services({'llm_service': mock_llm_service})

        assert agent.has_service('llm_service')
        assert agent.get_service('llm_service') == mock_llm_service
        assert hasattr(agent, 'llm_service')
        assert agent.llm_service == mock_llm_service

    def test_inject_multiple_services(self, mock_llm_service, mock_fiber_app, mock_oauth_service):
        """Test injecting multiple services."""
        agent = MultiServiceAgent()

        services = {
            'llm_service': mock_llm_service,
            'fiber_app': mock_fiber_app,
            'oauth_service': mock_oauth_service
        }
        agent.inject_services(services)

        assert agent.has_service('llm_service')
        assert agent.has_service('fiber_app')
        assert agent.has_service('oauth_service')

        assert agent.llm_service == mock_llm_service
        assert agent.fiber_app == mock_fiber_app
        assert agent.oauth_service == mock_oauth_service

    def test_get_injected_services(self, mock_llm_service):
        """Test retrieving all injected services."""
        agent = ClassStyleAgent()
        agent.inject_services({'llm_service': mock_llm_service})

        injected = agent.get_injected_services()
        assert 'llm_service' in injected
        assert injected['llm_service'] == mock_llm_service

    def test_get_service_not_injected(self):
        """Test getting service that wasn't injected raises error."""
        agent = ClassStyleAgent()

        with pytest.raises(KeyError) as exc_info:
            agent.get_service('nonexistent_service')

        assert 'nonexistent_service' in str(exc_info.value)

    def test_has_service(self, mock_llm_service):
        """Test has_service checks correctly."""
        agent = ClassStyleAgent()

        assert not agent.has_service('llm_service')

        agent.inject_services({'llm_service': mock_llm_service})

        assert agent.has_service('llm_service')
        assert not agent.has_service('other_service')

    def test_agent_can_use_injected_services(self):
        """Test agent can use injected services during execution."""
        agent = MultiServiceAgent()

        agent.inject_services({
            'llm_service': Mock(),
            'fiber_app': Mock(),
            'oauth_service': Mock()
        })

        result = agent.run_agent({})

        assert result['count'] == 3
        assert 'llm_service' in result['services']
        assert 'fiber_app' in result['services']
        assert 'oauth_service' in result['services']


# ============================================================================
# Test Execution with Validation
# ============================================================================

class TestExecutionWithValidation:
    """Test execute_with_validation method."""

    @pytest.mark.asyncio
    async def test_execute_with_input_validation_success(self):
        """Test execution with successful input validation."""
        agent = AgentWithSchemas()

        result = await agent.execute_with_validation({
            'name': 'Alice',
            'age': 25,
            'email': 'alice@example.com'
        })

        assert result['success'] is True
        assert result['input_validation']['valid'] is True
        assert result['output_validation']['valid'] is True
        assert 'output' in result
        assert result['output']['message'] == 'Hello Alice'

    @pytest.mark.asyncio
    async def test_execute_with_input_validation_failure(self):
        """Test execution fails with invalid input."""
        agent = AgentWithSchemas()

        result = await agent.execute_with_validation({
            'name': 'Alice'
            # Missing required 'age' field
        })

        assert result['success'] is False
        assert result['input_validation']['valid'] is False
        assert 'error' in result
        assert result['error'] == 'Input validation failed'

    @pytest.mark.asyncio
    async def test_execute_without_validation(self):
        """Test execution without validation."""
        agent = AgentWithSchemas()

        # Should execute even with invalid input when validation disabled
        result = await agent.execute_with_validation(
            {'name': 'Alice'},  # Missing age
            validate_input=False,
            validate_output=False
        )

        assert result['success'] is True

    @pytest.mark.asyncio
    async def test_execution_time_tracked(self):
        """Test that execution time is tracked."""
        agent = ClassStyleAgent()

        result = await agent.execute_with_validation({'prompt': 'test'})

        assert 'execution_time' in result
        assert isinstance(result['execution_time'], float)
        assert result['execution_time'] >= 0

    @pytest.mark.asyncio
    async def test_agent_info_in_result(self):
        """Test agent info is included in execution result."""
        agent = ClassStyleAgent()

        result = await agent.execute_with_validation({'prompt': 'test'})

        assert 'agent_info' in result
        assert result['agent_info']['class_name'] == 'ClassStyleAgent'
        assert 'dependencies_used' in result['agent_info']


# ============================================================================
# Test Backward Compatibility
# ============================================================================

class TestBackwardCompatibility:
    """Test backward compatibility with existing agents."""

    @pytest.mark.asyncio
    async def test_old_agent_with_run_agent_async(self):
        """Test old agents with run_agent_async still work."""
        class OldStyleAgent(FiberAgent):
            async def run_agent_async(self, input_data: Any, **kwargs) -> Any:
                return {'legacy': True, 'result': 'old style'}

        agent = OldStyleAgent()
        result = await agent.execute({'test': 'data'})

        assert result['legacy'] is True
        assert result['result'] == 'old style'

    def test_agent_without_dependencies(self):
        """Test agent without get_dependencies method."""
        class SimpleAgent(FiberAgent):
            def run_agent(self, input_data: Any, **kwargs) -> Any:
                return {'result': 'simple'}

        agent = SimpleAgent()
        dependencies = agent.get_dependencies()

        # Should return empty list by default
        assert dependencies == []

    def test_agent_without_custom_schemas(self):
        """Test agent without custom schema methods."""
        agent = ClassStyleAgent()

        # Should have default schemas
        input_schema = agent.get_input_schema()
        output_schema = agent.get_output_schema()

        assert input_schema is not None
        assert output_schema is not None


# ============================================================================
# Test FiberInjectable Interface
# ============================================================================

class TestFiberInjectable:
    """Test FiberInjectable base class."""

    def test_injectable_initialization(self):
        """Test Injectable initializes service dict."""
        class TestInjectable(FiberInjectable):
            def get_dependencies(self) -> List[str]:
                return []

        obj = TestInjectable()
        assert hasattr(obj, '_injected_services')
        assert isinstance(obj._injected_services, dict)

    def test_injectable_inject_services(self):
        """Test Injectable can inject services."""
        class TestInjectable(FiberInjectable):
            def get_dependencies(self) -> List[str]:
                return ['service1', 'service2']

        obj = TestInjectable()
        services = {'service1': 'value1', 'service2': 'value2'}
        obj.inject_services(services)

        assert obj.service1 == 'value1'
        assert obj.service2 == 'value2'

    def test_injectable_get_service(self):
        """Test Injectable can retrieve injected services."""
        class TestInjectable(FiberInjectable):
            def get_dependencies(self) -> List[str]:
                return ['test_service']

        obj = TestInjectable()
        obj.inject_services({'test_service': 'test_value'})

        assert obj.get_service('test_service') == 'test_value'

    def test_injectable_has_service(self):
        """Test Injectable can check service availability."""
        class TestInjectable(FiberInjectable):
            def get_dependencies(self) -> List[str]:
                return []

        obj = TestInjectable()
        obj.inject_services({'service1': 'value1'})

        assert obj.has_service('service1') is True
        assert obj.has_service('service2') is False


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling in agent execution."""

    @pytest.mark.asyncio
    async def test_execution_error_caught(self):
        """Test that execution errors are caught and returned."""
        class ErrorAgent(FiberAgent):
            async def run_agent_async(self, input_data: Any, **kwargs) -> Any:
                raise ValueError("Test error")

        agent = ErrorAgent()
        result = await agent.execute_with_validation({})

        assert result['success'] is False
        assert 'error' in result
        assert 'Test error' in result['error']
        assert result['error_type'] == 'ValueError'

    @pytest.mark.asyncio
    async def test_validation_error_format(self):
        """Test validation error format is consistent."""
        agent = AgentWithSchemas()

        result = await agent.execute_with_validation({})

        assert result['success'] is False
        assert 'input_validation' in result
        assert 'errors' in result['input_validation']
        assert len(result['input_validation']['errors']) > 0


# ============================================================================
# Test Agent Loading
# ============================================================================

class TestAgentLoading:
    """Test agent loading from module path."""

    def test_from_module_not_implemented(self):
        """Test from_module is defined but may raise for invalid paths."""
        # This is a placeholder - actual implementation may vary
        with pytest.raises(Exception):
            FiberAgent.from_module('invalid.module.path')


# ============================================================================
# Integration-like Tests (within unit test scope)
# ============================================================================

class TestIntegrationScenarios:
    """Test realistic agent usage scenarios."""

    @pytest.mark.asyncio
    async def test_complete_agent_lifecycle(self, mock_llm_service):
        """Test complete agent lifecycle from init to execution."""
        # 1. Create agent with config
        config = {'model': 'gpt-4', 'temperature': 0.7}
        agent = ClassStyleAgent(config)

        # 2. Set metadata
        agent.agent_id = "agent-123"
        agent.agent_type = "class"

        # 3. Inject services
        agent.inject_services({'llm_service': mock_llm_service})

        # 4. Execute with validation
        result = await agent.execute_with_validation({
            'prompt': 'test prompt'
        })

        # 5. Verify complete flow
        assert result['success'] is True
        assert 'output' in result
        assert result['output']['used_llm'] is True
        assert 'execution_time' in result
        assert result['agent_info']['class_name'] == 'ClassStyleAgent'

    @pytest.mark.asyncio
    async def test_agent_with_multiple_services(self, mock_llm_service, mock_fiber_app):
        """Test agent using multiple services together."""
        agent = MultiServiceAgent()

        agent.inject_services({
            'llm_service': mock_llm_service,
            'fiber_app': mock_fiber_app
        })

        result = agent.run_agent({})

        assert result['count'] == 2
        assert 'llm_service' in result['services']
        assert 'fiber_app' in result['services']
