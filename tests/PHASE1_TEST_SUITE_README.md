# Phase 1 Integration Test Suite

Comprehensive test suite for Phase 1 integration covering FiberAgent, LLMProviderFactory, and the complete activation flow.

## Overview

This test suite provides comprehensive coverage for the Phase 1 FiberWise integration, ensuring:

- **FiberAgent** works correctly with CLASS and LLM agent types
- **LLMProviderFactory** creates providers correctly for all supported types
- **Activation flow** works end-to-end with proper service injection
- **Backward compatibility** with existing agents is maintained
- **Error handling** is robust and consistent

## Test Files

### Unit Tests

Located in `tests/unit/`:

#### 1. `test_fiber_agent.py`

**Purpose**: Unit tests for the FiberAgent base class and its functionality.

**Coverage**:
- FiberAgent base functionality (initialization, properties, metadata)
- CLASS agent execution (both class and function styles)
- LLM agent execution with service injection
- Schema validation (input/output)
- Service injection patterns
- Execution with validation
- Backward compatibility with legacy agents
- Error handling

**Test Classes**:
- `TestFiberAgentBase` - Base class functionality
- `TestClassAgentExecution` - CLASS agent execution
- `TestSchemaValidation` - Input/output validation
- `TestServiceInjection` - Service dependency injection
- `TestExecutionWithValidation` - Validation during execution
- `TestBackwardCompatibility` - Legacy agent support
- `TestFiberInjectable` - Injectable interface
- `TestErrorHandling` - Error scenarios
- `TestIntegrationScenarios` - Complete workflows

**Key Tests**:
```bash
# Run all FiberAgent tests
pytest tests/unit/test_fiber_agent.py -v

# Run specific test class
pytest tests/unit/test_fiber_agent.py::TestClassAgentExecution -v

# Run with coverage
pytest tests/unit/test_fiber_agent.py --cov=fiberwise_common.entities.fiber_agent
```

#### 2. `test_llm_service_factory.py`

**Purpose**: Unit tests for LLMServiceFactory and provider implementations.

**Coverage**:
- Provider creation for all types (OpenAI, Anthropic, Google, Ollama, etc.)
- Provider configuration handling
- Provider-specific features and parameters
- Error handling for API calls
- BaseLLMService interface compliance
- Factory pattern implementation

**Test Classes**:
- `TestLLMServiceFactoryCreation` - Factory provider creation
- `TestProviderConfigurations` - Configuration handling
- `TestOpenAIProvider` - OpenAI specific tests
- `TestAnthropicProvider` - Anthropic specific tests
- `TestGoogleAIProvider` - Google AI specific tests
- `TestOllamaProvider` - Ollama specific tests
- `TestProviderErrorHandling` - Error scenarios
- `TestBaseLLMServiceInterface` - Interface compliance
- `TestProviderParameters` - Parameter handling
- `TestFactoryPattern` - Factory pattern
- `TestEdgeCases` - Edge cases and unusual scenarios
- `TestProviderSpecificFeatures` - Provider-specific features

**Key Tests**:
```bash
# Run all LLM service factory tests
pytest tests/unit/test_llm_service_factory.py -v

# Run OpenAI provider tests only
pytest tests/unit/test_llm_service_factory.py::TestOpenAIProvider -v

# Test all providers
pytest tests/unit/test_llm_service_factory.py::TestLLMServiceFactoryCreation -v
```

### Integration Tests

Located in `tests/integration/`:

#### 3. `test_activation_integration.py`

**Purpose**: End-to-end integration tests for the complete activation flow.

**Coverage**:
- Complete activation flow (database → agent → execution → result)
- Backward compatibility with existing agents
- Error scenarios in real-world conditions
- Agent lifecycle management
- Different agent types (CLASS, function, LLM)
- Service injection patterns
- Database operations
- Performance characteristics

**Test Classes**:
- `TestActivationFlowIntegration` - End-to-end activation
- `TestBackwardCompatibility` - Legacy agent support
- `TestErrorScenarios` - Error handling
- `TestAgentLifecycle` - Complete lifecycle
- `TestAgentTypes` - Different agent types
- `TestServiceInjection` - Service injection patterns
- `TestDatabaseOperations` - Database interactions
- `TestPerformance` - Performance tests

**Key Tests**:
```bash
# Run all integration tests
pytest tests/integration/test_activation_integration.py -v

# Run end-to-end flow tests
pytest tests/integration/test_activation_integration.py::TestActivationFlowIntegration -v

# Run backward compatibility tests
pytest tests/integration/test_activation_integration.py::TestBackwardCompatibility -v

# Run with real database (requires setup)
pytest tests/integration/test_activation_integration.py --db-url=sqlite:///test.db
```

## Running Tests

### Run All Phase 1 Tests

```bash
# Run all unit and integration tests
pytest tests/unit/test_fiber_agent.py tests/unit/test_llm_service_factory.py tests/integration/test_activation_integration.py -v

# With coverage report
pytest tests/unit/test_fiber_agent.py tests/unit/test_llm_service_factory.py tests/integration/test_activation_integration.py --cov=fiberwise_common --cov-report=html
```

### Run by Test Type

```bash
# Unit tests only
pytest tests/unit/test_fiber_agent.py tests/unit/test_llm_service_factory.py -v

# Integration tests only
pytest tests/integration/test_activation_integration.py -v

# Specific markers
pytest -m integration  # All integration tests
pytest -m asyncio      # All async tests
pytest -m slow         # Slow tests (performance)
```

### Run Specific Test Scenarios

```bash
# Test CLASS agent execution
pytest tests/unit/test_fiber_agent.py::TestClassAgentExecution -v

# Test schema validation
pytest tests/unit/test_fiber_agent.py::TestSchemaValidation -v

# Test service injection
pytest tests/unit/test_fiber_agent.py::TestServiceInjection -v

# Test OpenAI provider
pytest tests/unit/test_llm_service_factory.py::TestOpenAIProvider -v

# Test complete lifecycle
pytest tests/integration/test_activation_integration.py::TestAgentLifecycle -v
```

## Test Coverage Requirements

| Component | Target Coverage | Status |
|-----------|----------------|---------|
| FiberAgent | 90% | ✓ Achieved |
| LLMServiceFactory | 85% | ✓ Achieved |
| Provider Implementations | 80% | ✓ Achieved |
| Service Injection | 90% | ✓ Achieved |
| Activation Flow | 85% | ✓ Achieved |
| Backward Compatibility | 100% | ✓ Achieved |

### Check Coverage

```bash
# Generate coverage report
pytest tests/unit/test_fiber_agent.py --cov=fiberwise_common.entities.fiber_agent --cov-report=term-missing

pytest tests/unit/test_llm_service_factory.py --cov=fiberwise_common.services.llm_service_factory --cov-report=term-missing

# Full coverage report
pytest tests/ --cov=fiberwise_common --cov-report=html
open htmlcov/index.html
```

## Test Scenarios Covered

### 1. FiberAgent Tests

- ✅ Agent initialization with and without config
- ✅ CLASS agent with class-based implementation
- ✅ CLASS agent with function-based implementation
- ✅ LLM agent execution with injected services
- ✅ Input schema validation (success and failure)
- ✅ Output schema validation (success and failure)
- ✅ Service injection (single and multiple services)
- ✅ Execution with validation
- ✅ Error handling and logging
- ✅ Backward compatibility with legacy agents
- ✅ Agent metadata and properties
- ✅ Complete agent lifecycle

### 2. LLM Provider Factory Tests

- ✅ Create OpenAI provider
- ✅ Create Anthropic provider
- ✅ Create Google AI (Gemini) provider
- ✅ Create Ollama provider
- ✅ Create Hugging Face provider
- ✅ Create OpenRouter provider
- ✅ Create Cloudflare Workers AI provider
- ✅ Custom OpenAI-compatible providers
- ✅ Provider configuration handling
- ✅ API parameter handling (temperature, max_tokens)
- ✅ Error handling for API calls
- ✅ Provider-specific features
- ✅ Edge cases (empty keys, invalid types)

### 3. Activation Integration Tests

- ✅ End-to-end CLASS agent activation
- ✅ End-to-end function agent activation
- ✅ Activation with ServiceRegistry
- ✅ Legacy agent compatibility
- ✅ Agent type migration
- ✅ Error handling in activation flow
- ✅ Missing service handling
- ✅ LLM service error handling
- ✅ Complete agent lifecycle
- ✅ Multiple activations of same agent
- ✅ Different agent types (CLASS, function, LLM)
- ✅ Service injection patterns
- ✅ Database operations
- ✅ Concurrent activations (performance)

## CI/CD Integration

### GitHub Actions Workflow

Add to `.github/workflows/test.yml`:

```yaml
name: Phase 1 Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements-test.txt

      - name: Run Unit Tests
        run: |
          pytest tests/unit/test_fiber_agent.py -v --cov=fiberwise_common.entities.fiber_agent
          pytest tests/unit/test_llm_service_factory.py -v --cov=fiberwise_common.services.llm_service_factory

      - name: Run Integration Tests
        run: |
          pytest tests/integration/test_activation_integration.py -v --cov=fiberwise_common

      - name: Generate Coverage Report
        run: |
          pytest tests/ --cov=fiberwise_common --cov-report=xml

      - name: Upload Coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

## Test Data and Fixtures

### Mock Services

All tests use mock services to avoid external dependencies:

- **mock_llm_service**: Simulates LLM API calls with predictable responses
- **mock_fiber_app**: Mock FiberApp instance
- **mock_oauth_service**: Mock OAuth service
- **mock_database_provider**: Mock database for unit tests

### Test Agents

Custom agent implementations for testing:

- **TestClassAgent**: CLASS-style agent with LLM service
- **TestFunctionAgent**: Function-style agent
- **LegacyAgent**: Legacy agent using old patterns
- **AgentWithSchemas**: Agent with input/output schemas
- **MultiServiceAgent**: Agent using multiple services
- **ErrorAgent**: Agent that raises errors for testing

### Database Fixtures

- **sqlite_db**: SQLite database for integration tests
- Test data setup functions:
  - `setup_test_database()` - Create schema
  - `insert_test_agent()` - Insert test agent
  - `insert_test_llm_provider()` - Insert test provider

## Debugging Tests

### Run Tests with Debug Output

```bash
# Verbose output
pytest tests/unit/test_fiber_agent.py -vv

# Show print statements
pytest tests/unit/test_fiber_agent.py -s

# Show local variables on failure
pytest tests/unit/test_fiber_agent.py -l

# Run specific test with debugging
pytest tests/unit/test_fiber_agent.py::TestClassAgentExecution::test_class_style_async_execution -vv -s
```

### Common Issues

1. **Import errors**: Ensure `fiberwise-common` is installed in development mode
   ```bash
   cd fiberwise-common
   pip install -e .
   ```

2. **Async test failures**: Make sure `pytest-asyncio` is installed
   ```bash
   pip install pytest-asyncio
   ```

3. **Database errors**: Integration tests create temporary databases
   - No external database setup required
   - Tests clean up automatically

4. **Mock service errors**: Verify mock objects are properly configured
   - Check AsyncMock vs Mock usage
   - Verify return_value vs side_effect

## Continuous Improvement

### Adding New Tests

1. Follow existing patterns in test files
2. Use descriptive test names: `test_<what>_<condition>_<expected>`
3. Add docstrings explaining what the test verifies
4. Group related tests in classes
5. Use appropriate fixtures and mocks

### Test Organization

```
tests/
├── unit/                          # Unit tests
│   ├── test_fiber_agent.py       # FiberAgent tests
│   └── test_llm_service_factory.py  # Provider factory tests
├── integration/                   # Integration tests
│   └── test_activation_integration.py  # End-to-end tests
└── PHASE1_TEST_SUITE_README.md   # This file
```

## Success Criteria

Phase 1 testing is complete when:

- ✅ All unit tests pass with >85% coverage
- ✅ All integration tests pass
- ✅ Backward compatibility tests pass (100%)
- ✅ CI/CD pipeline runs tests automatically
- ✅ Documentation is complete and accurate
- ✅ No regression in existing functionality

## Next Steps

After Phase 1 testing is complete:

1. ✅ Run full test suite: `pytest tests/ -v --cov=fiberwise_common`
2. ✅ Review coverage report: `pytest tests/ --cov=fiberwise_common --cov-report=html`
3. ✅ Fix any failing tests or coverage gaps
4. ✅ Add tests to CI/CD pipeline
5. ✅ Update main README with test information
6. ✅ Proceed to Phase 2 implementation

## References

- [PHASE1_READINESS_CHECKLIST.md](../../../PHASE1_READINESS_CHECKLIST.md) - Complete test plan
- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio documentation](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)

---

**Last Updated**: 2026-02-03
**Test Suite Version**: 1.0.0
**Phase**: 1 (FiberAgent Integration)
