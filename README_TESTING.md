# Testing Guide for Fiberwise-Common

This guide covers the comprehensive test suite for the `fiberwise-common` package.

## Test Structure

```
tests/
├── conftest.py              # Package-specific fixtures
├── pytest.ini              # Pytest configuration
├── requirements-test.txt    # Test dependencies
├── run_tests.py            # Test runner script
├── unit/                   # Unit tests
│   ├── test_file_utils.py
│   ├── test_code_validators.py
│   ├── test_agent_utils.py
│   ├── test_llm_response_utils.py
│   ├── test_base_service.py
│   └── test_database_providers_enhanced.py
├── integration/            # Integration tests
│   └── test_full_stack_integration.py
└── fixtures/               # Test data and fixtures
```

## Running Tests

### Quick Start

```bash
# Install test dependencies
python run_tests.py --install

# Run all tests
python run_tests.py

# Run with coverage
python run_tests.py --coverage --html
```

### Test Categories

#### Unit Tests
```bash
# Run only unit tests
python run_tests.py --type unit

# Run specific test file
pytest tests/unit/test_file_utils.py -v

# Run specific test class
pytest tests/unit/test_file_utils.py::TestChecksumFunctions -v

# Run specific test method
pytest tests/unit/test_file_utils.py::TestChecksumFunctions::test_calculate_file_checksum_valid_file -v
```

#### Integration Tests
```bash
# Run only integration tests
python run_tests.py --type integration

# Run integration tests with verbose output
python run_tests.py --type integration --verbose

# Include slow tests
python run_tests.py --type integration --slow
```

### Advanced Options

#### Parallel Execution
```bash
# Run tests in parallel (4 workers)
python run_tests.py --parallel 4

# Run specific tests in parallel
pytest tests/unit/ -n 4
```

#### Coverage Reporting
```bash
# Generate terminal coverage report
python run_tests.py --coverage

# Generate HTML coverage report
python run_tests.py --coverage --html

# Open coverage report (after generation)
open htmlcov/index.html  # macOS
start htmlcov/index.html # Windows
```

#### Test Markers
```bash
# Run only database tests
pytest -m database

# Run only fast tests (exclude slow)
pytest -m "not slow"

# Run only network-requiring tests
pytest -m network

# Run parametrized tests
pytest -m parametrized
```

## Test Categories Overview

### 1. File Utilities Tests (`test_file_utils.py`)
- **Checksum Functions**: SHA256 calculation, verification, safe operations
- **Path Operations**: Normalization, directory creation
- **Manifest Loading**: JSON/YAML parsing, format detection
- **JSON Utilities**: Safe loading/dumping with error handling
- **Integration**: Combined file operations workflows

**Key Features Tested:**
- File integrity verification
- Cross-platform path handling
- Error handling and edge cases
- Performance with large files

### 2. Code Validators Tests (`test_code_validators.py`)
- **Input Validation**: Various data types, empty/None handling
- **Code Validation**: Python code analysis, warning generation
- **Integration**: Validation pipelines

**Key Features Tested:**
- Comprehensive input validation
- Code snippet analysis
- Multi-step validation workflows

### 3. Agent Utils Tests (`test_agent_utils.py`)
- **MetadataMixin**: Runtime metadata management
- **Agent Extraction**: File introspection, metadata extraction
- **Integration**: Combined metadata workflows

**Key Features Tested:**
- Class-based metadata management
- Function/class agent detection
- File analysis and introspection
- Error recovery and fallbacks

### 4. LLM Response Utils Tests (`test_llm_response_utils.py`)
- **Provider Support**: OpenAI, Anthropic, Google, Ollama, HuggingFace, Cloudflare
- **Response Standardization**: Consistent format across providers
- **Schema Integration**: Structured data extraction
- **Error Handling**: Malformed response recovery

**Key Features Tested:**
- Multi-provider response parsing
- Schema-based data extraction
- Cross-provider consistency
- Error resilience

### 5. Database Providers Tests (`test_database_providers_enhanced.py`)
- **SQLite Provider**: Connection management, query execution
- **DuckDB Provider**: Basic operations (when available)
- **Provider Factory**: Dynamic provider creation
- **Transactions**: Begin, commit, rollback operations
- **Concurrency**: Parallel operations, multiple connections

**Key Features Tested:**
- CRUD operations
- Transaction integrity
- Concurrent access
- Performance with large datasets
- Error handling and recovery

### 6. Base Service Tests (`test_base_service.py`)
- **Service Foundation**: Database delegation, method inheritance
- **Error Propagation**: Database error handling
- **Concurrency**: Parallel service operations
- **Integration**: Service composition and workflows

**Key Features Tested:**
- Service-database integration
- Error handling across layers
- Concurrent service usage
- Service inheritance patterns

### 7. Full Stack Integration Tests (`test_full_stack_integration.py`)
- **Complete Workflows**: End-to-end functionality
- **Service Integration**: Multiple services working together
- **Performance**: Large-scale operations, stress testing
- **Real-world Scenarios**: Practical usage patterns

**Key Features Tested:**
- Agent management workflows
- Manifest processing pipelines
- Cross-service data sharing
- Performance under load
- Complete integration scenarios

## Test Configuration

### Pytest Configuration (`pytest.ini`)
- **Coverage**: 80% minimum coverage requirement
- **Markers**: Organized test categorization
- **Reporting**: Multiple output formats
- **Filtering**: Warning management

### Fixtures (`conftest.py`)
- **Database Providers**: SQLite, DuckDB test instances
- **Sample Data**: JSON, YAML, Python code samples
- **File Management**: Temporary directories, cleanup
- **Mock Objects**: LLM responses, database providers

## Coverage Goals

- **Overall**: 80% minimum coverage
- **Unit Tests**: 90%+ coverage for utilities
- **Integration Tests**: Focus on workflows and interactions
- **Edge Cases**: Comprehensive error condition testing

## Performance Benchmarks

### Expected Performance
- **100 agent files**: < 30 seconds processing
- **Concurrent operations**: < 10 seconds for 100 parallel operations
- **Database operations**: Sub-second for typical queries
- **File operations**: Efficient handling of large files

### Performance Tests
- Large-scale agent processing
- Concurrent service operations
- Database stress testing
- Memory usage validation

## Debugging Tests

### Verbose Output
```bash
# Maximum verbosity
pytest -vvv tests/unit/test_file_utils.py

# Show print statements
pytest -s tests/unit/test_file_utils.py

# Show warnings
pytest --disable-warnings tests/
```

### Failed Test Analysis
```bash
# Stop on first failure
pytest -x tests/

# Run last failed tests
pytest --lf

# Run tests that failed in last run
pytest --ff

# Debug mode (drop into pdb on failure)
pytest --pdb tests/unit/test_file_utils.py
```

## Test Data

### Fixtures Available
- `temp_dir`: Temporary directory for test files
- `sample_json_data`: Standard JSON test data
- `sample_manifest_data`: Application manifest data
- `mock_database_provider`: Mocked database for unit tests
- `sqlite_provider`: Real SQLite provider for integration tests
- `sample_agent_file`: Python agent file for testing

### Creating Test Data
```python
def test_example(temp_dir, sample_json_data):
    # Use temp_dir for file operations
    test_file = temp_dir / "test.json"
    
    # Use sample_json_data for consistent test data
    assert sample_json_data["name"] == "test_agent"
```

## Best Practices

### Writing Tests
1. **Use descriptive names**: `test_calculate_checksum_with_valid_file`
2. **Test edge cases**: Empty inputs, None values, malformed data
3. **Use fixtures**: Avoid duplication, ensure cleanup
4. **Mark appropriately**: Use `@pytest.mark.integration`, `@pytest.mark.slow`
5. **Assert meaningfully**: Check specific conditions, not just "no exception"

### Test Organization
1. **Group related tests**: Use classes to organize related test methods
2. **Separate concerns**: Unit vs integration tests
3. **Use parametrize**: Test multiple scenarios efficiently
4. **Mock external dependencies**: Focus on code under test

### Performance Considerations
1. **Mark slow tests**: Use `@pytest.mark.slow` for long-running tests
2. **Use fixtures wisely**: Session scope for expensive setup
3. **Parallel-safe tests**: Avoid shared state, use temporary files
4. **Clean up resources**: Ensure proper fixture cleanup

## Continuous Integration

Tests are designed to run in CI/CD environments:
- **Parallel execution**: Tests can run concurrently
- **No external dependencies**: All tests use mocks or embedded databases
- **Deterministic**: Tests produce consistent results
- **Fast feedback**: Unit tests complete quickly

## Contributing Tests

When adding new functionality:
1. **Add unit tests** for all new functions/classes
2. **Add integration tests** for new workflows
3. **Update fixtures** if new test data is needed
4. **Document complex tests** with clear docstrings
5. **Run full test suite** before submitting changes

## Troubleshooting

### Common Issues
1. **Import errors**: Check `sys.path` and package installation
2. **Database locks**: Ensure proper cleanup of database connections
3. **File permissions**: Use temporary directories for test files
4. **Async tests**: Use `@pytest.mark.asyncio` for async functions
5. **Mock issues**: Verify mock configurations and reset between tests

### Getting Help
- Check test output for specific error messages
- Use `pytest --tb=long` for detailed tracebacks
- Review fixture usage and dependencies
- Run individual tests to isolate issues