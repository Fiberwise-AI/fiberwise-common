"""
Integration tests for fiberwise-common full stack functionality.

Tests the integration between utilities, database providers, and services
working together in realistic scenarios.
"""
import pytest
import tempfile
import json
import yaml
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# Import all major components
from fiberwise_common.database.providers import SQLiteProvider, create_database_provider
from fiberwise_common.services.base_service import BaseService
from fiberwise_common.utils.file_utils import (
    load_manifest, safe_json_dumps, calculate_file_checksum, ensure_directory_exists
)
from fiberwise_common.utils.agent_utils import MetadataMixin, extract_agent_metadata
from fiberwise_common.utils.code_validators import validate_input, validate_code_snippet
from fiberwise_common.utils.llm_response_utils import standardize_response


class AgentManagementService(BaseService):
    """Example service that manages agents using utilities."""
    
    async def create_agent_from_file(self, agent_file_path: str) -> Dict[str, Any]:
        """Create agent record from file analysis."""
        # Extract metadata from file
        metadata = extract_agent_metadata(agent_file_path)
        
        # Calculate file checksum for integrity
        checksum = calculate_file_checksum(agent_file_path)
        
        # Store in database
        await self._execute("""
            INSERT INTO agents (name, description, type, file_path, checksum)
            VALUES (?, ?, ?, ?, ?)
        """, (metadata["name"], metadata["description"], metadata["type"], agent_file_path, checksum))
        
        # Get the created agent
        result = await self._fetch_one("SELECT last_insert_rowid()")
        agent_id = result[0] if result else 0
        
        return {
            "id": agent_id,
            "metadata": metadata,
            "checksum": checksum,
            "file_path": agent_file_path
        }
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents."""
        results = await self._fetch_all("SELECT id, name, description, type, file_path FROM agents")
        return [
            {
                "id": row[0],
                "name": row[1], 
                "description": row[2],
                "type": row[3],
                "file_path": row[4]
            } for row in results
        ]
    
    async def validate_agent_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Validate agent code and return analysis."""
        # Basic input validation
        if not validate_input(code):
            return {"valid": False, "error": "Invalid code input"}
        
        # Code validation
        validated_code, warnings = validate_code_snippet(code, language)
        
        return {
            "valid": len(warnings) == 0,
            "code": validated_code,
            "warnings": warnings,
            "language": language
        }


class ManifestProcessingService(BaseService):
    """Service for processing application manifests."""
    
    async def process_manifest_file(self, manifest_path: str) -> Dict[str, Any]:
        """Process a manifest file and store its contents."""
        # Load manifest using utility
        manifest_data = load_manifest(Path(manifest_path))
        
        # Calculate integrity checksum
        checksum = calculate_file_checksum(manifest_path)
        
        # Store manifest record
        await self._execute("""
            INSERT INTO manifests (name, version, file_path, checksum, data)
            VALUES (?, ?, ?, ?, ?)
        """, (
            manifest_data.get("name", "unknown"),
            manifest_data.get("version", "0.0.0"),
            manifest_path,
            checksum,
            safe_json_dumps(manifest_data)
        ))
        
        # Process agents from manifest
        agents_processed = 0
        if "agents" in manifest_data:
            for agent_info in manifest_data["agents"]:
                await self._execute("""
                    INSERT INTO manifest_agents (manifest_name, agent_name, agent_file, description)
                    VALUES (?, ?, ?, ?)
                """, (
                    manifest_data.get("name"),
                    agent_info.get("name"),
                    agent_info.get("file"),
                    agent_info.get("description", "")
                ))
                agents_processed += 1
        
        return {
            "manifest": manifest_data,
            "checksum": checksum,
            "agents_processed": agents_processed
        }


@pytest.mark.integration
class TestFullStackIntegration:
    """Full stack integration tests."""
    
    @pytest.fixture
    async def setup_database(self, temp_dir: Path):
        """Setup test database with schema."""
        db_path = temp_dir / "integration_test.db"
        provider = SQLiteProvider(f'sqlite:///{db_path}')
        await provider.connect()
        
        # Create test schema
        await provider.execute("""
            CREATE TABLE agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                checksum TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await provider.execute("""
            CREATE TABLE manifests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                file_path TEXT NOT NULL,
                checksum TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await provider.execute("""
            CREATE TABLE manifest_agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                manifest_name TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                agent_file TEXT,
                description TEXT
            )
        """)
        
        yield provider
        await provider.disconnect()
    
    @pytest.fixture
    def sample_agent_files(self, temp_dir: Path) -> Dict[str, Path]:
        """Create sample agent files for testing."""
        agents_dir = temp_dir / "agents"
        ensure_directory_exists(agents_dir)
        
        # Function-based agent
        function_agent = agents_dir / "function_agent.py"
        function_agent.write_text('''
def run_agent(input_data):
    """A simple function-based agent for testing."""
    return {"status": "success", "processed": input_data}
''')
        
        # Class-based agent
        class_agent = agents_dir / "class_agent.py"
        class_agent.write_text('''
class ProcessorAgent:
    """A class-based agent that processes data."""
    
    def __init__(self):
        self.name = "processor"
    
    def run_agent(self, input_data):
        return {"processed": True, "data": input_data}
''')
        
        # Agent with metadata mixin (simulated)
        metadata_agent = agents_dir / "metadata_agent.py"
        metadata_agent.write_text('''
class MetadataAgent:
    """An advanced agent with metadata management."""
    
    def __init__(self):
        self._description = "Agent with advanced metadata capabilities"
    
    def execute(self, input_data):
        return {"result": input_data, "agent": "metadata_agent"}
''')
        
        return {
            "function": function_agent,
            "class": class_agent,
            "metadata": metadata_agent
        }
    
    @pytest.fixture
    def sample_manifest_files(self, temp_dir: Path, sample_agent_files: Dict[str, Path]) -> Dict[str, Path]:
        """Create sample manifest files."""
        manifests_dir = temp_dir / "manifests"
        ensure_directory_exists(manifests_dir)
        
        # JSON manifest
        json_manifest_data = {
            "name": "test-app",
            "version": "1.0.0",
            "description": "Test application for integration tests",
            "agents": [
                {
                    "name": "function_agent",
                    "file": str(sample_agent_files["function"]),
                    "description": "Function-based test agent"
                },
                {
                    "name": "class_agent", 
                    "file": str(sample_agent_files["class"]),
                    "description": "Class-based test agent"
                }
            ],
            "functions": [
                {
                    "name": "utility_function",
                    "file": "functions/utils.py",
                    "description": "Utility functions"
                }
            ]
        }
        
        json_manifest = manifests_dir / "app_manifest.json"
        json_manifest.write_text(json.dumps(json_manifest_data, indent=2))
        
        # YAML manifest
        yaml_manifest_data = {
            "name": "yaml-test-app",
            "version": "2.0.0",
            "description": "YAML-based test application",
            "agents": [
                {
                    "name": "metadata_agent",
                    "file": str(sample_agent_files["metadata"]),
                    "description": "Agent with metadata support"
                }
            ]
        }
        
        yaml_manifest = manifests_dir / "app_manifest.yaml"
        yaml_manifest.write_text(yaml.dump(yaml_manifest_data, default_flow_style=False))
        
        return {
            "json": json_manifest,
            "yaml": yaml_manifest
        }

    async def test_agent_management_workflow(self, setup_database, sample_agent_files):
        """Test complete agent management workflow."""
        db_provider = setup_database
        service = AgentManagementService(db_provider)
        
        created_agents = []
        
        # Process each agent file
        for agent_type, agent_file in sample_agent_files.items():
            result = await service.create_agent_from_file(str(agent_file))
            created_agents.append(result)
            
            assert result["id"] > 0
            assert agent_type in result["metadata"]["type"]
            assert len(result["checksum"]) == 64  # SHA256 hex length
        
        # List all created agents
        agent_list = await service.list_agents()
        assert len(agent_list) == 3
        
        # Verify agent types were detected correctly
        agent_types = [agent["type"] for agent in agent_list]
        assert "function" in agent_types
        assert "class" in agent_types  # Both class agents should be detected

    async def test_manifest_processing_workflow(self, setup_database, sample_manifest_files):
        """Test manifest processing workflow."""
        db_provider = setup_database
        service = ManifestProcessingService(db_provider)
        
        processed_manifests = []
        
        # Process each manifest file
        for manifest_type, manifest_file in sample_manifest_files.items():
            result = await service.process_manifest_file(str(manifest_file))
            processed_manifests.append(result)
            
            assert "manifest" in result
            assert result["agents_processed"] > 0
            assert len(result["checksum"]) == 64
        
        # Verify JSON manifest processing
        json_result = processed_manifests[0]
        assert json_result["manifest"]["name"] == "test-app"
        assert json_result["agents_processed"] == 2
        
        # Verify YAML manifest processing  
        yaml_result = processed_manifests[1]
        assert yaml_result["manifest"]["name"] == "yaml-test-app"
        assert yaml_result["agents_processed"] == 1
        
        # Verify database records
        manifests = await service._fetch_all("SELECT name, version FROM manifests")
        manifest_names = [row[0] for row in manifests]
        assert "test-app" in manifest_names
        assert "yaml-test-app" in manifest_names
        
        # Verify agent records from manifests
        manifest_agents = await service._fetch_all("SELECT agent_name FROM manifest_agents")
        agent_names = [row[0] for row in manifest_agents]
        assert "function_agent" in agent_names
        assert "class_agent" in agent_names
        assert "metadata_agent" in agent_names

    async def test_code_validation_integration(self, setup_database):
        """Test code validation with agent management."""
        db_provider = setup_database
        service = AgentManagementService(db_provider)
        
        # Test valid Python code
        valid_code = '''
def run_agent(input_data):
    """Valid agent function."""
    return {"success": True, "data": input_data}
'''
        
        result = await service.validate_agent_code(valid_code, "python")
        assert result["valid"] is True
        assert len(result["warnings"]) == 0
        assert result["language"] == "python"
        
        # Test invalid code (empty)
        invalid_result = await service.validate_agent_code("", "python")
        assert invalid_result["valid"] is False
        assert len(invalid_result["warnings"]) == 1
        assert "empty" in invalid_result["warnings"][0].lower()
        
        # Test invalid input
        none_result = await service.validate_agent_code(None)
        assert none_result["valid"] is False
        assert "error" in none_result

    async def test_file_integrity_workflow(self, setup_database, sample_agent_files, temp_dir: Path):
        """Test file integrity checking throughout the workflow."""
        db_provider = setup_database
        agent_service = AgentManagementService(db_provider)
        
        # Create agent from file
        agent_file = sample_agent_files["function"]
        original_checksum = calculate_file_checksum(agent_file)
        
        result = await agent_service.create_agent_from_file(str(agent_file))
        assert result["checksum"] == original_checksum
        
        # Modify file and verify checksum changes
        modified_content = agent_file.read_text() + "\n# Modified comment"
        agent_file.write_text(modified_content)
        
        new_checksum = calculate_file_checksum(agent_file)
        assert new_checksum != original_checksum
        
        # Create another agent from modified file
        modified_result = await agent_service.create_agent_from_file(str(agent_file))
        assert modified_result["checksum"] == new_checksum
        assert modified_result["checksum"] != original_checksum

    async def test_concurrent_operations_integration(self, setup_database, sample_agent_files):
        """Test concurrent operations across multiple services."""
        db_provider = setup_database
        agent_service = AgentManagementService(db_provider)
        
        # Process multiple agents concurrently
        async def process_agent(agent_file_path: str) -> Dict[str, Any]:
            return await agent_service.create_agent_from_file(agent_file_path)
        
        tasks = [
            process_agent(str(agent_file)) 
            for agent_file in sample_agent_files.values()
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(result["id"] > 0 for result in results)
        
        # Verify all agents were created
        agent_list = await agent_service.list_agents()
        assert len(agent_list) == 3

    async def test_cross_service_data_sharing(self, setup_database, sample_manifest_files):
        """Test data sharing between different services."""
        db_provider = setup_database
        manifest_service = ManifestProcessingService(db_provider)
        agent_service = AgentManagementService(db_provider)
        
        # Process manifest
        manifest_file = sample_manifest_files["json"]
        manifest_result = await manifest_service.process_manifest_file(str(manifest_file))
        
        # Verify manifest agents were stored
        manifest_agents = await manifest_service._fetch_all(
            "SELECT agent_name, agent_file FROM manifest_agents WHERE manifest_name = ?",
            ("test-app",)
        )
        assert len(manifest_agents) == 2
        
        # Create actual agent records for the agents referenced in manifest
        temp_agent_file = Path(manifest_file.parent.parent / "temp_agent.py")
        temp_agent_file.write_text('''
def run_agent(data):
    """Temporary agent for integration test."""
    return data
''')
        
        agent_result = await agent_service.create_agent_from_file(str(temp_agent_file))
        assert agent_result["id"] > 0
        
        # Verify both services can access their respective data
        agents_from_agent_service = await agent_service.list_agents()
        manifests_from_manifest_service = await manifest_service._fetch_all("SELECT name FROM manifests")
        
        assert len(agents_from_agent_service) == 1
        assert len(manifests_from_manifest_service) == 1


@pytest.mark.integration
class TestUtilitiesIntegration:
    """Integration tests focusing on utility function interactions."""
    
    async def test_file_utilities_integration(self, temp_dir: Path):
        """Test integration between different file utilities."""
        # Create a project structure
        project_dir = temp_dir / "test_project"
        agents_dir = project_dir / "agents"
        manifests_dir = project_dir / "manifests"
        
        # Use utility to create directories
        ensure_directory_exists(agents_dir)
        ensure_directory_exists(manifests_dir)
        
        # Create agent file
        agent_file = agents_dir / "test_agent.py"
        agent_code = '''
class TestAgent:
    """Integration test agent."""
    
    def run_agent(self, data):
        return {"processed": data}
'''
        agent_file.write_text(agent_code)
        
        # Calculate checksum
        checksum = calculate_file_checksum(agent_file)
        
        # Extract metadata
        metadata = extract_agent_metadata(str(agent_file))
        
        # Create manifest with agent info
        manifest_data = {
            "name": "integration-project",
            "version": "1.0.0",
            "agents": [
                {
                    "name": metadata["name"],
                    "file": str(agent_file),
                    "description": metadata["description"],
                    "checksum": checksum,
                    "type": metadata["type"]
                }
            ]
        }
        
        # Save manifest using JSON utility
        manifest_file = manifests_dir / "project_manifest.json"
        manifest_json = safe_json_dumps(manifest_data)
        manifest_file.write_text(manifest_json)
        
        # Load and verify manifest
        loaded_manifest = load_manifest(manifest_file)
        assert loaded_manifest["name"] == "integration-project"
        assert len(loaded_manifest["agents"]) == 1
        assert loaded_manifest["agents"][0]["type"] == "class"
        
        # Verify file integrity
        stored_checksum = loaded_manifest["agents"][0]["checksum"]
        current_checksum = calculate_file_checksum(agent_file)
        assert stored_checksum == current_checksum

    def test_llm_response_standardization_integration(self):
        """Test LLM response standardization with various providers."""
        # Simulate responses from different LLM providers
        provider_responses = {
            "openai": {
                "choices": [
                    {
                        "message": {"content": "OpenAI response for integration test"},
                        "finish_reason": "stop"
                    }
                ]
            },
            "anthropic": {
                "content": [
                    {"type": "text", "text": "Anthropic response for integration test"}
                ],
                "stop_reason": "end_turn"
            },
            "google": {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Google response for integration test"}]
                        },
                        "finishReason": "STOP"
                    }
                ]
            }
        }
        
        standardized_responses = []
        for provider, raw_response in provider_responses.items():
            standardized = standardize_response(raw_response, provider, f"{provider}-model-v1")
            standardized_responses.append(standardized)
        
        # Verify all responses have consistent structure
        for response in standardized_responses:
            assert "text" in response
            assert "provider" in response
            assert "model" in response
            assert "finish_reason" in response
            assert "integration test" in response["text"].lower()
        
        # Verify provider-specific information is preserved
        assert standardized_responses[0]["provider"] == "openai"
        assert standardized_responses[1]["provider"] == "anthropic"
        assert standardized_responses[2]["provider"] == "google"

    def test_validation_pipeline_integration(self, temp_dir: Path):
        """Test complete validation pipeline integration."""
        # Create test cases with various validation scenarios
        test_cases = [
            {
                "name": "valid_agent",
                "code": '''
def run_agent(input_data):
    """Valid agent implementation."""
    if validate_input(input_data):
        return {"success": True, "data": input_data}
    return {"success": False, "error": "Invalid input"}
''',
                "expected_valid": True
            },
            {
                "name": "empty_agent",
                "code": "",
                "expected_valid": False
            },
            {
                "name": "whitespace_agent", 
                "code": "   \n\t  ",
                "expected_valid": False
            }
        ]
        
        validation_results = []
        
        for test_case in test_cases:
            # Step 1: Input validation
            input_valid = validate_input(test_case["code"])
            
            # Step 2: Code validation (if input is valid)
            code_result = None
            if input_valid:
                validated_code, warnings = validate_code_snippet(test_case["code"], "python")
                code_result = {
                    "code": validated_code,
                    "warnings": warnings,
                    "valid": len(warnings) == 0
                }
            
            # Step 3: Create agent file and extract metadata (if code is valid)
            metadata_result = None
            if code_result and code_result["valid"]:
                agent_file = temp_dir / f"{test_case['name']}.py"
                agent_file.write_text(test_case["code"])
                metadata_result = extract_agent_metadata(str(agent_file))
            
            validation_results.append({
                "name": test_case["name"],
                "input_valid": input_valid,
                "code_result": code_result,
                "metadata_result": metadata_result,
                "expected_valid": test_case["expected_valid"]
            })
        
        # Verify results match expectations
        for result in validation_results:
            if result["expected_valid"]:
                assert result["input_valid"] is True
                assert result["code_result"] is not None
                assert result["code_result"]["valid"] is True
                assert result["metadata_result"] is not None
                assert result["metadata_result"]["type"] == "function"
            else:
                # Either input invalid or code invalid
                assert not (result["input_valid"] and 
                           result["code_result"] and 
                           result["code_result"]["valid"])


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance-oriented integration tests."""
    
    async def test_large_scale_agent_processing(self, temp_dir: Path):
        """Test processing large numbers of agents."""
        # Create database
        db_path = temp_dir / "performance_test.db"
        provider = SQLiteProvider(f'sqlite:///{db_path}')
        await provider.connect()
        
        try:
            # Setup schema
            await provider.execute("""
                CREATE TABLE agents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    checksum TEXT NOT NULL
                )
            """)
            
            # Create many agent files
            agents_dir = temp_dir / "many_agents"
            ensure_directory_exists(agents_dir)
            
            agent_files = []
            for i in range(100):  # Create 100 agent files
                agent_file = agents_dir / f"agent_{i:03d}.py"
                agent_file.write_text(f'''
def run_agent_{i}(data):
    """Agent number {i} for performance testing."""
    return {{"agent_id": {i}, "data": data}}
''')
                agent_files.append(agent_file)
            
            # Process all agents
            service = AgentManagementService(provider)
            
            start_time = asyncio.get_event_loop().time()
            
            # Process in batches for better performance
            batch_size = 20
            for i in range(0, len(agent_files), batch_size):
                batch = agent_files[i:i+batch_size]
                
                tasks = [
                    service.create_agent_from_file(str(agent_file))
                    for agent_file in batch
                ]
                
                await asyncio.gather(*tasks)
            
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time
            
            # Verify all agents were processed
            agent_count = await provider.fetch_one("SELECT COUNT(*) FROM agents")
            assert agent_count[0] == 100
            
            # Performance assertion (should process 100 agents in reasonable time)
            assert processing_time < 30.0  # 30 seconds max for 100 agents
            
            print(f"Processed 100 agents in {processing_time:.2f} seconds")
            
        finally:
            await provider.disconnect()

    async def test_concurrent_service_stress(self, temp_dir: Path):
        """Stress test with concurrent service operations."""
        # Create database
        db_path = temp_dir / "stress_test.db"  
        provider = SQLiteProvider(f'sqlite:///{db_path}')
        await provider.connect()
        
        try:
            # Setup schema
            await provider.execute("""
                CREATE TABLE stress_test (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create multiple services
            service = BaseService(provider)
            
            # Define concurrent operations
            async def operation_batch(operation_id: int, batch_size: int = 10):
                results = []
                for i in range(batch_size):
                    await service._execute(
                        "INSERT INTO stress_test (operation_type, data) VALUES (?, ?)",
                        (f"op_{operation_id}", f"data_{operation_id}_{i}")
                    )
                    results.append(f"op_{operation_id}_{i}")
                return results
            
            # Execute many concurrent operations
            num_concurrent = 20
            tasks = [operation_batch(i, 5) for i in range(num_concurrent)]
            
            start_time = asyncio.get_event_loop().time()
            results = await asyncio.gather(*tasks)
            end_time = asyncio.get_event_loop().time()
            
            # Verify all operations completed
            total_records = await provider.fetch_one("SELECT COUNT(*) FROM stress_test")
            expected_records = num_concurrent * 5  # 20 operations * 5 records each
            assert total_records[0] == expected_records
            
            stress_time = end_time - start_time
            print(f"Completed {expected_records} concurrent operations in {stress_time:.2f} seconds")
            
            # Performance assertion
            assert stress_time < 10.0  # Should complete in under 10 seconds
            
        finally:
            await provider.disconnect()


@pytest.mark.integration
def test_complete_integration_workflow(temp_dir: Path):
    """Test complete integration workflow combining all components."""
    
    # This test demonstrates how all components work together
    # in a real-world scenario
    
    # 1. Setup project structure
    project_root = temp_dir / "complete_integration"
    ensure_directory_exists(project_root / "agents")
    ensure_directory_exists(project_root / "manifests")
    ensure_directory_exists(project_root / "data")
    
    # 2. Create various agent files
    agent_types = ["processor", "analyzer", "reporter"]
    agent_files = {}
    
    for agent_type in agent_types:
        agent_file = project_root / "agents" / f"{agent_type}_agent.py"
        agent_code = f'''
class {agent_type.title()}Agent:
    """Agent for {agent_type} operations."""
    
    def __init__(self):
        self.agent_type = "{agent_type}"
    
    def run_agent(self, data):
        """Process data using {agent_type} logic."""
        return {{
            "type": "{agent_type}",
            "processed": True,
            "data": data
        }}
'''
        agent_file.write_text(agent_code)
        agent_files[agent_type] = agent_file
    
    # 3. Extract metadata from all agents
    agent_metadata = {}
    for agent_type, agent_file in agent_files.items():
        metadata = extract_agent_metadata(str(agent_file))
        checksum = calculate_file_checksum(agent_file)
        
        agent_metadata[agent_type] = {
            **metadata,
            "checksum": checksum,
            "file_path": str(agent_file)
        }
    
    # 4. Create comprehensive manifest
    manifest_data = {
        "name": "complete-integration-app",
        "version": "1.0.0", 
        "description": "Complete integration test application",
        "agents": [
            {
                "name": agent_type,
                "file": metadata["file_path"],
                "type": metadata["type"],
                "description": metadata["description"],
                "checksum": metadata["checksum"]
            }
            for agent_type, metadata in agent_metadata.items()
        ],
        "workflow": [
            {"step": 1, "agent": "processor", "description": "Process input data"},
            {"step": 2, "agent": "analyzer", "description": "Analyze processed data"}, 
            {"step": 3, "agent": "reporter", "description": "Generate reports"}
        ]
    }
    
    # 5. Save manifest
    manifest_file = project_root / "manifests" / "app_manifest.json"
    manifest_json = safe_json_dumps(manifest_data)
    manifest_file.write_text(manifest_json)
    
    # 6. Verify manifest can be loaded
    loaded_manifest = load_manifest(manifest_file)
    assert loaded_manifest["name"] == "complete-integration-app"
    assert len(loaded_manifest["agents"]) == 3
    assert len(loaded_manifest["workflow"]) == 3
    
    # 7. Validate all agent checksums
    for agent_info in loaded_manifest["agents"]:
        current_checksum = calculate_file_checksum(agent_info["file"])
        assert current_checksum == agent_info["checksum"]
    
    # 8. Simulate LLM response processing for agent results
    mock_llm_responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": f"Agent {agent_type} completed successfully with processed data."
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        for agent_type in agent_types
    ]
    
    # 9. Standardize all responses
    standardized_responses = []
    for i, response in enumerate(mock_llm_responses):
        standardized = standardize_response(response, "openai", f"gpt-4-{agent_types[i]}")
        standardized_responses.append(standardized)
    
    # 10. Final verification
    assert all(
        "completed successfully" in response["text"] 
        for response in standardized_responses
    )
    
    # 11. Generate summary report
    summary = {
        "project": loaded_manifest["name"],
        "agents_created": len(agent_files),
        "agents_validated": len([
            agent for agent in loaded_manifest["agents"] 
            if agent["type"] in ["function", "class"]
        ]),
        "workflow_steps": len(loaded_manifest["workflow"]),
        "llm_responses_processed": len(standardized_responses),
        "all_checksums_valid": True
    }
    
    assert summary["agents_created"] == 3
    assert summary["agents_validated"] == 3
    assert summary["workflow_steps"] == 3
    assert summary["llm_responses_processed"] == 3
    assert summary["all_checksums_valid"] is True
    
    print(f"Complete integration test successful: {summary}")