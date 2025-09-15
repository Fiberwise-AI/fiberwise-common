import asyncio
import logging
import os
import uuid
import json
from datetime import datetime
from typing import Optional, List, Any, Dict
from .base_service import BaseService
from ..database.providers import DatabaseProvider
from .connection_manager import ConnectionManager
from ..entities.user import User
from ..entities.pipeline import Pipeline, PipelineExecution, PipelineExecutionResult, PipelineStep, PipelineExecutionContext
from ..entities.pipeline_events import PipelineExecutionUpdate, PipelineStepUpdate, PipelineStatus, StepStatus

logger = logging.getLogger(__name__)

class PipelineService(BaseService):
    """
    Service for managing pipelines and their executions.
    """

    def __init__(self, database_provider: DatabaseProvider, connection_manager: ConnectionManager):
        super().__init__(database_provider)
        self.db = database_provider
        self.connection_manager = connection_manager
        self._injected_services = {}

    def _convert_db_record_to_pipeline(self, record) -> Pipeline:
        """Convert database record to Pipeline Pydantic model with proper JSON parsing."""
        if not record:
            return None
            
        # Convert record to dict for manipulation
        data = dict(record)
        
        # Parse JSON fields that come as strings from database
        if data.get('definition') and isinstance(data['definition'], str):
            try:
                data['definition'] = json.loads(data['definition'])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse pipeline definition JSON for pipeline {data.get('pipeline_id', 'unknown')}: {e}")
                data['definition'] = {}

        if data.get('config') and isinstance(data['config'], str):
            try:
                data['config'] = json.loads(data['config'])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse pipeline config JSON for pipeline {data.get('pipeline_id', 'unknown')}: {e}")
                data['config'] = {}

        # Handle None file_path
        if data.get('file_path') is None:
            data['file_path'] = ""


        # Let Pydantic handle the conversion and validation
        return Pipeline(**data)

    def _convert_db_record_to_pipeline_execution(self, record) -> Optional[PipelineExecution]:
        """Convert database record to PipelineExecution Pydantic model with proper JSON parsing."""
        if not record:
            return None
            
        # Convert record to dict for manipulation
        data = dict(record)
        
        # Parse JSON fields that come as strings from database
        if data.get('input_data') and isinstance(data['input_data'], str):
            try:
                data['input_data'] = json.loads(data['input_data'])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse execution input_data JSON for execution {data.get('execution_id', 'unknown')}: {e}")
                data['input_data'] = {}
                
        if data.get('results') and isinstance(data['results'], str):
            try:
                data['results'] = json.loads(data['results'])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse execution results JSON for execution {data.get('execution_id', 'unknown')}: {e}")
                data['results'] = {}
                
        if data.get('human_input_config') and isinstance(data['human_input_config'], str):
            try:
                data['human_input_config'] = json.loads(data['human_input_config'])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse human_input_config JSON for execution {data.get('execution_id', 'unknown')}: {e}")
                data['human_input_config'] = {}
                
        if data.get('human_input_data') and isinstance(data['human_input_data'], str):
            try:
                data['human_input_data'] = json.loads(data['human_input_data'])
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse human_input_data JSON for execution {data.get('execution_id', 'unknown')}: {e}")
                data['human_input_data'] = {}
        
        # Let Pydantic handle the conversion and validation
        return PipelineExecution(**data)

    async def get_pipeline_by_filepath(self, file_path: str) -> Optional[Pipeline]:
        """
        Get a pipeline by its file path.
        
        Args:
            file_path: The relative path to the pipeline file within the app.
            
        Returns:
            Pipeline record or None if not found.
        """
        try:
            # Normalize path separators for cross-platform compatibility
            normalized_path = file_path.replace('\\', '/')
            
            query = "SELECT * FROM pipelines WHERE file_path = $1 OR file_path = $2"
            result = await self.db.fetch_one(query, file_path, normalized_path)
            
            return self._convert_db_record_to_pipeline(result)
        except Exception as e:
            logger.error(f"Error getting pipeline by filepath '{file_path}': {e}")
            return None

    async def get_pipeline_by_id(self, pipeline_id: uuid.UUID) -> Optional[Pipeline]:
        """
        Get a pipeline by its ID.
        
        Args:
            pipeline_id: The UUID of the pipeline.
            
        Returns:
            Pipeline record or None if not found.
        """
        try:
            query = "SELECT * FROM pipelines WHERE pipeline_id = $1"
            result = await self.db.fetch_one(query, str(pipeline_id))
            
            return self._convert_db_record_to_pipeline(result)
        except Exception as e:
            logger.error(f"Error getting pipeline by ID '{pipeline_id}': {e}")
            return None

    async def get_pipelines(self, app_id: Optional[uuid.UUID] = None, user_id: Optional[int] = None, 
                          limit: int = 100, offset: int = 0) -> List[Pipeline]:
        """
        Get a list of pipelines with optional filtering.
        
        Args:
            app_id: Optional app ID to filter by
            user_id: Optional user ID to filter by (for access control)
            limit: Maximum number of pipelines to return
            offset: Number of pipelines to skip
            
        Returns:
            List of pipeline records including pipeline_id field.
        """
        try:
            conditions = []
            params = []
            param_count = 0
            
            if app_id is not None:
                param_count += 1
                conditions.append(f"app_id = ${param_count}")
                params.append(str(app_id))
            
            if user_id is not None:
                param_count += 1
                conditions.append(f"created_by = ${param_count}")
                params.append(user_id)
            
            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
            
            param_count += 1
            limit_clause = f"LIMIT ${param_count}"
            params.append(limit)
            
            param_count += 1
            offset_clause = f"OFFSET ${param_count}"
            params.append(offset)
            
            query = f"""
                SELECT pipeline_id, pipeline_slug, name, description, file_path, 
                       definition, config, app_id, is_active, created_by, 
                       created_at, updated_at
                FROM pipelines 
                {where_clause}
                ORDER BY created_at DESC
                {limit_clause} {offset_clause}
            """
            
            results = await self.db.fetch_all(query, *params)
            
            if results:
                pipelines = []
                for result in results:
                    pipeline = self._convert_db_record_to_pipeline(result)
                    if pipeline:
                        pipelines.append(pipeline)
                return pipelines
            return []
        except Exception as e:
            logger.error(f"Error getting pipelines: {e}")
            raise

    async def execute_pipeline(self, pipeline_id: str, input_data: dict, user: User, execution_context: PipelineExecutionContext) -> PipelineExecutionResult:
        """
        Execute a pipeline with graph-based execution following the comprehensive implementation plan.

        Args:
            pipeline_id: The UUID string of the pipeline to execute
            input_data: Input data for the pipeline
            user: User object who initiated the execution (contains id, etc.)
            db: Database provider
            execution_context: Context information for the pipeline execution
            
        Returns:
            Execution result dictionary
        """
        try:
          
            # Extract user_id from user object
            user_id = user.id
            
            if not user_id:
                raise ValueError("User ID is required for pipeline execution")

            # Extract organization_id from execution_context - required for pipeline execution
            if not execution_context or not execution_context.organization_id:
                raise ValueError("Organization ID is required for pipeline execution")
            organization_id = execution_context.organization_id

            # Create execution record
            execution_id = str(uuid.uuid4())
            logger.info(f"Starting pipeline execution {execution_id} for pipeline {pipeline_id}")
            
            # Insert execution record into database
            start_time = datetime.now().isoformat()
            insert_execution_query = """
                INSERT INTO pipeline_executions (
                    execution_id, pipeline_id, status, input_data, created_by, 
                    started_at, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """
            
            # Insert execution record directly using database provider
            await self.db.execute(
                insert_execution_query,
                execution_id, pipeline_id, 'running', json.dumps(input_data), 
                user_id, start_time, start_time, start_time
            )
            
            # Get pipeline definition from database
            pipeline_query = "SELECT * FROM pipelines WHERE pipeline_id = $1"
            pipeline_record = await self.db.fetch_one(pipeline_query, pipeline_id)
            
            if not pipeline_record:
                raise ValueError(f"Pipeline {pipeline_id} not found")
            
            # Convert to Pipeline object using Pydantic
            pipeline = self._convert_db_record_to_pipeline(pipeline_record)
            if not pipeline:
                raise ValueError(f"Failed to convert pipeline {pipeline_id} to Pipeline object")
            
            # Parse pipeline definition
            definition = pipeline.definition
            if not definition:
                raise ValueError(f"Pipeline {pipeline_id} missing definition")

            try:
                structure = definition if isinstance(definition, dict) else json.loads(definition)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in pipeline definition: {str(e)}")

            if not structure or 'steps' not in structure:
                raise ValueError("Invalid pipeline structure - missing 'steps'")

            # No service creation at pipeline level - create at runtime in steps

            # Send pipeline start update
            await self._send_pipeline_update(
                execution_id, "running",
                f"Pipeline execution started", organization_id, str(pipeline.app_id), {"input_data": input_data}
            )
            
            # Execute the structured pipeline
            result = await self._execute_structured_pipeline(
                structure, input_data, execution_id, pipeline, self.db, user_id, organization_id
            )
            
            # Send pipeline completion update
            await self._send_pipeline_update(
                execution_id, "completed",
                f"Pipeline completed successfully", organization_id, str(pipeline.app_id), result.dict()
            )
            
            # Update execution record as completed
            end_time = datetime.now().isoformat()
            duration_ms = int((datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds() * 1000)
            
            update_execution_query = """
                UPDATE pipeline_executions 
                SET status = 'completed', completed_at = $1, duration_ms = $2, 
                    results = $3, updated_at = $4
                WHERE execution_id = $5
            """
            await self.db.execute(
                update_execution_query, 
                end_time, duration_ms, json.dumps(result.dict()), end_time, execution_id
            )
            
            logger.info(f"Pipeline {pipeline_id} execution {execution_id} completed successfully")
            
            # Result is already a PipelineExecutionResult object
            return result
            
        except Exception as e:
            # Update execution record as failed
            try:
                end_time = datetime.now().isoformat()
                duration_ms = int((datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds() * 1000)
                
                update_execution_query = """
                    UPDATE pipeline_executions 
                    SET status = 'failed', completed_at = $1, duration_ms = $2, 
                        error = $3, updated_at = $4
                    WHERE execution_id = $5
                """
                await self.db.execute(
                    update_execution_query, 
                    end_time, duration_ms, str(e), end_time, execution_id
                )
            except Exception as update_error:
                logger.error(f"Error updating failed execution record: {update_error}")
            
            logger.error(f"Error executing pipeline {pipeline_id}: {e}")
            raise

    async def _execute_structured_pipeline(self, structure: dict, input_data: dict,
                                         execution_id: str, pipeline: Pipeline, db, user_id: str, organization_id: int) -> PipelineExecutionResult:
        """Execute a pipeline as a directed graph with conditional branching."""
        try:
            from datetime import datetime
            
            steps_map = {step['id']: PipelineStep(**step) for step in structure.get('steps', [])}
            flow = structure.get('flow', {})
            paths = flow.get('paths', [])
            
            if not steps_map:
                raise ValueError("Pipeline has no steps defined")
            
            step_results = {}
            start_time = datetime.utcnow()
            
            # Graph traversal logic
            current_step_id = flow.get('start_at')
            if not current_step_id:
                raise ValueError("Pipeline flow missing 'start_at' field")
            
            current_data = input_data.copy()
            context = {'pipeline_input': input_data, 'steps': step_results}

            while current_step_id and current_step_id not in ['end', 'end_with_success']:
                if current_step_id in ['end_with_failure']:
                    raise Exception("Pipeline flow ended in a failure state.")

                step_def = steps_map.get(current_step_id)
                if not step_def:
                    raise ValueError(f"Step '{current_step_id}' not found in pipeline definition.")

                logger.info(f"Executing pipeline step: {current_step_id}")
                
                # Handle Human Input Steps (not implemented in this version)
                if step_def.type == 'human_input':
                    logger.warning(f"Human input step {current_step_id} not yet implemented - skipping")
                    # For now, create a mock result
                    step_results[current_step_id] = {
                        'success': True,
                        'result': {'mock_human_input': True, 'message': 'Human input step skipped'}
                    }
                else:
                    # Execute regular task step
                    try:
                        # Resolve dynamic parameters
                        resolved_params = self._resolve_dynamic_parameters(
                            step_def.parameters, context
                        )
                        
                        # Load and execute the real step class (following custom agent pattern)
                        step_result = await self._execute_pipeline_step(
                            current_step_id, step_def, resolved_params, pipeline, execution_id, organization_id, user_id
                        )
                        
                        step_results[current_step_id] = step_result
                        context['steps'] = step_results  # Update context
                        
                        if not step_result.get('success'):
                            error_msg = step_result.get('error', 'Unknown error')
                            raise Exception(f"Step {current_step_id} failed: {error_msg}")
                        
                        current_data = step_result.get('result', {})
                        
                    except Exception as e:
                        logger.error(f"Step {current_step_id} execution error: {str(e)}")
                        step_results[current_step_id] = {'success': False, 'error': str(e)}
                        raise
                
                # Find the next step based on evaluating path conditions
                next_step_id = self._find_next_step(current_step_id, paths, step_results.get(current_step_id, {}))
                current_step_id = next_step_id
            
            end_time = datetime.utcnow()
            
            return PipelineExecutionResult(
                execution_id=execution_id,
                pipeline_id=str(pipeline.pipeline_id),
                status="completed",
                input_data=input_data,
                output_data=current_data,
                step_results=step_results,
                created_by=user_id,
                started_at=start_time.isoformat() + "Z",
                completed_at=end_time.isoformat() + "Z"
            )
            
        except Exception as e:
            logger.error(f"Error in structured pipeline execution: {str(e)}")
            raise

    def _resolve_dynamic_parameters(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve dynamic parameter values using template substitution."""
        resolved = {}
        
        for key, value in parameters.items():
            if isinstance(value, str) and ('${' in value or '{' in value):
                # Simple template resolution using nested key access
                try:
                    resolved_value = value
                    
                    # Handle pipeline_input references (both ${} and {} syntax)
                    if 'pipeline_input.' in value:
                        import re
                        # Match both ${pipeline_input.field} and {pipeline_input.field}
                        matches = re.findall(r'\$?\{pipeline_input\.([^}]+)\}', value)
                        for match in matches:
                            nested_value = self._get_nested_value(context.get('pipeline_input', {}), match)
                            if nested_value is not None:
                                # Replace both possible formats
                                resolved_value = resolved_value.replace(f'${{pipeline_input.{match}}}', str(nested_value))
                                resolved_value = resolved_value.replace(f'{{pipeline_input.{match}}}', str(nested_value))
                    
                    # Handle step result references (both ${} and {} syntax)
                    if 'steps.' in value:
                        import re
                        # Match both ${steps.step.field} and {steps.step.field}
                        matches = re.findall(r'\$?\{steps\.([^}]+)\}', value)
                        for match in matches:
                            nested_value = self._get_nested_value(context.get('steps', {}), match)
                            if nested_value is not None:
                                # Replace both possible formats
                                resolved_value = resolved_value.replace(f'${{steps.{match}}}', str(nested_value))
                                resolved_value = resolved_value.replace(f'{{steps.{match}}}', str(nested_value))
                    
                    # Handle pipeline_execution references
                    if 'pipeline_execution.' in value:
                        import re
                        matches = re.findall(r'\$?\{pipeline_execution\.([^}]+)\}', value)
                        for match in matches:
                            # For now, provide mock execution context
                            if match == 'session_id':
                                execution_value = f"session_{hash(str(context)) % 10000}"
                                resolved_value = resolved_value.replace(f'${{pipeline_execution.{match}}}', execution_value)
                                resolved_value = resolved_value.replace(f'{{pipeline_execution.{match}}}', execution_value)
                    
                    resolved[key] = resolved_value
                except Exception as e:
                    logger.warning(f"Failed to resolve parameter {key}: {e}")
                    resolved[key] = value
            else:
                resolved[key] = value
        
        return resolved

    def _get_nested_value(self, obj: Dict, path: str):
        """Extract nested value using dot notation (e.g., 'result.score')."""
        try:
            keys = path.split('.')
            current = obj
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key)
                else:
                    return None
            return current
        except:
            return None

    def _find_next_step(self, current_step_id: str, paths: List[Dict], step_output: Dict) -> Optional[str]:
        """Evaluate outgoing paths from the current step to find the next one."""
        outgoing_paths = [p for p in paths if p.get('from') == current_step_id]
        
        for path in outgoing_paths:  # Order in manifest is the priority
            condition = path.get('condition', {'type': 'always'})
            
            if self._evaluate_path_condition(condition, step_output):
                logger.info(f"Condition met for path from '{current_step_id}' to '{path.get('to')}'")
                return path.get('to')
                
        logger.warning(f"No outgoing path condition met from step '{current_step_id}'. Ending pipeline.")
        return None

    def _evaluate_path_condition(self, condition: Dict, previous_step_output: Dict) -> bool:
        """Evaluate a path condition (simplified version for now)."""
        condition_type = condition.get('type', 'always')
        config = condition.get('config', {})

        if condition_type == 'always':
            return True
        
        elif condition_type == 'expression':
            # Simple expression evaluation
            source = config.get('source', '')
            operator = config.get('operator', '')
            value = config.get('value')
            
            # Extract value using dot notation
            actual_value = self._get_nested_value(previous_step_output, source)
            
            if operator == 'equals':
                return actual_value == value
            elif operator == 'greater_than':
                try:
                    return float(actual_value) > float(value)
                except (ValueError, TypeError):
                    return False
            elif operator == 'greater_than_or_equal':
                try:
                    return float(actual_value) >= float(value)
                except (ValueError, TypeError):
                    return False
            elif operator == 'less_than':
                try:
                    return float(actual_value) < float(value)
                except (ValueError, TypeError):
                    return False
            elif operator == 'equals_ignore_case':
                return str(actual_value).lower() == str(value).lower()
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False
        
        # TODO: Implement agent and function-based conditions
        logger.warning(f"Condition type '{condition_type}' not yet implemented")
        return False

    async def _execute_pipeline_step(self, step_id: str, step_def: PipelineStep, resolved_params: Dict[str, Any],
                                   pipeline: Pipeline, execution_id: str, organization_id: int, user_id: int) -> Dict[str, Any]:
        """
        Execute a pipeline step: Load Python class from pipeline_code table and inject services.
        Uses database-based loading with local_data configuration.
        """
        try:
            from datetime import datetime
            
            start_time = datetime.now()
            step_class_name = step_def.step_class
            pipeline_id = pipeline.pipeline_id
            
            if not step_class_name:
                raise ValueError(f"Step {step_id} missing step_class")
            
            if not pipeline_id:
                raise ValueError(f"Pipeline missing pipeline_id")
            
            logger.info(f"Loading step class {step_class_name} from pipeline_code table")
            
            # 1. Use self.db database provider
            # Database provider available as self.db
            
            # 2. Load step class using stored file_path (same pattern as agents)
            step_instance = await self._load_step_class_from_stored_path(
                pipeline_id, step_class_name, pipeline
            )
            
            # 3. Send realtime update - step starting
            await self._send_step_update(
                execution_id, step_id, "running",
                f"Executing step: {step_id}", organization_id, str(pipeline.app_id), resolved_params
            )
            
            # 4. Execute step with proper service injection (same pattern as agents)
            logger.info(f"Executing step {step_id} with parameters: {resolved_params}")
            
            # Execute step with runtime service injection
            if hasattr(step_instance, 'execute'):
                import inspect
                execute_method = getattr(step_instance, 'execute')
                sig = inspect.signature(execute_method)

                # Create FiberApp at runtime if needed
                dependencies = {}
                for param_name, param in sig.parameters.items():
                    if param_name in ['parameters', 'input_data']:
                        continue  # Skip the parameters/input_data parameter

                    # Skip **kwargs parameters
                    if param.kind == inspect.Parameter.VAR_KEYWORD:
                        continue

                    # Create FiberApp on demand for fiber parameter - reuse existing service creation
                    if param_name in ['fiber', 'fiber_app']:
                        logger.info(f"🔄 Creating FiberApp at runtime for step {step_id}")

                        # Use ExecutionKeyService for consistent key creation/validation
                        try:
                            from .execution_key_service import ExecutionKeyService
                            from fiberwise_sdk import FiberWiseConfig, FiberApp
                            import os

                            # Create execution key service instance
                            execution_key_service = ExecutionKeyService(self.db)

                            execution_key_result = await execution_key_service.create_execution_key(
                                app_id=str(pipeline.app_id),
                                organization_id=organization_id,
                                executor_type_id='pipeline',
                                executor_id=str(pipeline.pipeline_id),
                                created_by=user_id,
                                expiration_minutes=60
                            )
                            execution_key = execution_key_result['key_value'] if execution_key_result else None
                            if not execution_key:
                                logger.error(f"Unable to obtain execution key for FiberApp")
                                dependencies[param_name] = None
                                continue

                            # Create FiberApp config
                            config_dict = {
                                'app_id': str(pipeline.app_id),
                                'api_key': execution_key,
                                'base_url': os.getenv('FIBER_API_BASE_URL', 'http://localhost:5757/api/v1'),
                                'version': '1.0.0'
                            }

                            if organization_id:
                                config_dict['organization_id'] = organization_id

                            context_config = FiberWiseConfig(config_dict)
                            fiber_app = FiberApp(context_config)
                            dependencies[param_name] = fiber_app

                            logger.info(f"✅ Created runtime FiberApp with key: {execution_key[:10]}...")

                        except Exception as e:
                            logger.error(f"Failed to create runtime FiberApp: {e}")
                            dependencies[param_name] = None
                    # TODO: Add other services as needed

                logger.info(f"Injecting services into step {step_id}: {list(dependencies.keys())}")

                # Execute with proper service injection
                result = execute_method(resolved_params, **dependencies)
                
                # Handle async results
                if asyncio.iscoroutine(result):
                    result = await result
            else:
                raise AttributeError(f"Step class '{step_class_name}' missing execute() method")
            
            # Send realtime update - step completed (disabled in static method)
            logger.info(f"Step {step_id} completed successfully")
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.info(f"Step {step_id} completed in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing pipeline step {step_id}: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'result': {}
            }

    async def _find_step_class_file(self, step_class_name: str) -> Optional[str]:
        """
        Find step class file in fiber-apps directory.
        Searches recursively for Python files containing the step class.
        """
        import os
        import ast
        
        try:
            # Get the fiber-apps directory path
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            fiber_apps_dir = os.path.join(current_dir, 'fiber-apps')
            
            if not os.path.exists(fiber_apps_dir):
                logger.error(f"fiber-apps directory not found at {fiber_apps_dir}")
                return None
            
            logger.info(f"Searching for step class '{step_class_name}' in {fiber_apps_dir}")
            
            # Walk through all Python files in fiber-apps
            for root, dirs, files in os.walk(fiber_apps_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        
                        try:
                            # Parse the Python file to find class definitions
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Parse AST to find class names
                            tree = ast.parse(content)
                            
                            for node in ast.walk(tree):
                                if isinstance(node, ast.ClassDef) and node.name == step_class_name:
                                    logger.info(f"Found step class '{step_class_name}' in {file_path}")
                                    return file_path
                                    
                        except Exception as e:
                            # Skip files that can't be parsed
                            logger.debug(f"Skipping file {file_path}: {e}")
                            continue
            
            logger.warning(f"Step class '{step_class_name}' not found in fiber-apps directory")
            return None
            
        except Exception as e:
            logger.error(f"Error searching for step class {step_class_name}: {e}")
            return None

    async def _load_step_class_from_file(self, file_path: str, step_class_name: str):
        """
        Load step class from file using dynamic import.
        """
        import os
        import sys
        import importlib.util
        
        try:
            # Create module spec from file path
            module_name = f"step_{step_class_name}_{hash(file_path) % 10000}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            
            if not spec or not spec.loader:
                raise ImportError(f"Could not load module spec from {file_path}")
            
            # Import the module
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Get the step class from the module
            if not hasattr(module, step_class_name):
                raise AttributeError(f"Class '{step_class_name}' not found in module {file_path}")
            
            step_class = getattr(module, step_class_name)
            
            # Create instance of the step class
            step_instance = step_class()
            
            logger.info(f"Successfully loaded step class '{step_class_name}' from {file_path}")
            return step_instance
            
        except Exception as e:
            logger.error(f"Error loading step class from {file_path}: {e}")
            raise

    async def _load_step_class_from_stored_path(self, pipeline_id: str, step_class_name: str, pipeline: Pipeline):
        """
        Load a pipeline step class using the stored file_path from pipeline_versions.
        For steps with individual implementation_path, use that; otherwise use bundle directory.
        """
        try:
            # First, try to find the step's specific implementation_path in the pipeline structure
            step_implementation_path = None
            if hasattr(pipeline, 'definition') and pipeline.definition:
                definition = pipeline.definition
                if isinstance(definition, dict) and 'steps' in definition:
                    steps = definition['steps']
                    for step in steps:
                        if step.get('step_class') == step_class_name:
                            step_implementation_path = step.get('implementation_path')
                            break

            if step_implementation_path:
                logger.info(f"Loading step {step_class_name} from step-specific implementation_path: {step_implementation_path}")
                # Load directly from the step's implementation path
                step_instance = await self._load_step_class_from_file_path(step_implementation_path, step_class_name)
                return step_instance

            # Fallback: Get the bundle directory from the pipeline version
            version_query = """
                SELECT file_path FROM pipeline_versions
                WHERE pipeline_id = $1 AND is_active = 1
                ORDER BY created_at DESC LIMIT 1
            """
            version_record = await self.db.fetch_one(version_query, str(pipeline_id))
            if not version_record or not version_record['file_path']:
                raise ValueError(f"No active version with file_path found for pipeline {pipeline_id}")

            bundle_path = version_record['file_path']
            logger.info(f"Loading step {step_class_name} from bundle directory: {bundle_path}")

            # Load the step class from the bundle directory
            step_instance = await self._load_step_class_from_file_path(bundle_path, step_class_name)
            return step_instance

        except Exception as e:
            logger.error(f"Error loading step class {step_class_name} from stored path: {str(e)}")
            raise

    async def _load_step_class_from_file_path(self, file_path: str, step_class_name: str):
        """
        Load step class from a specific file path (like agents do).
        """
        try:
            import os
            import sys
            import importlib.util
            from fiberwise_common.config import BaseWebSettings

            # Get entity bundles directory
            try:
                settings = BaseWebSettings()
                entity_bundles_dir = settings.ENTITY_BUNDLES_DIR
            except:
                entity_bundles_dir = os.getenv('ENTITY_BUNDLES_DIR')
                if not entity_bundles_dir:
                    entity_bundles_dir = os.path.join(os.getcwd(), 'local_data', 'entity_bundles')

            # Build the full file path
            full_file_path = os.path.join(entity_bundles_dir, file_path)

            # For directory paths, find the step class file inside
            if os.path.isdir(full_file_path):
                # Search for step class in the directory (fallback to old behavior)
                step_file_path = await self._find_step_file_in_bundle(full_file_path, step_class_name)
                if not step_file_path:
                    raise FileNotFoundError(f"Step class '{step_class_name}' not found in directory: {full_file_path}")
                full_file_path = step_file_path

            if not os.path.exists(full_file_path):
                raise FileNotFoundError(f"Step file not found: {full_file_path}")

            # Load the Python module from file
            spec = importlib.util.spec_from_file_location("step_module", full_file_path)
            if spec is None or spec.loader is None:
                raise ValueError(f"Could not load spec from {full_file_path}")

            step_module = importlib.util.module_from_spec(spec)

            # Add to sys.modules to handle imports
            sys.modules["step_module"] = step_module

            try:
                spec.loader.exec_module(step_module)
            finally:
                # Clean up sys.modules
                if "step_module" in sys.modules:
                    del sys.modules["step_module"]

            # Get the step class from the module
            if not hasattr(step_module, step_class_name):
                raise AttributeError(f"Step class '{step_class_name}' not found in {full_file_path}")

            step_class = getattr(step_module, step_class_name)
            step_instance = step_class()

            logger.info(f"Successfully loaded step class {step_class_name} from {full_file_path}")
            return step_instance

        except Exception as e:
            logger.error(f"Error loading step class from file path {file_path}: {e}")
            raise

    async def _load_step_class_from_entity_bundle(self, pipeline_id: str, step_class_name: str, pipeline: Pipeline):
        """
        Load a pipeline step class from entity bundle directory (same pattern as _execute_custom_agent).
        Uses file-based loading instead of corrupted database code.
        """
        try:
            import os
            import sys
            import importlib.util
            
            # Get the entity bundle path for this pipeline
            entity_bundle_path = await self._get_pipeline_entity_bundle_path(
                pipeline_id, pipeline
            )
            
            if not entity_bundle_path:
                raise ValueError(f"Entity bundle path not found for pipeline {pipeline_id}")
            
            # Find the step class file in the entity bundle
            step_file_path = await self._find_step_file_in_bundle(
                entity_bundle_path, step_class_name
            )
            
            if not step_file_path:
                raise FileNotFoundError(f"Step class '{step_class_name}' not found in entity bundle: {entity_bundle_path}")
            
            logger.info(f"Loading step class {step_class_name} from entity bundle: {step_file_path}")
            
            # Use the same dynamic import pattern as _execute_custom_agent
            step_dir = os.path.dirname(os.path.abspath(step_file_path))
            step_filename = os.path.basename(step_file_path)
            
            # Add step directory to Python path
            if step_dir not in sys.path:
                sys.path.insert(0, step_dir)
            
            try:
                # Import the step module
                module_name = os.path.splitext(step_filename)[0]
                
                # Remove module from cache if it exists to get fresh import
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
                # Dynamic import (same as custom agents)
                spec = importlib.util.spec_from_file_location(module_name, step_file_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load module from {step_file_path}")
                
                step_module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = step_module
                spec.loader.exec_module(step_module)
                
                # Get the step class from the module
                if hasattr(step_module, step_class_name):
                    step_class = getattr(step_module, step_class_name)
                    step_instance = step_class()
                    logger.info(f"Successfully loaded and instantiated step class: {step_class_name}")
                    return step_instance
                else:
                    raise AttributeError(f"Step class '{step_class_name}' not found in module")
                    
            finally:
                # Clean up sys.path
                if step_dir in sys.path:
                    sys.path.remove(step_dir)
                
        except Exception as e:
            logger.error(f"Error loading step class {step_class_name} from entity bundle: {str(e)}")
            raise

    async def _get_pipeline_entity_bundle_path(self, pipeline_id: str, pipeline: Pipeline) -> str:
        """
        Get the entity bundle path for a pipeline (same pattern as agents).
        Path format: {ENTITY_BUNDLES_DIR}/apps/{app_id}/pipeline/{pipeline_id}/{version}/
        """
        try:
            import os
            from fiberwise_common.config import BaseWebSettings
            
            # Get settings for ENTITY_BUNDLES_DIR (same as agents)
            try:
                settings = BaseWebSettings()
                entity_bundles_dir = settings.ENTITY_BUNDLES_DIR
            except:
                entity_bundles_dir = os.getenv('ENTITY_BUNDLES_DIR')
                if not entity_bundles_dir:
                    # Fallback to local_data/entity_bundles
                    entity_bundles_dir = os.path.join(os.getcwd(), 'local_data', 'entity_bundles')
            
            # Get app_id from pipeline database record if not in pipeline dict
            app_id = pipeline.app_id
            if not app_id:
                # Query database to get app_id for this pipeline
                query = "SELECT app_id FROM pipelines WHERE pipeline_id = $1"
                pipeline_record = await self.db.fetch_one(query, pipeline_id)
                if pipeline_record:
                    app_id = pipeline_record['app_id']
                else:
                    raise ValueError(f"Pipeline {pipeline_id} not found in database")
            
            # Get active version_id from pipeline_versions table (like agents)
            version_query = """
                SELECT version_id FROM pipeline_versions 
                WHERE pipeline_id = $1 AND is_active = 1 
                ORDER BY created_at DESC LIMIT 1
            """
            version_record = await self.db.fetch_one(version_query, pipeline_id)
            if not version_record:
                raise ValueError(f"No active version found for pipeline {pipeline_id}")
            
            version_id = version_record['version_id']
            logger.info(f"Using active pipeline version: {version_id}")
            
            # Construct entity bundle path with proper version_id
            bundle_path = os.path.join(
                entity_bundles_dir,
                'apps',
                str(app_id),
                'pipeline',
                str(pipeline_id),
                str(version_id)  # Use actual version_id UUID
            )
            
            if os.path.exists(bundle_path):
                logger.info(f"Found entity bundle for pipeline {pipeline_id}: {bundle_path}")
                return bundle_path
            else:
                logger.warning(f"Entity bundle path does not exist: {bundle_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting entity bundle path for pipeline {pipeline_id}: {str(e)}")
            return None

    async def _find_step_file_in_bundle(self, bundle_path: str, step_class_name: str) -> str:
        """
        Find the step class file in the entity bundle directory.
        Searches recursively for Python files containing the step class.
        """
        try:
            import os
            
            # Search all Python files in the bundle
            for root, dirs, files in os.walk(bundle_path):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        file_path = os.path.join(root, file)
                        
                        # Quick check if file contains the class definition
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if f'class {step_class_name}' in content:
                                    logger.info(f"Found step class {step_class_name} in {file_path}")
                                    return file_path
                        except Exception:
                            continue  # Skip files that can't be read
            
            return None
            
        except Exception as e:
            logger.error(f"Error searching for step class {step_class_name} in bundle: {str(e)}")
            return None

    # async def _load_step_class_from_database(self, db_provider, pipeline_id: str, step_class_name: str):
    #     """
    #     Load a pipeline step class from pipeline_code table using exec() pattern.
    #     This follows the same approach as _load_pipeline_step_class() in activation_processor.py
    #     """
    #     try:
    #         # Query pipeline_code table to get step implementation (adapted for SQLite)
    #         query = """
    #             SELECT step_id, step_class, implementation_code, language 
    #             FROM pipeline_code 
    #             WHERE pipeline_id = $1 AND step_class = $2 AND is_active = true
    #             ORDER BY created_at DESC 
    #             LIMIT 1
    #         """
            
    #         step_record = await db_provider.fetch_one(query, pipeline_id, step_class_name)
            
    #         if not step_record:
    #             raise ValueError(f"Step class '{step_class_name}' not found in pipeline_code table for pipeline {pipeline_id}")
            
    #         implementation_code = step_record['implementation_code']
    #         step_id = step_record['step_id']
    #         language = step_record['language']
            
    #         logger.info(f"Loading step class {step_class_name} for step {step_id} from pipeline_code table")
            
    #         if language != 'python':
    #             raise ValueError(f"Only Python step implementations are supported, got: {language}")
            
    #         # Execute the implementation code to define the class (same as activation processor)
    #         namespace = {
    #             '__name__': f'pipeline_step_{step_id}',
    #             '__builtins__': __builtins__
    #         }
            
    #         try:
    #             # Execute the step class code
    #             exec(implementation_code, namespace)
                
    #             # Find the step class in the namespace
    #             if step_class_name in namespace:
    #                 step_class = namespace[step_class_name]
    #                 step_instance = step_class()
    #                 logger.info(f"Successfully loaded and instantiated step class: {step_class_name}")
    #                 return step_instance
    #             else:
    #                 raise AttributeError(f"Step class '{step_class_name}' not found in executed code")
                    
    #         except Exception as e:
    #             logger.error(f"Error executing step implementation code: {str(e)}")
    #             logger.error(f"Code that failed: {implementation_code[:500]}...")
    #             raise ValueError(f"Failed to execute step implementation: {str(e)}")
                
    #     except Exception as e:
    #         logger.error(f"Error loading step class {step_class_name}: {str(e)}")
    #         raise
    
    def _prepare_comprehensive_dependencies(self, parameters) -> Dict[str, Any]:
        """
        Prepare comprehensive dependencies for pipeline step execution based on function signature.
        Uses the same pattern as ActivationProcessor for consistency.
        
        Args:
            parameters: Function parameters from inspect.signature().parameters
            
        Returns:
            Dict of kwargs to inject into the step function
        """
        dependencies = {}
        
        import inspect
        
        for param_name, param in parameters.items():
            if param_name in ['parameters', 'input_data']:
                continue  # Skip the parameters/input_data parameter
            
            # Skip **kwargs parameters (VAR_KEYWORD)
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            
            # Map parameter names to injected services using the established pattern
            if param_name in ['fiber', 'fiber_app']:
                dependencies[param_name] = self._injected_services.get('fiber')
            elif param_name in ['llm_service', 'llm_provider_service', 'llm']:
                dependencies[param_name] = self._injected_services.get('llm_service')
            elif param_name in ['storage', 'agent_storage', 'storage_provider']:
                dependencies[param_name] = self._injected_services.get('storage')
            elif param_name in ['oauth_service', 'oauth', 'credentials']:
                dependencies[param_name] = self._injected_services.get('oauth_service')
            else:
                # Check if any other service matches this parameter name
                dependencies[param_name] = self._injected_services.get(param_name)
            
            # Log service injection status
            if dependencies[param_name] is not None:
                logger.info(f"✅ Injecting {param_name}: {type(dependencies[param_name]).__name__}")
            else:
                logger.error(f"❌ Service '{param_name}' not available, using None")
                logger.error(f"❌ _injected_services contents: {list(self._injected_services.keys())}")
                logger.error(f"❌ _injected_services['fiber']: {self._injected_services.get('fiber')}")

        return dependencies
    
    async def _send_step_update(self, execution_id: str, step_id: str, status: str, message: str, organization_id: int, app_id: str, data: Any = None):
        """Send realtime updates for pipeline step progress using ConnectionManager"""
        try:
            # Use the ConnectionManager instance from constructor
            if not self.connection_manager:
                logger.debug(f"Step update (no ConnectionManager): {step_id} - {status}")
                return

            # Prepare update payload
            update_data = {
                "type": "pipeline_step_update",
                "execution_id": execution_id,
                "step_id": step_id,
                "status": status,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add data if provided (but limit size)
            if data:
                # Truncate large data objects
                if isinstance(data, dict):
                    # Include key info but limit string lengths
                    truncated_data = {}
                    for key, value in data.items():
                        if isinstance(value, str) and len(value) > 100:
                            truncated_data[key] = value[:100] + "..."
                        else:
                            truncated_data[key] = value
                    update_data["data"] = truncated_data
                else:
                    update_data["data"] = str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
            
            # Send WebSocket message using ConnectionManager - broadcast to both org_app and app
            try:
                # Primary broadcast to org+app
                await self.connection_manager.broadcast_to_org_app(update_data, app_id, organization_id)
                logger.debug(f"Sent step broadcast to org {organization_id} app {app_id}: {step_id} - {status}")
                
                # Also broadcast to app-wide for broader coverage
                await self.connection_manager.broadcast_to_app(update_data, app_id)
                logger.debug(f"Sent step broadcast to app {app_id}: {step_id} - {status}")
                
            except Exception as e:
                logger.warning(f"Failed to broadcast step update: {e}")
            
        except Exception as e:
            # Don't fail pipeline execution if realtime fails
            logger.warning(f"Failed to send realtime update for step {step_id}: {e}")

    async def _send_pipeline_update(self, execution_id: str, status: str, message: str, organization_id: int, app_id: str, result: Any = None):
        """Send realtime updates for overall pipeline progress using ConnectionManager"""
        try:
            # Use the ConnectionManager instance from constructor
            if not self.connection_manager:
                logger.debug(f"Pipeline update (no ConnectionManager): {execution_id} - {status}")
                return

            # Prepare update payload
            update_data = {
                "type": "pipeline_execution_update",
                "execution_id": execution_id,
                "status": status,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add result if provided (but limit size)
            if result:
                if isinstance(result, dict):
                    # Include key results but limit string lengths
                    truncated_result = {}
                    for key, value in result.items():
                        if isinstance(value, str) and len(value) > 200:
                            truncated_result[key] = value[:200] + "..."
                        else:
                            truncated_result[key] = value
                    update_data["result"] = truncated_result
                else:
                    update_data["result"] = str(result)[:300] + "..." if len(str(result)) > 300 else str(result)
            
            # Send WebSocket message using ConnectionManager - broadcast to both org_app and app
            try:
                logger.info(f"🚀 SENDING PIPELINE BROADCAST: execution_id={execution_id}, status={status}, app_id={app_id}, org_id={organization_id}")
                
                # Primary broadcast to org+app
                await self.connection_manager.broadcast_to_org_app(update_data, app_id, organization_id)
                logger.debug(f"Sent pipeline broadcast to org {organization_id} app {app_id}: {execution_id} - {status}")
                
                # Also broadcast to app-wide for broader coverage
                await self.connection_manager.broadcast_to_app(update_data, app_id)
                logger.debug(f"Sent pipeline broadcast to app {app_id}: {execution_id} - {status}")
                
            except Exception as e:
                logger.warning(f"Failed to broadcast pipeline update: {e}")
            
        except Exception as e:
            # Don't fail pipeline execution if realtime fails
            logger.warning(f"Failed to send pipeline realtime update: {e}")



    async def create_execution(self, execution_data: PipelineExecution) -> PipelineExecution:
        """
        Create a new pipeline execution record.
        
        Args:
            execution_data: Dictionary containing execution details.
            
        Returns:
            The created execution record.
        """
        try:
            execution_id = str(uuid.uuid4())
            pipeline_id = execution_data.pipeline_id
            input_data = execution_data.input_data or {}
            status = execution_data.status or 'queued'
            created_by = execution_data.created_by
            priority = execution_data.priority or 10
            
            insert_query = """
                INSERT INTO pipeline_executions (
                    execution_id, pipeline_id, status, input_data, created_by, priority
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """
            
            params = (
                execution_id,
                pipeline_id,
                status,
                json.dumps(input_data),
                created_by,
                priority
            )
            await self.db.execute(insert_query, *params)
            
            select_query = "SELECT * FROM pipeline_executions WHERE execution_id = $1"
            result = await self.db.fetch_one(select_query, execution_id)
            
            if not result:
                raise Exception("Failed to create and retrieve pipeline execution record")
                
            return self._convert_db_record_to_pipeline_execution(result)
        except Exception as e:
            logger.error(f"Error creating pipeline execution: {e}")
            raise

    async def get_all_executions(self,
        limit: int = 10,
        offset: int = 0,
        status: Optional[str] = None,
        app_id: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get all pipeline executions with pagination and filtering.
        
        Args:
            db: Database provider
            limit: Maximum number of results
            offset: Number of results to skip
            status: Filter by execution status
            app_id: Filter by app ID
            user_id: Filter by user ID
            
        Returns:
            Dictionary with items, total, limit, offset
        """
        try:
            # Build WHERE clause
            where_conditions = []
            params = []
            param_count = 1
            
            if status:
                where_conditions.append(f"pe.status = ${param_count}")
                params.append(status)
                param_count += 1
            
            if app_id:
                where_conditions.append(f"p.app_id = ${param_count}")
                params.append(app_id)
                param_count += 1
                
            if user_id:
                where_conditions.append(f"pe.created_by = ${param_count}")
                params.append(user_id)
                param_count += 1
                
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            # Count query - need JOIN if filtering by app_id or user_id
            if app_id or user_id:
                count_query = f"""
                    SELECT COUNT(*) as total 
                    FROM pipeline_executions pe
                    LEFT JOIN pipelines p ON pe.pipeline_id = p.pipeline_id
                    {where_clause}
                """
            else:
                count_query = f"SELECT COUNT(*) as total FROM pipeline_executions pe {where_clause}"
            count_result = await self.db.fetch_one(count_query, *params)
            total = count_result['total'] if count_result else 0
            
            # Main query with pagination
            main_query = f"""
                SELECT 
                    pe.execution_id, pe.pipeline_id, pe.status, pe.priority,
                    pe.input_data, pe.context, pe.results, pe.error,
                    pe.started_at, pe.completed_at, pe.duration_ms,
                    pe.created_by, pe.created_at, pe.updated_at,
                    pe.human_input_config, pe.human_input_data, pe.waiting_step_id,
                    p.name as pipeline_name, p.app_id 
                FROM pipeline_executions pe
                LEFT JOIN pipelines p ON pe.pipeline_id = p.pipeline_id
                {where_clause}
                ORDER BY pe.created_at DESC 
                LIMIT ${param_count} OFFSET ${param_count + 1}
            """
            
            params.extend([limit, offset])
            results = await self.db.fetch_all(main_query, *params)
            items = [dict(row) for row in results] if results else []
            
            return {
                'items': items,
                'total': total,
                'limit': limit,
                'offset': offset
            }
            
        except Exception as e:
            logger.error(f"Error getting all pipeline executions: {e}")
            raise

    async def get_pipeline_executions(self,
        pipeline_id: str,
        limit: int = 10,
        offset: int = 0,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get executions for a specific pipeline.
        
        Args:
            pipeline_id: The pipeline ID
            limit: Maximum number of results
            offset: Number of results to skip  
            status: Filter by execution status
            db: Database provider
            
        Returns:
            Dictionary with items, total, limit, offset
        """
        try:
            # Build WHERE clause
            where_conditions = ["pipeline_id = $1"]
            params = [pipeline_id]
            param_count = 2
            
            if status:
                where_conditions.append(f"status = ${param_count}")
                params.append(status)
                param_count += 1
                
            where_clause = "WHERE " + " AND ".join(where_conditions)
            
            # Count query
            count_query = f"SELECT COUNT(*) as total FROM pipeline_executions {where_clause}"
            count_result = await self.db.fetch_one(count_query, *params)
            total = count_result['total'] if count_result else 0
            
            # Main query with pagination
            main_query = f"""
                SELECT 
                    pe.execution_id, pe.pipeline_id, pe.status, pe.priority,
                    pe.input_data, pe.context, pe.results, pe.error,
                    pe.started_at, pe.completed_at, pe.duration_ms,
                    pe.created_by, pe.created_at, pe.updated_at,
                    pe.human_input_config, pe.human_input_data, pe.waiting_step_id,
                    p.name as pipeline_name
                FROM pipeline_executions pe
                LEFT JOIN pipelines p ON pe.pipeline_id = p.pipeline_id
                {where_clause}
                ORDER BY pe.created_at DESC 
                LIMIT ${param_count} OFFSET ${param_count + 1}
            """
            
            params.extend([limit, offset])
            results = await self.db.fetch_all(main_query, *params)
            items = [dict(row) for row in results] if results else []
            
            return {
                'items': items,
                'total': total,
                'limit': limit,
                'offset': offset
            }
            
        except Exception as e:
            logger.error(f"Error getting pipeline executions for pipeline {pipeline_id}: {e}")
            raise

    async def get_execution_by_id(self, execution_id: str) -> Optional[PipelineExecution]:
        """
        Get a specific pipeline execution by ID.
        
        Args:
            execution_id: The execution ID
            db: Database provider
            
        Returns:
            Execution record or None if not found
        """
        try:
            query = """
                SELECT pe.*, p.name as pipeline_name, p.app_id
                FROM pipeline_executions pe
                LEFT JOIN pipelines p ON pe.pipeline_id = p.pipeline_id
                WHERE pe.execution_id = $1
            """
            
            result = await self.db.fetch_one(query, execution_id)
            
            return self._convert_db_record_to_pipeline_execution(result)
            
        except Exception as e:
            logger.error(f"Error getting execution by id {execution_id}: {e}")
            raise
