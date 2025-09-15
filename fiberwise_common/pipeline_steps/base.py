"""
FiberWise Pipeline Steps Base Classes

Shared base classes and utilities for creating reusable pipeline steps
that maintain consistency with FiberWise platform patterns.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from datetime import datetime
import asyncio
from pydantic import BaseModel, ValidationError

# from .schemas import StepInputSchema, StepOutputSchema  # Removed - schemas not required


class StepResult:
    """Standardized result format for pipeline steps"""
    
    def __init__(self, success: bool, data: Dict[str, Any], 
                 step_name: str, execution_time: Optional[float] = None,
                 error: Optional[str] = None, metadata: Optional[Dict] = None):
        self.success = success
        self.data = data
        self.step_name = step_name
        self.execution_time = execution_time or 0.0
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format"""
        return {
            'success': self.success,
            'data': self.data,
            'step_name': self.step_name,
            'execution_time': self.execution_time,
            'error': self.error,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }


class PipelineStep(ABC):
    """Base class for all FiberWise pipeline steps"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    
    
    @abstractmethod
    async def execute_step(self, input_data: Dict[str, Any], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the core step logic"""
        pass
    
    async def execute(self, input_data: Dict[str, Any], 
                     context: Dict[str, Any]) -> StepResult:
        """Execute step with timing and error handling"""
        start_time = datetime.now()
        
        try:
            # Execute step logic
            result_data = await self.execute_step(input_data, context)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return StepResult(
                success=True,
                data=result_data,
                step_name=self.name,
                execution_time=execution_time,
                metadata={
                    'description': self.description
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return StepResult(
                success=False,
                data={},
                step_name=self.name,
                execution_time=execution_time,
                error=str(e),
                metadata={
                    'description': self.description
                }
            )


class FiberWiseStep(PipelineStep):
    """Pipeline step that integrates with FiberWise SDK"""
    
    def __init__(self, name: str, description: str, fiber_sdk=None):
        super().__init__(name, description)
        self.fiber = fiber_sdk
    
    def set_fiber_sdk(self, fiber_sdk):
        """Set the FiberWise SDK instance"""
        self.fiber = fiber_sdk


class FunctionStep(FiberWiseStep):
    """Step that calls a FiberWise function"""
    
    def __init__(self, name: str, description: str, function_name: str, 
                 fiber_sdk=None):
        super().__init__(name, description, fiber_sdk)
        self.function_name = function_name
    
    async def execute_step(self, input_data: Dict[str, Any], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute FiberWise function"""
        if not self.fiber:
            raise ValueError("FiberWise SDK not configured")
        
        # Prepare function parameters from input_data
        function_params = self.prepare_function_params(input_data, context)
        
        # Call function
        result = await self.fiber.func.activate(self.function_name, function_params)
        
        return result
    
    def prepare_function_params(self, input_data: Dict[str, Any], 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare parameters for function call - override in subclasses"""
        return input_data


class AgentStep(FiberWiseStep):
    """Step that calls a FiberWise agent"""
    
    def __init__(self, name: str, description: str, agent_name: str, 
                 fiber_sdk=None):
        super().__init__(name, description, fiber_sdk)
        self.agent_name = agent_name
    
    async def execute_step(self, input_data: Dict[str, Any], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute FiberWise agent"""
        if not self.fiber:
            raise ValueError("FiberWise SDK not configured")
        
        # Prepare agent parameters from input_data
        agent_params = self.prepare_agent_params(input_data, context)
        
        # Call agent
        result = await self.fiber.agent.activate(self.agent_name, agent_params)
        
        return result
    
    def prepare_agent_params(self, input_data: Dict[str, Any], 
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare parameters for agent call - override in subclasses"""
        return input_data


class PipelineRunner:
    """Utility for running sequences of pipeline steps"""
    
    def __init__(self, fiber_sdk=None):
        self.fiber = fiber_sdk
        self.execution_log = []
    
    async def run_steps(self, steps: List[PipelineStep], 
                       initial_data: Dict[str, Any],
                       context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a sequence of steps"""
        context = context or {}
        current_data = initial_data.copy()
        
        for step in steps:
            # Set FiberWise SDK if step supports it
            if isinstance(step, FiberWiseStep) and self.fiber:
                step.set_fiber_sdk(self.fiber)
            
            # Execute step
            result = await step.execute(current_data, context)
            
            # Log execution
            self.execution_log.append(result.to_dict())
            
            # Handle result
            if result.success:
                # Merge result data into current data
                current_data.update(result.data)
                print(f"✅ {step.name} completed in {result.execution_time:.2f}s")
            else:
                print(f"❌ {step.name} failed: {result.error}")
                raise Exception(f"Step {step.name} failed: {result.error}")
        
        return {
            'success': True,
            'data': current_data,
            'execution_log': self.execution_log
        }
    
    async def run_parallel_steps(self, steps: List[PipelineStep], 
                                input_data: Dict[str, Any],
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run steps in parallel"""
        context = context or {}
        
        # Set FiberWise SDK for all steps
        for step in steps:
            if isinstance(step, FiberWiseStep) and self.fiber:
                step.set_fiber_sdk(self.fiber)
        
        # Execute all steps in parallel
        tasks = [step.execute(input_data, context) for step in steps]
        results = await asyncio.gather(*tasks)
        
        # Combine results
        combined_data = input_data.copy()
        for result in results:
            self.execution_log.append(result.to_dict())
            
            if result.success:
                combined_data.update(result.data)
                print(f"✅ {result.step_name} completed in {result.execution_time:.2f}s")
            else:
                print(f"❌ {result.step_name} failed: {result.error}")
                # Continue with other results even if one fails
        
        return {
            'success': all(result.success for result in results),
            'data': combined_data,
            'execution_log': self.execution_log,
            'parallel_results': [result.to_dict() for result in results]
        }