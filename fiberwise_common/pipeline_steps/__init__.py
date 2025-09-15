"""
FiberWise Pipeline Steps Library

Shared library for reusable pipeline steps that maintain consistency 
with FiberWise platform syntax and patterns.
"""

# Base classes
from .base import (
    PipelineStep, 
    FiberWiseStep, 
    FunctionStep, 
    AgentStep, 
    StepResult, 
    PipelineRunner
)

# All schemas
from .schemas import (
    # Enums
    ResearchScope,
    AnalysisType, 
    SynthesisMode,
    ScrapingMode,
    ProcessingMode,
    
    # Base schemas
    StepInputSchema,
    StepOutputSchema,
    
    # Data collection schemas
    WikipediaResearchInputSchema,
    WikipediaResearchOutputSchema,
    WikipediaArticle,
    WikipediaReference,
    WebScrapingInputSchema, 
    WebScrapingOutputSchema,
    ScrapedContent,
    ScrapedResult,
    
    # Processing schemas
    DataProcessingInputSchema,
    DataProcessingOutputSchema,
    ProcessingResult,
    
    # Agent collaboration schemas
    AgentConversationInputSchema,
    AgentConversationOutputSchema,
    AgentInsight,
    AgentRecommendation,
    
    # Knowledge synthesis schemas
    KnowledgeSynthesisInputSchema,
    KnowledgeSynthesisOutputSchema,
    KnowledgeElement,
    KeyFinding,
    ResearchRecommendation,
    ConfidenceScores,
    KnowledgeBase,
    
    # Utility functions
    schema_to_json_schema,
    create_schema_from_json,
    schema_to_dict,
    validate_json_against_schema,
    get_schema_class,
    list_available_schemas,
    STEP_SCHEMAS
)

# Version info
__version__ = "2.0.0"
__author__ = "FiberWise Platform"