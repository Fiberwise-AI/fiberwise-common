"""Tool Registry Service — lists available tools from ia_modules."""

import logging
from typing import List, Dict, Any
from .base_service import BaseService

logger = logging.getLogger(__name__)


class ToolRegistryService(BaseService):
    """Lists tools available in the ia_modules tool registry."""

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Return all registered tools with metadata from ia_modules."""
        try:
            # Import ia_modules builtin tools
            from ia_modules.tools.builtin_tools import (
                calculator_tool,
                web_search_tool,
                code_executor_tool,
                file_operations_tool,
                api_caller_tool
            )

            # Build tool list from actual ia_modules tools
            tools = [
                {
                    "name": "calculator",
                    "description": "Perform mathematical calculations and statistical operations (mean, median, std, etc.)",
                    "category": "computation",
                    "icon": "calculator",
                    "parameters": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate",
                            "required": True
                        }
                    },
                    "examples": ["2 + 2 * 3", "sqrt(16) + pow(2, 3)", "mean([1, 2, 3, 4, 5])"]
                },
                {
                    "name": "web_search",
                    "description": "Search the web for information (returns mock results in demo mode)",
                    "category": "research",
                    "icon": "search",
                    "parameters": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                            "required": True
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10)",
                            "required": False,
                            "default": 10
                        }
                    },
                    "examples": ["artificial intelligence trends", "Python async programming best practices"]
                },
                {
                    "name": "code_executor",
                    "description": "Execute Python code in a sandboxed environment",
                    "category": "development",
                    "icon": "code",
                    "requires_approval": True,
                    "parameters": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute",
                            "required": True
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Execution timeout in seconds (default: 5)",
                            "required": False,
                            "default": 5
                        }
                    },
                    "examples": ["print('Hello, World!')", "result = sum([1, 2, 3, 4, 5])\\nprint(result)"]
                },
                {
                    "name": "file_operations",
                    "description": "Read, write, list, and delete files with access controls",
                    "category": "filesystem",
                    "icon": "file-alt",
                    "requires_approval": True,
                    "parameters": {
                        "operation": {
                            "type": "string",
                            "description": "Operation type: read, write, read_json, write_json, list, delete",
                            "required": True,
                            "enum": ["read", "write", "read_json", "write_json", "list", "delete"]
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Path to file or directory",
                            "required": True
                        },
                        "data": {
                            "type": "string",
                            "description": "Content to write (for write operations)",
                            "required": False
                        }
                    },
                    "examples": [
                        'read: {"operation": "read", "file_path": "/data/config.json"}',
                        'write: {"operation": "write", "file_path": "/data/output.txt", "data": "Hello"}'
                    ]
                },
                {
                    "name": "api_caller",
                    "description": "Make HTTP API requests with support for GET, POST, PUT, DELETE",
                    "category": "integration",
                    "icon": "globe",
                    "parameters": {
                        "method": {
                            "type": "string",
                            "description": "HTTP method",
                            "required": True,
                            "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]
                        },
                        "url": {
                            "type": "string",
                            "description": "API endpoint URL",
                            "required": True
                        },
                        "headers": {
                            "type": "object",
                            "description": "HTTP headers",
                            "required": False
                        },
                        "json": {
                            "type": "object",
                            "description": "JSON request body (for POST/PUT)",
                            "required": False
                        },
                        "params": {
                            "type": "object",
                            "description": "URL query parameters",
                            "required": False
                        }
                    },
                    "examples": [
                        '{"method": "GET", "url": "https://api.example.com/data"}',
                        '{"method": "POST", "url": "https://api.example.com/create", "json": {"name": "test"}}'
                    ]
                }
            ]

            logger.info(f"✅ Loaded {len(tools)} ia_modules builtin tools")
            return tools

        except ImportError as e:
            logger.error(f"❌ ia_modules builtin tools not available: {e}")
            logger.error("Please ensure ia_modules is installed: pip install ia_modules")
            return []
        except Exception as e:
            logger.error(f"❌ Error loading tool registry: {e}")
            return []
