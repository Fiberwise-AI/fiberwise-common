"""Tool Registry Service — lists available tools from ia_modules."""

import logging
from typing import List, Dict, Any
from .base_service import BaseService

logger = logging.getLogger(__name__)


class ToolRegistryService(BaseService):
    """Lists tools available in the ia_modules tool registry."""

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Return all registered tools with metadata."""
        tools = []
        try:
            from ia_modules.tools.registry import ToolRegistry
            registry = ToolRegistry()
            for name, tool in registry.get_all_tools().items():
                tools.append({
                    "name": name,
                    "description": getattr(tool, "description", ""),
                    "category": getattr(tool, "category", "built-in"),
                    "parameters": getattr(tool, "parameters", {}),
                    "icon": getattr(tool, "icon", "cog"),
                    "example": getattr(tool, "example", None),
                })
        except ImportError:
            logger.info("ia_modules tool registry not available, returning built-in defaults")
            tools = self._builtin_tools()
        except Exception as e:
            logger.warning(f"Could not load tool registry: {e}")
            tools = self._builtin_tools()
        return tools

    def _builtin_tools(self) -> List[Dict[str, Any]]:
        """Fallback built-in tool list when ia_modules is unavailable."""
        return [
            {"name": "web_search", "description": "Search the web for information", "category": "built-in", "icon": "search", "parameters": {"query": {"type": "string", "description": "Search query"}}},
            {"name": "calculator", "description": "Evaluate mathematical expressions", "category": "built-in", "icon": "calculator", "parameters": {"expression": {"type": "string", "description": "Math expression"}}},
            {"name": "code_interpreter", "description": "Execute Python code in a sandboxed environment", "category": "built-in", "icon": "code", "parameters": {"code": {"type": "string", "description": "Python code to execute"}}},
            {"name": "file_reader", "description": "Read contents of a file", "category": "built-in", "icon": "file-alt", "parameters": {"path": {"type": "string", "description": "File path"}}},
            {"name": "http_request", "description": "Make HTTP requests to external APIs", "category": "built-in", "icon": "globe", "parameters": {"url": {"type": "string", "description": "URL"}, "method": {"type": "string", "description": "HTTP method"}}},
            {"name": "json_parser", "description": "Parse and extract data from JSON", "category": "built-in", "icon": "brackets-curly", "parameters": {"data": {"type": "string", "description": "JSON string"}, "path": {"type": "string", "description": "JSONPath expression"}}},
        ]
