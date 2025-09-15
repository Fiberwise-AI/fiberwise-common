import re
import json
import logging
from typing import Any, Dict, Optional, List, Union

logger = logging.getLogger(__name__)

def _apply_output_schema(text: str, schema: Dict[str, Any]) -> Optional[Union[Dict, List, Any]]:
    """
    Parses a string of text to extract structured data based on a provided schema.

    This function handles three schema types:
    - 'json': Extracts and parses JSON from the text
    - 'object': Extracts key-value pairs based on schema properties
    - 'array': Extracts list items from the text

    Args:
        text: The text to parse
        schema: The schema definition dict

    Returns:
        Parsed structured data or None if parsing fails
    """
    if not schema or not text:
        return None

    try:
        schema_type = schema.get('type', 'object')

        if schema_type == 'json':
            # For JSON schema, extract JSON from text
            # Find JSON blocks in the text - look for content between ```json and ```
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)

            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # If no code blocks, try to extract the entire text as JSON
                json_str = text.strip()

            # Try to parse the extracted text as JSON
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If parsing fails, use regex to find JSON-like patterns
                curly_match = re.search(r'\{[\s\S]*\}', json_str)
                if curly_match:
                    try:
                        return json.loads(curly_match.group(0))
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse JSON from response")
                        return None

        elif schema_type == 'object' and schema.get('properties'):
            # For object schema with properties, attempt to extract values
            properties = schema.get('properties', {})
            structured_data = {}

            for prop_name, prop_def in properties.items():
                # Try to extract property from text using pattern matching
                pattern = rf'{prop_name}[:\s]+([^\n]+)'
                matches = re.search(pattern, text, re.IGNORECASE)

                if matches:
                    value = matches.group(1).strip()

                    # Convert value based on type
                    prop_type = prop_def.get('type')
                    if prop_type == 'number':
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    elif prop_type == 'integer':
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                    elif prop_type == 'boolean':
                        value = value.lower() in ('true', 'yes', '1')

                    structured_data[prop_name] = value

            return structured_data if structured_data else None

        elif schema_type == 'array':
            # For array schema, split by lines or listed items
            items = []

            # Try to extract bullet points or numbered lists
            list_items = re.findall(r'(?:^|\n)[*\-•]?\s*(\d+\.?)?\s*([^\n]+)', text)

            if list_items:
                items = [item[1].strip() for item in list_items if item[1].strip()]
            else:
                # Fallback to splitting by newlines
                items = [line.strip() for line in text.split('\n')
                        if line.strip() and not line.strip().startswith('#')]

            return items if items else None

    except Exception as e:
        logger.warning(f"Error applying output schema: {str(e)}")

    return None


# Public alias for backward compatibility
apply_output_schema = _apply_output_schema


__all__ = ['_apply_output_schema', 'apply_output_schema']
