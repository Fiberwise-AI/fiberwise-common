"""
Configuration utilities for FiberWise - Pure utilities without CLI dependencies.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Configuration constants
FIBERWISE_DIR = Path.home() / ".fiberwise"
CONFIG_DIR = FIBERWISE_DIR / "configs"
DEFAULT_CONFIG_MARKER_FILE = FIBERWISE_DIR / "default_config.txt"

def get_default_config_name() -> Optional[str]:
    """Reads the default configuration name from the marker file."""
    if DEFAULT_CONFIG_MARKER_FILE.exists():
        try:
            content = DEFAULT_CONFIG_MARKER_FILE.read_text().strip()
            if content:
                return content
        except Exception:
            pass
    return None

def load_config(config_name: str) -> Optional[Dict[str, Any]]:
    """Loads configuration data from the specified config name."""
    # Sanitize config name for filename - same approach as commands.py
    safe_filename = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in config_name)
    config_filename = f"{safe_filename}.json"
    config_path = CONFIG_DIR / config_filename

    if not config_path.exists():
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except (json.JSONDecodeError, Exception):
        return None

    # Extract required information
    api_key = config_data.get("api_key") or config_data.get("fiberwise_api_key")
    api_endpoint = config_data.get("base_url") or config_data.get("fiberwise_base_url")

    # Validate required keys
    if not api_key or not api_endpoint:
        return None

    # Clean up endpoint URL (remove trailing slash if present)
    config_data["base_url"] = api_endpoint.rstrip('/')
    config_data["api_key"] = api_key
    return config_data

def save_config(config_data: Dict[str, Any], config_name: str) -> bool:
    """Save configuration data to the specified config name."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Sanitize config name for filename
        safe_filename = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in config_name)
        config_filename = f"{safe_filename}.json"
        config_path = CONFIG_DIR / config_filename
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        
        return True
    except Exception:
        return False
