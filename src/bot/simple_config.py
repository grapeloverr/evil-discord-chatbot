"""
Simple configuration loader for testing
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any

def load_simple_config() -> Dict[str, Any]:
    """Load configuration with simple approach"""
    environment = os.getenv("BOT_ENV", "production")

    def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_config(base[key], value)
            else:
                base[key] = value
        return base

    def replace_env_vars(value: str) -> str:
        pattern = r"\$\{([^:}]+)(?::([^}]*))?\}"

        def _sub(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            env_value = os.getenv(var_name)
            return env_value if env_value is not None else default_value

        return re.sub(pattern, _sub, value)

    def substitute_vars(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = substitute_vars(v)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = substitute_vars(item)
        elif isinstance(obj, str) and "${" in obj:
            obj = replace_env_vars(obj)
        return obj

    # Load base config
    config_file = Path("config") / "settings.yaml"
    base_config: Dict[str, Any] = {}
    if config_file.exists():
        with open(config_file, "r") as f:
            base_config = yaml.safe_load(f) or {}

    # Load environment-specific config
    env_config_file = Path("config") / f"settings.{environment}.yaml"
    if env_config_file.exists():
        with open(env_config_file, "r") as f:
            env_config = yaml.safe_load(f) or {}
            merge_config(base_config, env_config)

    # Environment variable substitution
    base_config = substitute_vars(base_config)

    return base_config
