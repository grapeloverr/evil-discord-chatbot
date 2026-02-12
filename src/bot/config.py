"""
Configuration management for Discord bot
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    """Unified configuration management"""
    
    def __init__(self, config_path: Optional[str] = None, environment: str = None):
        self.environment = environment or os.getenv("BOT_ENV", "production")
        self.config_dir = Path(config_path or "config").resolve()
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML files"""
        try:
            # Load base configuration
            base_config_file = self.config_dir / "settings.yaml"
            if base_config_file.exists():
                with open(base_config_file, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
            
            # Load environment-specific configuration
            env_config_file = self.config_dir / f"settings.{self.environment}.yaml"
            logger.debug(f"Looking for env config file: {env_config_file}")
            logger.debug(f"Env config file exists: {env_config_file.exists()}")
            
            if env_config_file.exists():
                with open(env_config_file, 'r') as f:
                    env_config = yaml.safe_load(f) or {}
                    logger.debug(f"Loaded env config: {env_config}")
                    logger.debug(f"Before merge - base config: {self.config}")
                    self._merge_config(self.config, env_config)
                    logger.debug(f"After merge - config: {self.config}")
            else:
                logger.debug(f"No env config file found for {self.environment}")
            
            # Substitute environment variables
            self._substitute_env_vars(self.config)
            
            # Validate configuration (only after loading everything)
            self._validate_config()  # Re-enabled for testing
            
            logger.info(f"Configuration loaded for environment: {self.environment}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge configuration dictionaries"""
        logger.debug(f"Merging config. Base: {base}")
        logger.debug(f"Override: {override}")
        
        for key, value in override.items():
            logger.debug(f"Processing key: {key}, value: {value}")
            logger.debug(f"Base has key {key}: {key in base}")
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                logger.debug(f"Recursive merge for {key}")
                self._merge_config(base[key], value)
            else:
                logger.debug(f"Direct assignment for {key}")
                base[key] = value
        
        logger.debug(f"Result after merge: {base}")
    
    def _substitute_env_vars(self, config: Any):
        """Replace ${VAR:default} patterns with environment variables"""
        if isinstance(config, dict):
            for key, value in config.items():
                config[key] = self._substitute_env_vars(value)
        elif isinstance(config, list):
            for i, item in enumerate(config):
                config[i] = self._substitute_env_vars(item)
        elif isinstance(config, str) and "${" in config:
            config = self._replace_env_var(config)
        return config
    
    def _replace_env_var(self, value: str) -> str:
        """Replace a single environment variable pattern"""
        import re
        
        pattern = r'\$\{([^:}]+):?([^}]*)\}'
        match = re.search(pattern, value)
        
        if match:
            var_name = match.group(1)
            default_value = match.group(2)
            env_value = os.getenv(var_name)
            
            # Only substitute if environment variable is actually set
            # or use default value
            if env_value is not None:
                return env_value
            else:
                return default_value
        
        return value
    
    def _validate_config(self):
        """Validate required configuration values"""
        logger.debug(f"Validating config for environment: {self.environment}")
        logger.debug(f"Current config: {self.config}")
        
        required_keys = [
            "llm.provider",
            "llm.model"
        ]
        
        # Only validate Discord token if not in test mode
        # Check if we're in development mode and test_mode is enabled
        is_dev_mode = self.environment == "development"
        test_mode_enabled = self.config.get("development", {}).get("test_mode", False)
        
        logger.debug(f"Dev mode: {is_dev_mode}, Test mode enabled: {test_mode_enabled}")
        
        if not (is_dev_mode and test_mode_enabled):
            required_keys.append("discord.token")
        
        logger.debug(f"Required keys: {required_keys}")
        
        for key in required_keys:
            if not self.get(key):
                raise ValueError(f"Required configuration key missing: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_discord_config(self) -> Dict[str, Any]:
        """Get Discord-specific configuration"""
        return self.get("discord", {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM-specific configuration"""
        return self.get("llm", {})
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory-specific configuration"""
        return self.get("memory", {})
    
    def get_voice_config(self) -> Dict[str, Any]:
        """Get voice-specific configuration"""
        return self.get("voice", {})
    
    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled"""
        return self.get(f"plugins.{plugin_name}", False)
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.get("development.debug", False)
    
    def get_log_level(self) -> str:
        """Get logging level"""
        return self.get("logging.level", "INFO")
    
    def get_log_format(self) -> str:
        """Get logging format"""
        return self.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# Global configuration instance (lazy-loaded)
config = None

def get_config():
    """Get global configuration instance"""
    global config
    if config is None:
        config = Config()
    return config
