"""
LLM model management and integration
"""

import aiohttp
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from .simple_config import load_simple_config

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from the model"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy"""
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM provider"""
    
    def __init__(self, base_url: str, model: str, **kwargs):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.config = kwargs
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Ollama API"""
        try:
            session = await self._get_session()
            
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.config.get("temperature", 0.8)),
                    "num_predict": kwargs.get("max_tokens", self.config.get("max_tokens", 150)),
                    "stop": kwargs.get("stop", ["User:", "Human:", "\n\n"]),
                }
            }
            
            timeout = aiohttp.ClientTimeout(total=self.config.get("timeout", 30))
            
            async with session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "").strip()
                else:
                    logger.error(f"Ollama API error: {response.status}")
                    return "Sorry, my brain is lagging."
                    
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return "Having trouble thinking right now."
    
    async def health_check(self) -> bool:
        """Check Ollama health"""
        try:
            session = await self._get_session()
            timeout = aiohttp.ClientTimeout(total=5)
            
            async with session.get(
                f"{self.base_url}/api/tags",
                timeout=timeout
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """List available models"""
        try:
            session = await self._get_session()
            timeout = aiohttp.ClientTimeout(total=5)
            
            async with session.get(
                f"{self.base_url}/api/tags",
                timeout=timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return [model["name"] for model in result.get("models", [])]
                return []
                
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.config = kwargs
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API"""
        try:
            session = await self._get_session()
            
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", self.config.get("max_tokens", 150)),
                "temperature": kwargs.get("temperature", self.config.get("temperature", 0.8)),
            }
            
            timeout = aiohttp.ClientTimeout(total=self.config.get("timeout", 30))
            
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=data,
                timeout=timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    logger.error(f"OpenAI API error: {response.status}")
                    return "Sorry, my brain is lagging."
                    
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return "Having trouble thinking right now."
    
    async def health_check(self) -> bool:
        """Check OpenAI API health"""
        try:
            session = await self._get_session()
            timeout = aiohttp.ClientTimeout(total=5)
            
            async with session.get(
                "https://api.openai.com/v1/models",
                timeout=timeout
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()


class ModelManager:
    """Manages LLM providers and model switching"""
    
    def __init__(self):
        self.providers = {}
        self.current_provider = None
        self.current_model = None
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the configured LLM provider"""
        llm_config = load_simple_config().get("llm", {})
        provider_type = llm_config.get("provider", "ollama")
        
        if provider_type == "ollama":
            self.current_provider = OllamaProvider(
                llm_config.get("base_url", "http://127.0.0.1:11434"),
                llm_config.get("model", "dolphin-llama3"),
                **{k: v for k, v in llm_config.items() if k not in ['base_url', 'model']}
            )
        elif provider_type == "openai":
            api_key = llm_config.get("api_key") or load_simple_config().get("openai.api_key")
            if not api_key:
                raise ValueError("OpenAI API key not configured")
            
            self.current_provider = OpenAIProvider(
                api_key=api_key,
                model=llm_config.get("model", "gpt-3.5-turbo"),
                **llm_config
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_type}")
        
        self.current_model = llm_config.get("model")
        logger.info(f"Initialized {provider_type} provider with model: {self.current_model}")
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using current provider"""
        if not self.current_provider:
            raise RuntimeError("No LLM provider initialized")
        
        return await self.current_provider.generate_response(prompt, **kwargs)
    
    async def health_check(self) -> bool:
        """Check current provider health"""
        if not self.current_provider:
            return False
        
        return await self.current_provider.health_check()
    
    async def list_models(self) -> List[str]:
        """List available models (if supported by provider)"""
        if not self.current_provider:
            return []
        
        if hasattr(self.current_provider, 'list_models'):
            return await self.current_provider.list_models()
        return []
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different model (if supported)"""
        if not self.current_provider:
            return False
        
        # For now, only support model switching in Ollama
        if isinstance(self.current_provider, OllamaProvider):
            available_models = await self.current_provider.list_models()
            if model_name in available_models:
                self.current_provider.model = model_name
                self.current_model = model_name
                logger.info(f"Switched to model: {model_name}")
                return True
        
        return False
    
    def get_current_model(self) -> str:
        """Get current model name"""
        return self.current_model
    
    def get_provider_type(self) -> str:
        """Get current provider type"""
        if isinstance(self.current_provider, OllamaProvider):
            return "ollama"
        elif isinstance(self.current_provider, OpenAIProvider):
            return "openai"
        return "unknown"
    
    async def close(self):
        """Close provider connections"""
        if self.current_provider:
            await self.current_provider.close()


# Global model manager instance (lazy-loaded)
model_manager = None

def get_model_manager():
    """Get global model manager instance"""
    global model_manager
    if model_manager is None:
        model_manager = ModelManager()
    return model_manager
