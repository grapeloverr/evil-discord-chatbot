"""
Tests for Discord bot functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
import sys
from pathlib import Path

# Add project root to path for testing
# so imports like `from src.bot...` resolve correctly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bot.core import DiscordBot
from src.bot.config import Config


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "discord": {
            "token": "test_token",
            "prefix": "!",
            "mention_prefix": True,
            "case_sensitive": False,
            "intents": ["messages", "guilds"]
        },
        "llm": {
            "provider": "ollama",
            "base_url": "http://test:11434",
            "model": "test-model",
            "temperature": 0.8,
            "max_tokens": 150
        },
        "memory": {
            "type": "memory",
            "max_conversations": 100,
            "persistence": False
        },
        "plugins": {
            "commands": False,
            "voice": False
        }
    }


@pytest.fixture
def bot(mock_config, monkeypatch):
    """Create a test bot instance"""
    from src.bot import core as core_module

    mock_model_manager = AsyncMock()
    mock_model_manager.generate_response = AsyncMock(return_value="Test response")
    mock_model_manager.health_check = AsyncMock(return_value=True)

    monkeypatch.setattr(core_module, "load_simple_config", lambda: mock_config)
    monkeypatch.setattr(core_module, "get_model_manager", lambda: mock_model_manager)
    return DiscordBot()


class TestDiscordBot:
    """Test Discord bot functionality"""
    
    def test_bot_initialization(self, bot):
        """Test bot initializes correctly"""
        assert bot is not None
        assert hasattr(bot, 'memory')
        assert hasattr(bot, 'model_manager')
    
    @pytest.mark.asyncio
    async def test_generate_response(self, bot):
        """Test response generation"""
        # Mock the model manager
        bot.model_manager.generate_response = AsyncMock(return_value="Test response")
        
        response = await bot._generate_response(
            "Hello",
            "test_channel",
            "test_user",
            "Test context"
        )
        
        assert response == "Test response"
        bot.model_manager.generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_mention(self, bot):
        """Test mention handling"""
        # Create mock message
        mock_message = Mock()
        mock_message.author.bot = False
        mock_message.author.id = 67890
        mock_message.author.display_name = "TestUser"
        mock_message.content = "hello"
        mock_message.guild.name = "TestGuild"
        mock_message.mentions = []
        typing_cm = AsyncMock()
        mock_message.channel.typing = Mock(return_value=typing_cm)
        mock_message.channel.send = AsyncMock()

        bot._generate_response = AsyncMock(return_value="hello back")

        # Test the method
        await bot._handle_mention(mock_message)
        
        # Verify typing was triggered
        mock_message.channel.typing.assert_called_once()


class TestConfiguration:
    """Test configuration system"""
    
    def test_config_loading(self, monkeypatch):
        """Test configuration loading"""
        monkeypatch.setenv("DISCORD_TOKEN", "test_token")
        config = Config()
        assert config is not None
        assert hasattr(config, 'config')
    
    def test_get_nested_config(self, monkeypatch):
        """Test getting nested configuration values"""
        monkeypatch.setenv("DISCORD_TOKEN", "test_token")
        config = Config()
        # Mock config data
        config.config = {
            "discord": {
                "token": "test_token",
                "prefix": "!"
            }
        }
        
        assert config.get("discord.token") == "test_token"
        assert config.get("discord.prefix") == "!"
        assert config.get("nonexistent.key", "default") == "default"


@pytest.mark.asyncio
class TestMemory:
    """Test memory system"""
    
    async def test_memory_manager_initialization(self):
        """Test memory manager initializes"""
        from src.bot.memory import MemoryManager
        
        config = {
            "type": "memory",
            "persistence": False,
            "max_conversations": 100
        }
        
        memory = MemoryManager(config)
        assert memory is not None
        assert memory.persistence == False
    
    async def test_add_conversation_memory_only(self):
        """Test adding conversation in memory-only mode"""
        from src.bot.memory import MemoryManager
        
        config = {
            "type": "memory",
            "persistence": False,
            "max_conversations": 100
        }
        
        memory = MemoryManager(config)
        
        # Should not raise an error
        memory.add_conversation(
            "channel1", "user1", "hello", "hi there", "context"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
