"""
Voice plugin for Discord bot (optional)
"""

import logging

logger = logging.getLogger(__name__)


def setup_voice(bot):
    """Setup voice features if dependencies are available"""
    try:
        # Try to import voice dependencies
        import speech_recognition as sr
        import edge_tts
        from discord.ext import voice_recv
        
        # Import voice commands
        from .commands import setup_voice_commands
        
        # Setup voice commands
        setup_voice_commands(bot)
        
        logger.info("Voice plugin loaded successfully")
        return True
        
    except ImportError as e:
        logger.warning(f"Voice dependencies not available: {e}")
        logger.info("Voice features disabled. Install optional dependencies to enable:")
        logger.info("- SpeechRecognition>=3.10.0")
        logger.info("- edge-tts>=6.1.0")
        logger.info("- discord-ext-voice-recv>=0.5.0")
        return False
    except Exception as e:
        logger.error(f"Failed to setup voice plugin: {e}")
        return False


__all__ = ["setup_voice"]
