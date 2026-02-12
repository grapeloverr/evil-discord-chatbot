"""
Error handling utilities
"""

import logging
import traceback
from typing import Optional, Callable, Any
import discord

logger = logging.getLogger(__name__)


class BotError(Exception):
    """Base exception for bot-related errors"""
    pass


class ConfigurationError(BotError):
    """Configuration-related errors"""
    pass


class LLMError(BotError):
    """LLM provider errors"""
    pass


class MemoryError(BotError):
    """Memory system errors"""
    pass


def handle_error(error: Exception, context: str = "") -> str:
    """Convert exceptions to user-friendly messages"""
    error_type = type(error).__name__
    error_msg = str(error)
    
    logger.error(f"{context}: {error_type}: {error_msg}")
    logger.debug(traceback.format_exc())
    
    # User-friendly error messages
    if isinstance(error, discord.Forbidden):
        return "I don't have permission to do that."
    elif isinstance(error, discord.HTTPException):
        return "Something went wrong with Discord. Try again later."
    elif isinstance(error, ConfigurationError):
        return "Bot configuration error. Please check the setup."
    elif isinstance(error, LLMError):
        return "My brain is not working right now. Try again later."
    elif isinstance(error, MemoryError):
        return "I'm having trouble remembering things. Try again later."
    elif "timeout" in error_msg.lower():
        return "Request timed out. Try again."
    elif "connection" in error_msg.lower():
        return "Connection problem. Check your internet and try again."
    else:
        return "Something went wrong. Please try again later."


async def safe_execute(func: Callable, *args, **kwargs) -> tuple[bool, Any]:
    """Safely execute a function and return (success, result)"""
    try:
        result = await func(*args, **kwargs)
        return True, result
    except Exception as e:
        logger.error(f"Safe execute failed: {e}")
        return False, handle_error(e)


def error_embed(title: str, description: str, color: discord.Color = discord.Color.red()) -> discord.Embed:
    """Create a standardized error embed"""
    return discord.Embed(
        title=f"❌ {title}",
        description=description,
        color=color
    )


def success_embed(title: str, description: str, color: discord.Color = discord.Color.green()) -> discord.Embed:
    """Create a standardized success embed"""
    return discord.Embed(
        title=f"✅ {title}",
        description=description,
        color=color
    )


def info_embed(title: str, description: str, color: discord.Color = discord.Color.blue()) -> discord.Embed:
    """Create a standardized info embed"""
    return discord.Embed(
        title=f"ℹ️ {title}",
        description=description,
        color=color
    )
