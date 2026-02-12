#!/usr/bin/env python3
"""
Main entry point for the Discord bot
"""

import sys
import os
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))
# Add root directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from bot import VoiceBot
from src.bot.simple_config import load_simple_config
from src.utils.errors import ConfigurationError

logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    try:
        # Load configuration
        config = load_simple_config()

        # Check for required configuration
        discord_token = os.getenv("DISCORD_TOKEN", "")
        if not discord_token or discord_token == "test_token_here":
            logger.error(
                "DISCORD_TOKEN environment variable not set or using test token!"
            )
            logger.error(
                "Please set your Discord bot token in environment or update the config."
            )
            logger.error(
                "Get your token from: https://discord.com/developers/applications"
            )
            sys.exit(1)

        # Create and run bot
        bot = VoiceBot()

        logger.info("Starting Discord bot...")
        bot.run(discord_token)

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
