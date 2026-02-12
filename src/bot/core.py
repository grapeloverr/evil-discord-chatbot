"""
Core Discord bot implementation
"""

import discord
from discord.ext import commands
import logging
from typing import Optional

from .simple_config import load_simple_config
from .memory import MemoryManager
from .models import get_model_manager
from ..utils.errors import handle_error, error_embed, success_embed
from ..utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class DiscordBot(commands.Bot):
    """Main Discord bot class"""
    
    def __init__(self):
        # Get configuration
        self.config = load_simple_config()
        discord_config = self.config.get("discord", {})
        
        prefix = discord_config.get("prefix", "!")
        mention_prefix = discord_config.get("mention_prefix", True)
        case_sensitive = discord_config.get("case_sensitive", False)
        
        # Setup intents
        intents = discord.Intents.default()
        intent_names = discord_config.get("intents", ["messages", "guilds"])
        
        for intent_name in intent_names:
            if hasattr(discord.Intents, intent_name):
                setattr(intents, intent_name, True)
        
        # Setup command prefix
        if mention_prefix:
            command_prefix = commands.when_mentioned_or(prefix)
        else:
            command_prefix = prefix
        
        super().__init__(
            command_prefix=command_prefix,
            intents=intents,
            case_insensitive=not case_sensitive,
            help_command=None  # We'll create custom help
        )
        
        # Initialize components
        self.memory = MemoryManager(self.config.get("memory", {}))
        self.model_manager = get_model_manager()
        
        # Load plugins
        self._load_plugins()
        
        logger.info(f"Bot initialized with prefix: '{prefix}'")
        logger.info(f"Mention prefix: {'enabled' if mention_prefix else 'disabled'}")
        logger.info(f"Case sensitive: {'yes' if case_sensitive else 'no'}")
    
    def _load_plugins(self):
        """Load enabled plugins"""
        # Load command plugin
        if (self.config.get("plugins", {}) or {}).get("commands", False):
            try:
                from ..plugins.commands import setup_commands
                setup_commands(self)
                logger.info("Commands plugin loaded")
            except Exception as e:
                logger.error(f"Failed to load commands plugin: {e}")
        
        # Load scrape command
        try:
            from ..plugins.commands import setup_scrape_command
            setup_scrape_command(self)
            logger.info("Scrape command loaded")
        except Exception as e:
            logger.error(f"Failed to load scrape command: {e}")
        
        # Load voice plugin
        if (self.config.get("plugins", {}) or {}).get("voice", False):
            try:
                from ..plugins.voice import setup_voice
                setup_voice(self)
                logger.info("Voice plugin loaded")
            except Exception as e:
                logger.error(f"Failed to load voice plugin: {e}")
    
    async def on_ready(self):
        """Called when bot is ready"""
        logger.info(f"Bot logged in as {self.user}")
        logger.info(f"Bot ID: {self.user.id}")
        logger.info(f"Connected to {len(self.guilds)} guilds")
        
        # Check LLM health
        if await self.model_manager.health_check():
            logger.info("LLM provider is healthy")
        else:
            logger.warning("LLM provider health check failed")
        
        # Set bot status
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name=f"{(self.config.get('discord', {}) or {}).get('prefix', '!')}chat"
            )
        )
    
    async def on_message(self, message):
        """Handle incoming messages"""
        # Don't respond to own messages
        if message.author == self.user:
            return
        
        # Don't respond to other bots
        if message.author.bot:
            return
        
        # Handle mentions
        if self.user.mentioned_in(message) and not message.mention_everyone:
            await self._handle_mention(message)
            return
        
        # Process commands
        await self.process_commands(message)
    
    async def _handle_mention(self, message):
        """Handle bot mentions"""
        try:
            # Extract content after mention
            content = message.content
            for mention in message.mentions:
                if mention == self.user:
                    content = content.replace(f"<@{mention.id}>", "").replace(
                        f"<@!{mention.id}>", ""
                    )
            
            content = content.strip()
            if not content:
                return
            
            logger.info(f"Bot mentioned by {message.author.display_name}: {content}")
            
            async with message.channel.typing():
                response = await self._generate_response(
                    content,
                    str(message.channel.id),
                    str(message.author.id),
                    f"User: {message.author.display_name} in {message.guild.name}"
                )
                
                await message.channel.send(response)
                
        except Exception as e:
            error_msg = handle_error(e, "Mention handler")
            await message.channel.send(embed=error_embed("Error", error_msg))
    
    async def _generate_response(self, user_input: str, channel_id: str, 
                              user_id: str, context: str = "") -> str:
        """Generate AI response with context"""
        try:
            # Get recent conversations for context
            recent_conversations = self.memory.get_recent_conversations(
                channel_id, user_id, limit=3
            )
            
            # Build context prompt
            context_text = ""
            if recent_conversations:
                context_text = "\\n--- Recent Conversation ---\\n"
                for conv in recent_conversations:
                    context_text += f"User: {conv['user_input']}\\n"
                    context_text += f"Bot: {conv['bot_response']}\\n\\n"
            
            # Build full prompt
            full_prompt = f"""You are a helpful AI assistant. Be concise but informative.

{context_text}
Current message: {user_input}

Respond as the AI assistant:"""
            
            # Generate response
            response = await self.model_manager.generate_response(full_prompt)
            
            # Store conversation
            self.memory.add_conversation(
                channel_id, user_id, user_input, response, context
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I'm having trouble thinking right now. Please try again later."
    
    async def on_command_error(self, ctx, error):
        """Handle command errors"""
        if isinstance(error, commands.CommandNotFound):
            return  # Don't respond to unknown commands
        
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(embed=error_embed(
                "Missing Argument",
                f"Missing required argument: `{error.param.name}`"
            ))
        elif isinstance(error, commands.MissingPermissions):
            await ctx.send(embed=error_embed(
                "Missing Permissions",
                "You don't have permission to use this command."
            ))
        elif isinstance(error, commands.BotMissingPermissions):
            await ctx.send(embed=error_embed(
                "Bot Missing Permissions",
                "I don't have permission to do that."
            ))
        else:
            error_msg = handle_error(error, f"Command {ctx.command}")
            await ctx.send(embed=error_embed("Command Error", error_msg))
    
    async def close(self):
        """Clean shutdown"""
        logger.info("Shutting down bot...")
        await self.model_manager.close()
        await super().close()
