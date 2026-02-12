"""
Server Scraper - Fetches real conversations from Discord server for training data.

This module scrapes message history from Discord channels and creates
high-quality training data from real human conversations.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging

import discord
from discord.ext import commands

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration
DEFAULT_MIN_MESSAGE_LENGTH = 10
DEFAULT_MAX_MESSAGE_LENGTH = 2000
MIN_CONVERSATION_LENGTH = 2
CONVERSATION_GAP_MINUTES = 5


@dataclass
class TrainingSample:
    """Represents a single training sample from scraped conversation."""
    input_text: str
    output_text: str
    source_channel: str
    source_channel_id: int
    source_author: str
    source_author_id: int
    target_author: str
    target_author_id: int
    timestamp: str
    quality_score: float = 0.0
    conversation_context: str = ""
    tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Conversation:
    """Represents a multi-message conversation thread."""
    messages: list = field(default_factory=list)
    channel_id: int = 0
    channel_name: str = ""

    def add_message(self, message) -> None:
        self.messages.append(message)

    def is_valid(self) -> bool:
        return len(self.messages) >= MIN_CONVERSATION_LENGTH

    def to_training_samples(self, min_length: int = DEFAULT_MIN_MESSAGE_LENGTH) -> list[TrainingSample]:
        """Convert conversation to training samples (consecutive message pairs)."""
        samples = []
        
        for i in range(len(self.messages) - 1):
            msg1 = self.messages[i]
            msg2 = self.messages[i + 1]
            
            # Skip if messages are too short or too long
            if not (min_length <= len(msg1.content) <= DEFAULT_MAX_MESSAGE_LENGTH):
                continue
            if not (min_length <= len(msg2.content) <= DEFAULT_MAX_MESSAGE_LENGTH):
                continue
            
            # Create sample (msg1 as input, msg2 as response)
            sample = TrainingSample(
                input_text=clean_text(msg1.content),
                output_text=clean_text(msg2.content),
                source_channel=self.channel_name,
                source_channel_id=self.channel_id,
                source_author=str(msg1.author),
                source_author_id=msg1.author.id,
                target_author=str(msg2.author),
                target_author_id=msg2.author.id,
                timestamp=msg2.created_at.isoformat(),
                quality_score=calculate_quality(msg1.content, msg2.content),
                conversation_context=f"Conversation in #{self.channel_name}",
                tags=["scraped", "real_conversation"]
            )
            samples.append(sample)
        
        return samples


def clean_text(text: str) -> str:
    """Clean and normalize text for training."""
    # Remove Discord-specific formatting
    text = re.sub(r'<@\d+>', '@user', text)  # User mentions
    text = re.sub(r'<#\d+>', '#channel', text)  # Channel mentions
    text = re.sub(r'<:\w+:\d+>', '[emoji]', text)  # Custom emojis
    text = re.sub(r':\w+:', '[emoji]', text)  # Text emojis
    text = re.sub(r'http\S+', '[link]', text)  # URLs
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def calculate_quality(input_text: str, output_text: str) -> float:
    """Calculate quality score for a conversation pair."""
    score = 0.0
    
    # Length score (prefer medium-length, coherent messages)
    input_len = len(input_text)
    output_len = len(output_text)
    
    if 20 <= input_len <= 500:
        score += 0.2
    elif 500 < input_len <= 1000:
        score += 0.1
    
    if 20 <= output_len <= 500:
        score += 0.2
    elif 500 < output_len <= 1000:
        score += 0.1
    
    # Variety score (avoid repetitive content)
    input_words = set(input_text.lower().split())
    output_words = set(output_text.lower().split())
    overlap = len(input_words & output_words)
    vocab_ratio = len(output_words) / max(len(output_words), 1)
    
    if vocab_ratio > 0.3:  # Some vocabulary reuse is good
        score += 0.1
    if overlap < len(output_words) * 0.5:  # But not too much repetition
        score += 0.1
    
    # Question-response bonus
    if '?' in input_text:
        score += 0.2
    
    # Code block or technical content bonus
    if '```' in input_text or '`' in input_text:
        score += 0.1
    
    return min(score, 1.0)


class ServerScraper:
    """
    Discord server scraper for generating training data.
    
    Features:
    - Fetch message history from all/specified channels
    - Filter spam, commands, and low-quality content
    - Group messages into conversations
    - Generate training samples with quality scores
    - Review mode for manual approval
    """
    
    def __init__(
        self,
        token: str,
        min_message_length: int = DEFAULT_MIN_MESSAGE_LENGTH,
        max_messages_per_channel: int = 10000,
        exclude_channels: Optional[list] = None,
        exclude_patterns: Optional[list] = None
    ):
        self.token = token
        self.min_message_length = min_message_length
        self.max_messages_per_channel = max_messages_per_channel
        self.exclude_channels = set(exclude_channels or [])
        self.exclude_patterns = exclude_patterns or [
            r'^!', r'^/', r'^\.', r'^--', r'^\?'
        ]
        
        self.bot: Optional[commands.Bot] = None
        self.samples: list[TrainingSample] = []
        self.stats: dict = defaultdict(int)
        
    async def _setup_bot(self) -> None:
        """Initialize Discord bot for scraping."""
        intents = discord.Intents.default()
        intents.message_content = True
        
        self.bot = commands.Bot(
            command_prefix='!',
            intents=intents,
            help_command=None
        )
        
        @self.bot.event
        async def on_ready():
            logger.info(f"Logged in as {self.bot.user}")
        
        await self.bot.login(self.token)
    
    def _should_exclude_message(self, message: discord.Message) -> bool:
        """Check if message should be excluded."""
        # Skip bot messages
        if message.author.bot:
            self.stats['bot_messages'] += 1
            return True
        
        # Skip commands (starts with prefix)
        content = message.content.strip()
        for pattern in self.exclude_patterns:
            if re.match(pattern, content):
                self.stats['command_messages'] += 1
                return True
        
        # Skip too short/long messages
        if len(content) < self.min_message_length:
            self.stats['too_short'] += 1
            return True
        if len(content) > DEFAULT_MAX_MESSAGE_LENGTH:
            self.stats['too_long'] += 1
            return True
        
        # Skip messages that are just links/images
        if is_low_content(content):
            self.stats['low_content'] += 1
            return True
        
        return False
    
    async def _fetch_channel_history(
        self,
        channel: discord.TextChannel,
        limit: int = None
    ) -> list[discord.Message]:
        """Fetch message history from a channel."""
        messages = []
        
        try:
            async for message in channel.history(limit=limit or self.max_messages_per_channel):
                if not self._should_exclude_message(message):
                    messages.append(message)
        except Exception as e:
            logger.error(f"Error fetching from {channel.name}: {e}")
            self.stats['fetch_errors'] += 1
        
        return messages
    
    async def _group_into_conversations(
        self,
        messages: list[discord.Message]
    ) -> list[Conversation]:
        """Group messages into conversation threads based on time gaps."""
        if not messages:
            return []
        
        # Sort by timestamp (newest first for fetching, but we process oldest first)
        sorted_messages = sorted(messages, key=lambda m: m.created_at)
        
        conversations = []
        current_conv = Conversation(
            channel_id=messages[0].channel.id,
            channel_name=messages[0].channel.name
        )
        
        last_time = None
        
        for message in sorted_messages:
            msg_time = message.created_at
            
            # Check if this is a new conversation (time gap)
            if last_time and (msg_time - last_time) > timedelta(minutes=CONVERSATION_GAP_MINUTES):
                if current_conv.is_valid():
                    conversations.append(current_conv)
                current_conv = Conversation(
                    channel_id=message.channel.id,
                    channel_name=message.channel.name
                )
            
            current_conv.add_message(message)
            last_time = msg_time
        
        # Don't forget the last conversation
        if current_conv.is_valid():
            conversations.append(current_conv)
        
        logger.info(f"Grouped {len(messages)} messages into {len(conversations)} conversations")
        return conversations
    
    async def _fetch_all_channels(self, guild: discord.Guild) -> dict:
        """Fetch and process all applicable text channels."""
        channels_data = {}
        
        logger.info(f"Checking {len(guild.text_channels)} text channels...")
        
        for channel in guild.text_channels:
            logger.info(f"  Checking #{channel.name}...")
            
            # Check exclusions
            if channel.name in self.exclude_channels:
                logger.info(f"    Skipping excluded channel: {channel.name}")
                continue
            
            # Check permissions
            perms = channel.permissions_for(guild.me)
            if not perms.read_message_history:
                logger.warning(f"    No permission to read history in {channel.name}")
                continue
            
            if not perms.read_messages:
                logger.warning(f"    No permission to read messages in {channel.name}")
                continue
            
            logger.info(f"    Fetching from #{channel.name}...")
            
            messages = await self._fetch_channel_history(channel)
            conversations = await self._group_into_conversations(messages)
            
            channels_data[channel.name] = {
                'messages': len(messages),
                'conversations': len(conversations)
            }
            
            logger.info(f"    Found {len(messages)} messages, {len(conversations)} conversations")
            
            # Generate samples from all conversations
            for conv in conversations:
                samples = conv.to_training_samples(self.min_message_length)
                self.samples.extend(samples)
        
        return channels_data
    
    async def scrape(
        self,
        guild_id: int,
        channels: Optional[list[int]] = None,
        limit_per_channel: int = 5000
    ) -> dict:
        """
        Main scraping function.
        
        Args:
            guild_id: Discord server ID to scrape
            channels: Optional list of channel IDs to scrape (None = all)
            limit_per_channel: Max messages per channel
        
        Returns:
            dict with stats and samples
        """
        await self._setup_bot()
        
        try:
            # Get guild - try fetch first, then cache
            guild = None
            
            # Try to get from cache first (bot must be in the guild)
            guild = self.bot.get_guild(guild_id)
            
            if guild and len(guild.text_channels) > 0:
                logger.info(f"Found guild {guild.name} in cache with {len(guild.text_channels)} channels")
            else:
                # Fetch guild info from API
                logger.info(f"Guild not fully cached, fetching from API...")
                try:
                    fetched_guild = await self.bot.fetch_guild(guild_id)
                    # fetch_guild returns a PartialGuild, we need to ensure we have the full object
                    # The bot should be in the guild, so get_guild should work after this
                    guild = self.bot.get_guild(guild_id)
                except discord.NotFound:
                    raise ValueError(f"Guild with ID {guild_id} not found")
                except discord.Forbidden:
                    raise ValueError(f"Bot doesn't have access to guild {guild_id}")
            
            if not guild:
                raise ValueError(f"Could not access guild {guild_id}")
            
            # If still no channels, try fetching channels directly
            if len(guild.text_channels) == 0:
                logger.info("No channels in cache, fetching directly...")
                try:
                    # Fetch channels using REST API
                    async for channel in guild.fetch_channels():
                        if isinstance(channel, discord.TextChannel):
                            # Accessing channel.name triggers a fetch for partial channels
                            _ = channel.name
                    # Refresh guild reference
                    guild = self.bot.get_guild(guild_id)
                except Exception as e:
                    logger.warning(f"Could not fetch channels: {e}")
            
            logger.info(f"Scraping server: {guild.name}")
            logger.info(f"Found {len(guild.text_channels)} text channels")
            
            if channels:
                # Filter to specified channels
                target_channels = [c for c in guild.text_channels if c.id in channels]
            else:
                target_channels = list(guild.text_channels)
            
            # Fetch all channels
            channels_data = await self._fetch_all_channels(guild)
            
            # Calculate stats
            self.stats['total_samples'] = len(self.samples)
            self.stats['channels_processed'] = len(channels_data)
            
            # Sort samples by quality
            self.samples.sort(key=lambda s: s.quality_score, reverse=True)
            
            result = {
                'samples': [s.to_dict() for s in self.samples],
                'stats': dict(self.stats),
                'channels': channels_data,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Scraping complete: {len(self.samples)} samples generated")
            
            return result
            
        finally:
            await self.bot.close()
    
    def export_samples(
        self,
        output_path: Path,
        format: str = 'json',
        min_quality: float = 0.0,
        limit: int = None
    ) -> None:
        """Export samples to file in specified format."""
        samples = [s for s in self.samples if s.quality_score >= min_quality]
        if limit:
            samples = samples[:limit]
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([s.to_dict() for s in samples], f, indent=2, ensure_ascii=False)
        
        elif format == 'alpaca':
            # Export in Alpaca fine-tuning format
            alpaca_data = []
            for s in samples:
                alpaca_data.append({
                    'instruction': s.input_text,
                    'input': '',
                    'output': s.output_text
                })
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
        
        elif format == 'sharegpt':
            # Export in ShareGPT format
            sharegpt_data = []
            for s in samples:
                sharegpt_data.append({
                    'conversations': [
                        {'from': 'human', 'value': s.input_text},
                        {'from': 'gpt', 'value': s.output_text}
                    ],
                    'source': f"discord_{s.source_channel}",
                    'quality_score': s.quality_score
                })
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sharegpt_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(samples)} samples to {output_path}")
    
    def get_review_queue(self, min_quality: float = 0.3, max_samples: int = 100) -> list[TrainingSample]:
        """Get samples needing review (quality between 0.3 and 0.7)."""
        review_queue = [
            s for s in self.samples
            if min_quality <= s.quality_score <= 0.7
        ][:max_samples]
        return review_queue
    
    def approve_sample(self, sample: TrainingSample) -> None:
        """Mark sample as approved (increase quality)."""
        sample.quality_score = min(1.0, sample.quality_score + 0.1)
        sample.tags.append('approved')
    
    def reject_sample(self, sample: TrainingSample) -> None:
        """Mark sample as rejected."""
        sample.tags.append('rejected')
        sample.quality_score = 0.0


def is_low_content(text: str) -> bool:
    """Check if message is low-quality content (just links, emojis, etc.)."""
    if not text:
        return True
    
    # Remove common elements
    cleaned = re.sub(r'http\S+', '', text)  # Remove URLs
    cleaned = re.sub(r'<:\w+:\d+>', '', cleaned)  # Remove emojis
    cleaned = re.sub(r':\w+:', '', cleaned)  # Remove text emojis
    
    cleaned = cleaned.strip()
    
    # If less than 3 words or all emojis
    words = cleaned.split()
    if len(words) < 3:
        return True
    
    # If mostly punctuation/emoji
    alpha_chars = sum(c.isalpha() for c in cleaned)
    if alpha_chars / max(len(cleaned), 1) < 0.3:
        return True
    
    return False


async def run_scraper(
    token: str,
    guild_id: int,
    output_file: str = 'scraped_training_data.json',
    format: str = 'json',
    dry_run: bool = False
) -> dict:
    """Convenience function to run the scraper."""
    scraper = ServerScraper(token)
    
    result = await scraper.scrape(guild_id)
    
    if not dry_run:
        output_path = Path(output_file)
        scraper.export_samples(output_path, format=format)
    
    return result


# Interactive review functions
def review_samples(samples: list[TrainingSample]) -> tuple[list, list]:
    """
    Review samples interactively.
    
    Returns:
        tuple of (approved_samples, rejected_samples)
    """
    approved = []
    rejected = []
    
    for i, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{len(samples)} (Quality: {sample.quality_score:.2f})")
        print(f"Channel: {sample.source_channel}")
        print(f"{'='*60}")
        print(f"\nINPUT:")
        print(sample.input_text[:500])
        print(f"\nOUTPUT:")
        print(sample.output_text[:500])
        print(f"\n{'='*60}")
        
        while True:
            choice = input("[a]pprove, [r]eject, [s]kip: ").lower().strip()
            if choice in ('a', 'approve'):
                approved.append(sample)
                break
            elif choice in ('r', 'reject'):
                rejected.append(sample)
                break
            elif choice in ('s', 'skip'):
                break
    
    return approved, rejected


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape Discord server for training data')
    parser.add_argument('token', help='Discord bot token')
    parser.add_argument('guild_id', type=int, help='Server ID to scrape')
    parser.add_argument('--output', '-o', default='scraped_training_data.json',
                        help='Output file path')
    parser.add_argument('--format', choices=['json', 'alpaca', 'sharegpt'], default='json')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without saving')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Max messages per channel')
    parser.add_argument('--min-length', type=int, default=10,
                        help='Minimum message length')
    
    args = parser.parse_args()
    
    result = asyncio.run(run_scraper(
        token=args.token,
        guild_id=args.guild_id,
        output_file=args.output,
        format=args.format,
        dry_run=args.dry_run
    ))
    
    print(f"\nScraping complete: {len(result.get('samples', []))} samples generated")
    if not args.dry_run:
        print(f"Data saved to: {args.output}")