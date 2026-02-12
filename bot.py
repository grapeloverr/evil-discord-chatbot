#!/usr/bin/env python3
"""
Discord Webhook Bot - Production-ready implementation
Implements /msg slash command that sends messages via webhooks while impersonating the bot
"""

import discord
from discord.ext import commands
from discord import app_commands
import logging
import sys
import os
import json
import math
import re
import random
from dataclasses import dataclass
import io
import asyncio
from datetime import datetime, timezone, timedelta
import aiohttp
from typing import Optional, Dict, Any, List, Union
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants
DISCORD_MESSAGE_LIMIT = 2000
DISCORD_FILE_SIZE_LIMIT = 8 * 1024 * 1024  # 8MB per file
SUPPORTED_IMAGE_TYPES = [
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/gif",
    "image/webp",
]
SUPPORTED_ATTACHMENT_TYPES = [
    "image/",
    "video/",
    "audio/",
    "application/pdf",
    "text/",
]  # Prefixes for supported types
MAX_ATTACHMENTS = 10
MAX_CHANNEL_MESSAGES = 30  # Maximum messages to keep in channel
PIN_RESEND_CHANNEL_ID = None  # Channel to resend pinned messages to
WEBHOOK_CACHE_TTL = timedelta(hours=1)  # Webhook cache expires after 1 hour
REQUIRED_PERMISSIONS = discord.Permissions(
    manage_webhooks=True,
    send_messages=True,
    read_message_history=True,
    embed_links=True,
    attach_files=True,
    manage_messages=True,  # Required for message deletion
    read_messages=True,
)

# Storage files
AUTODELETE_FILE = "autodelete_settings.json"
PREVIOUS_ROLES_FILE = "previous_roles.json"
RESENT_PINS_FILE = "resent_pins.json"
CENSOR_SETTINGS_FILE = "censor_settings.json"
PIN_SETTINGS_FILE = "pin_settings.json"
GUILD_SETTINGS_FILE = "guild_settings.json"
STYLE_REWARD_FILE = "style_reward_model.json"
HUMANIZE_TIMEOUT_SECONDS = 600
AUTO_TRAIN_DEFAULT_RATING = 8
AUTO_TRAIN_DEFAULT_SAVE_EVERY = 20
AUTO_TRAIN_SAVE_INTERVAL_SECONDS = 120
AUTO_TRAIN_MIN_CHARS = 6
AUTO_TRAIN_MAX_CHARS = 280

AI_STYLE_PHRASES = [
    "as an ai",
    "how can i assist",
    "i can help",
    "let me know if you need anything else",
    "i apologize",
    "certainly",
    "i understand your concern",
    "please feel free",
    "thank you for your patience",
    "i'm here to help",
    "i'm happy to help",
    "if you have any questions",
    "i hope this helps",
]

FORMAL_PHRASES = {
    "please",
    "kindly",
    "assist",
    "ensure",
    "certainly",
    "furthermore",
    "moreover",
    "additionally",
    "therefore",
    "thus",
    "however",
}
SLANG_WORDS = {"bro", "ngl", "lol", "wtf", "lmao", "idk", "fr", "rn", "yo", "nah"}
GREETINGS = {"hello", "hi", "hey", "greetings"}
CLOSINGS = {"sincerely", "regards", "best", "thanks", "thank you"}

FALLBACK_SERVER_STATEMENTS = [
    "why is this bot acting weird rn",
    "who pushed to prod without testing",
    "server lagging again after the update",
    "someone explain this bug in plain english",
    "i swear this worked yesterday",
    "we need better logs for this",
]


@dataclass
class CachedWebhook:
    """Cached webhook with expiration time"""

    webhook: discord.Webhook
    expires_at: datetime

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at


class MsgCommand:
    """Command to send messages via webhook"""

    @staticmethod
    async def callback(
        bot: "WebhookBot",
        interaction: discord.Interaction,
        message: str,
        attachment: Optional[discord.Attachment] = None,
    ):
        """Send a message via webhook impersonating the user"""
        # Acknowledge immediately to prevent timeout - Publicly to allow fallback to be public
        try:
            await interaction.response.defer(ephemeral=False)
        except discord.NotFound:
            # Interaction already expired
            return

        # Prepare content and attachment first
        files = []
        if attachment:
            if attachment.size <= DISCORD_FILE_SIZE_LIMIT:
                content_type = attachment.content_type or ""
                if any(
                    content_type.startswith(prefix)
                    for prefix in SUPPORTED_ATTACHMENT_TYPES
                ):
                    try:
                        file_data = await asyncio.wait_for(
                            attachment.read(), timeout=5.0
                        )
                        files.append(
                            discord.File(
                                io.BytesIO(file_data), filename=attachment.filename
                            )
                        )
                    except Exception as e:
                        logger.error(f"Failed to download attachment: {e}")
            else:
                # Large file - upload to cloud hosting
                link = await bot._upload_large_file(attachment)
                if link:
                    message += f"\n📎 **{attachment.filename}** (Large file): {link}"
                else:
                    message += (
                        f"\n❌ **{attachment.filename}** (Too large and upload failed)"
                    )

        try:
            # Check for webhook permissions - SAFELY
            can_use_webhooks = False
            if (
                isinstance(interaction.channel, discord.TextChannel)
                and interaction.app_permissions
                and interaction.app_permissions.manage_webhooks
            ):
                can_use_webhooks = True
            elif isinstance(interaction.channel, discord.TextChannel):
                # Fallback: Check guild permissions if app_permissions not available
                guild_perms = interaction.channel.permissions_for(interaction.guild.me)
                if guild_perms.manage_webhooks:
                    can_use_webhooks = True

            if can_use_webhooks:
                try:
                    # Try webhook path
                    webhook = await asyncio.wait_for(
                        bot._get_or_create_webhook(interaction.channel), timeout=2.0
                    )

                    webhook_data = {
                        "content": message[:DISCORD_MESSAGE_LIMIT],
                        "username": interaction.user.display_name,
                        "avatar_url": interaction.user.display_avatar.url
                        if interaction.user.display_avatar
                        else None,
                        "files": files,
                    }

                    await asyncio.wait_for(webhook.send(**webhook_data), timeout=5.0)

                    # Success via webhook - delete original thinking response
                    try:
                        await interaction.delete_original_response()
                    except Exception:
                        pass
                    return
                except Exception as e:
                    logger.warning(
                        f"Webhook failed, falling back to standard message: {e}"
                    )
                    # Re-create files for fallback if they were consumed?
                    # discord.py Files are consumable? Yes, usually.
                    # We should probably re-seek the BytesIO if possible.
                    for f in files:
                        f.fp.seek(0)

            # Fallback: Send as standard message (User Install / No Permission)
            # We cannot use webhooks here, so we use an Embed to mimic the user (impersonation)
            embed = discord.Embed(description=message, color=interaction.user.color)
            embed.set_author(
                name=interaction.user.display_name,
                icon_url=interaction.user.display_avatar.url
                if interaction.user.display_avatar
                else None,
            )

            # If there's an attachment but no message, we might need to adjust, but description is required?
            # actually logic handles message as arg, so it's fine.

            try:
                await interaction.edit_original_response(
                    content=None, embed=embed, attachments=files
                )
            except discord.NotFound:
                # Interaction died while processing
                logger.warning("Interaction expired before response could be sent")
                return

        except Exception as e:
            logger.error(f"Error in msg command: {e}")
            # If we failed, try to edit the original response with error message
            try:
                await interaction.edit_original_response(
                    content=f"❌ Error: {str(e)}", embed=None, attachments=[]
                )
            except discord.NotFound:
                # Interaction died, nothing we can do
                pass
            except discord.HTTPException:
                # Can't edit, try followup
                try:
                    await interaction.followup.send(
                        f"❌ Error: {str(e)}", ephemeral=True
                    )
                except Exception:
                    pass


class WebhookCache:
    """Thread-safe webhook caching system"""

    def __init__(self):
        self._cache: Dict[int, CachedWebhook] = {}
        self._lock = asyncio.Lock()

    async def get(self, channel_id: int) -> Optional[discord.Webhook]:
        """Get cached webhook if not expired"""
        async with self._lock:
            cached = self._cache.get(channel_id)
            if cached and not cached.is_expired():
                logger.debug(f"Using cached webhook for channel {channel_id}")
                return cached.webhook
            elif cached:
                # Remove expired webhook
                del self._cache[channel_id]
                logger.debug(f"Removed expired webhook for channel {channel_id}")
        return None

    async def set(self, channel_id: int, webhook: discord.Webhook) -> None:
        """Cache webhook with expiration"""
        async with self._lock:
            self._cache[channel_id] = CachedWebhook(
                webhook=webhook,
                expires_at=datetime.now(timezone.utc) + WEBHOOK_CACHE_TTL,
            )
            logger.debug(f"Cached webhook for channel {channel_id}")

    async def invalidate(self, channel_id: int) -> None:
        """Remove webhook from cache"""
        async with self._lock:
            if channel_id in self._cache:
                del self._cache[channel_id]
                logger.debug(f"Invalidated webhook cache for channel {channel_id}")


class WebhookBot(commands.Bot):
    """Main bot class with webhook functionality"""

    def __init__(
        self,
        pin_resend_channel_id: Optional[str] = None,
        auto_delete_count: Optional[int] = None,
        custom_prefix: Optional[str] = None,
        punish_role_id: Optional[str] = None,
    ):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True
        intents.reactions = True  # Ensure we can detect pin reactions

        # Normalize prefixes so internal parsing is always deterministic.
        if custom_prefix:
            if isinstance(custom_prefix, str):
                normalized_prefixes = [
                    p.strip() for p in custom_prefix.split(",") if p.strip()
                ]
            else:
                normalized_prefixes = [str(p).strip() for p in custom_prefix if str(p).strip()]
        else:
            normalized_prefixes = ["!", "~"]

        if not normalized_prefixes:
            normalized_prefixes = ["!", "~"]

        super().__init__(
            command_prefix=normalized_prefixes,
            intents=intents,
            help_command=None,
            activity=discord.Activity(
                type=discord.ActivityType.listening, name=f"/msg, !lq and ~lq"
            ),
        )

        self.pin_resend_channel_id = (
            int(pin_resend_channel_id)
        if pin_resend_channel_id
            else PIN_RESEND_CHANNEL_ID
        )

        # Use environment variable or fallback to hardcoded value
        self.auto_delete_count = (
            int(auto_delete_count) if auto_delete_count else MAX_CHANNEL_MESSAGES
        )

        # Store custom prefixes and punish role
        self.custom_prefix = normalized_prefixes
        self.punish_role_id = int(punish_role_id) if punish_role_id else None

        # Flexible configuration
        self.auto_delete_enabled_global = (
            os.getenv("AUTO_DELETE_ENABLED", "false").lower() == "true"
        )
        self.auto_delete_cooldown = int(
            os.getenv("AUTO_DELETE_COOLDOWN", "300")
        )  # 5 minutes
        self.auto_delete_rate_start = float(
            os.getenv("AUTO_DELETE_RATE_LIMIT_START", "1.0")
        )
        self.auto_delete_rate_max = float(
            os.getenv("AUTO_DELETE_RATE_LIMIT_MAX", "2.0")
        )
        self.auto_delete_bulk_delete = (
            os.getenv("AUTO_DELETE_BULK_DELETE", "true").lower() == "true"
        )
        self.auto_delete_exclude_pinned = (
            os.getenv("AUTO_DELETE_EXCLUDE_PINNED", "true").lower() == "true"
        )
        self.auto_delete_exclude_bots = (
            os.getenv("AUTO_DELETE_EXCLUDE_BOTS", "false").lower() == "true"
        )
        self.auto_delete_delete_age_hours = int(
            os.getenv("AUTO_DELETE_DELETE_AGE_HOURS", "0")
        )

        # Message filtering configuration
        self.filter_enabled = (
            os.getenv("FILTER_ENABLED", "false").lower() == "true"
        )
        self.filter_delete_instead = (
            os.getenv("FILTER_DELETE_INSTEAD", "false").lower() == "true"
        )
        self.filter_words = [
            w.strip()
            for w in os.getenv("FILTER_WORDS", "").split(",")
            if w.strip()
        ]

        # Censor Cover (Auto-replace)
        self.censor_cover_words = [
            w.strip()
            for w in os.getenv("CENSOR_COVER_WORDS", "").split(",")
            if w.strip()
        ]
        self._censor_settings = {}  # {guild_id: bool}
        self._pin_settings = {}  # {guild_id: channel_id}
        self._guild_settings = {}  # {guild_id: {setting_name: setting_value}}

        # Store previous roles for users who get punished
        self._previous_roles = {}  # {user_id: [role_ids]}

        self.webhook_cache = WebhookCache()
        self._ready_event = asyncio.Event()
        self._resent_pins = {}  # Track recently resent pinned messages per guild: {guild_id: set(message_ids)}
        self._autodelete_enabled = {}  # Track autodelete status per channel: {channel_id: bool}
        self._autodelete_limits = {}  # Track autodelete limits per channel: {channel_id: int}
        self._last_cleanup = {}  # Track last cleanup time per channel: {channel_id: datetime}

        # Task tracking
        self._cleanup_task: Optional[asyncio.Task] = None
        self._startup_task: Optional[asyncio.Task] = None
        self._style_reward_model = self._default_style_reward_model()
        self._humanize_sessions: Dict[tuple[int, int, int], Dict[str, Any]] = {}
        self._auto_train_updates = 0
        self._auto_train_last_save_ts = datetime.now(timezone.utc).timestamp()

    def _primary_prefix(self) -> str:
        return self.custom_prefix[0] if self.custom_prefix else "!"

    def _prefix_list_display(self) -> str:
        return ", ".join(f"`{p}`" for p in self.custom_prefix)

    def _default_style_reward_model(self) -> Dict[str, Any]:
        return {
            "version": 2,
            "learning_rate": 0.06,
            "bias": -1.35,
            "weights": {
                "ai_phrase": 2.80,
                "formality": 1.35,
                "avg_sentence": 1.05,
                "long_word_ratio": 1.00,
                "punct_burst": 0.70,
                "ellipsis": 0.45,
                "caps_ratio": 0.55,
                "contraction": -1.10,
                "slang": -0.90,
                "greeting": 0.50,
                "closing": 0.65,
                "repetition": 0.90,
                "unique_ratio": -0.60,
                "emoji_ratio": -0.40,
                "newline_ratio": 0.40,
                "number_ratio": 0.30,
                "quote_ratio": 0.45,
            },
            "strategy_stats": {
                "identity": {"count": 0, "mean_reward": 0.0},
                "casualize": {"count": 0, "mean_reward": 0.0},
                "trim_fluff": {"count": 0, "mean_reward": 0.0},
                "shorten": {"count": 0, "mean_reward": 0.0},
                "lowercase": {"count": 0, "mean_reward": 0.0},
                "drop_greeting": {"count": 0, "mean_reward": 0.0},
            },
            "generation_count": 0,
            "running_score": 0.0,
            "last_score": 0.0,
        }

    def _extract_ai_features(self, text: str) -> Dict[str, float]:
        cleaned = text or ""
        lowered = cleaned.lower()
        words = re.findall(r"[a-zA-Z']+", lowered)
        letters = [c for c in cleaned if c.isalpha()]
        sentence_chunks = [s for s in re.split(r"[.!?]+", cleaned) if s.strip()]
        word_count = max(1, len(words))
        sentence_count = max(1, len(sentence_chunks))
        avg_sentence_len = len(words) / sentence_count

        ai_phrase_hits = sum(1 for phrase in AI_STYLE_PHRASES if phrase in lowered)
        formality_hits = sum(1 for word in FORMAL_PHRASES if re.search(rf"\b{re.escape(word)}\b", lowered))
        slang_hits = sum(1 for word in SLANG_WORDS if re.search(rf"\b{re.escape(word)}\b", lowered))
        greeting_hits = sum(1 for word in GREETINGS if re.search(rf"\b{re.escape(word)}\b", lowered))
        closing_hits = sum(1 for word in CLOSINGS if re.search(rf"\b{re.escape(word)}\b", lowered))
        contractions = len(re.findall(r"\b[a-zA-Z]+('[a-zA-Z]+)\b", cleaned))
        long_word_ratio = sum(1 for w in words if len(w) >= 8) / word_count
        caps_ratio = (
            sum(1 for c in letters if c.isupper()) / max(1, len(letters))
            if letters
            else 0.0
        )
        punct_burst_hits = len(re.findall(r"[!?]{2,}", cleaned))
        ellipsis_hits = cleaned.count("...")
        emoji_hits = len(re.findall(r"[\U0001F300-\U0001FAFF]", cleaned))
        newline_count = cleaned.count("\n")
        number_hits = len(re.findall(r"\b\d+\b", cleaned))
        quote_hits = cleaned.count('"') + cleaned.count("'")
        unique_ratio = len(set(words)) / word_count
        repetition = max(0.0, 1.0 - unique_ratio)
        total_chars = max(1, len(cleaned))

        features = {
            "ai_phrase": min(1.0, ai_phrase_hits / 2.0),
            "formality": min(1.0, formality_hits / 4.0),
            "avg_sentence": min(1.0, avg_sentence_len / 26.0),
            "long_word_ratio": min(1.0, long_word_ratio * 2.5),
            "punct_burst": min(1.0, punct_burst_hits / 3.0),
            "ellipsis": min(1.0, ellipsis_hits / 2.0),
            "caps_ratio": min(1.0, caps_ratio * 3.0),
            "contraction": min(1.0, contractions / max(1.0, word_count / 5.0)),
            "slang": min(1.0, slang_hits / 3.0),
            "greeting": min(1.0, greeting_hits / 2.0),
            "closing": min(1.0, closing_hits / 2.0),
            "repetition": min(1.0, repetition * 1.5),
            "unique_ratio": min(1.0, unique_ratio * 1.2),
            "emoji_ratio": min(1.0, (emoji_hits / max(1, word_count)) * 2.0),
            "newline_ratio": min(1.0, newline_count / max(1, sentence_count)),
            "number_ratio": min(1.0, number_hits / max(1, word_count)),
            "quote_ratio": min(1.0, quote_hits / max(1, total_chars / 50)),
        }
        return features

    def _score_ai_text(self, text: str) -> float:
        features = self._extract_ai_features(text)
        weights = self._style_reward_model["weights"]
        raw = float(self._style_reward_model.get("bias", -1.0))
        for name, value in features.items():
            raw += float(weights.get(name, 0.0)) * float(value)
        # Logistic to 0..10
        score = 10.0 / (1.0 + math.exp(-raw))
        # Hard clamp if AI phrases detected.
        if features.get("ai_phrase", 0.0) > 0.45:
            score = max(score, 7.0)
        return round(max(0.0, min(10.0, score)), 1)

    def _apply_style_strategy(self, text: str, strategy: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return cleaned

        if strategy == "identity":
            return cleaned

        if strategy == "casualize":
            replacements = {
                r"\bi am\b": "i'm",
                r"\bi do not\b": "i don't",
                r"\bcannot\b": "can't",
                r"\bwill not\b": "won't",
                r"\bthat is\b": "that's",
                r"\bthere is\b": "there's",
                r"\blet me know\b": "tell me",
                r"\bi apologize\b": "my bad",
            }
            out = cleaned
            for pattern, repl in replacements.items():
                out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
            return out

        if strategy == "trim_fluff":
            out = re.sub(
                r"\s*(let me know if you need anything else|please feel free to ask).*?$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()
            return out or cleaned

        if strategy == "shorten":
            parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
            if len(parts) <= 2:
                return cleaned
            return " ".join(parts[:2])

        if strategy == "lowercase":
            return cleaned.lower()

        if strategy == "drop_greeting":
            out = re.sub(r"^(hey|hi|hello|greetings)[,!.\s]+", "", cleaned, flags=re.IGNORECASE)
            return out.strip() or cleaned

        return cleaned

    def _choose_humanized_text(self, text: str) -> tuple[str, float, str]:
        strategies = list(self._style_reward_model["strategy_stats"].keys())
        candidates = {}
        base_text = (text or "").strip()
        for strategy in strategies:
            candidate = self._apply_style_strategy(base_text, strategy)
            if candidate:
                candidates[strategy] = candidate
        if not candidates:
            return base_text, self._score_ai_text(base_text), "identity"

        scored = {}
        for strategy, candidate in candidates.items():
            score = self._score_ai_text(candidate)
            stats = self._style_reward_model["strategy_stats"].get(
                strategy, {"count": 0, "mean_reward": 0.0}
            )
            # Bandit-like selection: reward low scores and underexplored strategies.
            adjusted = score - (stats.get("mean_reward", 0.0) * 2.2) - (
                0.35 / math.sqrt(1 + stats.get("count", 0))
            )
            scored[strategy] = (candidate, score, adjusted)

        best_strategy = min(scored.keys(), key=lambda key: scored[key][2])
        best_text, best_score, _ = scored[best_strategy]
        return best_text, best_score, best_strategy

    def _learn_from_generation(self, text: str, ai_score: float, strategy: str) -> None:
        features = self._extract_ai_features(text)
        model = self._style_reward_model
        lr = float(model.get("learning_rate", 0.06))
        target = 0.15  # 1.5/10 AI score target
        error = (ai_score / 10.0) - target

        # Update classifier weights toward lower AI score.
        for name, value in features.items():
            current = float(model["weights"].get(name, 0.0))
            updated = current - (lr * error * float(value))
            # mild L2 regularization toward 0
            updated *= 0.995
            model["weights"][name] = max(-3.0, min(3.0, round(updated, 6)))

        bias = float(model.get("bias", -1.0))
        model["bias"] = max(-3.0, min(3.0, round((bias - (lr * error)) * 0.998, 6)))

        reward = max(0.0, 1.0 - (ai_score / 10.0))
        stats = model["strategy_stats"].setdefault(
            strategy, {"count": 0, "mean_reward": 0.0}
        )
        count = int(stats.get("count", 0)) + 1
        mean_reward = float(stats.get("mean_reward", 0.0))
        mean_reward += (reward - mean_reward) / count
        stats["count"] = count
        stats["mean_reward"] = round(mean_reward, 6)

        model["generation_count"] = int(model.get("generation_count", 0)) + 1
        prev_avg = float(model.get("running_score", 0.0))
        n = model["generation_count"]
        model["running_score"] = round(prev_avg + ((ai_score - prev_avg) / n), 6)
        model["last_score"] = ai_score

    def _format_scored_text(self, text: str, ai_score: float) -> str:
        return f"{text}\n\nis this ai: {ai_score:.1f}/10"

    def _prepare_scored_text(self, text: str) -> str:
        best_text, best_score, strategy = self._choose_humanized_text(text)
        self._learn_from_generation(best_text, best_score, strategy)
        self._save_persistent_data()
        return self._format_scored_text(best_text, best_score)

    async def _reply_scored(self, message: discord.Message, text: str, **kwargs):
        return await message.reply(self._prepare_scored_text(text), **kwargs)

    def _humanize_session_key(self, guild_id: int, channel_id: int, user_id: int) -> tuple[int, int, int]:
        return (guild_id, channel_id, user_id)

    def _parse_humanize_choice(self, text: str) -> Optional[int]:
        raw = (text or "").strip()
        if raw in {"1", "2", "3"}:
            return int(raw)
        return None

    def _parse_humanize_rating(self, text: str) -> Optional[int]:
        raw = (text or "").strip().lower()
        # Accept: 5, 5/10, 5 / 10, 10/10
        match = re.match(r"^(10|[1-9])(?:\s*/\s*10)?$", raw)
        if not match:
            return None
        return int(match.group(1))

    def _humanize_user_statement_from_text(self, text: str) -> str:
        cleaned = " ".join((text or "").strip().split())
        if not cleaned:
            cleaned = random.choice(FALLBACK_SERVER_STATEMENTS)
        return cleaned[:180]

    async def _pick_random_server_statement(self, channel: discord.TextChannel) -> str:
        candidates: List[str] = []
        try:
            async for msg in channel.history(limit=200):
                if msg.author.bot:
                    continue
                if not msg.content:
                    continue
                if any(msg.content.startswith(p) for p in self.custom_prefix):
                    continue
                if msg.content.startswith("/"):
                    continue
                cleaned = self._humanize_user_statement_from_text(msg.content)
                if len(cleaned) >= 4:
                    candidates.append(cleaned)
        except Exception as exc:
            logger.warning(f"Could not read channel history for /humanize: {exc}")

        if candidates:
            return random.choice(candidates)
        return random.choice(FALLBACK_SERVER_STATEMENTS)

    def _build_humanize_option(self, statement: str, strategy: str) -> str:
        base = statement.strip()
        starters = [
            "fair",
            "real talk",
            "ngl",
            "yeah",
            "honestly",
            "true",
            "i get you",
        ]
        tails = [
            "we should fix it before it blows up",
            "that’s exactly why this keeps breaking",
            "someone needs to own this and close it",
            "let's patch it properly this time",
            "at least now we know where it fails",
            "that explains the mess from earlier",
        ]
        composed = f"{random.choice(starters)}, {base}. {random.choice(tails)}"
        humanized = self._apply_style_strategy(composed, strategy)
        return " ".join(humanized.split())

    def _generate_humanize_candidates(self, statement: str, count: int = 3) -> List[Dict[str, Any]]:
        strategies = list(self._style_reward_model.get("strategy_stats", {}).keys())
        if not strategies:
            strategies = ["identity", "casualize", "shorten"]

        random.shuffle(strategies)
        chosen = strategies[: max(3, count)]
        options: List[Dict[str, Any]] = []
        seen = set()
        for strategy in chosen:
            text = self._build_humanize_option(statement, strategy)
            if text in seen:
                continue
            seen.add(text)
            score = self._score_ai_text(text)
            options.append({"text": text, "strategy": strategy, "ai_score": score})
            if len(options) >= count:
                break

        while len(options) < count:
            text = self._build_humanize_option(statement, "identity")
            if text in seen:
                text = f"{text} rn"
            seen.add(text)
            options.append({"text": text, "strategy": "identity", "ai_score": self._score_ai_text(text)})

        return options

    def _format_humanize_statement(self, statement: str) -> str:
        return f"user: {statement}"

    def _format_humanize_options(self, options: List[Dict[str, Any]]) -> str:
        lines = ["bot:"]
        for idx, item in enumerate(options, start=1):
            lines.append(f"{idx}. {item['text']}")
        lines.append("which did you like best? reply with 1, 2, or 3")
        return "\n".join(lines)

    def _update_strategy_reward_from_rating(self, strategy: str, rating_10: int) -> None:
        stats = self._style_reward_model["strategy_stats"].setdefault(
            strategy, {"count": 0, "mean_reward": 0.0}
        )
        reward = max(0.0, min(1.0, rating_10 / 10.0))
        count = int(stats.get("count", 0)) + 1
        mean_reward = float(stats.get("mean_reward", 0.0))
        mean_reward += (reward - mean_reward) / count
        stats["count"] = count
        stats["mean_reward"] = round(mean_reward, 6)

    def _learn_from_human_rating(
        self, text: str, strategy: str, rating_10: int, persist: bool = True
    ) -> float:
        # Higher user rating means the reply should feel less AI-like.
        predicted = self._score_ai_text(text)
        desired_ai_norm = 1.0 - (max(1, min(10, rating_10)) / 10.0)
        features = self._extract_ai_features(text)

        lr = float(self._style_reward_model.get("learning_rate", 0.06)) * 1.35
        error = (predicted / 10.0) - desired_ai_norm
        for name, value in features.items():
            current = float(self._style_reward_model["weights"].get(name, 0.0))
            updated = (current - (lr * error * float(value))) * 0.995
            self._style_reward_model["weights"][name] = max(-3.0, min(3.0, round(updated, 6)))

        bias = float(self._style_reward_model.get("bias", -1.0))
        self._style_reward_model["bias"] = max(
            -3.0, min(3.0, round((bias - (lr * error)) * 0.998, 6))
        )

        self._update_strategy_reward_from_rating(strategy, rating_10)
        rescored = self._score_ai_text(text)
        self._style_reward_model["last_score"] = rescored
        if persist:
            self._save_persistent_data()
        return rescored

    def _normalize_auto_train_text(self, text: str) -> Optional[str]:
        cleaned = " ".join((text or "").strip().split())
        if not cleaned:
            return None
        if len(cleaned) < AUTO_TRAIN_MIN_CHARS or len(cleaned) > AUTO_TRAIN_MAX_CHARS:
            return None
        # Ignore messages that are mostly symbols/noise.
        if not re.search(r"[A-Za-z0-9]", cleaned):
            return None
        if cleaned.lower().startswith(("http://", "https://", "www.")):
            return None
        if cleaned.count("```") >= 2:
            return None
        return cleaned

    def _auto_train_config_for_guild(self, guild_id: int) -> Dict[str, Any]:
        enabled = bool(self.get_guild_setting(guild_id, "auto_train_enabled", False))
        rating_raw = self.get_guild_setting(
            guild_id, "auto_train_target_rating", AUTO_TRAIN_DEFAULT_RATING
        )
        save_every_raw = self.get_guild_setting(
            guild_id, "auto_train_save_every", AUTO_TRAIN_DEFAULT_SAVE_EVERY
        )
        strategy = str(
            self.get_guild_setting(guild_id, "auto_train_strategy", "identity")
            or "identity"
        )
        if strategy not in self._style_reward_model.get("strategy_stats", {}):
            strategy = "identity"

        try:
            rating = int(rating_raw)
        except Exception:
            rating = AUTO_TRAIN_DEFAULT_RATING
        rating = max(1, min(10, rating))

        try:
            save_every = int(save_every_raw)
        except Exception:
            save_every = AUTO_TRAIN_DEFAULT_SAVE_EVERY
        save_every = max(1, min(500, save_every))

        return {
            "enabled": enabled,
            "rating": rating,
            "save_every": save_every,
            "strategy": strategy,
        }

    async def _maybe_auto_train_from_message(self, message: discord.Message) -> None:
        if not message.guild:
            return
        if getattr(message.author, "bot", False):
            return
        if getattr(message, "webhook_id", None):
            return

        config = self._auto_train_config_for_guild(message.guild.id)
        if not config["enabled"]:
            return

        content = (message.content or "").strip()
        if not content:
            return
        if any(content.startswith(prefix) for prefix in self.custom_prefix):
            return

        mentions = getattr(message, "mentions", [])
        if self.user and any(getattr(m, "id", None) == self.user.id for m in mentions):
            return

        normalized = self._normalize_auto_train_text(content)
        if not normalized:
            return

        try:
            self._learn_from_human_rating(
                normalized,
                config["strategy"],
                config["rating"],
                persist=False,
            )
            self._auto_train_updates += 1
            now_ts = datetime.now(timezone.utc).timestamp()
            should_persist = (
                self._auto_train_updates % config["save_every"] == 0
                or (now_ts - self._auto_train_last_save_ts) >= AUTO_TRAIN_SAVE_INTERVAL_SECONDS
            )
            if should_persist:
                self._save_persistent_data()
                self._auto_train_last_save_ts = now_ts
        except Exception as exc:
            logger.error(f"Auto-train update failed for guild {message.guild.id}: {exc}")

    async def _handle_humanize_session_message(self, message: discord.Message) -> bool:
        if not message.guild:
            return False

        key = self._humanize_session_key(message.guild.id, message.channel.id, message.author.id)
        session = self._humanize_sessions.get(key)
        if not session:
            return False

        if datetime.now(timezone.utc).timestamp() - float(session.get("created_ts", 0.0)) > HUMANIZE_TIMEOUT_SECONDS:
            self._humanize_sessions.pop(key, None)
            return False

        content = message.content.strip()
        stage = session.get("stage")

        if stage == "select":
            choice = self._parse_humanize_choice(content)
            if choice is None:
                return False
            session["selected"] = choice - 1
            session["stage"] = "rate"
            await message.reply("rate 1/10")
            return True

        if stage == "rate":
            rating = self._parse_humanize_rating(content)
            if rating is None:
                return False
            idx = int(session.get("selected", 0))
            options = session.get("options", [])
            if not options or idx < 0 or idx >= len(options):
                self._humanize_sessions.pop(key, None)
                return False
            selected = options[idx]
            rescored = self._learn_from_human_rating(
                selected["text"], selected.get("strategy", "identity"), rating
            )
            self._humanize_sessions.pop(key, None)
            await message.reply(
                f"logged {rating}/10 on option {idx + 1}. is this ai: {rescored:.1f}/10"
            )
            return True

        return False

    async def on_app_command_error(
        self,
        interaction: discord.Interaction,
        error: discord.app_commands.AppCommandError,
    ):
        """Handle application command errors"""
        if isinstance(error, discord.app_commands.CommandInvokeError):
            error = error.original

        # If interaction is unknown/not found, we can't do anything
        if isinstance(error, discord.NotFound):
            logger.warning(
                f"Interaction expired/not found for command {interaction.command.name if interaction.command else 'unknown'}"
            )
            return

        error_msg = "Unknown error occurred"
        if isinstance(error, discord.Forbidden):
            error_msg = "I don't have permission to do that"
        elif isinstance(error, discord.HTTPException):
            error_msg = "Discord API error occurred"
        else:
            error_msg = str(error)

        try:
            if not interaction.response.is_done():
                try:
                    await interaction.response.send_message(
                        f"❌ {error_msg}", ephemeral=True
                    )
                except discord.HTTPException as e:
                    # If it says already acknowledged, try followup
                    if e.code == 40060:
                        await interaction.followup.send(
                            f"❌ {error_msg}", ephemeral=True
                        )
                    else:
                        raise e
            else:
                # Response already done, use followup
                try:
                    await interaction.followup.send(f"❌ {error_msg}", ephemeral=True)
                except discord.NotFound:
                    # Interaction died, nothing we can do
                    pass
        except discord.NotFound:
            # Interaction died while we were trying to reply
            pass
        except Exception as e:
            logger.error(f"Failed to send error response: {e}")

        logger.error(f"Command error: {type(error).__name__}: {error}")

    def get_guild_setting(self, guild_id: int, setting_name: str, default=None):
        """Get a setting for a specific guild with automatic defaults"""
        guild_id_str = str(guild_id)
        guild_settings = self._guild_settings.get(guild_id_str, {})
        
        if isinstance(guild_settings, dict):
            return guild_settings.get(setting_name, default)
        elif isinstance(guild_settings, (int, str, bool)) and setting_name == "punish_role_id":
            return guild_settings if guild_settings is not None else default
        return default

    def set_guild_setting(self, guild_id: int, setting_name: str, value, persist=True):
        """Set a setting for a specific guild with optional persistence"""
        guild_id_str = str(guild_id)
        
        # Ensure guild settings is a dictionary
        if guild_id_str not in self._guild_settings or not isinstance(self._guild_settings[guild_id_str], dict):
            self._guild_settings[guild_id_str] = {}
        
        self._guild_settings[guild_id_str][setting_name] = value
        
        if persist:
            self._save_persistent_data()
        
        logger.info(f"Set guild setting for {guild_id}: {setting_name} = {value} ({'persisted' if persist else 'temporary'})")

    def get_all_guild_settings(self, guild_id: int):
        """Get all settings for a specific guild"""
        guild_id_str = str(guild_id)
        return self._guild_settings.get(guild_id_str, {})

    def delete_guild_setting(self, guild_id: int, setting_name: str):
        """Delete a setting for a specific guild"""
        guild_id_str = str(guild_id)
        if guild_id_str in self._guild_settings and isinstance(self._guild_settings[guild_id_str], dict):
            if setting_name in self._guild_settings[guild_id_str]:
                del self._guild_settings[guild_id_str][setting_name]
                self._save_persistent_data()
                logger.info(f"Deleted guild setting for {guild_id}: {setting_name}")
                return True
        return False

    def get_default_guild_settings(self):
        """Get default settings template for new servers"""
        return {
            "welcome_message": None,
            "auto_delete_enabled": False,
            "auto_delete_limit": 50,
            "pin_resend_channel": None,
            "censor_enabled": False,
            "log_level": "INFO",
            "auto_train_enabled": False,
            "auto_train_target_rating": AUTO_TRAIN_DEFAULT_RATING,
            "auto_train_save_every": AUTO_TRAIN_DEFAULT_SAVE_EVERY,
            "auto_train_strategy": "identity",
        }

    def apply_default_settings(self, guild_id: int):
        """Apply default settings to a new guild"""
        defaults = self.get_default_guild_settings()
        for setting_name, default_value in defaults.items():
            # Only set if not already configured
            if self.get_guild_setting(guild_id, setting_name) is None:
                self.set_guild_setting(guild_id, setting_name, default_value, persist=True)
        
        logger.info(f"Applied default settings to guild {guild_id}")

    async def _get_or_create_webhook(
        self, channel: discord.TextChannel
    ) -> discord.Webhook:
        """Get existing webhook or create new one for the channel"""
        # Try to get cached webhook first
        cached_webhook = await self.webhook_cache.get(channel.id)
        if cached_webhook:
            return cached_webhook

        # Try to find existing webhook in channel
        webhooks = await channel.webhooks()
        for webhook in webhooks:
            if webhook.name == f"{self.user.name} Webhook":
                await self.webhook_cache.set(channel.id, webhook)
                return webhook

        # Create new webhook
        webhook = await channel.create_webhook(
            name=f"{self.user.name} Webhook",
            reason="Auto-created webhook for message sending",
        )
        await self.webhook_cache.set(channel.id, webhook)
        return webhook

    async def _upload_to_catbox(self, attachment: discord.Attachment) -> Optional[str]:
        """Upload an attachment to catbox.moe and return the link"""
        try:
            # Catbox limit is 200MB
            if attachment.size > 200 * 1024 * 1024:
                logger.warning(
                    f"File {attachment.filename} is too large for Catbox (>{attachment.size} bytes)"
                )
                return None

            logger.info(
                f"Uploading {attachment.filename} ({attachment.size} bytes) to catbox.moe"
            )
            file_data = await attachment.read()

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            # Catbox API requires reqtype=fileupload
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field("reqtype", "fileupload")
                data.add_field("fileToUpload", file_data, filename=attachment.filename)

                async with session.post(
                    "https://catbox.moe/user/api.php", data=data, headers=headers
                ) as resp:
                    if resp.status == 200:
                        link = await resp.text()
                        if link.startswith("https://files.catbox.moe/"):
                            logger.info(f"Successfully uploaded {attachment.filename}: {link}")
                            return link.strip()
                        else:
                            logger.error(
                                f"Catbox returned unexpected response: {link[:100]}"
                            )
                    else:
                        resp_text = await resp.text()
                        logger.error(
                            f"Catbox upload failed with status {resp.status}: {resp_text[:500]}"
                        )
                    return None
        except Exception as e:
            logger.error(f"Error uploading to Catbox: {e}")
            return None

    async def _upload_to_file_io(self, attachment: discord.Attachment) -> Optional[str]:
        """Upload an attachment to file.io and return the link"""
        try:
            logger.info(
                f"Uploading {attachment.filename} ({attachment.size} bytes) to file.io"
            )
            file_data = await attachment.read()
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field("file", file_data, filename=attachment.filename)
                async with session.post(
                    "https://file.io/", data=data, headers=headers
                ) as resp:
                    if resp.status == 200:
                        try:
                            json_resp = await resp.json()
                            return json_resp.get("link")
                        except Exception as e:
                            resp_text = await resp.text()
                            logger.error(
                                f"file.io returned 200 but failed to parse JSON: {e}. Body: {resp_text[:500]}"
                            )
                            return None
                    else:
                        resp_text = await resp.text()
                        logger.error(
                            f"file.io upload failed with status {resp.status}: {resp_text[:500]}"
                        )
                        return None
        except Exception as e:
            logger.error(f"Error uploading to file.io: {e}")
            return None

    async def _upload_large_file(self, attachment: discord.Attachment) -> Optional[str]:
        """Try multiple services to upload a large file"""
        # 1. Try Catbox (200MB limit)
        if attachment.size <= 200 * 1024 * 1024:
            logger.info(f"Attempting to upload {attachment.filename} to Catbox ({attachment.size / 1024 / 1024:.1f}MB)")
            link = await self._upload_to_catbox(attachment)
            if link:
                logger.info(f"Successfully uploaded {attachment.filename} to Catbox: {link}")
                return link
            else:
                logger.warning(f"Catbox upload failed for {attachment.filename}, trying file.io")

        # 2. Try file.io as fallback
        logger.info(f"Attempting to upload {attachment.filename} to file.io ({attachment.size / 1024 / 1024:.1f}MB)")
        return await self._upload_to_file_io(attachment)

    def _load_persistent_data(self):
        """Load persistent data from JSON files"""
        try:
            # Load autodelete settings
            if os.path.exists(AUTODELETE_FILE):
                with open(AUTODELETE_FILE, "r") as f:
                    data = json.load(f)
                    # Handle both old format (boolean only) and new format (with limits)
                    self._autodelete_enabled = {}
                    self._autodelete_limits = {}
                    for k, v in data.items():
                        channel_id = int(k)
                        if isinstance(v, dict):
                            self._autodelete_enabled[channel_id] = v.get(
                                "enabled", False
                            )
                            self._autodelete_limits[channel_id] = v.get(
                                "limit", self.auto_delete_count
                            )
                        else:
                            # Legacy format - treat as enabled flag
                            self._autodelete_enabled[channel_id] = v
                            self._autodelete_limits[channel_id] = self.auto_delete_count
                    logger.info(
                        f"Loaded autodelete settings for {len(self._autodelete_enabled)} channels"
                    )

            # Load previous roles
            if os.path.exists(PREVIOUS_ROLES_FILE):
                with open(PREVIOUS_ROLES_FILE, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._previous_roles = {int(k): v for k, v in data.items()}
                        logger.info(
                            f"Loaded previous roles for {len(self._previous_roles)} users"
                        )
                    else:
                        self._previous_roles = {}
                        logger.warning(
                            f"previous_roles.json is not a dictionary, resetting"
                        )

            # Load resent pins
            if os.path.exists(RESENT_PINS_FILE):
                with open(RESENT_PINS_FILE, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # Convert lists back to sets and ensure guild_ids are integers
                        self._resent_pins = {int(k): set(v) for k, v in data.items()}
                        total_pins = sum(
                            len(pins) for pins in self._resent_pins.values()
                        )
                        logger.info(
                            f"Loaded {total_pins} resent pin entries across {len(self._resent_pins)} guilds"
                        )
                    else:
                        self._resent_pins = {}
                        logger.warning(
                            f"resent_pins.json is not a dictionary, resetting"
                        )

            # Load censor settings
            if os.path.exists(CENSOR_SETTINGS_FILE):
                with open(CENSOR_SETTINGS_FILE, "r") as f:
                    self._censor_settings = json.load(f)
                    logger.info(
                        f"Loaded censor settings for {len(self._censor_settings)} guilds"
                    )

            # Load pin settings
            if os.path.exists(PIN_SETTINGS_FILE):
                with open(PIN_SETTINGS_FILE, "r") as f:
                    self._pin_settings = json.load(f)
                    logger.info(
                        f"Loaded pin settings for {len(self._pin_settings)} guilds"
                    )

            # Load guild settings
            if os.path.exists(GUILD_SETTINGS_FILE):
                with open(GUILD_SETTINGS_FILE, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._guild_settings = data
                        logger.info(
                            f"Loaded guild settings for {len(self._guild_settings)} guilds"
                        )
                    else:
                        self._guild_settings = {}
                        logger.warning(
                            f"guild_settings.json is not a dictionary, resetting"
                        )

            # Load style reward model
            if os.path.exists(STYLE_REWARD_FILE):
                with open(STYLE_REWARD_FILE, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        merged = self._default_style_reward_model()
                        merged.update(
                            {k: v for k, v in data.items() if k in merged}
                        )
                        saved_weights = data.get("weights")
                        if isinstance(saved_weights, dict):
                            merged["weights"].update(saved_weights)
                        saved_stats = data.get("strategy_stats")
                        if isinstance(saved_stats, dict):
                            for name, stats in saved_stats.items():
                                if not isinstance(stats, dict):
                                    continue
                                if name not in merged["strategy_stats"]:
                                    merged["strategy_stats"][name] = {
                                        "count": 0,
                                        "mean_reward": 0.0,
                                    }
                                merged["strategy_stats"][name]["count"] = int(
                                    max(0, stats.get("count", 0))
                                )
                                merged["strategy_stats"][name]["mean_reward"] = float(
                                    stats.get("mean_reward", 0.0)
                                )
                        self._style_reward_model = merged
                        logger.info("Loaded style reward model")
                    else:
                        logger.warning(
                            "style_reward_model.json is not a dictionary, resetting"
                        )

        except Exception as e:
            logger.error(f"Error loading persistent data: {e}")

    async def setup_hook(self):
        """Setup hook called before bot starts"""

        # Register slash command using decorator approach
        @self.tree.command()
        @discord.app_commands.allowed_installs(guilds=True, users=True)
        @discord.app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
        @discord.app_commands.describe(
            message="The message to send", attachment="Optional image attachment"
        )
        async def msg(
            interaction: discord.Interaction,
            message: str,
            attachment: discord.Attachment = None,
        ):
            """Send a message via webhook impersonating the user"""
            await MsgCommand.callback(self, interaction, message, attachment)

        @self.tree.command()
        @discord.app_commands.describe(message_id="The message ID to resend")
        async def resend_pin(interaction: discord.Interaction, message_id: str):
            """Manually resend a pinned message"""
            if not interaction.guild:
                await interaction.response.send_message(
                    "❌ This command can only be used in a server.", ephemeral=True
                )
                return

            # Parse message ID
            try:
                msg_id = int(message_id)
            except ValueError:
                await interaction.response.send_message(
                    "Invalid message ID format. Please provide a valid message ID.",
                    ephemeral=True,
                )
                return

            # Find the message - only search within the current guild
            message = None
            for channel in interaction.guild.text_channels:
                try:
                    msg = await channel.fetch_message(msg_id)
                    if msg and msg.pinned:
                        message = msg
                        break
                except discord.NotFound:
                    continue
                except Exception as e:
                    logger.error(
                        f"Error fetching message {msg_id} for manual resend: {e}"
                    )

            if message:
                # Check if pin resend channel is configured for this guild
                target_channel_id = self._pin_settings.get(str(interaction.guild.id))
                if not target_channel_id:
                    await interaction.response.send_message(
                        "❌ No pin resend channel configured for this server. Use `/redirect_pins #channel` to set one.",
                        ephemeral=True,
                    )
                    return

                # Resend the message
                try:
                    target_channel = self.get_channel(int(target_channel_id))
                    if not target_channel:
                        await interaction.response.send_message(
                            "❌ Pin resend channel not found.", ephemeral=True
                        )
                        return

                    webhook = await self._get_or_create_webhook(target_channel)

                    content = message.content or ""
                    files = []

                    for attachment in message.attachments:
                        if attachment.size <= DISCORD_FILE_SIZE_LIMIT:
                            try:
                                file_data = await attachment.read()
                                files.append(
                                    discord.File(
                                        io.BytesIO(file_data),
                                        filename=attachment.filename,
                                    )
                                )
                            except Exception as e:
                                logger.error(f"Error reading attachment: {e}")
                                pass
                        else:
                            # Large file - upload to cloud hosting
                            logger.info(f"Processing large attachment: {attachment.filename} ({attachment.size / 1024 / 1024:.1f}MB)")
                            link = await self._upload_large_file(attachment)
                            if link:
                                content += f"\n📎 **{attachment.filename}** (Large file): {link}"
                                logger.info(f"Successfully uploaded large file {attachment.filename}: {link}")
                            else:
                                content += f"\n❌ **{attachment.filename}** (Too large to send and upload failed)"
                                logger.error(f"Failed to upload large file {attachment.filename}")

                    if not content and not files:
                        content = ""

                    # Send via webhook - SILENT operation
                    await webhook.send(
                        content=content,
                        username=message.author.display_name,
                        avatar_url=message.author.display_avatar.url if message.author.display_avatar else None,
                        files=files
                    )
                    
                    logger.info(f"✅ Pin {message.id} resent to {target_channel.name}")
                    
                    self._add_resent_pin(interaction.guild.id, message.id)

                    await interaction.response.send_message(
                        f"✅ Pinned message from {message.author.mention} resent successfully to {target_channel.mention}!",
                        ephemeral=True,
                    )
                except Exception as e:
                    logger.error(f"Error resending pin: {e}")
                    if not interaction.response.is_done():
                        await interaction.response.send_message(
                            "❌ Failed to resend message.", ephemeral=True
                        )
                    else:
                        await interaction.followup.send(
                            "❌ Failed to resend message.", ephemeral=True
                        )
            else:
                await interaction.response.send_message(
                    "❌ Could not find the specified message in this server or it's not pinned.",
                    ephemeral=True,
                )

        @self.tree.command()
        @discord.app_commands.describe(
            user="The user to ban",
            reason="Reason for banning (optional)",
            delete_message_days="Number of days of messages to delete (0-7)",
        )
        async def ban(
            interaction: discord.Interaction,
            user: discord.Member,
            reason: str = "No reason provided",
            delete_message_days: int = 0,
        ):
            """Ban a user from the server"""
            if not interaction.user.guild_permissions.ban_members:
                await interaction.response.send_message(
                    "❌ You need 'Ban Members' permission to use this command.",
                    ephemeral=True,
                )
                return

            # Check role hierarchy
            if (
                user.top_role >= interaction.user.top_role
                and interaction.user != interaction.guild.owner
            ):
                await interaction.response.send_message(
                    "❌ You cannot ban someone with equal or higher role than you.",
                    ephemeral=True,
                )
                return

            # Cannot ban the server owner
            if user == interaction.guild.owner:
                await interaction.response.send_message(
                    "❌ You cannot ban the server owner.", ephemeral=True
                )
                return

            # Validate delete_message_days
            if delete_message_days < 0 or delete_message_days > 7:
                await interaction.response.send_message(
                    "❌ Delete message days must be between 0 and 7.", ephemeral=True
                )
                return

            try:
                await interaction.guild.ban(
                    user,
                    reason=f"Banned by {interaction.user}: {reason}",
                    delete_message_days=delete_message_days,
                )
                await interaction.response.send_message(
                    f"✅ **{user.display_name}** has been banned.\n"
                    f"Reason: {reason}\n"
                    f"Messages deleted: {delete_message_days} days",
                    ephemeral=True,
                )
                logger.info(
                    f"User {user} banned by {interaction.user} - Reason: {reason}, Messages deleted: {delete_message_days} days"
                )

            except discord.Forbidden:
                await interaction.response.send_message(
                    "❌ I don't have permission to ban this user.", ephemeral=True
                )
            except Exception as e:
                logger.error(f"Error in ban command: {e}")
                await interaction.response.send_message(
                    "❌ Failed to ban user. Check logs for details.", ephemeral=True
                )

        @self.tree.command(description="Kick a user")
        @discord.app_commands.describe(user="The user to kick", reason="Reason for the kick")
        @discord.app_commands.default_permissions(kick_members=True)
        @discord.app_commands.checks.has_permissions(kick_members=True)
        async def kick(
            interaction: discord.Interaction,
            user: discord.Member,
            reason: str = "No reason provided",
        ):
            """Kick a user from the server"""
            if not interaction.guild:
                await interaction.response.send_message(
                    "❌ This command can only be used in a server.", ephemeral=True
                )
                return

            # Check role hierarchy
            if (
                user.top_role >= interaction.user.top_role
                and interaction.user != interaction.guild.owner
            ):
                await interaction.response.send_message(
                    "❌ You cannot kick someone with equal or higher role than you.",
                    ephemeral=True,
                )
                return

            try:
                await user.kick(reason=f"Kicked by {interaction.user}: {reason}")
                await interaction.response.send_message(
                    f"✅ **{user.display_name}** has been kicked.\nReason: {reason}",
                    ephemeral=False,
                )
            except discord.Forbidden:
                await interaction.response.send_message(
                    "❌ I don't have permission to kick this user.", ephemeral=True
                )
            except Exception as e:
                logger.error(f"Error in kick command: {e}")
                await interaction.response.send_message(
                    "❌ Failed to kick user.", ephemeral=True
                )

        @self.tree.command(description="Timeout a user")
        @discord.app_commands.describe(
            user="The user to timeout",
            duration="Duration (e.g. 10m, 1h, 1d)",
            reason="Reason for timeout",
        )
        @discord.app_commands.default_permissions(moderate_members=True)
        @discord.app_commands.checks.has_permissions(moderate_members=True)
        async def timeout(
            interaction: discord.Interaction,
            user: discord.Member,
            duration: str,
            reason: str = "No reason provided",
        ):
            """Timeout a user for a specified duration"""
            delta = self._parse_duration_str(duration)
            if not delta:
                await interaction.response.send_message(
                    "❌ Invalid duration format. Use '10m', '1h', '1d', etc.",
                    ephemeral=True,
                )
                return

            # Check hierarchy
            if (
                user.top_role >= interaction.user.top_role
                and interaction.user != interaction.guild.owner
            ):
                await interaction.response.send_message(
                    "❌ You cannot timeout someone with equal or higher role than you.",
                    ephemeral=True,
                )
                return

            try:
                await user.timeout(
                    delta, reason=f"Timeout by {interaction.user}: {reason}"
                )
                await interaction.response.send_message(
                    f"✅ **{user.display_name}** has been timed out for {duration}.\nReason: {reason}"
                )
            except discord.Forbidden:
                await interaction.response.send_message(
                    "❌ I don't have permission to timeout this user.", ephemeral=True
                )
            except Exception as e:
                logger.error(f"Error in timeout command: {e}")
                await interaction.response.send_message(
                    "❌ Failed to timeout user.", ephemeral=True
                )

        @self.tree.command(description="Timeout all members with a specific role")
        @discord.app_commands.describe(
            role="The role to target",
            duration="Duration (e.g. 10m, 1h)",
            reason="Reason for timeout",
        )
        @discord.app_commands.default_permissions(moderate_members=True)
        @discord.app_commands.checks.has_permissions(moderate_members=True)
        async def timeout_role(
            interaction: discord.Interaction,
            role: discord.Role,
            duration: str,
            reason: str = "No reason provided",
        ):
            """Mass timeout all members with a specific role"""
            if not interaction.guild:
                return

            delta = self._parse_duration_str(duration)
            if not delta:
                await interaction.response.send_message(
                    "❌ Invalid duration format. Use '10m', '1h', '1d', etc.",
                    ephemeral=True,
                )
                return

            # Check hierarchy
            if (
                role >= interaction.user.top_role
                and interaction.user != interaction.guild.owner
            ):
                await interaction.response.send_message(
                    "❌ You cannot target a role equal to or higher than your top role.",
                    ephemeral=True,
                )
                return

            await interaction.response.defer(ephemeral=False)

            count = 0
            errors = 0

            # Iterate members
            # Fetch members if needed? Role.members might be incomplete without intent, but usually okay for cache.
            # Assuming Chunking is enabled or cache is populated.

            for member in role.members:
                if member.bot:
                    continue
                # Skip if member is higher/equal to moderator
                if (
                    member.top_role >= interaction.user.top_role
                    and interaction.user != interaction.guild.owner
                ):
                    continue

                try:
                    await member.timeout(
                        delta,
                        reason=f"Mass timeout ({role.name}) by {interaction.user}: {reason}",
                    )
                    count += 1
                except Exception as e:
                    logger.error(f"Error timing out member {member}: {e}")
                    errors += 1

            await interaction.followup.send(
                f"✅ Mass Timeout Complete.\nTarget Role: **{role.name}**\nTimed out: {count} members\nErrors: {errors}\nDuration: {duration}"
            )

        @self.tree.command(description="Redirect pinned messages to a specific channel")
        @discord.app_commands.describe(channel="The channel to send pinned messages to")
        @discord.app_commands.default_permissions(manage_guild=True)
        @discord.app_commands.checks.has_permissions(manage_guild=True)
        async def redirect_pins(
            interaction: discord.Interaction, channel: discord.TextChannel
        ):
            """Set the channel where pinned messages are redirected"""
            if not interaction.guild:
                return

            self._pin_settings[str(interaction.guild.id)] = channel.id
            self._save_persistent_data()

            await interaction.response.send_message(
                f"✅ Pinned messages will now be redirected to {channel.mention}",
                ephemeral=False,
            )

        @self.tree.command(description="Show help and command list")
        async def help(interaction: discord.Interaction):
            """Show help and command list"""
            embed = discord.Embed(
                title="🤖 Bot Help",
                description="Here are the available commands:",
                color=discord.Color.blue(),
            )

            # Public Commands
            embed.add_field(
                name="/msg",
                value="Send a message as the bot (or invisible webhook)",
                inline=False,
            )

            # Moderation Commands
            embed.add_field(
                name="🛡️ Moderation",
                value="**/ban @user** - Ban a user\n"
                "**/kick @user** - Kick a user\n"
                "**/timeout @user [duration]** - Timeout a user (e.g. 10m, 1h)\n"
                "**/timeout_role @role [duration]** - Timeout ALL members in a role",
                inline=False,
            )

            # Configuration Commands (Admin)
            prefixes = " or ".join(self.custom_prefix) if isinstance(self.custom_prefix, list) else self.custom_prefix
            embed.add_field(
                name="⚙️ Configuration",
                value=f"**/censor_toggle [true/false]** - Enable/Disable censor cover\n"
                f"**/redirect_pins #channel** - Set where pins are resent\n"
                f"**/aiscore <text>** - Score text AI-likeness\n"
                f"**/humanize** - Generate 3 human-style reply options and collect feedback\n"
                f"**/auto-train** - Configure passive human-style learning per server\n"
                f"**/server_settings list** - Show all server settings\n"
                f"**/server_settings get <name>** - Get a specific setting\n"
                f"**/server_settings set <name> <value>** - Set a setting\n"
                f"**/server_settings delete <name>** - Delete a setting\n"
                f"**/apply_defaults** - Apply default settings to new server\n"
                f"**/set_punish_role @role** - Set jail role for this server\n"
                f"**/jailrole @user [duration]** - Jail a user (remove roles, add jail role)\n"
                f"**/unjailrole @user** - Unjail a user (restore previous roles)\n"
                f"**{prefixes}help** - Show this help message\n"
                f"**{prefixes}aiscore <text>** - Score text AI-likeness\n"
                f"**{prefixes}humanize** - Run humanization options + rating loop\n"
                f"**{prefixes}autotrain [on|off] [rating]** - Configure auto training\n"
                f"**{prefixes}lq @user** - Jail a user (remove roles, add jail role)\n"
                f"**{prefixes}ulq @user** - Unjail a user (restore previous roles)\n"
                f"**{prefixes}prefix** - Show current command prefixes\n"
                f"**{prefixes}censor_toggle true/false** - Same as /censor_toggle\n"
                f"**{prefixes}set_punish_role @role** - Set jail role for this server",
                inline=False,
            )

            await interaction.response.send_message(embed=embed, ephemeral=True)

        @self.tree.command(description="Score text with AI-likeness rating")
        @discord.app_commands.describe(text="Text to score")
        async def aiscore(interaction: discord.Interaction, text: str):
            score = self._score_ai_text(text)
            await interaction.response.send_message(
                f"`is this ai: {score:.1f}/10`\n{text}",
                ephemeral=True,
            )

        @self.tree.command(description="Generate 3 humanized reply options and collect feedback")
        async def humanize(interaction: discord.Interaction):
            if not interaction.guild or not isinstance(interaction.channel, discord.TextChannel):
                await interaction.response.send_message(
                    "❌ This command can only be used in a server text channel.",
                    ephemeral=True,
                )
                return

            statement = await self._pick_random_server_statement(interaction.channel)
            options = self._generate_humanize_candidates(statement, count=3)
            key = self._humanize_session_key(
                interaction.guild.id, interaction.channel.id, interaction.user.id
            )
            self._humanize_sessions[key] = {
                "created_ts": datetime.now(timezone.utc).timestamp(),
                "stage": "select",
                "statement": statement,
                "options": options,
                "selected": None,
            }
            await interaction.response.send_message(
                self._format_humanize_statement(statement),
                ephemeral=False,
            )
            await interaction.followup.send(
                self._format_humanize_options(options),
                ephemeral=False,
            )

        @self.tree.command(
            name="auto-train",
            description="Configure passive human-style training for this server",
        )
        @discord.app_commands.describe(
            enabled="Enable/disable auto training (leave empty to view status)",
            target_rating="Human target rating (1-10, default 8)",
        )
        @discord.app_commands.default_permissions(manage_guild=True)
        @discord.app_commands.checks.has_permissions(manage_guild=True)
        async def auto_train(
            interaction: discord.Interaction,
            enabled: Optional[bool] = None,
            target_rating: discord.app_commands.Range[int, 1, 10] = AUTO_TRAIN_DEFAULT_RATING,
        ):
            if not interaction.guild:
                await interaction.response.send_message(
                    "❌ This command can only be used in a server.", ephemeral=True
                )
                return

            guild_id = interaction.guild.id
            if enabled is None:
                cfg = self._auto_train_config_for_guild(guild_id)
                state = "enabled" if cfg["enabled"] else "disabled"
                await interaction.response.send_message(
                    f"Auto-train is currently **{state}**.\n"
                    f"Target rating: **{cfg['rating']}/10**\n"
                    f"Strategy: **{cfg['strategy']}**\n"
                    f"Save every: **{cfg['save_every']}** updates",
                    ephemeral=True,
                )
                return

            self.set_guild_setting(guild_id, "auto_train_enabled", bool(enabled), persist=False)
            self.set_guild_setting(
                guild_id, "auto_train_target_rating", int(target_rating), persist=False
            )
            if self.get_guild_setting(guild_id, "auto_train_strategy") is None:
                self.set_guild_setting(guild_id, "auto_train_strategy", "identity", persist=False)
            if self.get_guild_setting(guild_id, "auto_train_save_every") is None:
                self.set_guild_setting(
                    guild_id,
                    "auto_train_save_every",
                    AUTO_TRAIN_DEFAULT_SAVE_EVERY,
                    persist=False,
                )
            self._save_persistent_data()

            status = "enabled" if enabled else "disabled"
            await interaction.response.send_message(
                f"✅ Auto-train {status} for **{interaction.guild.name}**.\n"
                f"Target rating set to **{int(target_rating)}/10**.",
                ephemeral=True,
            )

        @self.tree.command(description="Apply default settings to this server")
        @discord.app_commands.default_permissions(manage_guild=True)
        @discord.app_commands.checks.has_permissions(manage_guild=True)
        async def apply_defaults(interaction: discord.Interaction):
            """Apply default settings to current server"""
            if not interaction.guild:
                await interaction.response.send_message(
                    "❌ This command can only be used in a server.", ephemeral=True
                )
                return
            
            guild_id = interaction.guild.id
            current_settings = self.get_all_guild_settings(guild_id)
            
            if current_settings:
                await interaction.response.send_message(
                    "⚠️ This server already has settings configured. Use `/server_settings list` to view them.", 
                    ephemeral=True
                )
                return
            
            # Apply defaults and show what was set
            self.apply_default_settings(guild_id)
            defaults = self.get_default_guild_settings()
            
            embed = discord.Embed(
                title="⚙️ Applied Default Settings",
                description="The following default settings have been applied:",
                color=discord.Color.green()
            )
            
            for setting_name, default_value in defaults.items():
                embed.add_field(
                    name=f"**{setting_name}**",
                    value=f"`{default_value}`",
                    inline=False
                )
            
            try:
                await interaction.response.send_message(embed=embed, ephemeral=True)
            except Exception as e:
                logger.error(f"Error in apply_defaults command: {e}")
                await interaction.response.send_message(
                    "❌ An error occurred while applying defaults.", ephemeral=True
                )

        @self.tree.command(description="Manage server settings")
        @discord.app_commands.describe(
            action="Action to perform (get/set/delete/list)",
            setting_name="Name of the setting (for set/delete)",
            value="Value to set (for set action)"
        )
        @discord.app_commands.choices(action=[
            discord.app_commands.Choice(name="get", value="get"),
            discord.app_commands.Choice(name="set", value="set"),
            discord.app_commands.Choice(name="delete", value="delete"),
            discord.app_commands.Choice(name="list", value="list")
        ])
        @discord.app_commands.default_permissions(manage_guild=True)
        @discord.app_commands.checks.has_permissions(manage_guild=True)
        async def server_settings(
            interaction: discord.Interaction,
            action: str,
            setting_name: str = None,
            value: str = None
        ):
            """Manage server settings"""
            if not interaction.guild:
                await interaction.response.send_message(
                    "❌ This command can only be used in a server.", ephemeral=True
                )
                return
            
            guild_id = interaction.guild.id
            
            if action == "list":
                settings = self.get_all_guild_settings(guild_id)
                if not settings:
                    await interaction.response.send_message(
                        "📋 No custom settings configured for this server.", ephemeral=True
                    )
                    return
                
                settings_text = "\n".join([f"**{k}**: {v}" for k, v in settings.items()])
                embed = discord.Embed(
                    title="⚙️ Server Settings",
                    description=settings_text,
                    color=discord.Color.blue()
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
            
            elif action == "get":
                if not setting_name:
                    await interaction.response.send_message(
                        "❌ Please provide a setting name.", ephemeral=True
                    )
                    return
                
                value = self.get_guild_setting(guild_id, setting_name)
                if value is None:
                    await interaction.response.send_message(
                        f"❌ Setting '{setting_name}' not found.", ephemeral=True
                    )
                    return
                
                await interaction.response.send_message(
                    f"⚙️ **{setting_name}**: `{value}`", ephemeral=True
                )
            
            elif action == "set":
                if not setting_name or value is None:
                    await interaction.response.send_message(
                        "❌ Please provide both setting name and value.", ephemeral=True
                    )
                    return
                
                # Try to parse as JSON for complex values
                try:
                    if value.startswith('{') or value.startswith('['):
                        import json
                        parsed_value = json.loads(value)
                    else:
                        # Try to parse as int, float, bool, or keep as string
                        if value.lower() in ('true', 'false'):
                            parsed_value = value.lower() == 'true'
                        elif value.isdigit():
                            parsed_value = int(value)
                        elif value.replace('.', '').isdigit():
                            parsed_value = float(value)
                        else:
                            parsed_value = value
                except:
                    parsed_value = value
                
                self.set_guild_setting(guild_id, setting_name, parsed_value)
                await interaction.response.send_message(
                    f"✅ Set **{setting_name}** to `{parsed_value}`", ephemeral=True
                )
            
            elif action == "delete":
                if not setting_name:
                    await interaction.response.send_message(
                        "❌ Please provide a setting name.", ephemeral=True
                    )
                    return
                
                if self.delete_guild_setting(guild_id, setting_name):
                    await interaction.response.send_message(
                        f"✅ Deleted setting **{setting_name}**", ephemeral=True
                    )
                else:
                    await interaction.response.send_message(
                        f"❌ Setting '{setting_name}' not found.", ephemeral=True
                    )

        @self.tree.command(description="Toggle Censor Cover usage in this server")
        @discord.app_commands.describe(enabled="Enable or disable Censor Cover for this server")
        @discord.app_commands.default_permissions(manage_guild=True)
        @discord.app_commands.checks.has_permissions(manage_guild=True)
        async def censor_toggle(interaction: discord.Interaction, enabled: bool):
            """Toggle Censor Cover usage in this server"""
            if not interaction.guild:
                await interaction.response.send_message(
                    "❌ This command can only be used in a server.", ephemeral=True
                )
                return

            self._censor_settings[str(interaction.guild.id)] = enabled
            self._save_persistent_data()

            status = "✅ Enabled" if enabled else "❌ Disabled"
            await interaction.response.send_message(
                f"{status} Censor Cover for **{interaction.guild.name}**",
                ephemeral=True,
            )

        @self.tree.command(
            description="Jail a user by removing all roles and giving jail role"
        )
        @discord.app_commands.describe(
            user="The user to jail",
            duration="Duration (e.g., 10m, 1h, 1d)",
            reason="Reason for jailing (optional)",
        )
        @discord.app_commands.default_permissions(manage_roles=True)
        @discord.app_commands.checks.has_permissions(manage_roles=True)
        async def jailrole(
            interaction: discord.Interaction,
            user: discord.Member,
            duration: str = None,
            reason: str = "No reason provided",
        ):
            """Jail a user by removing all roles and giving jail role"""
            if not interaction.guild:
                await interaction.response.send_message(
                    "❌ This command can only be used in a server.", ephemeral=True
                )
                return

            # Get punish role for this guild
            guild_id = str(interaction.guild.id)
            punish_role_id = None

            # Check guild settings first
            guild_settings = self._guild_settings.get(guild_id)
            if isinstance(guild_settings, dict):
                punish_role_id = guild_settings.get("punish_role_id")
            elif isinstance(guild_settings, (int, str)):
                punish_role_id = int(guild_settings)

            logger.info(
                f"Checking jail role for guild {guild_id}. Found in settings: {punish_role_id}. Settings keys: {list(self._guild_settings.keys())}"
            )

            # Fallback to environment variable if not in guild settings
            if not punish_role_id:
                punish_role_id_env = os.getenv("PUNISH_ROLE_ID")
                if punish_role_id_env:
                    try:
                        punish_role_id = int(punish_role_id_env)
                        logger.info(
                            f"Using environment variable PUNISH_ROLE_ID: {punish_role_id}"
                        )
                    except ValueError:
                        logger.warning(
                            f"Invalid PUNISH_ROLE_ID environment variable: {punish_role_id_env}"
                        )

            if not punish_role_id:
                await interaction.response.send_message(
                    "❌ Punish role not configured for this server. Please use `/set_punish_role @role`.",
                    ephemeral=True,
                )
                return

            # Parse duration if provided
            jail_until = None
            duration_text = ""
            if duration:
                jail_until = self._parse_duration_str(duration)
                if jail_until:
                    duration_text = f" for {duration}"
                else:
                    await interaction.response.send_message(
                        "❌ Invalid duration format. Use examples like: 10m, 1h, 1d",
                        ephemeral=True,
                    )
                    return

            # Check role hierarchy
            if (
                user.top_role >= interaction.user.top_role
                and interaction.user != interaction.guild.owner
            ):
                await interaction.response.send_message(
                    "❌ You cannot jail someone with equal or higher role than you.",
                    ephemeral=True,
                )
                return

            # Cannot jail the server owner
            if user == interaction.guild.owner:
                await interaction.response.send_message(
                    "❌ You cannot jail the server owner.", ephemeral=True
                )
                return

            try:
                jail_role = interaction.guild.get_role(punish_role_id)
                if not jail_role:
                    await interaction.response.send_message(
                        "❌ Jail role not found. Please check PUNISH_ROLE_ID.",
                        ephemeral=True,
                    )
                    return

                # Store current roles (excluding @everyone and jail role)
                current_roles = [
                    role
                    for role in user.roles
                    if role.id != user.guild.id and role.id != punish_role_id
                ]
                self._add_previous_roles(user.id, [role.id for role in current_roles])

                # Remove current roles and add jail role
                await user.remove_roles(
                    *current_roles, reason=f"Jailed by {interaction.user}: {reason}"
                )
                await user.add_roles(
                    jail_role, reason=f"Jailed by {interaction.user}: {reason}"
                )

                # Apply timeout if duration specified
                if jail_until:
                    await user.timeout(
                        jail_until,
                        reason=f"Jailed for {duration} by {interaction.user}",
                    )

                await interaction.response.send_message(
                    f"✅ **{user.display_name}** has been jailed{duration_text}.\n"
                    f"Reason: {reason}\n"
                    f"Previous roles stored and jail role applied.",
                    ephemeral=True,
                )
                logger.info(
                    f"User {user} jailed by {interaction.user} for {duration or 'indefinite'} - stored {len(current_roles)} previous roles"
                )

            except discord.Forbidden:
                await interaction.response.send_message(
                    "❌ I don't have permission to manage roles for this user.",
                    ephemeral=True,
                )
            except Exception as e:
                logger.error(f"Error in jailrole command: {e}")
                await interaction.response.send_message(
                    "❌ Failed to jail user. Check logs for details.", ephemeral=True
                )

        @self.tree.command(
            description="Unjail a user by removing jail role and restoring previous roles"
        )
        @discord.app_commands.describe(user="The user to unjail")
        @discord.app_commands.default_permissions(manage_roles=True)
        @discord.app_commands.checks.has_permissions(manage_roles=True)
        async def unjailrole(interaction: discord.Interaction, user: discord.Member):
            """Unjail a user by removing jail role and restoring previous roles"""
            if not interaction.guild:
                await interaction.response.send_message(
                    "❌ This command can only be used in a server.", ephemeral=True
                )
                return

            # Get punish role for this guild
            guild_id = str(interaction.guild.id)
            punish_role_id = None

            # Check guild settings first
            guild_settings = self._guild_settings.get(guild_id)
            if isinstance(guild_settings, dict):
                punish_role_id = guild_settings.get("punish_role_id")

            # Fallback to checking the keys directly if nested dict fails
            if not punish_role_id and guild_id in self._guild_settings:
                val = self._guild_settings[guild_id]
                if isinstance(val, int):
                    punish_role_id = val
                elif isinstance(val, dict):
                    punish_role_id = val.get("punish_role_id")

            # Fallback to environment variable if not in guild settings
            if not punish_role_id:
                punish_role_id_env = os.getenv("PUNISH_ROLE_ID")
                if punish_role_id_env:
                    try:
                        punish_role_id = int(punish_role_id_env)
                    except ValueError:
                        pass

            if not punish_role_id:
                await interaction.response.send_message(
                    "❌ Punish role not configured for this server. Please use `/set_punish_role @role`.",
                    ephemeral=True,
                )
                return

            # Check role hierarchy
            if (
                user.top_role >= interaction.user.top_role
                and interaction.user != interaction.guild.owner
            ):
                await interaction.response.send_message(
                    "❌ You cannot unjail someone with equal or higher role than you.",
                    ephemeral=True,
                )
                return

            try:
                jail_role = interaction.guild.get_role(punish_role_id)
                if not jail_role:
                    await interaction.response.send_message(
                        "❌ Jail role not found. Please check PUNISH_ROLE_ID.",
                        ephemeral=True,
                    )
                    return

                # Remove jail role
                await user.remove_roles(
                    jail_role, reason=f"Unjailed by {interaction.user}"
                )

                # Restore previous roles if they exist
                restored_roles = []
                if user.id in self._previous_roles:
                    for role_id in self._previous_roles[user.id]:
                        role = interaction.guild.get_role(role_id)
                        if role and role.id != user.guild.id:  # Skip @everyone
                            try:
                                await user.add_roles(
                                    role,
                                    reason=f"Restored previous role by {interaction.user}",
                                )
                                restored_roles.append(role.name)
                            except discord.Forbidden:
                                logger.warning(
                                    f"Could not restore role {role.name} to {user}"
                                )

                # Clean up stored roles
                self._remove_previous_roles(user.id)

                role_text = (
                    f" and restored roles: {', '.join(restored_roles)}"
                    if restored_roles
                    else ""
                )
                await interaction.response.send_message(
                    f"✅ **{user.display_name}** has been unjailed{role_text}.",
                    ephemeral=True,
                )
                logger.info(
                    f"User {user} unjailed by {interaction.user} - restored {len(restored_roles)} roles"
                )

            except discord.Forbidden:
                await interaction.response.send_message(
                    "❌ I don't have permission to manage roles for this user.",
                    ephemeral=True,
                )
            except Exception as e:
                logger.error(f"Error in unjailrole command: {e}")
                await interaction.response.send_message(
                    "❌ Failed to unjail user. Check logs for details.", ephemeral=True
                )
                # Clean up stored roles
                self._remove_previous_roles(user.id)

                role_text = (
                    f" and restored roles: {', '.join(restored_roles)}"
                    if restored_roles
                    else ""
                )
                await interaction.response.send_message(
                    f"✅ **{user.display_name}** has been unjailed{role_text}.",
                    ephemeral=True,
                )
                logger.info(
                    f"User {user} unjailed by {interaction.user} - restored {len(restored_roles)} roles"
                )

            except discord.Forbidden:
                await interaction.response.send_message(
                    "❌ I don't have permission to manage roles for this user.",
                    ephemeral=True,
                )
            except Exception as e:
                logger.error(f"Error in unjailrole command: {e}")
                await interaction.response.send_message(
                    "❌ Failed to unjail user. Check logs for details.", ephemeral=True
                )

        @self.tree.command(description="Set the punish role for this server")
        @discord.app_commands.describe(role="The role to use for jailing users")
        @discord.app_commands.default_permissions(manage_guild=True)
        @discord.app_commands.checks.has_permissions(manage_guild=True)
        async def set_punish_role(interaction: discord.Interaction, role: discord.Role):
            """Set the punish role for this server"""
            if not interaction.guild:
                await interaction.response.send_message(
                    "❌ This command can only be used in a server.", ephemeral=True
                )
                return

            # Store the punish role for this guild
            guild_id = str(interaction.guild.id)
            # Ensure it's a dictionary for this guild
            if guild_id not in self._guild_settings or not isinstance(
                self._guild_settings[guild_id], dict
            ):
                self._guild_settings[guild_id] = {}

            self._guild_settings[guild_id]["punish_role_id"] = role.id
            self._save_persistent_data()

            await interaction.response.send_message(
                f"✅ Punish role set to **{role.name}** for this server.",
                ephemeral=True,
            )
            logger.info(
                f"Punish role set to {role.id} ({role.name}) for guild {interaction.guild.id} by {interaction.user}"
            )

        @self.tree.command(
            description="Set autodelete limit for this channel (0 to disable)"
        )
        @discord.app_commands.describe(
            limit="Maximum number of messages to keep (0-1000, 0 = disable)"
        )
        @discord.app_commands.default_permissions(manage_messages=True)
        @discord.app_commands.checks.has_permissions(manage_messages=True)
        async def autodelete(
            interaction: discord.Interaction, limit: discord.app_commands.Range[int, 0, 1000]
        ):
            """Set autodelete limit for this channel. Immediately deletes excess messages. Use 0 to disable."""
            if not interaction.guild:
                await interaction.response.send_message(
                    "❌ This command can only be used in a server.", ephemeral=True
                )
                return

            channel_id = interaction.channel.id

            # Handle disable (limit = 0) - no need to defer for this quick operation
            if limit == 0:
                self._update_autodelete_setting(channel_id, False)
                await interaction.response.send_message(
                    "✅ Autodelete disabled for this channel.", ephemeral=True
                )
                logger.info(
                    f"Autodelete disabled for channel {channel_id} by {interaction.user}"
                )
                return

            # Defer response since message deletion can take time (> 3 seconds)
            await interaction.response.defer(ephemeral=True)

            # Update autodelete setting with custom limit
            self._update_autodelete_setting(channel_id, True, limit)

            # Immediately clean up excess messages
            try:
                deleted_count = await self._cleanup_channel_messages_immediate(
                    interaction.channel, limit
                )

                if deleted_count > 0:
                    await interaction.followup.send(
                        f"✅ Deleted **{deleted_count}** messages. Channel will keep maximum **{limit}** messages.",
                        ephemeral=True,
                    )
                else:
                    await interaction.followup.send(
                        f"✅ No messages to delete. Channel will keep maximum **{limit}** messages.",
                        ephemeral=True,
                    )
            except Exception as e:
                await interaction.followup.send(
                    f"⚠️ Autodelete enabled but cleanup failed: {str(e)}", ephemeral=True
                )
                return

            logger.info(
                f"Autodelete set to {limit} messages for channel {channel_id} by {interaction.user}"
            )

        @self.tree.command(
            description="Enable autodelete for this channel with default limit"
        )
        @discord.app_commands.default_permissions(manage_messages=True)
        @discord.app_commands.checks.has_permissions(manage_messages=True)
        async def enable_autodelete(interaction: discord.Interaction):
            """Enable autodelete for this channel with default limit"""
            if not interaction.guild:
                await interaction.response.send_message(
                    "❌ This command can only be used in a server.", ephemeral=True
                )
                return

            channel_id = interaction.channel.id
            
            # Enable autodelete with default limit
            self._update_autodelete_setting(channel_id, True)
            
            await interaction.response.send_message(
                f"✅ Autodelete enabled for this channel with limit of {self.auto_delete_count} messages.",
                ephemeral=True
            )
            logger.info(
                f"Autodelete enabled for channel {channel_id} by {interaction.user}"
            )

        @self.tree.command(
            description="Disable autodelete for this channel"
        )
        @discord.app_commands.default_permissions(manage_messages=True)
        @discord.app_commands.checks.has_permissions(manage_messages=True)
        async def disable_autodelete(interaction: discord.Interaction):
            """Disable autodelete for this channel"""
            if not interaction.guild:
                await interaction.response.send_message(
                    "❌ This command can only be used in a server.", ephemeral=True
                )
                return

            channel_id = interaction.channel.id
            
            # Disable autodelete
            self._update_autodelete_setting(channel_id, False)
            
            await interaction.response.send_message(
                "✅ Autodelete disabled for this channel.", ephemeral=True
            )
            logger.info(
                f"Autodelete disabled for channel {channel_id} by {interaction.user}"
            )

        @self.tree.command(description="Sync slash commands (admin only)")
        @discord.app_commands.default_permissions(administrator=True)
        @discord.app_commands.checks.has_permissions(administrator=True)
        async def sync(interaction: discord.Interaction):
            """Manually sync slash commands"""
            await interaction.response.defer(ephemeral=True)
            try:
                if hasattr(self.tree, "app_command_guild"):
                    guild = self.tree.app_command_guild
                    self.tree.copy_global_to(guild=guild)
                    synced = await self.tree.sync(guild=guild)
                    await interaction.followup.send(
                        f"✅ Synced {len(synced)} commands to this guild.",
                        ephemeral=True,
                    )
                else:
                    guild_synced_count = None
                    if interaction.guild:
                        self.tree.copy_global_to(guild=interaction.guild)
                        guild_synced = await self.tree.sync(guild=interaction.guild)
                        guild_synced_count = len(guild_synced)
                    global_synced = await self.tree.sync()
                    msg = (
                        f"✅ Synced {len(global_synced)} commands globally. "
                        f"Global propagation may take up to 1 hour."
                    )
                    if guild_synced_count is not None:
                        msg = (
                            f"✅ Synced {guild_synced_count} commands to this guild immediately "
                            f"and {len(global_synced)} globally."
                        )
                    await interaction.followup.send(msg, ephemeral=True)
                logger.info(f"Commands synced manually by {interaction.user}")
            except Exception as e:
                await interaction.followup.send(
                    f"❌ Failed to sync: {str(e)}", ephemeral=True
                )
                logger.error(f"Manual sync failed: {e}")

        # Error handler for application commands
        self.tree.on_error = self.on_app_command_error

        # Load persistent data
        self._load_persistent_data()

    def _save_persistent_data(self):
        """Save persistent data to JSON files"""
        try:
            # Save autodelete settings
            autodelete_data = {}
            for channel_id in self._autodelete_enabled:
                autodelete_data[str(channel_id)] = {
                    "enabled": self._autodelete_enabled[channel_id],
                    "limit": self._autodelete_limits.get(
                        channel_id, self.auto_delete_count
                    ),
                }
            with open(AUTODELETE_FILE, "w") as f:
                json.dump(autodelete_data, f, indent=2)

            # Save previous roles
            with open(PREVIOUS_ROLES_FILE, "w") as f:
                json.dump(self._previous_roles, f, indent=2)

            # Save resent pins
            with open(RESENT_PINS_FILE, "w") as f:
                # Convert sets to lists for JSON serialization
                serializable_pins = {
                    str(k): list(v) for k, v in self._resent_pins.items()
                }
                json.dump(serializable_pins, f, indent=2)

            # Save censor settings
            with open(CENSOR_SETTINGS_FILE, "w") as f:
                json.dump(self._censor_settings, f, indent=2)

            # Save pin settings
            with open(PIN_SETTINGS_FILE, "w") as f:
                json.dump(self._pin_settings, f, indent=2)

            # Save guild settings
            with open(GUILD_SETTINGS_FILE, "w") as f:
                json.dump(self._guild_settings, f, indent=2)

            # Save style reward model
            with open(STYLE_REWARD_FILE, "w") as f:
                json.dump(self._style_reward_model, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving persistent data: {e}")

    def _update_autodelete_setting(
        self, channel_id: int, enabled: bool, limit: int = None
    ):
        """Update autodelete setting and save to persistent storage"""
        self._autodelete_enabled[channel_id] = enabled
        if limit is not None:
            self._autodelete_limits[channel_id] = limit
        self._save_persistent_data()
        logger.info(
            f"Updated autodelete setting for channel {channel_id}: enabled={enabled}, limit={limit or self._autodelete_limits.get(channel_id, self.auto_delete_count)}"
        )

    def _add_previous_roles(self, user_id: int, role_ids: list):
        """Add previous roles for user and save to persistent storage"""
        self._previous_roles[user_id] = role_ids
        self._save_persistent_data()
        logger.info(f"Stored {len(role_ids)} previous roles for user {user_id}")

    def _remove_previous_roles(self, user_id: int):
        """Remove previous roles for user and save to persistent storage"""
        if user_id in self._previous_roles:
            del self._previous_roles[user_id]
            self._save_persistent_data()
            logger.info(f"Removed previous roles for user {user_id}")

    def _add_resent_pin(self, guild_id: int, message_id: int):
        """Add resent pin and save to persistent storage"""
        if guild_id not in self._resent_pins:
            self._resent_pins[guild_id] = set()

        self._resent_pins[guild_id].add(message_id)
        # Keep only last 100 entries per guild to prevent memory buildup
        if len(self._resent_pins[guild_id]) > 100:
            # Remove oldest 50 entries
            old_pins = list(self._resent_pins[guild_id])[:50]
            for pin_id in old_pins:
                self._resent_pins[guild_id].discard(pin_id)
        self._save_persistent_data()

    def _cleanup_old_data(self):
        """Clean up old data and save"""
        # Clean up old resent pins (keep last 100 total entries)
        total_pins = sum(len(pins) for pins in self._resent_pins.values())
        if total_pins > 100:
            # Remove oldest entries from each guild until we have under 100
            pins_to_remove = (
                total_pins - 50
            )  # Remove enough to get well under the limit
            removed_count = 0

            for guild_id, pin_set in list(self._resent_pins.items()):
                if removed_count >= pins_to_remove:
                    break

                # Convert to list to get oldest entries
                pin_list = list(pin_set)
                to_remove = min(len(pin_list), pins_to_remove - removed_count)

                for i in range(to_remove):
                    pin_set.discard(pin_list[i])
                    removed_count += 1

                # Remove empty guild entries
                if not pin_set:
                    del self._resent_pins[guild_id]

            self._save_persistent_data()
            logger.info(f"Cleaned up {removed_count} old resent pin entries")

    def _display_memory_status(self):
        """Display all remembered settings on startup"""
        print("\n" + "=" * 60)
        print("🧠 BOT MEMORY STATUS")
        print("=" * 60)

        # Auto-delete settings
        print(f"📋 Auto-Delete Settings:")
        if self._autodelete_enabled:
            enabled_channels = [
                ch_id for ch_id, enabled in self._autodelete_enabled.items() if enabled
            ]
            if enabled_channels:
                print(f"   Enabled in {len(enabled_channels)} channels:")
                for channel_id in enabled_channels:
                    channel = self.get_channel(channel_id)
                    channel_name = (
                        channel.name if channel else f"Unknown ({channel_id})"
                    )
                    print(f"   - #{channel_name} ({channel_id})")
                print(f"   Message Limit: {self.auto_delete_count} (deletes oldest)")
            else:
                print("   No channels have autodelete enabled")
        else:
            print("   No autodelete settings found")

        # Previous roles (punished users)
        print(f"\n👥 Punished Users ({len(self._previous_roles)}):")
        if self._previous_roles:
            for user_id, roles in self._previous_roles.items():
                print(f"   User {user_id}: {len(roles)} roles stored")
        else:
            print("   No punished users")

        # Resent pins
        print(f"\n📌 Resent Pins:")
        if self._resent_pins:
            total_pins = sum(len(pins) for pins in self._resent_pins.values())
            print(
                f"   Tracking {total_pins} resent pins across {len(self._resent_pins)} guilds"
            )
            for guild_id, pins in self._resent_pins.items():
                guild = self.get_guild(guild_id)
                guild_name = guild.name if guild else f"Unknown ({guild_id})"
                print(f"   - {guild_name}: {len(pins)} pins")
        else:
            print("   No resent pins tracked")

        # Bot configuration
        print(f"\n⚙️  Bot Configuration:")
        print(f"   Pin Resend Channel: {self.pin_resend_channel_id}")
        print(f"   Custom Prefix: '{self.custom_prefix}'")
        print(f"   Punish Role ID: {self.punish_role_id}")
        print(f"   Auto-Delete Limit: {self.auto_delete_count}")

        # Flexible autodelete settings
        print(f"\n🔧 Flexible Autodelete Settings:")
        print(f"   Global Enable: {'✅' if self.auto_delete_enabled_global else '❌'}")
        cooldown_minutes = self.auto_delete_cooldown / 60
        print(
            f"   Cooldown: {self.auto_delete_cooldown}s ({cooldown_minutes:.1f} minutes)"
        )
        print(
            f"   Rate Limit: {self.auto_delete_rate_start}s → {self.auto_delete_rate_max}s"
        )
        print(f"   Bulk Delete: {'✅' if self.auto_delete_bulk_delete else '❌'}")
        print(f"   Exclude Pinned: {'✅' if self.auto_delete_exclude_pinned else '❌'}")
        print(f"   Exclude Bots: {'✅' if self.auto_delete_exclude_bots else '❌'}")
        print(
            f"   Age Limit: {self.auto_delete_delete_age_hours}h ({'any age' if self.auto_delete_delete_age_hours == 0 else f'older than {self.auto_delete_delete_age_hours}h'})"
        )

        # Message filtering
        print(f"\n📝 Message Filtering:")
        print(f"   Filter Enabled: {'✅' if self.filter_enabled else '❌'}")
        if self.filter_enabled:
            print(f"   Filter Words: {len(self.filter_words)} configured")
            print(
                f"   Delete Filtered: {'✅' if self.filter_delete_instead else '❌ (log only)'}"
            )

        # Censor Cover
        print(f"\n🤐 Censor Cover:")
        if self.censor_cover_words:
            print(f"   Words: {len(self.censor_cover_words)} configured in .env")
            enabled_count = sum(1 for v in self._censor_settings.values() if v)
            print(f"   Enabled in {enabled_count} guilds")
        else:
            print(f"   Not configured (set CENSOR_COVER_WORDS in .env)")

        # Pin Redirects
        print(f"\n📌 Pin Redirects:")
        print(f"   Global Fallback: {self.pin_resend_channel_id}")
        if self._pin_settings:
            print(f"   Redirects configured for {len(self._pin_settings)} guilds")
        else:
            print(f"   No per-server redirects configured")

        print("=" * 60)
        print("✅ All settings loaded successfully!")
        print("=" * 60 + "\n")

    async def on_ready(self):
        """Called when bot is ready"""
        logger.info(f"Bot logged in as {self.user} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guilds")
        self._ready_event.set()

        # Apply default settings to new guilds
        for guild in self.guilds:
            if len(self.get_all_guild_settings(guild.id)) == 0:
                self.apply_default_settings(guild.id)

        # Display all remembered settings
        self._display_memory_status()

        # Perform initial cleanup for all guilds (in background to not block startup)
        if self._startup_task and not self._startup_task.done():
            self._startup_task.cancel()
        self._startup_task = asyncio.create_task(self._perform_startup_cleanup())

        # Start periodic cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

        # Sync slash commands
        # Sync slash commands
        try:
            # CLEANUP: Clear commands from the old debug guild to fix duplicates
            # (This ID was found in your .env previously)
            old_debug_guild_id = 1443930906325684336
            try:
                cleanup_guild = discord.Object(id=old_debug_guild_id)
                self.tree.clear_commands(guild=cleanup_guild)
                await self.tree.sync(guild=cleanup_guild)
                logger.info(
                    f"Cleared old guild-specific commands from {old_debug_guild_id} to fix duplicates"
                )
            except Exception as e:
                logger.warning(
                    f"Could not clear old guild commands (might not be in that guild): {e}"
                )

            if hasattr(self.tree, "app_command_guild"):
                guild = self.tree.app_command_guild
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                logger.info(
                    f"Synced {len(synced)} slash commands to guild {guild.id} (Debug Mode)"
                )
            else:
                global_synced = await self.tree.sync()
                logger.info(f"Synced {len(global_synced)} slash commands globally")
                # Also sync each connected guild for immediate availability.
                for guild in self.guilds:
                    try:
                        self.tree.copy_global_to(guild=guild)
                        guild_synced = await self.tree.sync(guild=guild)
                        logger.info(
                            f"Synced {len(guild_synced)} slash commands to guild {guild.id} for immediate visibility"
                        )
                    except Exception as guild_sync_error:
                        logger.warning(
                            f"Failed guild-specific sync for {guild.id}: {guild_sync_error}"
                        )
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")

    async def _periodic_cleanup(self):
        """Periodic cleanup task to maintain data and save"""
        while not self.is_closed():
            try:
                # Clean up old data every hour
                self._cleanup_old_data()
                await asyncio.sleep(3600)  # 1 hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(300)  # 5 minutes on error

    async def _perform_startup_cleanup(self):
        """Perform cleanup on startup for all channels where autodelete is enabled"""
        for guild in self.guilds:
            for channel in guild.text_channels:
                if self._autodelete_enabled.get(channel.id, False):
                    logger.info(
                        f"Performing startup cleanup for channel {channel.id} ({channel.name})"
                    )
                    try:
                        await self._cleanup_channel_messages(channel)
                    except Exception as e:
                        logger.error(
                            f"Startup cleanup failed for channel {channel.id}: {e}"
                        )

                await asyncio.sleep(1)  # Small delay to avoid rate limits

    async def _check_and_censor_message(self, message: discord.Message) -> bool:
        """Check message content for censor words and cover if necessary. Returns True if censored."""
        if (
            not self.censor_cover_words
            or message.author.bot
            or not isinstance(message.channel, discord.TextChannel)
        ):
            return False

        # Check if enabled in this guild
        if not self._censor_settings.get(str(message.guild.id), False):
            return False

        content_lower = message.content.lower()
        if any(word.lower() in content_lower for word in self.censor_cover_words):
            # Check permissions
            perms = message.channel.permissions_for(message.guild.me)
            if perms.manage_messages and perms.manage_webhooks:
                try:
                    # Prepare content and files
                    content = message.content or ""
                    files = []
                    for attachment in message.attachments:
                        if attachment.size <= DISCORD_FILE_SIZE_LIMIT:
                            try:
                                file_data = await attachment.read()
                                files.append(
                                    discord.File(
                                        io.BytesIO(file_data),
                                        filename=attachment.filename,
                                    )
                                )
                            except Exception as e:
                                logger.error(f"Error reading attachment: {e}")
                                pass
                        else:
                            # Large file - upload to file.io
                            link = await self._upload_to_file_io(attachment)
                            if link:
                                content += f"\n📎 **{attachment.filename}** (Large file): {link}"
                            else:
                                content += f"\n❌ **{attachment.filename}** (Too large to send and upload failed)"

                    # Get webhook
                    webhook = await self._get_or_create_webhook(message.channel)

                    # Send via webhook
                    await webhook.send(
                        content=content,
                        username=message.author.display_name,
                        avatar_url=message.author.display_avatar.url
                        if message.author.display_avatar
                        else None,
                        files=files,
                    )

                    # Delete original message
                    await message.delete()

                    logger.info(
                        f"Censor-covered message from {message.author} in {message.channel.name}"
                    )
                    return True
                except Exception as e:
                    logger.error(f"Failed to censor-cover message: {e}")
        return False

    async def on_message(self, message: discord.Message):
        """Handle message events for channel management and prefix commands"""
        # Ignore DMs
        if not isinstance(message.channel, discord.TextChannel):
            return

        # Never process bot-authored messages to avoid bot loops/doubles.
        if message.author.bot:
            return

        # Ignore messages in the pin resend channel
        if message.channel.id == self.pin_resend_channel_id:
            return

        # Handle active /humanize interaction sessions.
        if await self._handle_humanize_session_message(message):
            return

        # Handle prefix commands
        has_prefix = any(message.content.startswith(prefix) for prefix in self.custom_prefix)
        if has_prefix:
            await self._handle_prefix_command(message)
            return

        # Passive per-guild style learning from real user messages.
        await self._maybe_auto_train_from_message(message)

        # Censor Cover Logic (Auto-replace with Webhook)
        if await self._check_and_censor_message(message):
            return  # Stop processing (prevents double filtering)

        # Message filtering
        if self.filter_enabled and not message.author.bot:
            message_lower = message.content.lower()
            for word in self.filter_words:
                if word.strip().lower() in message_lower:
                    logger.warning(
                        f"Filtered message from {message.author}: {message.content[:50]}..."
                    )
                    if self.filter_delete_instead:
                        try:
                            await message.delete()
                            logger.info(
                                f"Deleted filtered message from {message.author}"
                            )
                        except discord.Forbidden:
                            logger.warning("No permission to delete filtered message")
                    return  # Stop processing further

        # Check if message is pinned (handles cases where message was already pinned before bot started)
        if message.pinned:
            await self._resend_pinned_message(message)
            return

        # Clean up old messages if autodelete is enabled for this channel
        is_enabled = self._autodelete_enabled.get(message.channel.id, False)
        global_enabled = self.auto_delete_enabled_global
        logger.info(
            f"Autodelete check: channel={message.channel.id}, enabled={is_enabled}, global={global_enabled}"
        )

        if is_enabled and global_enabled:
            channel_id = message.channel.id
            logger.info(f"Running autodelete cleanup for channel {channel_id}")

            # Always run cleanup - no cooldown
            try:
                await self._cleanup_channel_messages(message.channel)
            except Exception as e:
                logger.error(f"Autodelete cleanup failed for channel {channel_id}: {e}")
                import traceback

                logger.error(traceback.format_exc())

    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent):
        """Handle message edits (including pin changes)"""
        try:
            # Get the updated message
            channel = self.get_channel(payload.channel_id)
            if not channel:
                return

            try:
                message = await channel.fetch_message(payload.message_id)
            except discord.NotFound:
                # Message was deleted, ignore
                return

            # Check if message was pinned and hasn't been resent yet
            if message.pinned:
                guild_id = message.guild.id if message.guild else None
                if guild_id and (
                    guild_id not in self._resent_pins
                    or message.id not in self._resent_pins[guild_id]
                ):
                    await self._resend_pinned_message(message)

            # Censor check for edited message
            try:
                await self._check_and_censor_message(message)
            except Exception as e:
                logger.error(f"Error checking edited message: {e}")

        except Exception as e:
            logger.error(f"Error handling message edit: {e}")

    async def on_guild_channel_pins_update(
        self, channel: discord.abc.GuildChannel, last_pin: Optional[datetime]
    ):
        """Handle pin updates in channels"""
        try:
            # Get recent messages to find newly pinned ones
            async for message in channel.history(limit=50):
                if message.pinned:
                    guild_id = message.guild.id
                    if (
                        guild_id not in self._resent_pins
                        or message.id not in self._resent_pins[guild_id]
                    ):
                        await self._resend_pinned_message(message)
                        break  # Only handle the most recent newly pinned message

        except Exception as e:
            logger.error(f"Error handling pins update in {channel.id}: {e}")

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        """Handle reaction additions (including pin reactions)"""
        try:
            # Check if this is a pin reaction (📌)
            if payload.emoji.name == "📌":
                channel = self.get_channel(payload.channel_id)
                if not channel:
                    return

                try:
                    message = await channel.fetch_message(payload.message_id)
                except discord.NotFound:
                    # Message was deleted, ignore
                    return

                # Check if message is now pinned (after reaction)
                if message.pinned:
                    guild_id = message.guild.id
                    if (
                        guild_id not in self._resent_pins
                        or message.id not in self._resent_pins[guild_id]
                    ):
                        await self._resend_pinned_message(message)

        except Exception as e:
            logger.error(f"Error handling reaction add: {e}")

    def _parse_duration_str(self, duration_str: str) -> Optional[timedelta]:
        """Parse duration string (e.g., '1h', '30m', '1d') into timedelta"""
        duration_str = duration_str.lower()
        try:
            if duration_str.endswith("s"):
                return timedelta(seconds=int(duration_str[:-1]))
            elif duration_str.endswith("m"):
                return timedelta(minutes=int(duration_str[:-1]))
            elif duration_str.endswith("h"):
                return timedelta(hours=int(duration_str[:-1]))
            elif duration_str.endswith("d"):
                return timedelta(days=int(duration_str[:-1]))
            else:
                return timedelta(minutes=int(duration_str))
        except ValueError:
            return None

    async def _handle_prefix_command(self, message: discord.Message):
        """Handle prefix commands"""
        # Find which prefix was used
        used_prefix = None
        for prefix in self.custom_prefix:
            if message.content.startswith(prefix):
                used_prefix = prefix
                break
        
        if not used_prefix:
            return
            
        content = message.content[len(used_prefix) :].strip()
        cmd_parts = content.split(" ", 1)
        command = cmd_parts[0].lower()
        args = cmd_parts[1] if len(cmd_parts) > 1 else ""

        if command == "lq":
            await self._handle_lq_command(message, used_prefix=used_prefix)
        elif command == "ulq":
            await self._handle_ulq_command(message)
        elif command in ["censor_toggle", "ct"]:
            # Manual handling for Censor Toggle to match previous logic logic
            # OR better, make a _handle_censor_toggle_command
            await self._handle_censor_toggle_command(message, args)
        elif command == "set_punish_role":
            await self._handle_set_punish_role_command(message)
        elif command == "msg":
            await self._handle_msg_command(message, args)
        elif command == "ban":
            await self._handle_ban_command(message, args)
        elif command == "kick":
            await self._handle_kick_command(message, args)
        elif command == "timeout":
            await self._handle_timeout_command(message, args)
        elif command == "timeout_role":
            await self._handle_timeout_role_command(message, args)
        elif command == "redirect_pins":
            await self._handle_redirect_pins_command(message, args)
        elif command == "prefix":
            await self._handle_prefix_command_info(message)
        elif command == "aiscore":
            await self._handle_aiscore_command(message, args)
        elif command == "humanize":
            await self._handle_humanize_command(message)
        elif command in ["autotrain", "auto-train"]:
            await self._handle_autotrain_command(message, args)
        elif command == "help":
            await self._handle_help_command(message)

    async def _handle_help_command(self, message: discord.Message):
        """Show help embed"""
        prefix = self._primary_prefix()
        embed = discord.Embed(
            title="🤖 Bot Help",
            description="Here are the available commands:",
            color=discord.Color.blue(),
        )
        # Public Commands
        embed.add_field(
            name=f"{prefix}msg <text>",
            value="Send a message as the bot (or invisible webhook)",
            inline=False,
        )
        embed.add_field(
            name="/help", value="Show this help message (slash command)", inline=False
        )
        # Moderation Commands
        embed.add_field(
            name="🛡️ Moderation",
            value=f"**{prefix}ban @user [reason]** - Ban a user\n"
            f"**{prefix}kick @user [reason]** - Kick a user\n"
            f"**{prefix}timeout @user [duration] [reason]** - Timeout a user\n"
            f"**{prefix}timeout_role @role [duration] [reason]** - Timeout role",
            inline=False,
        )
        # Configuration Commands
        embed.add_field(
            name="⚙️ Configuration",
            value=f"**{prefix}censor_toggle [true/false]** - Enable/Disable censor cover\n"
            f"**{prefix}redirect_pins #channel** - Set where pins are resent\n"
            f"**{prefix}set_punish_role @role** - Set jail role for this server\n"
            f"**{prefix}aiscore <text>** - Score text AI-likeness\n"
            f"**{prefix}humanize** - Generate 3 replies and collect rating\n"
            f"**{prefix}autotrain [on|off] [rating]** - Configure passive auto training\n"
            f"**{prefix}lq @user [duration]** - Jail a user (e.g., {prefix}lq @user 20m)\n"
            f"**{prefix}ulq @user** - Unjail a user (same as /unjailrole)\n"
            f"**{prefix}prefix** - Show current command prefixes",
            inline=False,
        )
        await message.reply(embed=embed)

    async def _handle_msg_command(self, message: discord.Message, args: str):
        """Handle ~msg command"""
        # Delete the command message first (if possible) for anonymity
        try:
            await message.delete()
        except:
            pass

        content = args or ""
        files = []
        for attachment in message.attachments:
            if attachment.size <= DISCORD_FILE_SIZE_LIMIT:
                try:
                    file_data = await attachment.read()
                    files.append(
                        discord.File(
                            io.BytesIO(file_data), filename=attachment.filename
                        )
                    )
                except Exception as e:
                    logger.error(f"Error reading attachment: {e}")
                    pass
            else:
                # Large file - upload
                link = await self._upload_large_file(attachment)
                if link:
                    content += f"\n📎 **{attachment.filename}** (Large file): {link}"
                else:
                    content += (
                        f"\n❌ **{attachment.filename}** (Too large and upload failed)"
                    )

        if not content and not files:
            return

        # Check permissions
        can_use_webhooks = False
        perms = message.channel.permissions_for(message.guild.me)
        if perms.manage_webhooks:
            can_use_webhooks = True

        if can_use_webhooks:
            try:
                webhook = await self._get_or_create_webhook(message.channel)
                await webhook.send(
                    content=content,
                    username=message.author.display_name,
                    avatar_url=message.author.display_avatar.url
                    if message.author.display_avatar
                    else None,
                    files=files,
                )
                return
            except Exception as e:
                logger.error(f"Webhook send failed in prefix msg: {e}")
                # Fallback

        # Fallback
        embed = discord.Embed(description=content, color=message.author.color)
        embed.set_author(
            name=message.author.display_name,
            icon_url=message.author.display_avatar.url
            if message.author.display_avatar
            else None,
        )
        await message.channel.send(embed=embed, files=files)

    async def _handle_ban_command(self, message: discord.Message, args: str):
        prefix = self._primary_prefix()
        if not message.author.guild_permissions.ban_members:
            await self._reply_scored(message, "❌ You need 'Ban Members' permission.")
            return
        if not message.mentions:
            await self._reply_scored(message, f"❌ Usage: `{prefix}ban @user [reason]`")
            return
        target = message.mentions[0]
        reason = args.replace(target.mention, "").strip() or "No reason provided"

        # Hierarchy check
        if (
            target.top_role >= message.author.top_role
            and message.author != message.guild.owner
        ):
            await self._reply_scored(message, "❌ Cannot ban equal/higher role.")
            return

        try:
            await target.ban(reason=f"Banned by {message.author}: {reason}")
            await self._reply_scored(
                message,
                f"✅ **{target.display_name}** has been banned.\nReason: {reason}"
            )
        except Exception as e:
            await self._reply_scored(message, f"❌ Failed to ban: {e}")

    async def _handle_kick_command(self, message: discord.Message, args: str):
        prefix = self._primary_prefix()
        if not message.author.guild_permissions.kick_members:
            await self._reply_scored(message, "❌ You need 'Kick Members' permission.")
            return
        if not message.mentions:
            await self._reply_scored(message, f"❌ Usage: `{prefix}kick @user [reason]`")
            return
        target = message.mentions[0]
        reason = args.replace(target.mention, "").strip() or "No reason provided"

        if (
            target.top_role >= message.author.top_role
            and message.author != message.guild.owner
        ):
            await self._reply_scored(message, "❌ Cannot kick equal/higher role.")
            return

        try:
            await target.kick(reason=f"Kicked by {message.author}: {reason}")
            await self._reply_scored(
                message,
                f"✅ **{target.display_name}** has been kicked.\nReason: {reason}"
            )
        except Exception as e:
            await self._reply_scored(message, f"❌ Failed to kick: {e}")

    async def _handle_timeout_command(self, message: discord.Message, args: str):
        prefix = self._primary_prefix()
        if not message.author.guild_permissions.moderate_members:
            await self._reply_scored(message, "❌ You need 'Timeout Members' permission.")
            return
        parts = args.split()
        if not message.mentions or len(parts) < 2:
            await self._reply_scored(
                message,
                f"❌ Usage: `{prefix}timeout @user <duration> [reason]`"
            )
            return
        target = message.mentions[0]

        # Find duration string (it's likely the part that isn't the mention)
        # Simple assumption: 2nd arg if mention is first
        duration_str = parts[1] if parts[0].startswith("<@") else parts[0]
        if duration_str.startswith("<@"):
            duration_str = parts[1]  # Trying to find non-mention

        delta = self._parse_duration_str(duration_str)
        if not delta:
            await self._reply_scored(message, "❌ Invalid duration. Use 10m, 1h, etc.")
            return

        reason = " ".join(parts[2:]) or "No reason provided"  # simplistic

        if (
            target.top_role >= message.author.top_role
            and message.author != message.guild.owner
        ):
            await self._reply_scored(message, "❌ Cannot timeout equal/higher role.")
            return

        try:
            await target.timeout(
                delta,
                reason=f"Timeout by {message.author}: {reason}",
            )
            await self._reply_scored(
                message,
                f"✅ **{target.display_name}** timed out for {duration_str}."
            )
        except Exception as e:
            await self._reply_scored(message, f"❌ Failed to timeout: {e}")

    async def _handle_timeout_role_command(self, message: discord.Message, args: str):
        prefix = self._primary_prefix()
        if not message.author.guild_permissions.moderate_members:
            await self._reply_scored(message, "❌ You need 'Timeout Members' permission.")
            return
        if not message.role_mentions:
            await self._reply_scored(
                message,
                f"❌ Usage: `{prefix}timeout_role @role <duration> [reason]`"
            )
            return
        role = message.role_mentions[0]
        parts = args.split()

        # Determine duration (similar simplistic logic)
        duration_str = None
        for part in parts:
            if not part.startswith("<@") and self._parse_duration_str(part):
                duration_str = part
                break

        if not duration_str:
            await self._reply_scored(message, "❌ Invalid duration.")
            return

        delta = self._parse_duration_str(duration_str)
        reason = (
            args.replace(role.mention, "").replace(duration_str, "").strip()
            or "Mass timeout"
        )

        if role >= message.author.top_role and message.author != message.guild.owner:
            await self._reply_scored(message, "❌ Cannot target equal/higher role.")
            return

        msg = await self._reply_scored(message, "⏳ Applying mass timeout...")
        count, errors = 0, 0
        for member in role.members:
            if member.bot:
                continue
            # Skip if member is higher/equal to moderator
            if (
                member.top_role >= message.author.top_role
                and message.author != message.guild.owner
            ):
                continue

            try:
                await member.timeout(
                    delta,
                    reason=f"Mass timeout by {message.author}: {reason}",
                )
                count += 1
            except Exception as e:
                logger.error(f"Error timing out member {member}: {e}")
                errors += 1

        await msg.edit(
            content=f"✅ Mass Timeout: {count} members timed out. {errors} failed."
        )

    async def _handle_redirect_pins_command(self, message: discord.Message, args: str):
        prefix = self._primary_prefix()
        if not message.author.guild_permissions.manage_guild:
            await self._reply_scored(message, "❌ You need 'Manage Server' permission.")
            return
        if not message.channel_mentions:
            await self._reply_scored(
                message,
                f"❌ Usage: `{prefix}redirect_pins #channel`"
            )
            return
        target = message.channel_mentions[0]
        self._pin_settings[str(message.guild.id)] = target.id
        self._save_persistent_data()
        await self._reply_scored(message, f"✅ Pinned messages redirected to {target.mention}")

    async def _handle_set_punish_role_command(self, message: discord.Message):
        """Handle prefix set_punish_role command"""
        if not message.guild:
            return

        if not message.author.guild_permissions.manage_roles:
            await self._reply_scored(
                message,
                "❌ You need 'Manage Roles' permission to use this command."
            )
            return

        # Parse role mention
        if len(message.role_mentions) == 0:
            await self._reply_scored(
                message,
                f"❌ Please mention a role. Usage: `{self._primary_prefix()}set_punish_role @role`"
            )
            return

        target_role = message.role_mentions[0]

        # Check role hierarchy
        if (
            target_role >= message.author.top_role
            and message.author != message.guild.owner
        ):
            await self._reply_scored(
                message,
                "❌ You cannot set a punish role equal to or higher than your highest role."
            )
            return

        try:
            # Store the punish role for this guild
            guild_id = str(message.guild.id)
            # Ensure it's a dictionary for this guild
            if guild_id not in self._guild_settings or not isinstance(
                self._guild_settings[guild_id], dict
            ):
                self._guild_settings[guild_id] = {}

            self._guild_settings[guild_id]["punish_role_id"] = target_role.id
            self._save_persistent_data()

            await self._reply_scored(
                message,
                f"✅ Punish role set to **{target_role.name}** for this server."
            )
            logger.info(
                f"Punish role set to {target_role.id} ({target_role.name}) for guild {message.guild.id} by {message.author}"
            )

        except discord.Forbidden:
            await self._reply_scored(
                message,
                "❌ I don't have permission to manage roles for this user."
            )
        except Exception as e:
            logger.error(f"Error in set_punish_role command: {e}")
            await self._reply_scored(message, "❌ Failed to set punish role. Check logs for details.")

    async def _handle_censor_toggle_command(self, message: discord.Message, args: str):
        if not message.author.guild_permissions.manage_guild:
            await self._reply_scored(message, "❌ You need 'Manage Server' permissions.")
            return

        if not args:
            current = self._censor_settings.get(str(message.guild.id), False)
            status = "Enabled" if current else "Disabled"
            await self._reply_scored(message, f"ℹ️ Censor Cover is currently **{status}**.")
            return

        enabled = args.lower() in ["true", "on", "enable", "yes", "1"]
        self._censor_settings[str(message.guild.id)] = enabled
        self._save_persistent_data()
        status_text = "✅ Enabled" if enabled else "❌ Disabled"
        await self._reply_scored(message, f"{status_text} Censor Cover for **{message.guild.name}**")

    async def _handle_lq_command(
        self, message: discord.Message, used_prefix: Optional[str] = None
    ):
        """Handle lq command - add punish role and store previous roles"""
        if not message.guild:
            return

        if not message.author.guild_permissions.manage_roles:
            await self._reply_scored(
                message,
                "❌ You need 'Manage Roles' permission to use this command."
            )
            return

        # Parse command: <prefix>lq @user [duration]
        prefix = used_prefix or next(
            (p for p in self.custom_prefix if message.content.startswith(p)),
            self._primary_prefix(),
        )
        content = message.content[len(prefix) :].strip()
        parts = content.split()

        if len(parts) < 2 or not message.mentions:
            await self._reply_scored(
                message,
                f"❌ Usage: `{prefix}lq @user [duration]` (e.g., 10m, 1h, 1d)"
            )
            return

        target_user = message.mentions[0]

        # Get punish role for this guild
        guild_id = str(message.guild.id)
        punish_role_id = None

        # Check guild settings first - handles both nested dict and direct mapping
        guild_settings = self._guild_settings.get(guild_id)
        if isinstance(guild_settings, dict):
            punish_role_id = guild_settings.get("punish_role_id")
        elif isinstance(guild_settings, (int, str)):
            punish_role_id = int(guild_settings)

        logger.info(
            f"Checking punish role for guild {guild_id}. Found in settings: {punish_role_id}. Settings keys: {list(self._guild_settings.keys())}"
        )

        # Fallback to environment variable if not in guild settings
        if not punish_role_id:
            punish_role_id_env = os.getenv("PUNISH_ROLE_ID")
            if punish_role_id_env:
                try:
                    punish_role_id = int(punish_role_id_env)
                    logger.info(
                        f"Using environment variable PUNISH_ROLE_ID: {punish_role_id}"
                    )
                except ValueError:
                    logger.warning(
                        f"Invalid PUNISH_ROLE_ID environment variable: {punish_role_id_env}"
                    )

        if not punish_role_id:
            await self._reply_scored(
                message,
                "❌ Punish role not configured for this server. Please ask an admin to use `/set_punish_role @role`."
            )
            return

        # Parse duration if provided (after the mention)
        duration = None
        duration_text = ""
        if len(parts) >= 3:
            duration_str = parts[2]  # Duration comes after @user
            duration = self._parse_duration_str(duration_str)
            if duration:
                duration_text = f" for {duration_str}"
            else:
                await self._reply_scored(
                    message,
                    "❌ Invalid duration format. Use examples like: 10m, 1h, 1d"
                )
                return

        # Check role hierarchy
        if (
            target_user.top_role >= message.author.top_role
            and message.author != message.guild.owner
        ):
            await self._reply_scored(
                message,
                "❌ You cannot punish someone with equal or higher role than you."
            )
            return

        try:
            punish_role = message.guild.get_role(punish_role_id)
            if not punish_role:
                await self._reply_scored(
                    message,
                    "❌ Punish role not found. Please check PUNISH_ROLE_ID."
                )
                return

            # Store current roles (excluding @everyone and punish role)
            current_roles = [
                role
                for role in target_user.roles
                if role.id != target_user.guild.id and role.id != punish_role_id
            ]
            self._add_previous_roles(
                target_user.id, [role.id for role in current_roles]
            )

            # Remove current roles and add punish role
            await target_user.remove_roles(
                *current_roles, reason=f"Punished by {message.author}"
            )
            await target_user.add_roles(
                punish_role, reason=f"Punished by {message.author}"
            )

            # Apply timeout if duration specified
            if duration:
                await target_user.timeout(
                    duration, reason=f"Punished for {duration_str} by {message.author}"
                )

            await self._reply_scored(
                message,
                f"✅ **{target_user.display_name}** has been punished{duration_text}.\n"
                f"Previous roles stored and punish role applied."
            )
            logger.info(
                f"User {target_user} punished by {message.author} for {duration_str or 'indefinite'} - stored {len(current_roles)} previous roles"
            )

        except discord.Forbidden:
            await self._reply_scored(
                message,
                "❌ I don't have permission to manage roles for this user."
            )
        except Exception as e:
            logger.error(f"Error in lq command: {e}")
            await self._reply_scored(message, "❌ Failed to punish user. Check logs for details.")

    async def _handle_ulq_command(self, message: discord.Message):
        """Handle ulq command - remove punish role and restore previous roles"""
        if not message.guild:
            return

        if not message.author.guild_permissions.manage_roles:
            await self._reply_scored(
                message,
                "❌ You need 'Manage Roles' permission to use this command."
            )
            return

        # Parse user mention
        if len(message.mentions) == 0:
            await self._reply_scored(
                message,
                f"❌ Please mention a user. Usage: `{self._primary_prefix()}ulq @user`"
            )
            return

        target_user = message.mentions[0]

        # Get punish role for this guild
        guild_id = str(message.guild.id)
        punish_role_id = None

        # Check guild settings first - handles both nested dict and direct mapping
        guild_settings = self._guild_settings.get(guild_id)
        if isinstance(guild_settings, dict):
            punish_role_id = guild_settings.get("punish_role_id")
        elif isinstance(guild_settings, (int, str)):
            punish_role_id = int(guild_settings)

        logger.info(
            f"Checking punish role for guild {guild_id}. Found in settings: {punish_role_id}. Settings keys: {list(self._guild_settings.keys())}"
        )

        # Fallback to environment variable if not in guild settings
        if not punish_role_id:
            punish_role_id_env = os.getenv("PUNISH_ROLE_ID")
            if punish_role_id_env:
                try:
                    punish_role_id = int(punish_role_id_env)
                    logger.info(
                        f"Using environment variable PUNISH_ROLE_ID: {punish_role_id}"
                    )
                except ValueError:
                    logger.warning(
                        f"Invalid PUNISH_ROLE_ID environment variable: {punish_role_id_env}"
                    )

        if not punish_role_id:
            await self._reply_scored(
                message,
                "❌ Punish role not configured for this server. Please ask an admin to use `/set_punish_role @role`."
            )
            return

        # Check role hierarchy
        if (
            target_user.top_role >= message.author.top_role
            and message.author != message.guild.owner
        ):
            await self._reply_scored(
                message,
                "❌ You cannot unpunish someone with equal or higher role than you."
            )
            return

        try:
            punish_role = message.guild.get_role(punish_role_id)
            if not punish_role:
                await self._reply_scored(
                    message,
                    "❌ Punish role not found. Please check PUNISH_ROLE_ID."
                )
                return

            # Remove punish role
            await target_user.remove_roles(
                punish_role, reason=f"Unpunished by {message.author}"
            )

            # Restore previous roles if they exist
            restored_roles = []
            if target_user.id in self._previous_roles:
                for role_id in self._previous_roles[target_user.id]:
                    role = message.guild.get_role(role_id)
                    if role and role.id != target_user.guild.id:  # Skip @everyone
                        try:
                            await target_user.add_roles(
                                role,
                                reason=f"Restored previous role by {message.author}",
                            )
                            restored_roles.append(role.name)
                        except discord.Forbidden:
                            logger.warning(
                                f"Could not restore role {role.name} to {target_user}"
                            )

                # Clean up stored roles
                self._remove_previous_roles(target_user.id)

            role_text = (
                f" and restored roles: {', '.join(restored_roles)}"
                if restored_roles
                else ""
            )
            await self._reply_scored(
                message,
                f"✅ **{target_user.display_name}** has been unpunished{role_text}."
            )
            logger.info(
                f"User {target_user} unpunished by {message.author} - restored {len(restored_roles)} roles"
            )

        except discord.Forbidden:
            await self._reply_scored(
                message,
                "❌ I don't have permission to manage roles for this user."
            )
        except Exception as e:
            logger.error(f"Error in ulq command: {e}")
            await self._reply_scored(message, "❌ Failed to unpunish user. Check logs for details.")

    async def _handle_prefix_command_info(self, message: discord.Message):
        """Handle prefix command - show current command prefixes"""
        prefix_text = self._prefix_list_display()
        
        embed = discord.Embed(
            title="🔧 Command Prefixes",
            description=f"Current command prefixes: {prefix_text}",
            color=discord.Color.blue()
        )
        embed.add_field(
            name="Usage",
            value=f"Use any of these prefixes followed by a command:\n"
                  f"• {prefix_text}help - Show this help\n"
                  f"• {prefix_text}lq @user [duration] - Jail a user\n"
                  f"• {prefix_text}ulq @user - Unjail a user\n"
                  f"• {prefix_text}msg <text> - Send a message via webhook\n"
                  f"• {prefix_text}aiscore <text> - Score AI-likeness\n"
                  f"• {prefix_text}humanize - 3 options + feedback\n"
                  f"• {prefix_text}autotrain [on|off] [rating] - Configure auto training",
            inline=False
        )
        await message.reply(embed=embed)

    async def _handle_aiscore_command(self, message: discord.Message, args: str):
        text = args.strip()
        if not text:
            await self._reply_scored(
                message,
                f"Usage: `{self._primary_prefix()}aiscore <text>`",
            )
            return

        score = self._score_ai_text(text)
        await message.reply(
            f"`is this ai: {score:.1f}/10`\n{text}"
        )

    async def _handle_humanize_command(self, message: discord.Message):
        if not message.guild or not isinstance(message.channel, discord.TextChannel):
            await self._reply_scored(
                message, "❌ This command can only be used in a server text channel."
            )
            return

        statement = await self._pick_random_server_statement(message.channel)
        options = self._generate_humanize_candidates(statement, count=3)
        key = self._humanize_session_key(
            message.guild.id, message.channel.id, message.author.id
        )
        self._humanize_sessions[key] = {
            "created_ts": datetime.now(timezone.utc).timestamp(),
            "stage": "select",
            "statement": statement,
            "options": options,
            "selected": None,
        }
        await message.reply(self._format_humanize_statement(statement))
        await message.channel.send(self._format_humanize_options(options))

    async def _handle_autotrain_command(self, message: discord.Message, args: str):
        if not message.guild:
            await self._reply_scored(
                message, "❌ This command can only be used in a server."
            )
            return
        if not message.author.guild_permissions.manage_guild:
            await self._reply_scored(
                message, "❌ You need 'Manage Server' permission to use this command."
            )
            return

        guild_id = message.guild.id
        parts = [p for p in (args or "").strip().split() if p]

        if not parts:
            cfg = self._auto_train_config_for_guild(guild_id)
            state = "enabled" if cfg["enabled"] else "disabled"
            await self._reply_scored(
                message,
                f"Auto-train is currently **{state}**.\n"
                f"Target rating: **{cfg['rating']}/10** | Strategy: **{cfg['strategy']}** | "
                f"Save every: **{cfg['save_every']}** updates",
            )
            return

        flag = parts[0].lower()
        if flag in {"on", "true", "1", "enable", "enabled"}:
            enabled = True
        elif flag in {"off", "false", "0", "disable", "disabled"}:
            enabled = False
        else:
            await self._reply_scored(
                message,
                f"Usage: `{self._primary_prefix()}autotrain [on|off] [rating 1-10]`",
            )
            return

        rating = AUTO_TRAIN_DEFAULT_RATING
        if len(parts) >= 2:
            try:
                rating = int(parts[1])
            except ValueError:
                await self._reply_scored(
                    message,
                    f"❌ Invalid rating. Usage: `{self._primary_prefix()}autotrain on 8`",
                )
                return
            if rating < 1 or rating > 10:
                await self._reply_scored(
                    message, "❌ Rating must be between 1 and 10."
                )
                return

        self.set_guild_setting(guild_id, "auto_train_enabled", enabled, persist=False)
        self.set_guild_setting(
            guild_id, "auto_train_target_rating", int(rating), persist=False
        )
        if self.get_guild_setting(guild_id, "auto_train_strategy") is None:
            self.set_guild_setting(guild_id, "auto_train_strategy", "identity", persist=False)
        if self.get_guild_setting(guild_id, "auto_train_save_every") is None:
            self.set_guild_setting(
                guild_id,
                "auto_train_save_every",
                AUTO_TRAIN_DEFAULT_SAVE_EVERY,
                persist=False,
            )
        self._save_persistent_data()

        status = "enabled" if enabled else "disabled"
        await self._reply_scored(
            message,
            f"✅ Auto-train {status} for **{message.guild.name}** "
            f"(target rating **{int(rating)}/10**).",
        )

    async def _resend_pinned_message(self, message: discord.Message):
        """Optimized pinned message resending with robust error handling"""
        # Skip if not in a guild (DM pins shouldn't be resent)
        if not message.guild:
            logger.debug("Skipping DM pin message")
            return

        guild_id = message.guild.id
        
        # Check if already processed - use set for O(1) lookup
        if guild_id in self._resent_pins and message.id in self._resent_pins[guild_id]:
            logger.debug(f"Pin {message.id} already processed for guild {guild_id}")
            return
        
        # Get target channel from server settings
        target_channel_id = self._pin_settings.get(str(guild_id))
        if not target_channel_id:
            logger.debug(f"No pin channel configured for guild {guild_id}")
            return
        
        # Get channel and validate
        target_channel = self.get_channel(int(target_channel_id))
        if not target_channel:
            logger.error(f"Pin channel {target_channel_id} not found for guild {guild_id}")
            return
        
        # Validate channel is in same guild
        if target_channel.guild.id != guild_id:
            logger.error(f"Pin channel {target_channel_id} belongs to different guild")
            return
        
        # Mark as processed immediately to prevent race conditions
        self._add_resent_pin(guild_id, message.id)
        
        # Process and send with comprehensive error handling
        try:
            # Prepare content and attachments
            content = await self._prepare_pin_content(message)
            files = await self._prepare_pin_files(message)
            
            # Get webhook and send
            webhook = await self._get_or_create_webhook(target_channel)
            
            await webhook.send(
                content=content,
                username=message.author.display_name,
                avatar_url=message.author.display_avatar.url if message.author.display_avatar else None,
                files=files
            )
            
            logger.info(f"✅ Pin {message.id} resent to {target_channel.name}")
            
        except discord.Forbidden as e:
            logger.error(f"Permission denied resending pin {message.id}: {e}")
        except discord.HTTPException as e:
            logger.error(f"HTTP error resending pin {message.id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error resending pin {message.id}: {e}")

    async def _prepare_pin_content(self, message: discord.Message) -> str:
        """Prepare pin content with attachment processing"""
        content = message.content or ""
        
        # Debug logging to trace content preparation
        logger.debug(f"DEBUG: _prepare_pin_content called - content: '{message.content}', attachments: {len(message.attachments) if message.attachments else 0}")
        
        if message.attachments:
            logger.info(f"Processing {len(message.attachments)} attachments for pin {message.id}")
            
            for attachment in message.attachments:
                try:
                    # Check file type and log appropriately
                    is_video = any(attachment.filename.lower().endswith(ext) for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'])
                    is_image = any(attachment.filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'])
                    
                    if attachment.size <= DISCORD_FILE_SIZE_LIMIT:
                        if is_video:
                            logger.debug(f"DEBUG: Processing video attachment {attachment.filename} - size: {attachment.size}")
                        elif is_image:
                            logger.debug(f"DEBUG: Processing image attachment {attachment.filename} - size: {attachment.size}")
                        else:
                            logger.debug(f"DEBUG: Processing file attachment {attachment.filename} - size: {attachment.size}")
                        continue
                    else:
                        # Large file - upload to cloud
                        file_type = "video" if is_video else ("image" if is_image else "file")
                        logger.debug(f"DEBUG: Processing large {file_type} {attachment.filename} - size: {attachment.size}")
                        link = await self._upload_large_file(attachment)
                        if link:
                            content += f"\n📎 **{attachment.filename}** (Large {file_type}): {link}"
                            logger.info(f"Uploaded {attachment.filename} ({file_type}): {link}")
                        else:
                            content += f"\n❌ **{attachment.filename}** (Upload failed)"
                            logger.error(f"Failed to upload {attachment.filename}")
                except Exception as e:
                    logger.error(f"Error processing attachment {attachment.filename}: {e}")
                    logger.debug(f"DEBUG: Exception in {attachment.filename} processing: {e}")
        
        # Handle empty content case - if no text AND no attachments, return empty string
        if not content and not message.attachments:
            logger.debug(f"DEBUG: Empty content detected - returning empty string")
            return ""
        
        if not content and not hasattr(self, '_prepared_files'):
            self._prepared_files = []
        
        logger.debug(f"DEBUG: _prepare_pin_content returning content: '{content}' (length: {len(content)})")
        return content or "*(Pinned message contained no text or sendable attachments)*"

    async def _prepare_pin_files(self, message: discord.Message) -> List[discord.File]:
        """Prepare files for pin sending"""
        files = []
        
        if not hasattr(self, '_prepared_files'):
            self._prepared_files = []
        
        if message.attachments:
            for attachment in message.attachments:
                if attachment.size <= DISCORD_FILE_SIZE_LIMIT:
                    try:
                        # Check if we already prepared this file
                        file_data = next((f for f in self._prepared_files if f.filename == attachment.filename), None)
                        
                        if file_data is None:
                            # Read and prepare new file
                            file_data = await attachment.read()
                            self._prepared_files.append(
                                discord.File(
                                    io.BytesIO(file_data), 
                                    filename=attachment.filename
                                )
                            )
                        files.append(discord.File(io.BytesIO(file_data), attachment.filename))
                    except Exception as e:
                        logger.error(f"Error reading attachment {attachment.filename}: {e}")
                else:
                    # Large file - upload and prepare link
                    link = await self._upload_large_file(attachment)
                    if link:
                        # Add link as text content, not as file
                        content = f"📎 **{attachment.filename}** (Large file): {link}"
                        logger.info(f"Prepared upload link for {attachment.filename}: {link}")
                    else:
                        logger.error(f"Failed to prepare upload for {attachment.filename}")
        
        return files

    async def _cleanup_channel_messages(
        self, channel: Union[discord.TextChannel, discord.Thread]
    ):
        """Clean up messages in channel to maintain limit"""
        try:
            # Get the limit for this channel (use per-channel limit or global default)
            limit = self._autodelete_limits.get(channel.id, self.auto_delete_count)

            # Get all messages in channel (newest first, then reverse to get oldest first)
            messages = []
            logger.info(f"Channel {channel.id}: Starting cleanup with limit {limit}")
            msg_count = 0
            async for message in channel.history(limit=None):
                msg_count += 1
                # Skip pinned messages if configured
                if self.auto_delete_exclude_pinned and message.pinned:
                    continue

                # Skip bot messages if configured
                if self.auto_delete_exclude_bots and message.author.bot:
                    continue

                # Age filtering
                if self.auto_delete_delete_age_hours > 0:
                    age_limit = datetime.now(timezone.utc) - timedelta(
                        hours=self.auto_delete_delete_age_hours
                    )
                    if message.created_at > age_limit:
                        continue

                messages.append(message)

            logger.info(
                f"Channel {channel.id}: Total messages fetched: {msg_count}, after filtering: {len(messages)}"
            )

            # Reverse to get oldest messages first
            messages.reverse()

            # If we exceed the limit, delete oldest messages
            if len(messages) > limit:
                messages_to_delete = messages[
                    : len(messages) - limit
                ]  # Delete oldest, keep newest
                deleted_count = 0

                logger.info(
                    f"Channel {channel.id}: Found {len(messages)} messages, limit {limit}, deleting {len(messages_to_delete)} oldest messages"
                )

                # Check if bulk delete is enabled and messages are young enough (14 days)
                if self.auto_delete_bulk_delete:
                    # Filter messages younger than 14 days
                    bulk_deletable = [
                        m
                        for m in messages_to_delete
                        if (datetime.now(timezone.utc) - m.created_at).days < 14
                    ]
                    if bulk_deletable:
                        try:
                            # Discord limits bulk delete to 100 messages at a time
                            for i in range(0, len(bulk_deletable), 100):
                                batch = bulk_deletable[i : i + 100]
                                await channel.delete_messages(batch)
                                deleted_count += len(batch)
                                logger.info(
                                    f"Bulk deleted {len(batch)} messages in channel {channel.id}"
                                )
                                await asyncio.sleep(1.5)  # Rate limit safety

                            # Remove bulk deleted messages from the list to process remaining
                            messages_to_delete = [
                                m for m in messages_to_delete if m not in bulk_deletable
                            ]
                        except Exception as e:
                            logger.error(f"Bulk delete failed: {e}")

                # Delete remaining messages (older than 14 days or if bulk delete failed/disabled) one by one
                for message in messages_to_delete:
                    try:
                        await message.delete()
                        deleted_count += 1

                        # Rate limiting
                        await asyncio.sleep(self.auto_delete_rate_start)

                    except discord.NotFound:
                        # Message already deleted
                        continue
                    except discord.Forbidden:
                        logger.warning(f"No permission to delete message {message.id}")
                        break
                    except discord.HTTPException as e:
                        if "rate limited" in str(e).lower():
                            # Handle rate limiting
                            retry_after = (
                                int(e.retry_after) if hasattr(e, "retry_after") else 5
                            )
                            logger.warning(
                                f"Rate limited, waiting {retry_after} seconds"
                            )
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            logger.error(f"Error deleting message {message.id}: {e}")
                            continue

                logger.info(
                    f"Cleaned up {deleted_count} messages in channel {channel.id}"
                )

        except discord.Forbidden:
            logger.warning(f"No permission to read messages in channel {channel.id}")
        except Exception as e:
            logger.error(f"Error cleaning up channel {channel.id}: {e}")

    async def _cleanup_channel_messages_immediate(
        self, channel: Union[discord.TextChannel, discord.Thread], limit: int
    ) -> int:
        """Immediately clean up messages to reach the specified limit. Returns count of deleted messages."""
        deleted_count = 0
        try:
            # Fetch messages up to a reasonable limit to avoid hanging on very active channels
            # We fetch limit + 1000 to ensure we can delete enough messages
            fetch_limit = limit + 1000
            messages = []
            async for message in channel.history(limit=fetch_limit):
                # Skip pinned messages if configured
                if self.auto_delete_exclude_pinned and message.pinned:
                    continue

                # Skip bot messages if configured
                if self.auto_delete_exclude_bots and message.author.bot:
                    continue

                messages.append(message)

            # Warn if we might have more messages than we fetched
            if len(messages) >= fetch_limit:
                logger.warning(
                    f"Channel {channel.id}: Hit fetch limit ({fetch_limit}), there may be more messages to delete"
                )

            # If we have more messages than limit, delete the oldest ones
            if len(messages) > limit:
                # Sort by creation time (oldest first)
                messages.sort(key=lambda m: m.created_at)
                messages_to_delete = messages[
                    : len(messages) - limit
                ]  # Delete oldest, keep newest

                logger.info(
                    f"Channel {channel.id}: Found {len(messages)} messages, limit {limit}, deleting {len(messages_to_delete)} oldest messages"
                )

                # Check if bulk delete is enabled and messages are young enough (14 days)
                if self.auto_delete_bulk_delete:
                    # Filter messages younger than 14 days
                    bulk_deletable = [
                        m
                        for m in messages_to_delete
                        if (datetime.now(timezone.utc) - m.created_at).days < 14
                    ]
                    if bulk_deletable:
                        try:
                            # Discord limits bulk delete to 100 messages at a time
                            for i in range(0, len(bulk_deletable), 100):
                                batch = bulk_deletable[i : i + 100]
                                await channel.delete_messages(batch)
                                deleted_count += len(batch)
                                logger.info(
                                    f"Bulk deleted {len(batch)} messages in channel {channel.id}"
                                )
                                await asyncio.sleep(1.5)  # Rate limit safety

                            # Remove bulk deleted messages from the list to process remaining
                            messages_to_delete = [
                                m for m in messages_to_delete if m not in bulk_deletable
                            ]
                        except Exception as e:
                            logger.error(f"Bulk delete failed: {e}")

                # Delete remaining messages (older than 14 days or if bulk delete failed/disabled) one by one
                for message in messages_to_delete:
                    try:
                        await message.delete()
                        deleted_count += 1

                        # Rate limiting
                        await asyncio.sleep(self.auto_delete_rate_start)

                    except discord.NotFound:
                        # Message already deleted
                        continue
                    except discord.Forbidden:
                        logger.warning(f"No permission to delete message {message.id}")
                        break
                    except discord.HTTPException as e:
                        if "rate limited" in str(e).lower():
                            # Handle rate limiting
                            retry_after = (
                                int(e.retry_after) if hasattr(e, "retry_after") else 5
                            )
                            logger.warning(
                                f"Rate limited, waiting {retry_after} seconds"
                            )
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            logger.error(f"Error deleting message {message.id}: {e}")
                            continue

                logger.info(
                    f"Immediately cleaned up {deleted_count} messages in channel {channel.id}"
                )

            return deleted_count

        except discord.Forbidden:
            logger.warning(f"No permission to read messages in channel {channel.id}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up channel {channel.id}: {e}")
            return deleted_count


async def main():
    """Main bot entry point"""
    load_dotenv()

    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        logger.error("DISCORD_BOT_TOKEN not found in environment variables")
        logger.error("Please create a .env file with your bot token")
        sys.exit(1)

    debug_guild = os.getenv("DEBUG_GUILD_ID")  # Optional: for testing in specific guild
    pin_resend_channel = os.getenv(
        "PIN_RESEND_CHANNEL_ID"
    )  # Optional: custom pin resend channel
    auto_delete_count = os.getenv("AUTO_DELETE_COUNT")  # Optional: custom message limit
    custom_prefix = os.getenv("CUSTOM_PREFIX")  # Optional: custom command prefix
    punish_role_id = os.getenv("PUNISH_ROLE_ID")  # Optional: role ID for punishment

    # Create bot instance
    bot = WebhookBot(
        pin_resend_channel, auto_delete_count, custom_prefix, punish_role_id
    )

    # Set debug guild if provided
    if debug_guild:
        try:
            bot.tree.app_command_guild = discord.Object(int(debug_guild))
            logger.info(f"Debug mode: Commands will only sync to guild {debug_guild}")
        except ValueError:
            logger.warning("Invalid DEBUG_GUILD_ID, ignoring...")

    # Check if running in Zeabur environment
    if os.getenv("ZEABUR_ENVIRONMENT"):
        logger.info("Running in Zeabur environment")
        # Zeabur will handle the web server via app.py
        # Bot will run in background thread
        return

    # Start bot directly for local development
    try:
        await bot.start(token)
    except discord.LoginFailure:
        logger.error("Invalid bot token provided")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Bot shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        if not bot.is_closed():
            await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
