#!/usr/bin/env python3
"""
Scrape Discord server messages with a bot token and write analysis reports.

Usage:
  python scrape_and_analyze_server.py --guild-id 1234567890
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import discord
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("scrape_analyze")


STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "for",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "is",
    "it",
    "this",
    "that",
    "these",
    "those",
    "be",
    "been",
    "are",
    "was",
    "were",
    "am",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "me",
    "my",
    "your",
    "our",
    "their",
    "its",
    "as",
    "not",
    "so",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "have",
    "has",
    "had",
    "just",
    "like",
    "im",
    "dont",
    "cant",
}


@dataclass
class ScrapeConfig:
    guild_id: int
    output_base_dir: Path
    per_channel_limit: int
    include_bots: bool


class ScrapeClient(discord.Client):
    def __init__(self, cfg: ScrapeConfig):
        intents = discord.Intents.default()
        intents.guilds = True
        intents.messages = True
        intents.message_content = True
        super().__init__(intents=intents)
        self.cfg = cfg
        self.done_event: asyncio.Event = asyncio.Event()
        self.error: Optional[str] = None

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (%s)", self.user, self.user.id if self.user else "?")
        try:
            await self._run_scrape()
        except Exception as exc:
            self.error = str(exc)
            logger.exception("Scrape failed: %s", exc)
        finally:
            self.done_event.set()
            await self.close()

    async def _run_scrape(self) -> None:
        guild = self.get_guild(self.cfg.guild_id)
        if guild is None:
            try:
                fetched = await self.fetch_guild(self.cfg.guild_id)
                guild = self.get_guild(fetched.id)
            except Exception as exc:
                raise RuntimeError(
                    f"Guild {self.cfg.guild_id} not accessible. Ensure bot is in server and has intents."
                ) from exc

        if guild is None:
            raise RuntimeError(
                f"Guild {self.cfg.guild_id} is not in local cache after fetch."
            )

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = self.cfg.output_base_dir / f"guild_{guild.id}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Scraping guild: %s (%s)", guild.name, guild.id)
        logger.info("Writing outputs to: %s", out_dir)

        messages_path = out_dir / "messages.jsonl"
        channels_report: List[Dict[str, Any]] = []
        total_messages = 0
        total_channels = 0
        word_counter: Counter[str] = Counter()
        user_counter: Counter[str] = Counter()
        channel_counter: Counter[str] = Counter()
        first_ts: Optional[datetime] = None
        last_ts: Optional[datetime] = None

        with messages_path.open("w", encoding="utf-8") as f:
            for channel in guild.text_channels:
                perms = channel.permissions_for(guild.me)
                if not (perms.view_channel and perms.read_message_history):
                    channels_report.append(
                        {
                            "channel_id": channel.id,
                            "channel_name": channel.name,
                            "status": "skipped_no_permission",
                            "count": 0,
                        }
                    )
                    continue

                channel_count = 0
                total_channels += 1
                logger.info("Reading #%s", channel.name)

                try:
                    async for msg in channel.history(limit=self.cfg.per_channel_limit, oldest_first=True):
                        if (not self.cfg.include_bots) and msg.author.bot:
                            continue

                        created = msg.created_at.astimezone(timezone.utc)
                        if first_ts is None or created < first_ts:
                            first_ts = created
                        if last_ts is None or created > last_ts:
                            last_ts = created

                        record = {
                            "message_id": msg.id,
                            "created_at": created.isoformat(),
                            "author_id": msg.author.id,
                            "author_name": str(msg.author),
                            "author_display_name": msg.author.display_name,
                            "author_is_bot": msg.author.bot,
                            "channel_id": channel.id,
                            "channel_name": channel.name,
                            "content": msg.content,
                            "attachments": [
                                {
                                    "filename": att.filename,
                                    "size": att.size,
                                    "content_type": att.content_type,
                                    "url": att.url,
                                }
                                for att in msg.attachments
                            ],
                            "embed_count": len(msg.embeds),
                            "jump_url": msg.jump_url,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                        text = msg.content.lower().strip()
                        if text:
                            for word in re.findall(r"[a-zA-Z']{3,}", text):
                                w = word.lower().strip("'")
                                if w and w not in STOP_WORDS:
                                    word_counter[w] += 1

                        user_counter[str(msg.author)] += 1
                        channel_counter[f"#{channel.name}"] += 1
                        channel_count += 1
                        total_messages += 1
                except discord.Forbidden:
                    channels_report.append(
                        {
                            "channel_id": channel.id,
                            "channel_name": channel.name,
                            "status": "forbidden",
                            "count": channel_count,
                        }
                    )
                    continue
                except discord.HTTPException as exc:
                    channels_report.append(
                        {
                            "channel_id": channel.id,
                            "channel_name": channel.name,
                            "status": f"http_error:{exc.status}",
                            "count": channel_count,
                        }
                    )
                    continue

                channels_report.append(
                    {
                        "channel_id": channel.id,
                        "channel_name": channel.name,
                        "status": "ok",
                        "count": channel_count,
                    }
                )

        summary = {
            "guild": {"id": guild.id, "name": guild.name},
            "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                "per_channel_limit": self.cfg.per_channel_limit,
                "include_bots": self.cfg.include_bots,
            },
            "stats": {
                "channels_scanned": total_channels,
                "messages_collected": total_messages,
                "time_range_start_utc": first_ts.isoformat() if first_ts else None,
                "time_range_end_utc": last_ts.isoformat() if last_ts else None,
            },
            "top_users": [{"user": u, "count": c} for u, c in user_counter.most_common(20)],
            "top_channels": [{"channel": ch, "count": c} for ch, c in channel_counter.most_common(20)],
            "top_words": [{"word": w, "count": c} for w, c in word_counter.most_common(100)],
            "channels": channels_report,
            "files": {
                "messages": str(messages_path),
            },
        }

        summary_path = out_dir / "analysis_summary.json"
        summary_txt_path = out_dir / "analysis_summary.txt"
        with summary_path.open("w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2, ensure_ascii=False)

        with summary_txt_path.open("w", encoding="utf-8") as tf:
            tf.write(f"Guild: {guild.name} ({guild.id})\n")
            tf.write(f"Scraped at (UTC): {summary['scraped_at_utc']}\n")
            tf.write(f"Channels scanned: {total_channels}\n")
            tf.write(f"Messages collected: {total_messages}\n")
            tf.write(f"Time range start: {summary['stats']['time_range_start_utc']}\n")
            tf.write(f"Time range end: {summary['stats']['time_range_end_utc']}\n\n")

            tf.write("Top users:\n")
            for entry in summary["top_users"][:10]:
                tf.write(f"- {entry['user']}: {entry['count']}\n")

            tf.write("\nTop channels:\n")
            for entry in summary["top_channels"][:10]:
                tf.write(f"- {entry['channel']}: {entry['count']}\n")

            tf.write("\nTop words:\n")
            for entry in summary["top_words"][:30]:
                tf.write(f"- {entry['word']}: {entry['count']}\n")

        logger.info("Scrape complete: %s messages across %s channels", total_messages, total_channels)
        logger.info("Saved: %s", summary_path)
        logger.info("Saved: %s", summary_txt_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape and analyze a Discord server")
    parser.add_argument("--guild-id", type=int, required=True, help="Discord guild/server ID")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("server_exports"),
        help="Base output directory",
    )
    parser.add_argument(
        "--per-channel-limit",
        type=int,
        default=2000,
        help="Max messages to fetch per text channel",
    )
    parser.add_argument(
        "--include-bots",
        action="store_true",
        help="Include bot-authored messages",
    )
    return parser.parse_args()


async def run() -> int:
    load_dotenv()
    args = parse_args()

    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        logger.error("DISCORD_BOT_TOKEN is missing in environment/.env")
        return 1

    cfg = ScrapeConfig(
        guild_id=args.guild_id,
        output_base_dir=args.output_dir,
        per_channel_limit=args.per_channel_limit,
        include_bots=args.include_bots,
    )

    client = ScrapeClient(cfg)
    try:
        await client.start(token)
        await client.done_event.wait()
    except discord.LoginFailure:
        logger.error(
            "Discord rejected DISCORD_BOT_TOKEN (401 Unauthorized). "
            "Update token in .env and retry."
        )
        return 1
    finally:
        if not client.is_closed():
            await client.close()

    if client.error:
        logger.error(client.error)
        return 1
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
