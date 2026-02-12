#!/usr/bin/env python3
"""
Quick scrape script - reads config from environment
Usage: python run_scrape.py
"""

import asyncio
import json
import os
from pathlib import Path

# Load env
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("SCRAPE_GUILD_ID", "0"))
OUTPUT_FILE = os.getenv("SCRAPE_OUTPUT", "scraped_training_data.json")
FORMAT = os.getenv("SCRAPE_FORMAT", "alpaca")
MAX_MSGS = int(os.getenv("SCRAPE_LIMIT", "1000"))
MIN_LEN = int(os.getenv("SCRAPE_MIN_LENGTH", "10"))

async def main():
    if not TOKEN:
        print("❌ DISCORD_TOKEN not set in .env")
        return
    if not GUILD_ID:
        print("❌ SCRAPE_GUILD_ID not set in .env")
        print("   Add: SCRAPE_GUILD_ID=1431511744748191950 to .env")
        return
    
    print(f"Initializing scraper...")
    print(f"  Guild ID: {GUILD_ID}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Format: {FORMAT}")
    print("-" * 40)
    
    from src.utils.server_scraper import ServerScraper
    
    scraper = ServerScraper(
        token=TOKEN,
        min_message_length=MIN_LEN,
        max_messages_per_channel=MAX_MSGS,
        exclude_channels=['bot-commands', 'spam', 'logs']
    )
    
    try:
        result = await scraper.scrape(GUILD_ID)
        
        print(f"\n{'='*50}")
        print("SCRAPING COMPLETE")
        print(f"{'='*50}")
        
        stats = result.get('stats', {})
        print(f"\nStats:")
        for key, value in sorted(stats.items()):
            print(f"  {key}: {value}")
        
        channels_data = result.get('channels', {})
        if channels_data:
            print(f"\nChannels:")
            for ch, data in channels_data.items():
                print(f"  #{ch}: {data.get('messages', 0)} msgs, {data.get('conversations', 0)} convs")
        
        samples = result.get('samples', [])
        print(f"\nTotal samples: {len(samples)}")
        
        # Quality breakdown
        quality_ranges = [
            (0.8, 1.0, "Excellent"),
            (0.6, 0.8, "Good"),
            (0.4, 0.6, "Medium"),
            (0.2, 0.4, "Low"),
        ]
        
        for low, high, label in quality_ranges:
            count = len([s for s in samples if low <= s.get('quality_score', 0) < high])
            print(f"  {label} ({low:.0f}-{high:.0f}%): {count}")
        
        # Export
        scraper.export_samples(OUTPUT_FILE, format=FORMAT)
        print(f"\n✅ Exported {len(samples)} samples to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    asyncio.run(main())
