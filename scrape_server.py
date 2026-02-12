#!/usr/bin/env python3
"""
Discord Server Scraper - CLI Tool

Scrapes real conversations from a Discord server and generates
training data for the bot.

Usage:
    python scrape_server.py <TOKEN> <GUILD_ID> [options]
    python scrape_server.py --interactive
    
Options:
    --output, -o       Output file (default: scraped_training_data.json)
    --format           Export format: json, alpaca, sharegpt (default: alpaca)
    --dry-run          Preview without saving
    --limit            Max messages per channel (default: 5000)
    --min-length       Minimum message length (default: 10)
    --channels         Comma-separated channel IDs to scrape
    --review           Review samples before exporting
    --min-quality      Minimum quality score (0.0-1.0)
    --help             Show this help

Examples:
    # Dry run to preview what would be scraped
    python scrape_server.py TOKEN 123456789 --dry-run
    
    # Full scrape with Alpaca format export
    python scrape_server.py TOKEN 123456789 --format alpaca --output train.json
    
    # Scrape specific channels only
    python scrape_server.py TOKEN 123456789 --channels 111,222,333
    
    # Review samples before exporting
    python scrape_server.py TOKEN 123456789 --review --min-quality 0.3
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from src.utils.server_scraper import ServerScraper, review_samples


def main():
    parser = argparse.ArgumentParser(
        description='Scrape Discord server for training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('token', nargs='?', help='Discord bot token')
    parser.add_argument('guild_id', nargs='?', type=int, help='Server ID to scrape')
    
    # Output options
    parser.add_argument('--output', '-o', default='scraped_training_data.json',
                        help='Output file path')
    parser.add_argument('--format', choices=['json', 'alpaca', 'sharegpt'], 
                        default='alpaca', help='Export format')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without saving')
    
    # Scraping options
    parser.add_argument('--limit', type=int, default=5000,
                        help='Max messages per channel')
    parser.add_argument('--min-length', type=int, default=10,
                        help='Minimum message length')
    parser.add_argument('--channels', default=None,
                        help='Comma-separated channel IDs to scrape')
    
    # Quality options
    parser.add_argument('--review', action='store_true',
                        help='Review samples before exporting')
    parser.add_argument('--min-quality', type=float, default=0.0,
                        help='Minimum quality score (0.0-1.0)')
    
    # Interactive mode
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive mode (prompt for missing args)')
    
    args = parser.parse_args()
    
    # Handle interactive mode
    if args.interactive:
        print("Discord Server Scraper - Interactive Mode")
        print("=" * 50)
        
        if not args.token:
            args.token = input("Discord Bot Token: ").strip()
            if not args.token:
                print("Error: Token is required")
                sys.exit(1)
        
        if not args.guild_id:
            guild_input = input("Server ID (GUILD_ID): ").strip()
            try:
                args.guild_id = int(guild_input)
            except ValueError:
                print("Error: Guild ID must be a number")
                sys.exit(1)
        
        args.output = input(f"Output file [{args.output}]: ").strip() or args.output
        args.dry_run = input("Dry run? (y/n) [n]: ").lower().strip() == 'y'
        
        if not args.dry_run:
            fmt = input("Format (json/alpaca/sharegpt) [alpaca]: ").strip() or 'alpaca'
            args.format = fmt if fmt in ('json', 'alpaca', 'sharegpt') else 'alpaca'
            
            if input("Review samples before export? (y/n) [n]: ").lower().strip() == 'y':
                args.review = True
                q = input("Minimum quality (0.0-1.0) [0.3]: ").strip()
                args.min_quality = float(q) if q else 0.3
    
    # Validate required args
    if not args.token or not args.guild_id:
        parser.print_help()
        print("\nError: Token and Guild ID are required (or use --interactive)")
        sys.exit(1)
    
    # Parse channel list
    channel_ids = None
    if args.channels:
        try:
            channel_ids = [int(c.strip()) for c in args.channels.split(',')]
        except ValueError:
            print("Error: Channel IDs must be numbers")
            sys.exit(1)
    
    # Run scraper
    async def run():
        print(f"Initializing scraper...")
        print(f"  Server ID: {args.guild_id}")
        print(f"  Output: {args.output}")
        print(f"  Format: {args.format}")
        print(f"  Max messages/channel: {args.limit}")
        print(f"  Min message length: {args.min_length}")
        print("-" * 50)
        
        scraper = ServerScraper(
            token=args.token,
            min_message_length=args.min_length,
            max_messages_per_channel=args.limit,
            exclude_channels=['bot-commands', 'spam', 'logs']  # Default exclusions
        )
        
        result = await scraper.scrape(
            guild_id=args.guild_id,
            channels=channel_ids,
            limit_per_channel=args.limit
        )
        
        # Print stats
        print("\n" + "=" * 50)
        print("SCRAPING COMPLETE")
        print("=" * 50)
        print(f"\nStats:")
        for key, value in result['stats'].items():
            print(f"  {key}: {value}")
        
        print(f"\nSamples by quality:")
        samples = result['samples']
        quality_ranges = [
            (0.8, 1.0, "Excellent"),
            (0.6, 0.8, "Good"),
            (0.4, 0.6, "Medium"),
            (0.2, 0.4, "Low"),
            (0.0, 0.2, "Very Low")
        ]
        
        for low, high, label in quality_ranges:
            count = len([s for s in samples if low <= s.get('quality_score', 0) < high])
            print(f"  {label} ({low:.0f}-{high:.0f}%): {count}")
        
        # Filter by min quality
        filtered_samples = [
            s for s in samples 
            if s.get('quality_score', 0) >= args.min_quality
        ]
        print(f"\nSamples above min quality ({args.min_quality}): {len(filtered_samples)}")
        
        # Review mode
        if args.review and filtered_samples:
            print("\n" + "-" * 50)
            print("Entering review mode...")
            print("Review samples and approve/reject them")
            print("-" * 50)
            
            from src.utils.server_scraper import TrainingSample
            
            # Convert dicts back to objects
            sample_objects = [
                TrainingSample(**s) for s in filtered_samples[:50]  # Review up to 50
            ]
            
            approved, rejected = review_samples(sample_objects)
            
            print(f"\nReview complete: {len(approved)} approved, {len(rejected)} rejected")
            
            # Update filtered samples - keep only approved ones
            approved_set = set(str(id(s)) for s in approved)
            filtered_samples = [
                s for s in filtered_samples 
                if hasattr(s, 'id') and str(id(s)) in approved_set or s in approved
            ]
        
        # Export
        if not args.dry_run and filtered_samples:
            output_path = Path(args.output)
            
            # Temporarily set samples for export
            scraper.samples = [type('Sample', (), s)() for s in filtered_samples]
            for i, sample_dict in enumerate(filtered_samples):
                sample_obj = scraper.samples[i]
                for k, v in sample_dict.items():
                    setattr(sample_obj, k, v)
            
            scraper.export_samples(
                output_path,
                format=args.format,
                min_quality=args.min_quality
            )
            
            print(f"\nExported {len(filtered_samples)} samples to {output_path}")
        elif args.dry_run:
            print(f"\n[DRY RUN] Would export {len(filtered_samples)} samples to {args.output}")
        
        return result
    
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nScraping cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
