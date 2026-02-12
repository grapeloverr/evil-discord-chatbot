"""
Chat commands for Discord bot
"""

import discord
from discord.ext import commands
import logging

logger = logging.getLogger(__name__)


def setup_commands(bot):
    """Setup chat commands"""
    
    @bot.command(name="chat")
    async def chat_cmd(ctx, *, message: str = None):
        """Chat with the AI bot"""
        if message is None:
            await ctx.send(embed=discord.Embed(
                title="❌ Missing Message",
                description="Please provide a message: `!chat <message>`",
                color=discord.Color.red()
            ))
            return
        
        async with ctx.typing():
            try:
                response = await bot._generate_response(
                    message,
                    str(ctx.channel.id),
                    str(ctx.author.id),
                    f"User: {ctx.author.display_name} in {ctx.guild.name}"
                )
                
                await ctx.send(response)
                
            except Exception as e:
                logger.error(f"Chat command error: {e}")
                await ctx.send(embed=discord.Embed(
                    title="❌ Error",
                    description="Sorry, I had trouble processing that.",
                    color=discord.Color.red()
                ))
    
    @bot.command(name="model")
    async def model_cmd(ctx, model_name: str = None):
        """Switch between different AI models or list available models"""
        if model_name is None:
            # Show current model and available models
            current_model = bot.model_manager.get_current_model()
            provider_type = bot.model_manager.get_provider_type()
            
            embed = discord.Embed(
                title="🤖 Model Information",
                color=discord.Color.blue()
            )
            
            embed.add_field(name="Current Model", value=current_model, inline=True)
            embed.add_field(name="Provider", value=provider_type.capitalize(), inline=True)
            
            # Try to list available models
            available_models = await bot.model_manager.list_models()
            if available_models:
                embed.add_field(
                    name="Available Models",
                    value="\\n".join(f"• {model}" for model in available_models[:10]),
                    inline=False
                )
            
            embed.set_footer(text=f"Use !model <name> to switch models")
            await ctx.send(embed=embed)
            return
        
        # Try to switch model
        success = await bot.model_manager.switch_model(model_name)
        if success:
            await ctx.send(embed=discord.Embed(
                title="✅ Model Switched",
                description=f"Switched to `{model_name}`",
                color=discord.Color.green()
            ))
            logger.info(f"Model switched to: {model_name}")
        else:
            await ctx.send(embed=discord.Embed(
                title="❌ Model Switch Failed",
                description=f"Could not switch to `{model_name}`. Check if the model is available.",
                color=discord.Color.red()
            ))
    
    @bot.command(name="status")
    async def status_cmd(ctx):
        """Show bot status and configuration"""
        embed = discord.Embed(
            title="🤖 Bot Status",
            color=discord.Color.blue()
        )
        
        # Model information
        embed.add_field(
            name="🧠 Current Model",
            value=bot.model_manager.get_current_model(),
            inline=True
        )
        embed.add_field(
            name="🔧 Provider",
            value=bot.model_manager.get_provider_type().capitalize(),
            inline=True
        )
        
        # Memory information
        memory_stats = bot.memory.get_memory_stats()
        embed.add_field(
            name="💾 Memory",
            value=f"{memory_stats['conversations']} conversations",
            inline=True
        )
        
        # LLM Health
        llm_healthy = await bot.model_manager.health_check()
        health_status = "🟢 Online" if llm_healthy else "🔴 Offline"
        embed.add_field(name="🌐 LLM Status", value=health_status, inline=True)
        
        # Guild information
        embed.add_field(
            name="📊 Guilds",
            value=str(len(bot.guilds)),
            inline=True
        )
        
        embed.set_footer(text=f"Bot ID: {bot.user.id}")
        await ctx.send(embed=embed)
    
    @bot.command(name="remember")
    async def remember_cmd(ctx, *, fact: str = None):
        """Learn a new fact"""
        if fact is None:
            await ctx.send(embed=discord.Embed(
                title="❌ Missing Fact",
                description="Use format: `!remember key := value`",
                color=discord.Color.red()
            ))
            return
        
        if ":=" in fact:
            key, value = fact.split(":=", 1)
            bot.memory.learn_fact(key.strip(), value.strip())
            await ctx.send(embed=discord.Embed(
                title="✅ Fact Learned",
                description=f"Learned: `{key.strip()}` = `{value.strip()}`",
                color=discord.Color.green()
            ))
        else:
            await ctx.send(embed=discord.Embed(
                title="❌ Invalid Format",
                description="Use format: `!remember key := value`",
                color=discord.Color.red()
            ))
    
    @bot.command(name="recall")
    async def recall_cmd(ctx, *, query: str = None):
        """Recall relevant memories and facts"""
        if query is None:
            await ctx.send(embed=discord.Embed(
                title="❌ Missing Query",
                description="Please provide a search query: `!recall <query>`",
                color=discord.Color.red()
            ))
            return
        
        # Search conversations
        conversations = bot.memory.search_conversations(query, limit=3)
        facts = bot.memory.search_facts(query, limit=3)
        
        embed = discord.Embed(
            title="🔍 Memory Search",
            description=f"Results for: `{query}`",
            color=discord.Color.blue()
        )
        
        if conversations:
            conv_text = "\\n".join([
                f"• {conv['user_input'][:50]}... → {conv['bot_response'][:50]}..."
                for conv in conversations
            ])
            embed.add_field(name="💬 Conversations", value=conv_text, inline=False)
        
        if facts:
            fact_text = "\\n".join([
                f"• {fact['key']}: {fact['value'][:50]}..."
                for fact in facts
            ])
            embed.add_field(name="📝 Facts", value=fact_text, inline=False)
        
        if not conversations and not facts:
            embed.description = f"No memories found for: `{query}`"
        
        await ctx.send(embed=embed)
    
    @bot.command(name="help")
    async def help_cmd(ctx):
        """Show help information"""
        embed = discord.Embed(
            title="🤖 Bot Help",
            description="Available commands:",
            color=discord.Color.blue()
        )
        
        commands_info = [
            ("!chat <message>", "Chat with the AI"),
            ("!model [name]", "Show or switch AI model"),
            ("!status", "Show bot status"),
            ("!remember key := value", "Learn a new fact"),
            ("!recall <query>", "Search memories"),
            ("!help", "Show this help")
        ]
        
        for cmd, desc in commands_info:
            embed.add_field(name=cmd, value=desc, inline=False)
        
        embed.add_field(
            name="💡 Tip",
            value="You can also mention the bot (@BotName) to chat without using commands!",
            inline=False
        )
        
        await ctx.send(embed=embed)
    
    @bot.command(name="memories")
    async def memories_cmd(ctx):
        """Show memory statistics and recent memories"""
        # Get memory stats
        if hasattr(bot.memory_engine, 'get_memory_stats'):
            stats = bot.memory_engine.get_memory_stats()
            
            embed = discord.Embed(
                title="🧠 Memory Statistics",
                color=discord.Color.blue()
            )
            
            embed.add_field(name="Total Conversations", value=str(stats.get("total_conversations", "N/A")), inline=True)
            embed.add_field(name="Total Facts", value=str(stats.get("total_facts", "N/A")), inline=True)
            embed.add_field(name="Users Tracked", value=str(stats.get("total_users_tracked", "N/A")), inline=True)
            embed.add_field(name="Unique Keywords", value=str(stats.get("unique_keywords", "N/A")), inline=True)
            
            # Date range
            date_range = stats.get("date_range", {})
            if date_range.get("first") != "Never":
                embed.add_field(name="First Memory", value=date_range["first"][:10], inline=True)
                embed.add_field(name="Last Memory", value=date_range["last"][:10], inline=True)
            
            await ctx.send(embed=embed)
        else:
            await ctx.send("Memory statistics not available.")
    
    @bot.command(name="train")
    async def train_cmd(ctx, action: str = None, *, args: str = None):
        """Training data management commands
        
        Usage:
        !train stats - Show training data statistics
        !train export [format] - Export training data (alpaca, sharegpt, raw)
        !train diversity - Show training data diversity report
        """
        if not hasattr(bot, 'training_engine'):
            await ctx.send("Training engine not available.")
            return
        
        if action is None or action == "stats":
            stats = bot.training_engine.get_training_stats()
            
            embed = discord.Embed(
                title="📊 Training Data Statistics",
                color=discord.Color.blue()
            )
            
            embed.add_field(name="Total Samples", value=str(stats.get("total_samples", 0)), inline=True)
            embed.add_field(name="Unique Conversations", value=str(stats.get("unique_conversations", 0)), inline=True)
            embed.add_field(name="Avg Quality", value=f"{stats.get('avg_quality', 0):.2f}", inline=True)
            
            quality_dist = stats.get("quality_distribution", {})
            embed.add_field(name="Excellent (≥0.8)", value=str(quality_dist.get("excellent", 0)), inline=True)
            embed.add_field(name="Good (0.5-0.8)", value=str(quality_dist.get("good", 0)), inline=True)
            embed.add_field(name="Needs Work (<0.5)", value=str(quality_dist.get("needs_work", 0)), inline=True)
            embed.add_field(name="Topic Diversity", value=str(stats.get("topic_diversity", 0)), inline=True)
            embed.add_field(name="Last Updated", value=str(stats.get("last_updated", "Never"))[:10], inline=True)
            
            await ctx.send(embed=embed)
            return
        
        if action == "export":
            format_type = args.strip().lower() if args else "alpaca"
            if format_type not in ["alpaca", "sharegpt", "raw"]:
                await ctx.send("Invalid format. Use: alpaca, sharegpt, or raw")
                return
            
            output_file = f"training_data_{format_type}.json"
            result = bot.training_engine.export_for_fine_tuning(output_file, format=format_type)
            
            if result:
                await ctx.send(embed=discord.Embed(
                    title="✅ Training Data Exported",
                    description=f"Exported to `{result}`",
                    color=discord.Color.green()
                ))
            else:
                await ctx.send("Failed to export training data.")
            return
        
        if action == "diversity":
            report = bot.training_engine.get_diversity_report()
            
            embed = discord.Embed(
                title="📈 Training Data Diversity",
                color=discord.Color.blue()
            )
            
            embed.add_field(name="Unique Topics", value=str(report.get("unique_topics", 0)), inline=True)
            
            # Top topics
            top_topics = report.get("top_topics", {})
            if top_topics:
                topics_text = "\n".join([f"• {k}: {v}" for k, v in list(top_topics.items())[:10]])
                embed.add_field(name="Top Topics", value=topics_text, inline=False)
            
            # Response length stats
            length_stats = report.get("response_length_stats", {})
            embed.add_field(name="Avg Response Length", value=f"{length_stats.get('avg', 0):.0f} chars", inline=True)
            embed.add_field(name="Samples with Feedback", value=str(report.get("samples_with_feedback", 0)), inline=True)
            
            await ctx.send(embed=embed)
            return
        
        if action == "feedback":
            # !train feedback <message_id> <rating>
            # or !train feedback <message_id> "<corrected response>"
            if not args:
                await ctx.send("Usage: `!train feedback <message_id> <rating/correction>`")
                return
            
            parts = args.split(" ", 1)
            if len(parts) < 2:
                await ctx.send("Usage: `!train feedback <message_id> <rating/correction>`")
                return
            
            message_id = parts[0]
            feedback_content = parts[1]
            
            # Check if it's a rating (number) or correction (text)
            if feedback_content.isdigit() and len(feedback_content) == 1:
                rating = int(feedback_content)
                success = bot.training_engine.add_feedback(message_id, rating=rating, user_id=str(ctx.author.id))
            else:
                success = bot.training_engine.add_feedback(message_id, correction=feedback_content, user_id=str(ctx.author.id))
            
            if success:
                await ctx.send(embed=discord.Embed(
                    title="✅ Feedback Recorded",
                    description="Thank you for helping improve the training data!",
                    color=discord.Color.green()
                ))
            else:
                await ctx.send("Could not find the conversation to add feedback to.")
            return
        
        # Default help
        await ctx.send(embed=discord.Embed(
            title="📚 Training Commands",
            description="""
`!train stats` - Show training data statistics
`!train export [format]` - Export training data (alpaca/sharegpt/raw)
`!train diversity` - Show training data diversity report
`!train feedback <message_id> <rating>` - Rate a response (1-5)
`!train feedback <message_id> "<correction>"` - Correct a response
            """,
            color=discord.Color.blue()
        ))
    
    @bot.command(name="personality")
    async def personality_cmd(ctx, action: str = None, *, args: str = None):
        """Personality and style management commands
        
        Usage:
        !personality show - Show current personality traits
        !personality set <trait> <value> - Set a trait (0.0-1.0)
        !personality user <user_id> - Show personality for a user
        !personality save - Save personality adaptations
        """
        if not hasattr(bot, 'personality'):
            await ctx.send("Personality system not available.")
            return
        
        if action is None or action == "show":
            stats = bot.personality.get_personality_stats()
            
            embed = discord.Embed(
                title="🎭 Personality Traits",
                description="Current personality configuration (0.0-1.0 scale)",
                color=discord.Color.blue()
            )
            
            traits = stats.get("default_traits", bot.personality.traits)
            trait_descriptions = {
                "formality": "0=c casual, 1=formal",
                "verbosity": "0=brief, 1=detailed",
                "humor": "0=serious, 1=humorous",
                "helpfulness": "How helpful to be",
                "creativity": "0=literal, 1=creative",
                "friendliness": "0=cold, 1=warm",
                "patience": "How patient to be",
                "confidence": "0=uncertain, 1=confident",
            }
            
            for trait, value in sorted(traits.items()):
                desc = trait_descriptions.get(trait, "")
                bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
                embed.add_field(
                    name=f"{trait.capitalize()} {bar}",
                    value=f"{value:.1f} - {desc}",
                    inline=False
                )
            
            embed.add_field(name="Users Adapted", value=str(stats.get("users_adapted", 0)), inline=True)
            embed.add_field(name="Corrections Learned", value=str(stats.get("corrections_learned", 0)), inline=True)
            
            await ctx.send(embed=embed)
            return
        
        if action == "set":
            # !personality set <trait> <value>
            if not args:
                await ctx.send("Usage: `!personality set <trait> <value>`")
                return
            
            parts = args.split()
            if len(parts) < 2:
                await ctx.send("Usage: `!personality set <trait> <value>`")
                return
            
            trait = parts[0].lower()
            try:
                value = float(parts[1])
                if not 0 <= value <= 1:
                    raise ValueError
            except ValueError:
                await ctx.send("Value must be a number between 0.0 and 1.0")
                return
            
            if trait in bot.personality.traits:
                bot.personality.traits[trait] = value
                await ctx.send(embed=discord.Embed(
                    title="✅ Trait Updated",
                    description=f"Set `{trait}` to {value}",
                    color=discord.Color.green()
                ))
            else:
                await ctx.send(f"Unknown trait: `{trait}`. Available: {', '.join(bot.personality.traits.keys())}")
            return
        
        if action == "user":
            # !personality user <user_id or @mention>
            user_id = None
            
            if args:
                # Try to extract user ID
                mention_match = re.match(r"<@!?(\d+)>", args)
                if mention_match:
                    user_id = int(mention_match.group(1))
                else:
                    try:
                        user_id = int(args.strip())
                    except ValueError:
                        pass
            
            if user_id is None:
                user_id = ctx.author.id
            
            stats = bot.personality.get_personality_stats(user_id)
            
            embed = discord.Embed(
                title=f"🎭 Personality for User {user_id}",
                color=discord.Color.blue()
            )
            
            traits = stats.get("traits", {})
            for trait, value in sorted(traits.items()):
                bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
                embed.add_field(name=f"{trait.capitalize()}", value=f"{bar} {value:.2f}", inline=True)
            
            embed.add_field(name="Conversations", value=str(stats.get("conversations", 0)), inline=True)
            
            await ctx.send(embed=embed)
            return
        
        if action == "save":
            bot.personality.save_adaptations()
            await ctx.send(embed=discord.Embed(
                title="✅ Personality Saved",
                description="Personality adaptations saved to file.",
                color=discord.Color.green()
            ))
            return
        
        # Default help
        await ctx.send(embed=discord.Embed(
            title="🎭 Personality Commands",
            description="""
`!personality show` - Show current personality traits
`!personality set <trait> <value>` - Set a trait (0.0-1.0)
`!personality user [@user]` - Show personality for a user
`!personality save` - Save personality adaptations
            """,
            color=discord.Color.blue()
        ))
    
    logger.info("Enhanced memory and personality commands registered")


def setup_scrape_command(bot):
    """Setup server scraping command"""
    
    @bot.command(name="scrape")
    async def scrape_cmd(ctx, action: str = None, *, args: str = None):
        """Scrape server conversations for training data
        
        Usage (Owner only):
        !scrape preview <guild_id> - Preview what would be scraped
        !scrape run <guild_id> [options] - Run the scraper
        !scrape import <file> - Import scraped data into training engine
        !scrape stats - Show scraped data statistics
        
        Options:
        --channels <ids> - Comma-separated channel IDs
        --limit <n> - Max messages per channel (default: 1000)
        --min-length <n> - Min message length (default: 10)
        --format <json/alpaca/sharegpt> - Output format
        """
        # Check if user is owner/admin
        if not await bot.is_owner(ctx.author):
            await ctx.send("❌ This command is owner-only.")
            return
        
        if action is None:
            await ctx.send(embed=discord.Embed(
                title="🔍 Server Scraper Commands",
                description="""
**Owner Commands:**
`!scrape preview <guild_id>` - Preview scraping results
`!scrape run <guild_id>` - Run the scraper
`!scrape import <file>` - Import into training data
`!scrape stats` - Show scraped data stats
                """,
                color=discord.Color.blue()
            ))
            return
        
        if action == "preview":
            # Preview mode - show what would be scraped without saving
            if not args:
                await ctx.send("Usage: `!scrape preview <guild_id> [options]`")
                return
            
            parts = args.split()
            try:
                guild_id = int(parts[0])
            except ValueError:
                await ctx.send("Invalid guild ID")
                return
            
            # Parse options
            options = {}
            for part in parts[1:]:
                if part.startswith("--"):
                    if "=" in part:
                        key, value = part[2:].split("=", 1)
                        options[key] = value
            
            await ctx.send(f"🔍 Previewing scrape for guild {guild_id}...\n(Note: Full preview requires bot permissions)")
            
            # Show channel list that would be scraped
            guild = bot.get_guild(guild_id)
            if guild:
                text_channels = [c for c in guild.text_channels if c.permissions_for(guild.me).read_message_history]
                embed = discord.Embed(
                    title=f"📊 Preview: {guild.name}",
                    color=discord.Color.blue()
                )
                embed.add_field(name="Total Text Channels", value=str(len(text_channels)), inline=True)
                embed.add_field(name="Members", value=str(guild.member_count), inline=True)
                
                channel_list = "\n".join([f"• #{c.name}" for c in text_channels[:20]])
                if len(text_channels) > 20:
                    channel_list += f"\n• ... and {len(text_channels) - 20} more"
                embed.add_field(name="Channels", value=channel_list, inline=False)
                
                await ctx.send(embed=embed)
            else:
                await ctx.send("Could not access guild. Make sure the bot is in the server.")
            return
        
        if action == "run":
            # Run the actual scraper
            if not args:
                await ctx.send("Usage: `!scrape run <guild_id> [options]`")
                return
            
            parts = args.split()
            try:
                guild_id = int(parts[0])
            except ValueError:
                await ctx.send("Invalid guild ID")
                return
            
            await ctx.send("⚠️ Starting server scrape... This may take a while.")
            
            # Import and run scraper
            try:
                from src.utils.server_scraper import ServerScraper
                
                # Parse options
                options = {
                    "limit": 1000,
                    "min-length": 10,
                    "format": "alpaca"
                }
                for part in parts[1:]:
                    if part.startswith("--"):
                        if "=" in part:
                            key, value = part[2:].split("=", 1)
                            if key in options:
                                if key == "limit" or key == "min-length":
                                    options[key] = int(value)
                                else:
                                    options[key] = value
                
                # Get token from environment
                from dotenv import load_dotenv
                load_dotenv()
                token = os.getenv("DISCORD_TOKEN")
                
                if not token:
                    await ctx.send("❌ DISCORD_TOKEN not found in environment")
                    return
                
                # Run scraper (this will take time, so we can't await in Discord command)
                # Instead, we'll create a task
                import asyncio
                
                async def run_scraper_task():
                    scraper = ServerScraper(
                        token=token,
                        min_message_length=options["min-length"],
                        max_messages_per_channel=options["limit"]
                    )
                    
                    result = await scraper.scrape(guild_id)
                    
                    # Save results
                    output_file = f"scraped_guild_{guild_id}.json"
                    scraper.export_samples(output_file, format=options["format"])
                    
                    return result, output_file
                
                # Schedule the task
                task = asyncio.create_task(run_scraper_task())
                
                # Store task for later reference
                if not hasattr(bot, 'scraper_tasks'):
                    bot.scraper_tasks = {}
                bot.scraper_tasks[f"scrape_{ctx.message.id}"] = task
                
                await ctx.send(f"✅ Scraping started in background. Check logs or use `!scrape stats` when complete.\nTask ID: `scrape_{ctx.message.id}`")
                
            except Exception as e:
                logger.error(f"Scraper error: {e}")
                await ctx.send(f"❌ Error starting scraper: {e}")
            return
        
        if action == "import":
            # Import scraped data into training engine
            if not args:
                await ctx.send("Usage: `!scrape import <file>`")
                return
            
            file_path = args.strip()
            
            if not os.path.exists(file_path):
                await ctx.send(f"File not found: {file_path}")
                return
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    scraped_data = json.load(f)
                
                if not hasattr(bot, 'training_engine'):
                    await ctx.send("Training engine not available.")
                    return
                
                # Import samples
                imported = 0
                for sample in scraped_data:
                    if isinstance(sample, dict):
                        bot.training_engine.add_training_sample(
                            user_input=sample.get('input_text', ''),
                            bot_response=sample.get('output_text', ''),
                            quality_score=sample.get('quality_score', 0.5)
                        )
                        imported += 1
                
                await ctx.send(embed=discord.Embed(
                    title="✅ Import Complete",
                    description=f"Imported {imported} samples from `{file_path}`",
                    color=discord.Color.green()
                ))
                
            except Exception as e:
                logger.error(f"Import error: {e}")
                await ctx.send(f"❌ Error importing: {e}")
            return
        
        if action == "stats":
            # Show scraped data statistics
            scraped_files = [f for f in os.listdir('.') if f.startswith('scraped_') and f.endswith('.json')]
            
            if not scraped_files:
                await ctx.send("No scraped data files found.")
                return
            
            embed = discord.Embed(
                title="📊 Scraped Data Statistics",
                color=discord.Color.blue()
            )
            
            total_samples = 0
            for f in scraped_files:
                try:
                    with open(f, 'r', encoding='utf-8') as fp:
                        data = json.load(fp)
                    count = len(data) if isinstance(data, list) else 0
                    total_samples += count
                    embed.add_field(name=f, value=f"{count} samples", inline=False)
                except Exception:
                    pass
            
            embed.add_field(name="Total Samples", value=str(total_samples), inline=True)
            
            # Show scraper tasks
            if hasattr(bot, 'scraper_tasks') and bot.scraper_tasks:
                active_tasks = len([t for t in bot.scraper_tasks.values() and not t.done()])
                embed.add_field(name="Active Tasks", value=str(active_tasks), inline=True)
            
            await ctx.send(embed=embed)
            return
        
        await ctx.send("Unknown action. Use `!scrape` for help.")
    
    logger.info("Server scrape command registered")
