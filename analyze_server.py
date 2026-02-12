#!/usr/bin/env python3
"""
Server Personality Analyzer
Analyzes server conversations and generates personality profile
"""

import asyncio
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv()


@dataclass
class PersonalityProfile:
    """Extracted personality profile from server conversations"""
    common_words: list = field(default_factory=list)
    slang: list = field(default_factory=list)
    use_emoji: float = 0.5
    use_caps: float = 0.0
    use_exclaim: float = 0.0
    use_laugh: float = 0.0
    avg_message_length: float = 0.0
    formality: float = 0.5
    humor: float = 0.5
    questions_ratio: float = 0.0
    urls_ratio: float = 0.0
    code_blocks_ratio: float = 0.0
    recommended_traits: dict = field(default_factory=dict)
    
    def to_personality_config(self) -> dict:
        return {
            "formality": 1.0 - self.formality,
            "verbosity": min(1.0, self.avg_message_length / 100),
            "humor": self.humor,
            "helpfulness": 0.8,
            "creativity": 0.7,
            "friendliness": 0.8,
            "patience": 0.7,
            "confidence": 0.8,
        }


class ServerAnalyzer:
    """Analyzes server conversations for personality extraction"""
    
    # Common slang and casual expressions
    SLANG_PATTERNS = [
        r'\blol\b', r'\blmao\b', r'\bbrb\b', r'\bimo\b', r'\bimho\b',
        r'\bbtw\b', r'\bidk\b', r'\bpls\b', r'\bthx\b', r'\bty\b',
        r'\bnp\b', r'\bomg\b', r'\bwtf\b', r'\bnvm\b', r'\bsmh\b',
        r'\byikes\b', r'\bcap\b', r'\bnocap\b', r'\bfr\b', r'\bfrfr\b',
        r'\bbased\b', r'\bpov\b', r'\bliterally\b', r'\bbasically\b',
        r'\bnah\b', r'\byeah\b', r'\byea\b', r'\bbet\b', r'\blowkey\b',
        r'\bhighkey\b', r'\bdope\b', r'\bsavage\b', r'\bGOAT\b',
    ]
    
    LAUGH_PATTERNS = [
        r'\blol\b', r'\blmao\b', r'\bha(?:ha)+', r'\bheh\b',
        r'\bkekw\b', r'\blul\b', r'\bomegalol\b', r'\brofl\b',
    ]
    
    # Words to filter out (URL parts, GIF links, Discord artifacts)
    FILTER_WORDS = {
        # URL parts
        'http', 'https', 'www', 'com', 'org', 'net', 'edu', 'gov',
        'tenor', 'giphy', 'imgur', 'discord', 'cdn', 'gg', 'ly', 'youtu',
        # GIF-related
        'gif', 'gifs', 'media',
        # Common non-words
        'view', 'click', 'link', 'url', 'embed', 'replied',
    }
    
    def __init__(self):
        self.messages = []
        self.word_counts = Counter()
        self.bigram_counts = Counter()
    
    async def fetch_messages(self, channel_ids: list, max_messages: int = 5000):
        """Fetch messages from channels via REST API"""
        import aiohttp
        
        token = os.getenv("DISCORD_TOKEN")
        if not token:
            raise ValueError("DISCORD_TOKEN not set")
        
        headers = {"Authorization": f"Bot {token}"}
        
        async with aiohttp.ClientSession() as session:
            for channel_id in channel_ids:
                print(f"Fetching channel {channel_id}...")
                
                async with session.get(
                    f"https://discord.com/api/v10/channels/{channel_id}/messages",
                    headers=headers,
                    params={"limit": min(100, max_messages // len(channel_ids))}
                ) as resp:
                    if resp.status == 200:
                        messages = await resp.json()
                        for msg in messages:
                            if msg.get('author', {}).get('bot'):
                                continue
                            self.messages.append(msg)
                            print(f"  Got {len(self.messages)} messages total")
                    else:
                        print(f"  Failed: {resp.status}")
        
        print(f"\nTotal messages collected: {len(self.messages)}")
        return self.messages
    
    def _clean_message(self, content: str) -> str:
        """Remove URLs, GIF links, and Discord artifacts from message"""
        # Remove URLs
        content = re.sub(r'https?://\S+', '', content)
        # Remove Discord custom emoji IDs
        content = re.sub(r'<a?:\w+:\d+>', '', content)
        # Remove markdown links
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        # Remove code blocks
        content = re.sub(r'```[\s\S]*?```', '', content)
        # Remove inline code
        content = re.sub(r'`[^`]+`', '', content)
        return content.strip()
    
    def analyze(self) -> PersonalityProfile:
        """Analyze collected messages"""
        if not self.messages:
            print("No messages to analyze!")
            return None
        
        profile = PersonalityProfile()
        
        # Stopwords to exclude from top words
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after',
            'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
            'neither', 'not', 'only', 'just', 'also', 'very', 'can',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our',
            'their', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'if', 'then', 'else', 'when', 'where',
            'why', 'how', 'all', 'each', 'every', 'no', 'any', 'some',
            'up', 'out', 'get', 'got', 'go', 'going', 'come', 'came',
            'make', 'made', 'know', 'knew', 'think', 'thought',
            'see', 'saw', 'look', 'looking', 'want', 'use', 'find',
            'give', 'tell', 'try', 'leave', 'call', 'keep', 'let',
            'put', 'seem', 'help', 'show', 'hear', 'play', 'run',
            'move', 'like', 'live', 'believe', 'hold', 'bring', 'happen',
        }
        
        total_chars = 0
        total_caps = 0
        total_exclaim = 0
        question_count = 0
        url_count = 0
        code_count = 0
        laugh_count = 0
        emoji_count = 0
        
        for msg in self.messages:
            content = msg.get('content', '')
            if not content:
                continue
            
            if content.startswith('!') or content.startswith('/'):
                continue
            
            # Clean message
            cleaned = self._clean_message(content)
            if not cleaned:
                continue
            
            content_lower = cleaned.lower()
            total_chars += len(content)
            
            # Caps ratio
            caps = sum(1 for c in content if c.isupper())
            total_caps += caps / max(len(content), 1)
            
            # Exclaim
            total_exclaim += content.count('!')
            
            # Questions
            if '?' in content:
                question_count += 1
            
            # URLs (before cleaning)
            if re.search(r'https?://|www\.', content):
                url_count += 1
            
            # Code blocks
            if '```' in content:
                code_count += 1
            
            # Laugh patterns
            for pattern in self.LAUGH_PATTERNS:
                if re.search(pattern, content_lower):
                    laugh_count += 1
                    break
            
            # Emoji (Discord custom)
            if re.search(r'<a?:\w+:\d+>', content):
                emoji_count += 1
            
            # Word analysis
            words = re.findall(r'\b[a-z]+\b', content_lower)
            # Filter out artifacts
            words = [w for w in words if w not in self.FILTER_WORDS 
                     and len(w) > 2 and w not in stopwords]
            
            for word in words:
                self.word_counts[word] += 1
            
            # Bigrams
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                self.bigram_counts[bigram] += 1
        
        n = len(self.messages)
        profile.avg_message_length = total_chars / max(n, 1)
        profile.use_caps = total_caps / max(n, 1)
        profile.use_exclaim = total_exclaim / max(n, 1)
        profile.questions_ratio = question_count / max(n, 1)
        profile.urls_ratio = url_count / max(n, 1)
        profile.code_blocks_ratio = code_count / max(n, 1)
        profile.use_laugh = laugh_count / max(n, 1)
        profile.use_emoji = emoji_count / max(n, 1)
        
        # Formality
        slang_matches = 0
        for pattern in self.SLANG_PATTERNS:
            for msg in self.messages:
                if re.search(pattern, msg.get('content', '').lower()):
                    slang_matches += 1
                    break
        profile.formality = 1.0 - (slang_matches / max(n, 1))
        profile.humor = min(1.0, profile.use_laugh * 3 + profile.use_emoji * 0.5)
        
        # Top words (real words only)
        top_words = [(w, c) for w, c in self.word_counts.most_common(100)
                     if w not in stopwords and len(w) > 2]
        profile.common_words = [w for w, c in top_words[:30]]
        
        # Slang detection
        profile.slang = [w for w, c in top_words[:20] 
                        if any(p in w for p in ['lol', 'lmao', 'bruh', 'tbh'])]
        
        profile.recommended_traits = profile.to_personality_config()
        
        return profile
    
    def print_report(self, profile: PersonalityProfile):
        print("\n" + "=" * 60)
        print("SERVER PERSONALITY ANALYSIS REPORT")
        print("=" * 60)
        
        print(f"\n📊 Message Statistics:")
        print(f"  Messages analyzed: {len(self.messages)}")
        print(f"  Avg message length: {profile.avg_message_length:.1f} chars")
        
        print(f"\n💬 Style Analysis:")
        print(f"  Formality: {profile.formality:.2f} (0=c casual, 1=formal)")
        print(f"  Humor: {profile.humor:.2f}")
        print(f"  Emoji usage: {profile.use_emoji:.2f}")
        print(f"  Laugh patterns: {profile.use_laugh:.2f}")
        print(f"  Questions: {profile.questions_ratio:.2f}")
        
        print(f"\n📝 Top Words (real words only):")
        print(f"  {' '.join(profile.common_words[:15])}")
        
        print(f"\n🎭 Recommended Personality Traits:")
        traits = profile.recommended_traits
        for trait, value in sorted(traits.items()):
            bar = "█" * int(value * 10) + "░" * (10 - int(value * 10))
            print(f"  {trait:12} {bar} {value:.2f}")
        
        print("\n" + "=" * 60)
    
    def save_profile(self, profile: PersonalityProfile, filename: str = "server_personality.json"):
        data = {
            "generated_at": datetime.now().isoformat(),
            "messages_analyzed": len(self.messages),
            "analysis": {
                "avg_message_length": profile.avg_message_length,
                "formality": profile.formality,
                "humor": profile.humor,
                "use_emoji": profile.use_emoji,
                "use_laugh": profile.use_laugh,
                "questions_ratio": profile.questions_ratio,
                "urls_ratio": profile.urls_ratio,
                "code_blocks_ratio": profile.code_blocks_ratio,
            },
            "common_words": profile.common_words,
            "slang": profile.slang,
            "recommended_traits": profile.recommended_traits,
            "top_bigrams": dict(self.bigram_counts.most_common(50)),
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved profile to {filename}")
        
        return data


async def main():
    print("Server Personality Analyzer")
    print("=" * 40)
    
    analyzer = ServerAnalyzer()
    
    channel_ids = []
    
    # Try to load from guild_channels.json
    if Path("guild_channels.json").exists():
        with open("guild_channels.json") as f:
            channels = json.load(f)
        text_channels = [c for c in channels if c['type'] == 0]
        channel_ids = [c['id'] for c in text_channels[:10]]
        print(f"Loaded {len(channel_ids)} channels from guild_channels.json")
    
    if not channel_ids:
        print("Enter channel IDs (comma-separated):")
        user_input = input("> ").strip()
        if user_input:
            channel_ids = [int(x.strip()) for x in user_input.split(',')]
    
    if not channel_ids:
        print("No channel IDs provided")
        return
    
    await analyzer.fetch_messages(channel_ids)
    profile = analyzer.analyze()
    
    if profile:
        analyzer.print_report(profile)
        analyzer.save_profile(profile)
        
        # Update .env with custom personality
        traits = profile.recommended_traits
        formality = traits.get("formality", 0.5)
        humor = traits.get("humor", 0.5)
        
        personality_parts = []
        
        if formality < 0.3:
            personality_parts.append("You are casual and conversational")
        elif formality > 0.7:
            personality_parts.append("You are formal and professional")
        else:
            personality_parts.append("You are friendly but approachable")
        
        if humor > 0.6:
            personality_parts.append("You have a good sense of humor and enjoy light banter")
        elif humor < 0.3:
            personality_parts.append("You are serious and focused")
        
        if profile.common_words:
            personality_parts.append(f"You often use words like: {', '.join(profile.common_words[:5])}")
        
        custom_personality = ". ".join(personality_parts) + "."
        
        # Update .env
        with open(".env", "r") as f:
            lines = f.readlines()
        
        updated_lines = []
        found_custom = False
        for line in lines:
            if line.startswith("CUSTOM_PERSONALITY="):
                updated_lines.append(f"CUSTOM_PERSONALITY={custom_personality}\n")
                found_custom = True
            else:
                updated_lines.append(line)
        
        if not found_custom:
            updated_lines.append(f"\nCUSTOM_PERSONALITY={custom_personality}\n")
        
        with open(".env", "w") as f:
            f.writelines(updated_lines)
        
        print("\nUpdated .env with custom personality")
        print("\nRestart your bot to apply the new personality!")


if __name__ == '__main__':
    asyncio.run(main())
