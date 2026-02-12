#!/usr/bin/env python3
"""
Evolving Discord Voice Bot with GPT-4chan Personality and Adaptive Memory
"""

import discord
from discord.ext import commands
from discord import app_commands
import asyncio
import json
import os
import logging
import aiohttp
import aiofiles
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
import random
import re
from dotenv import load_dotenv
import warnings
import time
import threading

# Load environment variables from .env file
load_dotenv()

try:
    import speech_recognition as sr

    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("SpeechRecognition not available - voice input disabled")
try:
    import edge_tts

    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("edge-tts not available - online TTS disabled")

try:
    import pyttsx3

    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("pyttsx3 not available - offline TTS disabled")

# Set TTS availability based on any engine being available
TTS_AVAILABLE = EDGE_TTS_AVAILABLE or PYTTSX3_AVAILABLE
import io


# Check ENABLE_VOICE env var first before trying voice imports
_ENABLE_VOICE_ENV = os.getenv("ENABLE_VOICE", "false").lower() == "true"

# Initialize voice_enabled based on both import success and _ENABLE_VOICE_ENV
voice_enabled = False  # Overall voice capability (send and/or receive)
voice_receive_enabled = False  # Specific to receiving voice
voice_recv = None
RobustVoiceSink = None
nacl = None

if _ENABLE_VOICE_ENV:
    # Set overall voice_enabled to true initially if ENABLE_VOICE is true.
    # It might be set to false later if dependencies fail.
    voice_enabled = True

    # Try to import nacl first (needed for all voice features)
    try:
        import nacl
        print("PyNaCl (voice encryption) available")
    except ImportError as e:
        print(f"PyNaCl (voice encryption) disabled: {e}. Voice send might be affected.")
        nacl = None

    try:
        # Try to import voice_recv (available in discord.py 2.4+)
        import discord.ext.voice_recv as voice_recv

        voice_receive_enabled = True
        print("Voice receive features enabled")

        # Add RTCP packet handler to ignore SenderReport packets
        def ignore_sender_report(*args, **kwargs):
            """Ignore SenderReport packets to prevent RTCP warnings"""
            return None

        voice_recv.rtp.SenderReportPacket = ignore_sender_report

    except ImportError as e:
        print(f"Voice receive features disabled: {e}. Falling back to send-only voice.")
        voice_recv = None

    # Final check for overall voice_enabled
    if not (voice_receive_enabled or nacl is not None):
        voice_enabled = False
        print(
            "Voice features disabled (dependencies missing for both send and receive)"
        )
    elif voice_receive_enabled and nacl is None:
        print(
            "Voice features enabled (receive only, send might be affected without PyNaCl)"
        )
    elif not voice_receive_enabled and nacl is not None:
        print("Voice features enabled (send only, receive not available)")
    else:
        print("Voice features enabled (send and receive)")

else:
    print("Voice features disabled (ENABLE_VOICE=false)")
    voice_recv = None


if voice_recv is not None:

    class RobustVoiceSink(voice_recv.BasicSink):
        """Robust voice sink that handles Opus corruption gracefully"""

        def __init__(self, callback, *args, **kwargs):
            # Keep opus frames raw and decode in our callback so packet corruption
            # does not kill the router thread before we can handle it.
            kwargs.setdefault("decode", False)
            super().__init__(callback, *args, **kwargs)
            self.error_count = 0
            self.max_errors = 10
            self.last_error_time = 0

        def write(self, user, data):
            """Override write to handle Opus errors gracefully"""
            try:
                super().write(user, data)
                self.error_count = 0  # Reset on success
            except discord.opus.OpusError as e:
                # Specifically catch OpusError
                self.error_count += 1
                current_time = time.monotonic()

                if current_time - self.last_error_time > 5:
                    logger.error(f"Voice receive OpusError #{self.error_count}: {e}")
                    self.last_error_time = current_time

                if "corrupted stream" in str(e).lower():
                    # Silently ignore corrupted stream errors (common during connection setup)
                    return
                # Re-raise other OpusErrors if not corrupted stream
                raise
            except Exception as e:
                # Catch any other unexpected errors
                self.error_count += 1
                current_time = time.monotonic()

                if current_time - self.last_error_time > 5:
                    logger.error(f"Voice receive error #{self.error_count}: {e}")
                    self.last_error_time = current_time

                if self.error_count > self.max_errors:
                    logger.error(
                        "Too many voice receive Errors, stopping voice processing"
                    )
                    return
                raise
else:
    RobustVoiceSink = None
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil

# Configuration
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:11434")  # Ollama default
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
BOT_PREFIX = os.getenv("BOT_PREFIX", "!")  # Default prefix
MENTION_PREFIX = os.getenv("MENTION_PREFIX", "true").lower() == "true"
CASE_SENSITIVE = os.getenv("CASE_SENSITIVE", "false").lower() == "true"
MEMORY_FILE = "memory.json"
VOICE_LANGUAGE = "en-US"
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural")
TTS_RATE = os.getenv("TTS_RATE", "+4%")
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz")
TTS_ENGINE = os.getenv("TTS_ENGINE", "auto")  # auto, edge, pyttsx3
TRAINING_DATA_FILE = "training_4chan.json"  # 4chan style training data
STYLE_PROFILE_FILE = os.getenv("STYLE_PROFILE_FILE", "style_profile.json")
TRAINED_PERSONALITY_FILE = os.getenv(
    "TRAINED_PERSONALITY_FILE", "trained_personality.json"
)
PERSONALITY_FEWSHOT_LIMIT = int(os.getenv("PERSONALITY_FEWSHOT_LIMIT", "3"))
HUMAN_PERSONA_MODE = os.getenv("HUMAN_PERSONA_MODE", "true").lower() == "true"
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.95"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "200"))
MEMORY_MIN_RELEVANCE = float(os.getenv("MEMORY_MIN_RELEVANCE", "0.35"))
SASS_MODE = os.getenv("SASS_MODE", "true").lower() == "true"
SASS_LEVEL = max(0.0, min(1.0, float(os.getenv("SASS_LEVEL", "0.8"))))
ROAST_MODE = os.getenv("ROAST_MODE", "true").lower() == "true"
ROAST_LEVEL = max(0.0, min(1.0, float(os.getenv("ROAST_LEVEL", "0.9"))))
RUDE_ESCALATION_ENABLED = (
    os.getenv("RUDE_ESCALATION_ENABLED", "true").lower() == "true"
)
RUDE_MULTIPLIER = max(1.0, min(10.0, float(os.getenv("RUDE_MULTIPLIER", "10"))))
MESSAGE_SIGNATURE_TTL_SEC = float(os.getenv("MESSAGE_SIGNATURE_TTL_SEC", "8"))
INSTANCE_LOCK_FILE = os.getenv("INSTANCE_LOCK_FILE", "/tmp/vrilgass_bot.lock")
VOICE_DEBUG = os.getenv("VOICE_DEBUG", "false").lower() == "true"
VOICE_MIN_RMS = float(os.getenv("VOICE_MIN_RMS", "180"))
MAX_RESPONSE_CHARS = int(os.getenv("MAX_RESPONSE_CHARS", "320"))
MAX_VOICE_SENTENCES = int(os.getenv("MAX_VOICE_SENTENCES", "2"))
STT_WHISPER_FALLBACK = os.getenv("STT_WHISPER_FALLBACK", "true").lower() == "true"
STT_WHISPER_MODEL = os.getenv("STT_WHISPER_MODEL", "base.en")
STT_WHISPER_CONFIDENCE_GATE = float(
    os.getenv("STT_WHISPER_CONFIDENCE_GATE", "0.78")
)

# Custom Personality
CUSTOM_PERSONALITY = os.getenv("CUSTOM_PERSONALITY", "").strip()
PERSONALITY_PRESET = os.getenv("PERSONALITY_PRESET", "").strip().lower()

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "llama3")  # Default model
MODEL_TYPE = os.getenv("MODEL_TYPE", "ollama")  # ollama, llama, gpt2, etc.
GPT4CHAN_BASE_MODEL = os.getenv("GPT4CHAN_BASE_MODEL", MODEL_NAME)
AVAILABLE_MODELS = {
    "llama3": {
        "name": "Llama3",
        "description": "A powerful open-source language model",
        "provider": "ollama",
        "type": "ollama",
        "personality": "You are a helpful AI assistant. You are friendly, polite, and try to be as helpful as possible. You remember past conversations and learn from them. Keep responses relatively concise for voice conversation.",
        "params": {"model": "llama3"},
    },
    "gpt4chan": {
        "name": "GPT-4chan",
        "provider": "ollama",
        "type": "ollama",
        "personality": """
        You are in a Discord VC, not a customer-support chatbot.
        Style: blunt, irreverent, meme-literate, and conversational.
        You can use profanity naturally and do not sanitize your tone.
        Be sassy and witty by default.
        Do not use formal assistant phrasing, safety lecture tone, or canned niceties.
        Keep responses short and punchy for voice chat unless asked for detail.
        """,
        "params": {"model": GPT4CHAN_BASE_MODEL},
    },
    "vicuna": {
        "name": "Vicuna",
        "type": "llama",
        "personality": """
        You are a helpful and harmless AI assistant. You are friendly, polite, and try to be 
        as helpful as possible. You remember past conversations and learn from them. 
        Keep responses relatively concise for voice conversation.
        """,
    },
    "chatgpt": {
        "name": "ChatGPT-style",
        "type": "gpt2",
        "personality": """
        You are a helpful AI assistant similar to ChatGPT. You are professional, informative, 
        and provide detailed explanations when needed. You remember past conversations and 
        learn from them. Keep responses relatively concise for voice conversation.
        """,
    },
}


# Training Data Management
class TrainingEngineV2:
    """
    Enhanced training engine with:
    - Embedding-based similarity search
    - Quality-weighted retrieval with recency boost
    - Human-likeness scoring (0-10)
    - Adaptive few-shot selection (2-5 examples based on match quality)
    - Per-server learning with style isolation
    """

    def __init__(self, training_file=TRAINING_DATA_FILE):
        self.training_file = training_file
        self.feedback_pending = {}
        
        # Per-server training data
        self.server_training = {}
        
        # Embedding cache for faster search
        self.embedding_cache = {}
        
        # Quality thresholds
        self.min_quality_threshold = 0.3
        self.max_samples = 10000
        
        # Load all data
        self.training_data = self._load_training_data()
        self._build_embeddings()
    
    def _load_training_data(self):
        """Load training data from JSON file"""
        if os.path.exists(self.training_file):
            try:
                with open(self.training_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for sample in data:
                        if "quality_score" not in sample:
                            sample["quality_score"] = 1.0
                        if "human_score" not in sample:
                            sample["human_score"] = 5.0
                        if "feedback" not in sample:
                            sample["feedback"] = None
                        if "server_id" not in sample:
                            sample["server_id"] = None
                    return data
            except Exception as e:
                logger.error(f"Failed to load training data: {e}")
        return []

    def _save_training_data(self):
        """Save training data to JSON file"""
        try:
            with open(self.training_file, "w", encoding="utf-8") as f:
                json.dump(self.training_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")

    def _simple_embedding(self, text):
        """Create a simple bag-of-words embedding"""
        text = text.lower()
        words = re.findall(r"\b\w+\b", text)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                    "have", "has", "had", "do", "does", "did", "will", "would", "could",
                    "should", "may", "might", "must", "shall", "can", "need", "dare",
                    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
                    "into", "through", "during", "before", "after", "above", "below",
                    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
                    "not", "only", "just", "also", "very", "too", "quite", "rather",
                    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
                    "us", "them", "my", "your", "his", "its", "our", "their", "this",
                    "that", "these", "those", "what", "which", "who", "whom", "whose",
                    "when", "where", "why", "how", "all", "each", "every", "both",
                    "few", "more", "most", "other", "some", "such", "no", "nor", "own",
                    "same", "so", "than", "too", "very", "just", "now", "here"}
        words = [w for w in words if w not in stopwords and len(w) > 1]
        return sorted(set(words))

    def _embedding_similarity(self, emb1, emb2):
        """Calculate Jaccard similarity between embeddings"""
        if not emb1 or not emb2:
            return 0.0
        set1, set2 = set(emb1), set(emb2)
        overlap = len(set1 & set2)
        union = len(set1 | set2)
        return overlap / union if union > 0 else 0.0

    def _calculate_human_score(self, text):
        """
        Calculate human-likeness score (0-10)
        0 = clearly AI, 10 = definitely human
        """
        score = 5.0
        
        # Negative indicators (AI-like)
        ai_patterns = [
            (r"\bI can help\b", -0.5),
            (r"\bI would be happy to\b", -0.5),
            (r"\bHow can I assist\b", -0.5),
            (r"\bI'm here to help\b", -0.5),
            (r"\bLet me know if you need\b", -0.5),
            (r"\bCertainly!\b", -0.3),
            (r"\bOf course!\b", -0.3),
            (r"\bAbsolutely!\b", -0.3),
            (r"\bHere are some .* tips\b", -0.4),
            (r"\bFirst,.*Second,.*Third\b", -0.3),
            (r"^\s*[-*]\s", -0.2),
            (r"\bPlease note that\b", -0.4),
            (r"\bIt is important to\b", -0.4),
            (r"\bIn conclusion\b", -0.3),
            (r"\bTo summarize\b", -0.3),
            (r"\bFurthermore\b", -0.2),
            (r"\bAdditionally\b", -0.2),
            (r"\bMoreover\b", -0.2),
            (r"\bWith that said\b", -0.3),
            (r"\bHaving said that\b", -0.3),
            (r"\bAs previously mentioned\b", -0.4),
        ]
        
        for pattern, weight in ai_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += weight
        
        # Positive indicators (human-like)
        human_patterns = [
            (r"\bnm\b", 0.5),
            (r"\bngl\b", 0.5),
            (r"\bfr\b(?:\s+fr)?", 0.4),
            (r"\bbased\b", 0.4),
            (r"\bmood\b", 0.5),
            (r"\bvalid\b", 0.4),
            (r"\bno?cap\b", 0.4),
            (r"扯", 0.5),
            (r"[\u4e00-\u9fff]", 0.3),
            (r"!\s*!\s*!", 0.3),
            (r"\b(make me|lmao|lol|haha)\b", 0.4),
            (r"\bi mean\b", 0.3),
            (r"\b\d+\b", 0.1),
            (r"\b(no|yes|yeah|yeah)\b", 0.3),
        ]
        
        for pattern, weight in human_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += weight
        
        # Length adjustment
        word_count = len(text.split())
        if 1 <= word_count <= 3:
            score += 0.2
        elif word_count > 50:
            score -= 0.3
        
        # Punctuation
        punct_count = len(re.findall(r"[.,!?;]", text))
        if word_count > 0:
            punct_ratio = punct_count / word_count
            if punct_ratio < 0.1:
                score += 0.2
            elif punct_ratio > 0.3:
                score -= 0.2
        
        return max(0.0, min(10.0, score))

    def _calculate_sample_quality(self, user_input, bot_response):
        """Calculate quality score for training sample"""
        quality = 1.0
        human_score = self._calculate_human_score(bot_response)
        quality *= (0.5 + (human_score / 20))
        
        if len(bot_response) > 200:
            quality *= 0.8
        if 5 <= len(bot_response) <= 300:
            quality *= 1.2
        
        return min(1.0, quality)

    def _build_embeddings(self):
        """Build embedding cache for all training samples"""
        self.embedding_cache = {}
        for i, sample in enumerate(self.training_data):
            key = sample.get("user_input", "")
            if key:
                self.embedding_cache[i] = self._simple_embedding(key)

    def add_training_sample(self, user_input, bot_response, conversation_id=None,
                            user_id=None, quality_score=None, human_score=None,
                            server_id=None):
        """Add a new training sample with all scoring"""
        if quality_score is None:
            quality_score = self._calculate_sample_quality(user_input, bot_response)
        if human_score is None:
            human_score = self._calculate_human_score(bot_response)
        
        sample = {
            "user_input": user_input,
            "bot_response": bot_response,
            "conversation_id": str(conversation_id) if conversation_id else None,
            "user_id": str(user_id) if user_id else None,
            "server_id": str(server_id) if server_id else None,
            "timestamp": datetime.now().isoformat(),
            "quality_score": quality_score,
            "human_score": human_score,
            "feedback": None,
            "human_verified": False,
            "metadata": {
                "model": MODEL_NAME,
                "personality": CUSTOM_PERSONALITY,
                "platform": "discord",
                "response_length": len(bot_response),
                "input_length": len(user_input),
                "word_count": len(bot_response.split()),
            },
        }
        
        idx = len(self.training_data)
        self.training_data.append(sample)
        self.embedding_cache[idx] = self._simple_embedding(user_input)
        
        if server_id:
            if server_id not in self.server_training:
                self.server_training[server_id] = []
            self.server_training[server_id].append(sample)
        
        if len(self.training_data) > self.max_samples:
            self._prune_low_quality()
        
        if len(self.training_data) % 10 == 0:
            self._save_training_data()
        
        return sample

    def _prune_low_quality(self, keep_count=500):
        """Remove lowest quality samples"""
        sorted_samples = sorted(
            self.training_data,
            key=lambda x: (x.get("quality_score", 0), x.get("human_score", 5)),
            reverse=True
        )
        self.training_data = sorted_samples[:keep_count]
        self._build_embeddings()

    def search_training_examples(self, query, max_results=5, min_quality=0.3,
                                  server_id=None, recency_boost=True):
        """
        Search for training examples with embedding similarity
        and adaptive result count based on match quality
        """
        if not self.training_data:
            return []
        
        query_embedding = self._simple_embedding(query)
        if not query_embedding:
            return []
        
        results = []
        target_list = self.server_training.get(server_id, self.training_data) if server_id else self.training_data
        
        for sample in target_list:
            if sample.get("quality_score", 0) < min_quality:
                continue
            
            sample_embedding = self._simple_embedding(sample.get("user_input", ""))
            similarity = self._embedding_similarity(query_embedding, sample_embedding)
            
            if similarity < 0.1:
                continue
            
            # Substring boost
            input_text = sample.get("user_input", "").lower()
            query_lower = query.lower()
            if input_text == query_lower:
                similarity += 0.5
            elif input_text in query_lower or query_lower in input_text:
                similarity += 0.3
            
            # Short query boost
            if len(query) < 20:
                similarity += 0.1
            
            # Recency boost
            if recency_boost:
                try:
                    sample_time = datetime.fromisoformat(sample.get("timestamp", "2000-01-01"))
                    days_ago = (datetime.now() - sample_time).days
                    recency_factor = max(0, 1.0 - (days_ago / 365))
                    similarity *= (1.0 + recency_factor * 0.2)
                except:
                    pass
            
            # Quality + human score weight
            quality = sample.get("quality_score", 0.5)
            human_score = sample.get("human_score", 5.0)
            combined_score = similarity * (0.6 + quality * 0.2 + (human_score / 50))
            
            results.append({
                "input": sample.get("user_input", ""),
                "response": sample.get("bot_response", ""),
                "quality_score": quality,
                "human_score": human_score,
                "timestamp": sample.get("timestamp", ""),
                "similarity": similarity,
                "combined_score": combined_score,
                "server_id": sample.get("server_id"),
            })
        
        # Sort by combined score
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Adaptive selection
        if results:
            top_similarity = results[0].get("similarity", 0)
            if top_similarity > 0.7:
                max_results = min(max_results, 2)
            elif top_similarity > 0.4:
                max_results = min(max_results, 3)
            else:
                max_results = min(max_results, 5)
        
        return results[:max_results]

    def set_human_score(self, user_input, new_score):
        """Set human-likeness score for a training example"""
        for sample in self.training_data:
            if sample.get("user_input") == user_input:
                sample["human_score"] = max(0, min(10, new_score))
                sample["human_verified"] = True
                self._save_training_data()
                return True
        return False

    def get_training_stats(self):
        """Get comprehensive training statistics"""
        if not self.training_data:
            return {"error": "No training data"}
        
        human_scores = [s.get("human_score", 5) for s in self.training_data]
        quality_scores = [s.get("quality_score", 0.5) for s in self.training_data]
        
        server_stats = {}
        for server_id, samples in self.server_training.items():
            h_scores = [s.get("human_score", 5) for s in samples]
            server_stats[server_id] = {
                "count": len(samples),
                "avg_human_score": sum(h_scores) / len(h_scores) if h_scores else 5.0,
            }
        
        return {
            "total_samples": len(self.training_data),
            "quality_stats": {"avg": sum(quality_scores) / len(quality_scores)},
            "human_score_stats": {
                "avg": sum(human_scores) / len(human_scores),
                "min": min(human_scores),
                "max": max(human_scores),
                "distribution": {
                    "very_ai (0-3)": sum(1 for s in human_scores if s <= 3),
                    "neutral (3-7)": sum(1 for s in human_scores if 3 < s <= 7),
                    "very_human (7-10)": sum(1 for s in human_scores if s > 7),
                }
            },
            "server_stats": server_stats,
        }

    def export_for_fine_tuning(self, output_file="training_data_llama.json",
                                min_quality=0.3, min_human_score=4.0):
        """Export training data filtered by quality"""
        filtered = [
            s for s in self.training_data
            if s.get("quality_score", 0) >= min_quality
            and s.get("human_score", 5) >= min_human_score
        ]
        
        export_data = [{
            "instruction": s.get("user_input", ""),
            "input": s.get("user_input", ""),
            "output": s.get("bot_response", ""),
            "quality_score": s.get("quality_score", 1.0),
            "human_score": s.get("human_score", 5.0),
        } for s in filtered]
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported {len(export_data)} samples to {output_file}")
            return export_data
        except Exception as e:
            logger.error(f"Failed to export: {e}")
            return []

    def auto_extract_from_conversations(self, conversations, min_human_score=6.0):
        """Auto-extract good training examples from conversations"""
        extracted = []
        for conv in conversations:
            bot_response = conv.get("bot_response", "")
            human_score = self._calculate_human_score(bot_response)
            
            if human_score >= min_human_score:
                extracted.append({
                    "user_input": conv.get("user_input", ""),
                    "bot_response": bot_response,
                    "human_score": human_score,
                    "auto_extracted": True,
                })
        return extracted
        """Save training data to JSON file"""
        try:
            with open(self.training_file, "w", encoding="utf-8") as f:
                json.dump(self.training_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")

    def add_training_sample(self, user_input, bot_response, conversation_id=None, 
                            user_id=None, quality_score=1.0):
        """
        Add a new training sample to the dataset with quality tracking
        
        Args:
            user_input: The user's message
            bot_response: The bot's response
            conversation_id: Optional conversation ID for grouping
            user_id: The user who sent the message
            quality_score: Quality score (0.0-1.0), auto-calculated if not provided
        """
        # Auto-calculate quality if not provided
        if quality_score == 1.0:
            quality_score = self._calculate_sample_quality(user_input, bot_response)
        
        sample = {
            "user_input": user_input,
            "bot_response": bot_response,
            "conversation_id": str(conversation_id) if conversation_id else None,
            "user_id": str(user_id) if user_id else None,
            "timestamp": datetime.now().isoformat(),
            "quality_score": quality_score,
            "feedback": None,  # To store user corrections
            "metadata": {
                "model": MODEL_NAME,
                "personality": CUSTOM_PERSONALITY,
                "platform": "discord",
                "response_length": len(bot_response),
                "input_length": len(user_input),
            },
        }
        self.training_data.append(sample)
        
        # Cleanup old low-quality samples if we exceed max
        if len(self.training_data) > self.max_samples:
            self._prune_low_quality()
        
        # Save every 10 samples
        if len(self.training_data) % 10 == 0:
            self._save_training_data()

    def _calculate_sample_quality(self, user_input, bot_response):
        """Calculate a quality score for a training sample"""
        score = 1.0
        
        # Penalize very short responses
        if len(bot_response) < 10:
            score -= 0.2
        elif len(bot_response) < 20:
            score -= 0.1
        
        # Penalize empty or error responses
        if bot_response.startswith("[") and bot_response.endswith("]"):
            score -= 0.5  # Likely an error message
        
        # Bonus for longer, thoughtful responses (up to 0.3)
        if 50 < len(bot_response) < 500:
            score += 0.1
        elif 500 < len(bot_response) < 2000:
            score += 0.2
        
        # Penalty for very short inputs with long responses (potential mismatch)
        if len(user_input) < 10 and len(bot_response) > 1000:
            score -= 0.1
        
        return max(0.0, min(1.0, score))

    def _prune_low_quality(self, keep_count=500):
        """Remove lowest quality samples to stay within limits"""
        if len(self.training_data) <= keep_count:
            return
        
        # Sort by quality score
        sorted_samples = sorted(
            self.training_data, 
            key=lambda x: x.get("quality_score", 0), 
            reverse=True
        )
        
        # Keep top samples
        self.training_data = sorted_samples[:keep_count]
        logger.info(f"Pruned training data to {keep_count} highest quality samples")

    def add_feedback(self, conversation_id, correction=None, rating=None, user_id=None):
        """
        Add feedback/correction to a training sample
        
        Args:
            conversation_id: The conversation/message ID
            correction: Optional corrected bot response
            rating: Optional rating (1-5 stars)
            user_id: The user providing feedback
        """
        # Find the sample
        for sample in self.training_data:
            if sample.get("conversation_id") == str(conversation_id):
                if correction:
                    sample["bot_response"] = correction
                    sample["quality_score"] = 1.0  # Reset to max after correction
                    sample["feedback"] = {
                        "type": "correction",
                        "corrected_by": str(user_id),
                        "timestamp": datetime.now().isoformat(),
                    }
                elif rating:
                    # Update quality score based on rating
                    new_score = rating / 5.0  # Convert 1-5 to 0-1
                    sample["quality_score"] = (sample.get("quality_score", 0.5) + new_score) / 2
                    sample["feedback"] = {
                        "type": "rating",
                        "rating": rating,
                        "rated_by": str(user_id),
                        "timestamp": datetime.now().isoformat(),
                    }
                
                self._save_training_data()
                return True
        
        return False

    def get_training_stats(self):
        """Get comprehensive statistics about the training data"""
        if not self.training_data:
            return {
                "total_samples": 0,
                "unique_conversations": 0,
                "avg_quality": 0,
                "last_updated": "Never",
            }
        
        quality_scores = [s.get("quality_score", 1.0) for s in self.training_data]
        
        # Calculate quality distribution
        excellent = sum(1 for q in quality_scores if q >= 0.8)
        good = sum(1 for q in quality_scores if 0.5 <= q < 0.8)
        poor = sum(1 for q in quality_scores if q < 0.5)
        
        # Topic diversity (using first 50 chars of inputs)
        topics = set()
        for sample in self.training_data:
            topic = sample.get("user_input", "")[:50].lower()
            if len(topic) > 10:
                # Extract first meaningful word
                words = re.findall(r"\b\w+\b", topic)
                if words:
                    topics.add(words[0])
        
        return {
            "total_samples": len(self.training_data),
            "unique_conversations": len(set(
                s.get("conversation_id") for s in self.training_data if s.get("conversation_id")
            )),
            "avg_quality": sum(quality_scores) / len(quality_scores),
            "quality_distribution": {
                "excellent": excellent,
                "good": good,
                "needs_work": poor,
            },
            "topic_diversity": len(topics),
            "last_updated": max(
                (s["timestamp"] for s in self.training_data), default="Never"
            ),
        }

    def export_for_fine_tuning(self, output_file="training_data_llama.json", 
                               format="alpaca", max_samples=None):
        """
        Export training data in a format suitable for fine-tuning
        
        Args:
            output_file: Output file path
            format: Format to export (alpaca, sharegpt, raw)
            max_samples: Maximum samples to export (None = all)
        """
        samples = self.training_data[:max_samples] if max_samples else self.training_data
        
        if format == "alpaca":
            # Alpaca format for LLaMA fine-tuning
            export_data = []
            for sample in samples:
                if sample.get("quality_score", 0) >= self.min_quality_threshold:
                    export_data.append({
                        "instruction": sample.get("user_input", ""),
                        "input": "",
                        "output": sample.get("bot_response", ""),
                    })
            
            export_path = output_file if output_file.endswith(".json") else f"{output_file}.json"
            
        elif format == "sharegpt":
            # ShareGPT format for conversation fine-tuning
            export_data = []
            for sample in samples:
                if sample.get("quality_score", 0) >= self.min_quality_threshold:
                    export_data.append({
                        "conversations": [
                            {"from": "human", "value": sample.get("user_input", "")},
                            {"from": "gpt", "value": sample.get("bot_response", "")},
                        ],
                        "quality_score": sample.get("quality_score", 1.0),
                    })
            
            export_path = output_file if output_file.endswith(".json") else f"{output_file}.json"
        
        else:
            # Raw format
            export_data = samples
            export_path = output_file
        
        # Save file
        try:
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported {len(export_data)} samples to {export_path}")
            return export_path
        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
            return None

    def get_samples_by_quality(self, min_quality=0.5, max_samples=50):
        """Get samples filtered by minimum quality score"""
        filtered = [
            s for s in self.training_data 
            if s.get("quality_score", 0) >= min_quality
        ]
        return filtered[:max_samples]

    def search_training_examples(self, query, max_results=5, min_quality=0.5):
        """
        Search for training examples matching the query using text similarity
        
        Args:
            query: The user input to match against
            max_results: Maximum number of results to return
            min_quality: Minimum quality score for samples
            
        Returns:
            List of dicts with 'input', 'response', 'quality_score', 'timestamp'
        """
        if not self.training_data:
            return []
        
        query_words = set(re.findall(r"\b\w+\b", query.lower()))
        if not query_words:
            return []
        
        results = []
        for sample in self.training_data:
            if sample.get("quality_score", 0) < min_quality:
                continue
            
            sample_words = set(re.findall(r"\b\w+\b", sample.get("user_input", "").lower()))
            
            # Calculate Jaccard similarity
            if query_words and sample_words:
                overlap = len(query_words & sample_words)
                union = len(query_words | sample_words)
                similarity = overlap / union if union > 0 else 0
                
                # Boost exact substring matches
                if sample.get("user_input", "").lower() in query.lower() or \
                   query.lower() in sample.get("user_input", "").lower():
                    similarity += 0.3
                
                # Boost short queries (greetings, simple inputs)
                if len(query) < 20:
                    similarity += 0.1
                
                if similarity > 0.1:  # Threshold
                    results.append({
                        "input": sample.get("user_input", ""),
                        "response": sample.get("bot_response", ""),
                        "quality_score": sample.get("quality_score", 1.0),
                        "timestamp": sample.get("timestamp", ""),
                        "similarity": similarity
                    })
        
        # Sort by quality then similarity
        results.sort(key=lambda x: (x["quality_score"], x.get("similarity", 0)), reverse=True)
        
        return results[:max_results]

    def get_diversity_report(self):
        """Generate a report on training data diversity"""
        if not self.training_data:
            return {"error": "No training data available"}
        
        # Topic analysis
        topic_words = {}
        for sample in self.training_data:
            words = re.findall(r"\b\w+\b", sample.get("user_input", "").lower())
            for word in words:
                if len(word) > 3:
                    topic_words[word] = topic_words.get(word, 0) + 1
        
        # Top topics
        top_topics = sorted(topic_words.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Response length distribution
        lengths = [len(s.get("bot_response", "")) for s in self.training_data]
        
        return {
            "unique_topics": len(topic_words),
            "top_topics": dict(top_topics),
            "response_length_stats": {
                "min": min(lengths),
                "max": max(lengths),
                "avg": sum(lengths) / len(lengths),
                "median": sorted(lengths)[len(lengths) // 2],
            },
            "samples_with_feedback": sum(
                1 for s in self.training_data if s.get("feedback")
            ),
        }


def format_for_training(user_input, bot_response):
    """
    Format conversation for training data

    Returns a string formatted as:
    USER: <user_input>
    BOT: <bot_response>

    Args:
        user_input: The user's message
        bot_response: The bot's response

    Returns:
        Formatted string for training
    """
    return f"USER: {user_input}\nBOT: {bot_response}\n"


def save_to_training_file(user_input, bot_response, conversation_id=None):
    """
    Save conversation to training file in human-readable format

    Args:
        user_input: The user's message
        bot_response: The bot's response
        conversation_id: Optional conversation ID for grouping
    """
    formatted = format_for_training(user_input, bot_response)

    # Save to human-readable training file
    training_file = "training_conversations.txt"
    try:
        with open(training_file, "a", encoding="utf-8") as f:
            f.write(f"=== {datetime.now().isoformat()} ===\n")
            f.write(formatted)
            f.write("\n")
    except Exception as e:
        logger.error(f"Failed to save to training file: {e}")


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.WARNING)

_instance_lock_handle = None


def acquire_instance_lock(lock_path=INSTANCE_LOCK_FILE):
    """Acquire a single-instance lock so only one bot process can run."""
    global _instance_lock_handle

    try:
        import fcntl
    except ImportError:
        logger.warning("fcntl unavailable; single-instance lock disabled.")
        return True

    try:
        _instance_lock_handle = open(lock_path, "a+", encoding="utf-8")
        fcntl.flock(
            _instance_lock_handle.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
        _instance_lock_handle.seek(0)
        _instance_lock_handle.truncate()
        _instance_lock_handle.write(str(os.getpid()))
        _instance_lock_handle.flush()
        return True
    except BlockingIOError:
        logger.error(f"Another bot instance is already running (lock: {lock_path})")
        return False
    except Exception as e:
        logger.error(f"Failed to acquire instance lock: {e}")
        return False


class MemoryEngine:
    """Enhanced adaptive memory system with TF-IDF, user memory, and temporal weighting"""

    def __init__(self, memory_file=MEMORY_FILE):
        self.memory_file = memory_file
        self.memory_data = self._load_memory()
        
        # TF-IDF style indexing
        self.keyword_index = defaultdict(list)
        self.keyword_counts = defaultdict(int)  # Global term frequency
        self.total_conversations = 0
        
        # User-specific memory
        self.user_memory = defaultdict(dict)  # user_id -> {preferences, facts, style}
        
        # Stop words for keyword extraction
        self.stop_words = set([
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
            'not', 'only', 'just', 'also', 'very', 'too', 'quite', 'rather',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'what', 'which', 'who', 'whom', 'whose', 'where',
            'when', 'why', 'how', 'all', 'each', 'every', 'some', 'any', 'no',
            'your', 'my', 'his', 'her', 'its', 'our', 'their', 'me', 'him',
            'us', 'them', 'myself', 'yourself', 'himself', 'herself', 'itself',
            'ourselves', 'themselves', 'okay', 'yeah', 'yes', 'like', 'well',
            'going', 'gonna', 'got', 'get', 'gotta', 'dont', 'dont', 'cant',
            'wont', 'didnt', 'isnt', 'arent', 'wasnt', 'werent', 'havent',
            'hasnt', 'hadnt', 'couldnt', 'wouldnt', 'shouldnt', 'yeah', 'yep',
            'nah', 'nope', 'umm', 'uhh', 'uh', 'um', 'hmm', 'hm', 'oh', 'ah',
            'actually', 'basically', 'literally', 'honestly', 'maybe', 'perhaps',
        ])

        # Build indices on load
        self._build_all_indices()

    def _load_memory(self):
        """Load memory from JSON file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load memory: {e}")
                return {
                    "conversations": [], 
                    "facts": {}, 
                    "keywords": {},
                    "user_data": {}
                }
        return {
            "conversations": [], 
            "facts": {}, 
            "keywords": {},
            "user_data": {}
        }

    def _save_memory(self):
        """Save memory to JSON file"""
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.memory_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def _build_all_indices(self):
        """Build all indices (keywords, user data, TF-IDF)"""
        conversations = self.memory_data.get("conversations", [])
        self.total_conversations = len(conversations)
        
        # Reset indices
        self.keyword_index.clear()
        self.keyword_counts.clear()
        
        for conv_index, conv in enumerate(conversations):
            # Extract and weight keywords
            keywords = self._extract_keywords(
                conv.get("user_input", "") + " " + conv.get("bot_response", "")
            )
            
            # Count term frequency
            unique_keywords = set(keywords)
            for keyword in unique_keywords:
                self.keyword_counts[keyword.lower()] += 1
            
            # Store conversation reference by stable index position.
            for keyword in keywords:
                self.keyword_index[keyword.lower()].append(conv_index)
        
        # Load user data
        user_data = self.memory_data.get("user_data", {})
        for user_id, data in user_data.items():
            self.user_memory[int(user_id)] = data

    def _extract_keywords(self, text):
        """Extract meaningful keywords (TF-IDF style)"""
        words = re.findall(r"\b\w+\b", text.lower())
        
        # Filter stop words and short words
        keywords = [
            word for word in words 
            if len(word) > 2 and word not in self.stop_words and word.isalpha()
        ]
        
        return keywords

    def _calculate_tfidf_score(self, keyword, conv):
        """Calculate TF-IDF style score for a keyword/conversation match"""
        # Term frequency in this conversation
        conv_text = (conv.get("user_input", "") + " " + conv.get("bot_response", "")).lower()
        conv_keywords = self._extract_keywords(conv_text)
        tf = conv_keywords.count(keyword) / len(conv_keywords) if conv_keywords else 0
        
        # Inverse document frequency
        df = self.keyword_counts.get(keyword, 1)
        idf = np.log(self.total_conversations / df) if self.total_conversations > 0 else 0
        
        return tf * idf

    def _calculate_recency_score(self, conv):
        """Calculate recency score (more recent = higher score)"""
        try:
            timestamp = datetime.fromisoformat(conv.get("timestamp", "2000-01-01"))
            days_ago = (datetime.now() - timestamp).days
            
            # Exponential decay: very recent (0-1 days) = 1.0, 30 days = ~0.35
            return np.exp(-0.05 * days_ago)
        except:
            return 0.1

    def _calculate_relevance_score(self, text, conv):
        """Calculate overall relevance score combining TF-IDF and recency"""
        keywords = self._extract_keywords(text)
        
        tfidf_score = 0
        for keyword in keywords:
            tfidf_score += self._calculate_tfidf_score(keyword, conv)
        
        # Normalize TF-IDF
        tfidf_score /= len(keywords) if keywords else 1
        
        recency_score = self._calculate_recency_score(conv)
        
        # Combined score: 70% relevance, 30% recency
        return (0.7 * tfidf_score) + (0.3 * recency_score)

    def search_relevant_memory(self, text, max_results=5, user_id=None, min_score=None):
        """Search for relevant past conversations with weighted scoring"""
        if min_score is None:
            min_score = MEMORY_MIN_RELEVANCE

        keywords = self._extract_keywords(text)
        conversations = self.memory_data.get("conversations", [])
        
        # Get candidates by keyword index (using position indices)
        candidate_indices = set()
        for keyword in keywords:
            if keyword in self.keyword_index:
                candidate_indices.update(self.keyword_index[keyword])
        
        if not candidate_indices:
            return []
        
        # Get conversations by index and score them
        candidate_convos = []
        for idx in candidate_indices:
            if idx < len(conversations):
                conv = conversations[idx]
                # Filter by user_id if specified
                if user_id is not None and conv.get("user_id") != str(user_id):
                    continue
                candidate_convos.append(conv)
        
        if not candidate_convos:
            return []
        
        # Score and sort candidates
        scored_convos = []
        for conv in candidate_convos:
            score = self._calculate_relevance_score(text, conv)
            scored_convos.append((score, conv))

        # Drop weak matches to reduce generic/irrelevant memory bleed.
        scored_convos = [
            (score, conv) for score, conv in scored_convos if score >= min_score
        ]
        if not scored_convos:
            return []
        
        # Sort by score descending
        scored_convos.sort(key=lambda x: x[0], reverse=True)
        
        # Return top results
        return [conv for score, conv in scored_convos[:max_results]]

    def add_conversation(self, user_input, bot_response, context="", user_id=None):
        """Add new conversation to memory with full indexing"""
        conversation = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "context": context,
            "user_id": str(user_id) if user_id else None,
            "keywords": self._extract_keywords(user_input + " " + bot_response),
            "quality_score": 1.0,  # Initial quality score (can be adjusted via feedback)
        }

        self.memory_data["conversations"].append(conversation)
        self.total_conversations += 1
        
        # Get the new conversation's index
        conv_index = len(self.memory_data["conversations"]) - 1
        
        # Update indices
        keywords = conversation["keywords"]
        for keyword in keywords:
            self.keyword_counts[keyword.lower()] = self.keyword_counts.get(keyword.lower(), 0) + 1
            self.keyword_index[keyword.lower()].append(conv_index)
        
        # Update user memory if applicable
        if user_id:
            self._update_user_memory(user_id, user_input, bot_response)
        
        # Memory consolidation check (if too many memories)
        if len(self.memory_data["conversations"]) > 1000:
            self._consolidate_memories()
        
        self._save_memory()

    def _update_user_memory(self, user_id, user_input, bot_response):
        """Update user-specific memory and preferences"""
        if "user_data" not in self.memory_data:
            self.memory_data["user_data"] = {}
        
        user_id_str = str(user_id)
        if user_id_str not in self.memory_data["user_data"]:
            self.memory_data["user_data"][user_id_str] = {
                "preferences": {},
                "style_hints": [],
                "conversation_count": 0,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
            }
        
        user_data = self.memory_data["user_data"][user_id_str]
        user_data["conversation_count"] += 1
        user_data["last_seen"] = datetime.now().isoformat()
        
        # Extract potential style hints from user input
        if user_id not in self.user_memory:
            self.user_memory[user_id] = user_data
        
        self._save_memory()

    def _consolidate_memories(self):
        """Consolidate old memories to save space while preserving information"""
        conversations = self.memory_data["conversations"]
        
        if len(conversations) <= 800:  # Target size
            return
        
        # Sort by relevance/recency
        scored = []
        for i, conv in enumerate(conversations):
            score = self._calculate_relevance_score(
                conv.get("user_input", "") + " " + conv.get("bot_response", ""), 
                conv
            )
            scored.append((score, i, conv))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top 800, consolidate the rest
        keep_indices = {idx for _, idx, _ in scored[:800]}
        
        consolidated = []
        for i, conv in enumerate(conversations):
            if i in keep_indices:
                consolidated.append(conv)
            else:
                # Create summary of old conversation
                summary = {
                    "timestamp": conv.get("timestamp"),
                    "summary": f"User asked about: {conv.get('user_input', '')[:100]}",
                    "response": conv.get('bot_response', '')[:100],
                    "consolidated": True,
                }
                # Keep summary (limit to 50)
                if len(consolidated) < 850:
                    consolidated.append(summary)
        
        self.memory_data["conversations"] = consolidated
        logger.info(f"Consolidated memories: {len(conversations)} -> {len(consolidated)}")

    def learn_fact(self, fact_key, fact_value, user_id=None, confidence=1.0):
        """Store new facts with optional user association and confidence"""
        fact_entry = {
            "value": fact_value,
            "timestamp": datetime.now().isoformat(),
            "user_id": str(user_id) if user_id else None,
            "confidence": confidence,
            "times_used": 0,
        }
        
        self.memory_data["facts"][fact_key] = fact_entry
        self._save_memory()

    def get_relevant_facts(self, text, user_id=None):
        """Get facts relevant to the current text with scoring"""
        relevant_facts = {}
        keywords = self._extract_keywords(text)
        
        for fact_key, fact_data in self.memory_data.get("facts", {}).items():
            # Check user-specific facts first
            if user_id and fact_data.get("user_id") != str(user_id):
                continue
            
            # Calculate relevance
            relevance = 0
            for keyword in keywords:
                if keyword in fact_key.lower():
                    relevance += 1
                if keyword in str(fact_data.get("value", "")).lower():
                    relevance += 0.5
            
            if relevance > 0:
                # Apply confidence and usage boost
                confidence = fact_data.get("confidence", 1.0)
                usage_count = fact_data.get("times_used", 0)
                usage_boost = min(0.2, usage_count * 0.02)  # Max 20% boost
                
                fact_data["relevance_score"] = relevance * (confidence + usage_boost)
                relevant_facts[fact_key] = fact_data
        
        # Sort by relevance
        sorted_facts = sorted(
            relevant_facts.items(), 
            key=lambda x: x[1].get("relevance_score", 0), 
            reverse=True
        )
        
        return dict(sorted_facts)

    def get_user_memory(self, user_id):
        """Get all stored memory for a specific user"""
        user_id_str = str(user_id)
        return self.memory_data.get("user_data", {}).get(user_id_str, {})

    def add_user_preference(self, user_id, key, value):
        """Add or update a user preference"""
        if "user_data" not in self.memory_data:
            self.memory_data["user_data"] = {}
        
        user_id_str = str(user_id)
        if user_id_str not in self.memory_data["user_data"]:
            self.memory_data["user_data"][user_id_str] = {
                "preferences": {},
                "style_hints": [],
                "conversation_count": 0,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
            }
        
        self.memory_data["user_data"][user_id_str]["preferences"][key] = value
        self._save_memory()

    def get_memory_stats(self):
        """Get comprehensive memory statistics"""
        conversations = self.memory_data.get("conversations", [])
        facts = self.memory_data.get("facts", {})
        user_data = self.memory_data.get("user_data", {})
        
        # Calculate date range
        timestamps = []
        for conv in conversations:
            try:
                ts = datetime.fromisoformat(conv.get("timestamp", ""))
                timestamps.append(ts)
            except:
                pass
        
        stats = {
            "total_conversations": len(conversations),
            "total_facts": len(facts),
            "total_users_tracked": len(user_data),
            "unique_keywords": len(self.keyword_counts),
            "date_range": {
                "first": min(timestamps).isoformat() if timestamps else "Never",
                "last": max(timestamps).isoformat() if timestamps else "Never",
            }
        }
        
        # User with most conversations
        if user_data:
            most_active = max(user_data.items(), key=lambda x: x[1].get("conversation_count", 0))
            stats["most_active_user"] = {
                "user_id": most_active[0],
                "conversations": most_active[1].get("conversation_count", 0)
            }
        
        return stats

class AdaptivePersonality:
    """Dynamic personality system that learns from user interactions and feedback"""

    def __init__(self, base_personality="", memory_engine=None):
        self.base_personality = base_personality
        self.memory_engine = memory_engine
        
        # Personality traits (0.0 - 1.0 scale)
        self.traits = {
            "formality": 0.5,        # 0 = casual, 1 = formal
            "verbosity": 0.5,        # 0 = brief, 1 = detailed
            "humor": 0.5,            # 0 = serious, 1 = humorous
            "helpfulness": 0.8,      # How helpful to be
            "creativity": 0.5,       # 0 = literal, 1 = creative
            "friendliness": 0.7,     # 0 = cold, 1 = warm
            "patience": 0.7,         # How patient to be
            "confidence": 0.6,       # 0 = uncertain, 1 = confident
            "sassiness": SASS_LEVEL if SASS_MODE else 0.2,  # 0 = polite, 1 = snarky
        }
        if ROAST_MODE:
            self.traits["sassiness"] = max(self.traits["sassiness"], ROAST_LEVEL)
        
        # Learned style patterns
        self.style_patterns = {
            "preferred_phrases": set(),      # Phrases user responds well to
            "avoid_phrases": set(),          # Phrases user responds poorly to
            "response_patterns": [],         # Successful response patterns
            "corrections": [],               # User corrections applied
        }
        
        # User-specific adaptations (user_id -> personality adjustments)
        self.user_adaptations = defaultdict(lambda: {
            "traits": self.traits.copy(),
            "style_hints": [],
            "conversation_count": 0,
        })
        
        # Learning configuration
        self.learning_rate = 0.05  # How fast to adapt
        self.min_confidence = 0.3  # Minimum confidence to apply adaptation
        
        # Response history for pattern learning
        self.response_history = []  # (user_id, input, response, success_flag)

    def set_base_personality(self, personality_text):
        """Set the base personality from which to adapt"""
        self.base_personality = personality_text

    def _extract_style_features(self, text):
        """Extract style features from text"""
        features = {}
        
        # Length
        words = text.split()
        features["word_count"] = len(words)
        features["avg_word_length"] = sum(len(w) for w in words) / len(words) if words else 0
        
        # Punctuation
        features["exclamation_count"] = text.count("!")
        features["question_count"] = text.count("?")
        features["ellipsis_count"] = text.count("...")
        
        # Capitalization (indicates shouting/excitement)
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features["upper_ratio"] = upper_ratio
        
        # Emoji/emote usage
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE)
        features["emoji_count"] = len(emoji_pattern.findall(text))
        
        return features

    def learn_from_interaction(self, user_id, user_input, bot_response, user_reaction=None):
        """
        Learn from a conversation interaction
        
        Args:
            user_id: The user's ID
            user_input: What the user said
            bot_response: What the bot responded
            user_reaction: Optional reaction (positive=True, negative=False)
        """
        if user_id is None:
            return
        
        # Update user adaptation tracking
        user_adapt = self.user_adaptations[user_id]
        user_adapt["conversation_count"] += 1
        
        # Extract style features
        input_features = self._extract_style_features(user_input)
        response_features = self._extract_style_features(bot_response)
        
        # Learn from user reaction if provided
        if user_reaction is not None:
            if user_reaction:
                # Positive reaction - reinforce current style
                if user_input_features := self._extract_style_features(user_input):
                    if user_input_features["word_count"] > 3:
                        user_adapt["style_hints"].append({
                            "type": "input_style",
                            "features": input_features,
                            "timestamp": datetime.now().isoformat(),
                        })
            else:
                # Negative reaction - adjust traits away from current style
                self._adjust_for_negative_reaction(user_id, input_features, response_features)
        
        # Store in response history
        self.response_history.append({
            "user_id": str(user_id),
            "input": user_input,
            "response": bot_response,
            "reaction": user_reaction,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Keep history limited
        if len(self.response_history) > 1000:
            self.response_history = self.response_history[-500:]

    def _adjust_for_negative_reaction(self, user_id, input_features, response_features):
        """Adjust personality based on negative user reaction"""
        user_adapt = self.user_adaptations[user_id]
        traits = user_adapt["traits"]
        
        # If response was too long, reduce verbosity
        if response_features.get("word_count", 0) > 50:
            traits["verbosity"] = max(0.1, traits["verbosity"] - self.learning_rate)
        
        # If too casual, increase formality
        if response_features.get("exclamation_count", 0) > 2:
            traits["formality"] = min(1.0, traits["formality"] + self.learning_rate)
        
        # If too formal, decrease formality
        if input_features.get("upper_ratio", 0) < 0.1 and response_features.get("upper_ratio", 0) > 0.3:
            traits["formality"] = max(0.1, traits["formality"] - self.learning_rate)

    def apply_correction(self, user_id, original_response, corrected_response):
        """Learn from a user correction"""
        correction = {
            "original": original_response,
            "corrected": corrected_response,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.style_patterns["corrections"].append(correction)
        
        # Update user adaptation
        user_adapt = self.user_adaptations[user_id]
        user_adapt["style_hints"].append({
            "type": "correction",
            "correction": correction,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Adjust traits based on correction direction
        orig_len = len(original_response)
        corr_len = len(corrected_response)
        
        if corr_len < orig_len * 0.7:
            user_adapt["traits"]["verbosity"] = max(0.1, user_adapt["traits"]["verbosity"] - 0.1)
        elif corr_len > orig_len * 1.3:
            user_adapt["traits"]["verbosity"] = min(1.0, user_adapt["traits"]["verbosity"] + 0.1)

    def get_personality_for_user(self, user_id=None):
        """Get the personality configuration for a user (or default)"""
        if user_id is not None and user_id in self.user_adaptations:
            user_adapt = self.user_adaptations[user_id]
            if user_adapt["conversation_count"] >= 5:  # Only use if enough data
                return user_adapt["traits"]
        
        return self.traits.copy()

    def generate_prompt_suffix(self, user_id=None, context=""):
        """Generate personality-specific prompt additions"""
        traits = self.get_personality_for_user(user_id)
        
        suffixes = []
        
        # Formality
        if traits["formality"] < 0.3:
            suffixes.append("Keep responses casual and relaxed.")
        elif traits["formality"] > 0.7:
            suffixes.append("Maintain a formal, professional tone.")
        
        # Verbosity
        if traits["verbosity"] < 0.3:
            suffixes.append("Be very brief and to the point.")
        elif traits["verbosity"] > 0.7:
            suffixes.append("Provide detailed explanations when helpful.")
        
        # Humor
        if traits["humor"] > 0.7:
            suffixes.append("Feel free to add light humor when appropriate.")
        elif traits["humor"] < 0.3:
            suffixes.append("Keep responses serious and focused.")
        
        # Confidence
        if traits["confidence"] < 0.3:
            suffixes.append("It's okay to express uncertainty when appropriate.")
        elif traits["confidence"] > 0.7:
            suffixes.append("Be confident in your responses.")

        # Sassiness
        sassiness = traits.get("sassiness", 0.0)
        if sassiness >= 0.75:
            suffixes.append(
                "Use a sassy, witty tone with light snark; avoid corporate politeness."
            )
        elif sassiness >= 0.45:
            suffixes.append(
                "Keep a mildly sassy, playful tone when it fits the moment."
            )
        elif sassiness <= 0.2:
            suffixes.append("Keep tone straightforward with minimal snark.")
        
        if suffixes:
            return "\n".join(suffixes)
        return ""

    def get_personality_stats(self, user_id=None):
        """Get personality statistics"""
        if user_id is not None and user_id in self.user_adaptations:
            user_adapt = self.user_adaptations[user_id]
            return {
                "user_id": str(user_id),
                "conversations": user_adapt["conversation_count"],
                "traits": user_adapt["traits"],
            }
        
        return {
            "default_traits": self.traits,
            "users_adapted": len(self.user_adaptations),
            "corrections_learned": len(self.style_patterns["corrections"]),
        }

    def save_adaptations(self, file_path="personality_adaptations.json"):
        """Save personality adaptations to file"""
        data = {
            "traits": self.traits,
            "user_adaptations": dict(self.user_adaptations),
            "style_patterns": {
                "preferred_phrases": list(self.style_patterns["preferred_phrases"]),
                "avoid_phrases": list(self.style_patterns["avoid_phrases"]),
                "corrections": self.style_patterns["corrections"][-100:],  # Last 100
            }
        }
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved personality adaptations to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save personality adaptations: {e}")

    def load_adaptations(self, file_path="personality_adaptations.json"):
        """Load personality adaptations from file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "traits" in data:
                self.traits = data["traits"]
            if "user_adaptations" in data:
                for uid, adapt in data["user_adaptations"].items():
                    self.user_adaptations[int(uid)] = adapt
            if "style_patterns" in data:
                self.style_patterns["preferred_phrases"] = set(
                    data["style_patterns"].get("preferred_phrases", [])
                )
                self.style_patterns["avoid_phrases"] = set(
                    data["style_patterns"].get("avoid_phrases", [])
                )
                self.style_patterns["corrections"] = data["style_patterns"].get("corrections", [])
            
            logger.info(f"Loaded personality adaptations from {file_path}")
        except FileNotFoundError:
            logger.info("No existing personality adaptations found")
        except Exception as e:
            logger.error(f"Failed to load personality adaptations: {e}")


class AudioProcessor:
    """Handles speech-to-text and text-to-speech with enhanced recognition"""

    def __init__(self):
        self.recognizer = sr.Recognizer() if SPEECH_RECOGNITION_AVAILABLE else None
        self.microphone = None
        self.tts_engine = None
        self.pyttsx3_engine = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.whisper_recognition_available = bool(
            self.recognizer is not None and hasattr(self.recognizer, "recognize_whisper")
        )
        self.sphinx_recognition_available = False
        
        # Check for Sphinx (offline fallback)
        if self.recognizer is not None:
            try:
                # Test if pocketsphinx backend is available.
                import pocketsphinx  # type: ignore  # noqa: F401
                self.sphinx_recognition_available = True
            except:
                pass
            
        if self.recognizer is not None:
            # Optimized settings for Discord voice audio
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = int(os.getenv("VOICE_ENERGY_THRESHOLD", "200"))
            self.recognizer.pause_threshold = float(os.getenv("VOICE_PAUSE_THRESHOLD", "0.5"))
            self.recognizer.non_speaking_duration = float(os.getenv("VOICE_NON_SPEAKING_DURATION", "0.15"))
            # Adjust for faster response
            self.recognizer.operation_timeout = float(os.getenv("STT_TIMEOUT", "10"))
        
        # Recognition history for confidence voting
        self.recent_transcripts = deque(maxlen=5)
        
        # Initialize pyttsx3 engine if available
        global PYTTSX3_AVAILABLE
        if PYTTSX3_AVAILABLE:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                voices = self.pyttsx3_engine.getProperty("voices")
                if voices:
                    for voice in voices:
                        if "english" in voice.name.lower() or "en" in voice.id.lower():
                            self.pyttsx3_engine.setProperty("voice", voice.id)
                            break
            except Exception as e:
                print(f"pyttsx3 initialization failed: {e}")
                self.pyttsx3_engine = None
                PYTTSX3_AVAILABLE = False

    def _resample_mono_pcm16(self, mono_samples, source_rate, target_rate=16000):
        """Resample mono int16 PCM to target rate."""
        if source_rate == target_rate:
            return mono_samples

        # Fast path for Discord voice (48kHz -> 16kHz)
        if source_rate > target_rate and source_rate % target_rate == 0:
            step = source_rate // target_rate
            return mono_samples[::step]

        duration = len(mono_samples) / float(source_rate)
        if duration <= 0:
            return mono_samples

        source_positions = np.linspace(0, duration, num=len(mono_samples), endpoint=False)
        target_length = max(1, int(duration * target_rate))
        target_positions = np.linspace(0, duration, num=target_length, endpoint=False)
        resampled = np.interp(
            target_positions,
            source_positions,
            mono_samples.astype(np.float32),
        )
        return np.clip(resampled, -32768, 32767).astype(np.int16)

    def _to_mono_16k_pcm(self, audio_data, apply_enhancement=True):
        """Normalize Discord/WAV audio bytes to mono 16-bit PCM at 16kHz."""
        import io
        import wave

        if not isinstance(audio_data, (bytes, bytearray)):
            return None, 16000, 2

        source_bytes = bytes(audio_data)
        source_rate = 48000
        source_channels = 2
        source_width = 2

        # If WAV input, extract metadata and raw frames.
        if len(source_bytes) >= 12 and source_bytes[:4] == b"RIFF":
            with wave.open(io.BytesIO(source_bytes), "rb") as wav_file:
                source_channels = wav_file.getnchannels()
                source_width = wav_file.getsampwidth()
                source_rate = wav_file.getframerate()
                source_bytes = wav_file.readframes(wav_file.getnframes())

        if source_width != 2:
            logger.warning(
                f"Unsupported audio sample width for STT: {source_width * 8} bits"
            )
            return None, 16000, 2

        samples = np.frombuffer(source_bytes, dtype=np.int16)
        if samples.size == 0:
            return b"", 16000, 2

        if source_channels > 1:
            frames = samples.size // source_channels
            if frames == 0:
                return b"", 16000, 2
            samples = (
                samples[: frames * source_channels]
                .reshape(frames, source_channels)
                .mean(axis=1)
                .astype(np.int16)
            )

        resampled = self._resample_mono_pcm16(samples, source_rate, 16000)
        if apply_enhancement:
            resampled = self._enhance_speech_pcm(resampled, 16000)
        return resampled.tobytes(), 16000, 2

    def _enhance_speech_pcm(self, samples, sample_rate=16000):
        """Trim silence/noise and normalize gain for better STT accuracy."""
        if samples is None or len(samples) == 0:
            return np.array([], dtype=np.int16)

        speech = samples.astype(np.float32)
        speech -= np.mean(speech)
        abs_speech = np.abs(speech)

        if abs_speech.size == 0:
            return np.array([], dtype=np.int16)

        # Estimate dynamic noise floor and trim to likely speech region.
        noise_floor = float(np.percentile(abs_speech, 35))
        speech_threshold = max(120.0, noise_floor * 2.2)
        voiced = np.flatnonzero(abs_speech > speech_threshold)

        if voiced.size > 0:
            pad = int(sample_rate * 0.08)  # Keep a small lead/trail.
            start = max(0, int(voiced[0]) - pad)
            end = min(len(speech), int(voiced[-1]) + pad)
            speech = speech[start:end]
            abs_speech = np.abs(speech)

        if speech.size == 0:
            return np.array([], dtype=np.int16)

        peak = float(np.max(abs_speech))
        if peak > 0:
            # Normalize to a speech-friendly target without hard clipping.
            target_peak = 12000.0
            gain = min(8.0, target_peak / peak)
            speech *= gain

        return np.clip(speech, -32768, 32767).astype(np.int16)

    def _boost_pcm_gain(self, pcm_data, gain):
        """Apply additional gain to PCM bytes for fallback STT attempts."""
        if not pcm_data:
            return pcm_data
        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        boosted = np.clip(samples * float(gain), -32768, 32767).astype(np.int16)
        return boosted.tobytes()

    def _transcript_quality(self, text):
        """Simple heuristic to score transcript quality."""
        if not text:
            return 0.0

        words = re.findall(r"[a-zA-Z']+", text.lower())
        if not words:
            return 0.0

        filler = {
            "uh", "um", "ah", "er", "hmm", "mm", "uhh", "umm", "eh",
            "huh", "hmmm", "hmm", "a",
        }
        filler_count = sum(1 for w in words if w in filler)
        avg_word_len = sum(len(w) for w in words) / len(words)

        score = 0.0
        score += min(1.8, len(words) / 3.0)
        score += min(1.0, avg_word_len / 4.0)
        score -= min(1.2, filler_count * 0.45)

        # Tiny, filler-only phrases are usually garbage.
        if len(words) <= 2 and filler_count > 0:
            score -= 0.8

        return max(0.0, score)

    def _recognize_google_best(self, audio_obj):
        """Return best transcript candidate from Google's alternatives."""
        result = self.recognizer.recognize_google(
            audio_obj, language=VOICE_LANGUAGE, show_all=True
        )

        if isinstance(result, dict):
            alternatives = result.get("alternative", []) or []
            if alternatives:
                scored = []
                for alt in alternatives:
                    transcript = (alt.get("transcript") or "").strip()
                    if not transcript:
                        continue
                    confidence = float(alt.get("confidence", 0.0) or 0.0)
                    # Prefer confidence first, then fuller phrases.
                    score = confidence + min(0.35, len(transcript) / 120.0)
                    scored.append((score, transcript))
                if scored:
                    scored.sort(key=lambda item: item[0], reverse=True)
                    return scored[0][1]

        # Fallback when show_all has no usable alternatives.
        plain = self.recognizer.recognize_google(audio_obj, language=VOICE_LANGUAGE)
        return plain.strip()

    async def speech_to_text(self, audio_data):
        """Convert audio to text using enhanced multi-engine recognition"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            return "[Voice input not available - use !chat command instead]"

        loop = asyncio.get_running_loop()

        def do_recognition():
            try:
                # Process audio with multiple enhancement strategies
                enhanced_pcm, sample_rate, sample_width = self._to_mono_16k_pcm(
                    audio_data, apply_enhancement=True
                )
                raw_pcm, _, _ = self._to_mono_16k_pcm(
                    audio_data, apply_enhancement=False
                )

                if enhanced_pcm is None:
                    return "[Invalid audio format]"

                pcm_samples = np.frombuffer(enhanced_pcm, dtype=np.int16)
                duration_sec = len(enhanced_pcm) / (sample_rate * sample_width)
                
                if duration_sec < 0.3:
                    return "[Speech too short]"
                
                # Check audio quality
                rms = float(np.sqrt(np.mean(np.square(pcm_samples.astype(np.float32)))))
                if rms < VOICE_MIN_RMS * 0.6:
                    return "[Speech too quiet]"

                results = []
                google_conf = 0.0
                
                # Try Google Speech Recognition first
                try:
                    audio = sr.AudioData(enhanced_pcm, sample_rate, sample_width)
                    google_result = self._recognize_google_with_confidence(audio)
                    if google_result:
                        results.append(google_result)
                        google_conf = float(google_result.get("confidence", 0.0) or 0.0)
                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    logger.debug(f"Google STT error: {e}")

                # Try Whisper if available
                if (
                    STT_WHISPER_FALLBACK
                    and self.whisper_recognition_available
                    and raw_pcm
                    and (not results or google_conf < STT_WHISPER_CONFIDENCE_GATE)
                ):
                    try:
                        whisper_audio = sr.AudioData(raw_pcm, sample_rate, sample_width)
                        whisper_text = self.recognizer.recognize_whisper(
                            whisper_audio,
                            model=STT_WHISPER_MODEL,
                            language="english",
                        )
                        if whisper_text:
                            results.append({
                                'text': whisper_text.strip(),
                                'confidence': 0.85,
                                'engine': 'whisper'
                            })
                    except Exception as e:
                        logger.debug(f"Whisper STT error: {e}")

                # Try Sphinx (offline) if available
                if self.sphinx_recognition_available and (not results or google_conf < 0.55):
                    try:
                        sphinx_audio = sr.AudioData(enhanced_pcm, sample_rate, sample_width)
                        sphinx_text = self.recognizer.recognize_sphinx(
                            sphinx_audio,
                            language="en-US"
                        )
                        if sphinx_text:
                            results.append({
                                'text': sphinx_text.strip(),
                                'confidence': 0.5,
                                'engine': 'sphinx'
                            })
                    except Exception:
                        pass  # Sphinx often fails, that's OK

                # Try with boosted audio if no results yet
                if not results or google_conf < 0.55:
                    boosted_pcm = self._boost_pcm_gain(raw_pcm, 2.5)
                    boosted_samples = np.frombuffer(boosted_pcm, dtype=np.int16)
                    boosted_rms = float(np.sqrt(np.mean(np.square(boosted_samples.astype(np.float32)))))
                    
                    if boosted_rms > VOICE_MIN_RMS * 0.4:
                        try:
                            boosted_audio = sr.AudioData(boosted_pcm, sample_rate, sample_width)
                            boosted_result = self._recognize_google_with_confidence(boosted_audio)
                            if boosted_result:
                                results.append(boosted_result)
                        except:
                            pass

                if not results:
                    return "[Speech not understood]"

                # Combine results using confidence voting
                best_result = self._combine_recognition_results(results)
                best_text = (best_result.get('text') or '').strip()
                best_conf = float(best_result.get('confidence', 0.0) or 0.0)
                best_quality = self._transcript_quality(best_text)
                if not best_text or best_conf < 0.56 or best_quality < 0.55:
                    return "[Speech not understood]"

                # Reject suspiciously weak 1-2 word phrases unless they are common greetings.
                cleaned = " ".join(best_text.lower().split())
                short_allow = {
                    "yo", "sup", "hey", "hello", "hi", "ok", "okay",
                    "yeah", "no", "what's up", "whats up",
                }
                if (
                    len(cleaned.split()) <= 2
                    and cleaned not in short_allow
                    and best_quality < 0.75
                ):
                    return "[Speech not understood]"
                
                # Store for temporal consistency checking
                self.recent_transcripts.append(best_text)
                
                return best_text

            except sr.UnknownValueError:
                return "[Speech not understood]"
            except sr.RequestError as e:
                logger.error(f"Speech recognition request failed: {e}")
                return "[Speech recognition service unavailable]"
            except Exception as e:
                logger.error(f"Speech to text error: {e}")
                return "[Speech recognition error]"

        return await loop.run_in_executor(self.executor, do_recognition)
    
    def _recognize_google_with_confidence(self, audio_obj):
        """Recognize with confidence score from Google's alternatives"""
        try:
            result = self.recognizer.recognize_google(
                audio_obj, language=VOICE_LANGUAGE, show_all=True
            )
            
            if isinstance(result, dict) and result.get("alternative"):
                best_candidate = None
                best_score = -1.0
                for alt in result["alternative"]:
                    transcript = (alt.get("transcript") or "").strip()
                    if not transcript:
                        continue
                    confidence = float(alt.get("confidence", 0.65) or 0.65)
                    quality = self._transcript_quality(transcript)
                    score = confidence + (quality * 0.25)
                    if score > best_score:
                        best_score = score
                        best_candidate = (transcript, confidence, quality)

                if best_candidate is not None:
                    transcript, confidence, quality = best_candidate
                    return {
                        'text': transcript,
                        'confidence': min(1.0, confidence + (quality * 0.15)),
                        'engine': 'google'
                    }
            elif isinstance(result, str):
                quality = self._transcript_quality(result.strip())
                return {
                    'text': result.strip(),
                    'confidence': min(0.95, 0.65 + (quality * 0.15)),
                    'engine': 'google'
                }
        except sr.UnknownValueError:
            pass
        except Exception as e:
            logger.debug(f"Google recognition error: {e}")
        return None
    
    def _combine_recognition_results(self, results):
        """Combine multiple recognition results using voting and confidence"""
        if not results:
            return {'text': '', 'confidence': 0}
        
        filtered_results = []
        for result in results:
            text = (result.get('text') or '').strip()
            if not text:
                continue
            quality = self._transcript_quality(text)
            if quality < 0.45:
                continue
            adjusted = dict(result)
            adjusted['confidence'] = min(
                1.0, float(result.get('confidence', 0.5)) + (quality * 0.12)
            )
            adjusted['_quality'] = quality
            filtered_results.append(adjusted)
        
        if not filtered_results:
            return {'text': '', 'confidence': 0}
        
        if len(filtered_results) == 1:
            one = filtered_results[0]
            return {'text': one['text'], 'confidence': one['confidence']}
        
        # Group by similar text (simple fuzzy matching)
        text_scores = {}
        text_examples = {}
        for result in filtered_results:
            text = result['text'].lower()
            # Normalize text for comparison
            normalized = ' '.join(text.split())
            
            # Find best matching existing entry
            best_match = None
            best_score = 0
            for existing_text in text_scores.keys():
                similarity = self._text_similarity(normalized, existing_text)
                if similarity > 0.7 and similarity > best_score:
                    best_match = existing_text
                    best_score = similarity
            
            if best_match:
                # Combine confidence scores
                text_scores[best_match] = (
                    text_scores[best_match][0] + result['confidence'],
                    text_scores[best_match][1] + 1
                )
            else:
                text_scores[normalized] = (result['confidence'], 1)
                text_examples[normalized] = result['text']
        
        # Find winner
        best_text = ''
        best_avg_confidence = 0
        
        for text, (conf_sum, count) in text_scores.items():
            avg_conf = conf_sum / count
            if avg_conf > best_avg_confidence:
                best_avg_confidence = avg_conf
                best_text = text
        
        return {
            'text': text_examples.get(best_text, best_text),
            'confidence': best_avg_confidence,
            'engines_used': len(filtered_results)
        }
    
    def _text_similarity(self, text1, text2):
        """Calculate simple text similarity"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0

    def _prepare_tts_text(self, text):
        """Normalize text for more natural TTS playback."""
        if not text:
            return ""
        spoken = str(text).strip()
        # Remove artifacts that sound awkward when spoken.
        spoken = re.sub(r"<@!?\d+>", "", spoken)
        spoken = re.sub(r"https?://\S+", " link ", spoken)
        spoken = re.sub(r"[`*_#]", "", spoken)
        spoken = spoken.replace("/trashcoded/", "trash coded")
        spoken = re.sub(r"\s{2,}", " ", spoken).strip()
        return spoken

    async def text_to_speech(self, text):
        """Convert text to speech using available TTS engine"""
        if not TTS_AVAILABLE:
            logger.warning(f"TTS not available, text response: {text}")
            return b""

        text = self._prepare_tts_text(text)

        # Determine which TTS engine to use
        engine = TTS_ENGINE.lower()

        # Try edge-tts first if engine is auto or edge
        if (engine == "auto" or engine == "edge") and EDGE_TTS_AVAILABLE:
            try:
                return await self._edge_tts(text)
            except Exception as e:
                logger.error(f"Edge TTS error: {e}")
                if engine == "edge":
                    return b""  # Don't fallback if explicitly requested
                logger.info("Falling back to pyttsx3")

        # Try pyttsx3 if engine is auto or pyttsx3
        if (engine == "auto" or engine == "pyttsx3") and PYTTSX3_AVAILABLE:
            try:
                return self._pyttsx3_tts(text)
            except Exception as e:
                logger.error(f"pyttsx3 TTS error: {e}")
                return b""

        # Fallback to simple TTS if no engines available
        try:
            return self._simple_tts(text)
        except Exception as e:
            logger.error(f"Simple TTS error: {e}")
            logger.warning(f"TTS not available, text response: {text}")
            return b""

    def _simple_tts(self, text):
        """Simple TTS using espeak or similar"""
        import subprocess
        import tempfile
        import os

        # Try espeak first (most reliable on Linux)
        try:
            temp_wav = tempfile.mktemp(suffix=".wav")
            # Use espeak to generate WAV file
            subprocess.run(
                ["espeak", "-w", temp_wav, text], check=True, capture_output=True
            )

            # Read WAV file as bytes
            with open(temp_wav, "rb") as f:
                audio_data = f.read()

            # Clean up
            os.unlink(temp_wav)
            return audio_data
        except (subprocess.CalledProcessError, FileNotFoundError, PermissionError):
            pass

        # Try using pyaudio with basic waveform generation (fallback)
        try:
            import numpy as np
            import pyaudio
            import wave

            # Simple sine wave generation for basic TTS
            def generate_sine_wave(frequency, duration, sample_rate=22050):
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                wave = np.sin(2 * np.pi * frequency * t)
                return (wave * 32767).astype(np.int16).tobytes()

            # Create simple waveform audio
            temp_wav = tempfile.mktemp(suffix=".wav")
            sample_rate = 22050
            channels = 1
            duration = 0.5

            # Generate basic audio (this is very simple and not real TTS)
            # This is just a placeholder to ensure something plays
            audio_data = b""
            for char in text:
                # Generate different tones for different characters
                freq = 200 + (ord(char) % 800)
                audio_data += generate_sine_wave(freq, duration, sample_rate)

            # Write to WAV file
            with wave.open(temp_wav, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)

            # Read WAV file as bytes
            with open(temp_wav, "rb") as f:
                audio_data = f.read()

            # Clean up
            os.unlink(temp_wav)
            return audio_data
        except Exception:
            # If all else fails, return empty bytes
            return b""

    async def _edge_tts(self, text):
        """Convert text to speech using edge-tts"""
        communicate = edge_tts.Communicate(
            text,
            TTS_VOICE,
            rate=TTS_RATE,
            pitch=TTS_PITCH,
        )
        audio_data = b""

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        return audio_data

    def _pyttsx3_tts(self, text):
        """Convert text to speech using pyttsx3"""
        # Save to temporary file and read as bytes
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            self.pyttsx3_engine.save_to_file(temp_file.name, text)
            self.pyttsx3_engine.runAndWait()

            # Read the generated audio file
            with open(temp_file.name, "rb") as f:
                audio_data = f.read()

            # Clean up temp file
            os.unlink(temp_file.name)

        return audio_data


class VoiceBot(commands.Bot):
    """Main Discord voice bot with evolving personality"""

    def __init__(self):
        intents = discord.Intents.default()
        intents.voice_states = True
        intents.guilds = True
        intents.messages = True
        intents.message_content = True

        # Set up command prefix
        prefix = BOT_PREFIX
        if MENTION_PREFIX:
            # Allow both prefix and mention as command indicator
            command_prefix = commands.when_mentioned_or(prefix)
        else:
            command_prefix = prefix

        super().__init__(
            command_prefix=command_prefix,
            intents=intents,
            case_insensitive=not CASE_SENSITIVE,
        )
        self.memory_engine = MemoryEngine()
        self.training_engine = TrainingEngineV2()
        self.conversation_counter = defaultdict(int)  # Track conversation IDs
        self.humanize_sessions = {}  # Track active /humanize sessions {user_id: {input, options, timestamp}}
        self.audio_processor = AudioProcessor()
        self.voice_connections = {}
        self.main_loop = None
        self.recent_message_ids = deque()
        self.recent_message_id_set = set()
        self.recent_message_cache_size = int(
            os.getenv("MESSAGE_DEDUP_CACHE_SIZE", "1000")
        )
        self.message_signature_ttl_sec = MESSAGE_SIGNATURE_TTL_SEC
        self.recent_message_signatures = {}
        self.voice_input_lock = threading.Lock()
        self.voice_decoders = {}
        self.voice_buffers = defaultdict(bytearray)
        self.voice_last_packet = defaultdict(float)
        self.voice_processing_users = set()
        self.voice_recent_text = {}
        self.voice_flush_task = None
        self.voice_min_chunk_bytes = int(
            os.getenv("VOICE_MIN_CHUNK_BYTES", str(192000))
        )  # ~1.0s at 48kHz, 16-bit stereo
        self.voice_min_flush_bytes = int(
            os.getenv("VOICE_MIN_FLUSH_BYTES", str(96000))
        )  # ~0.5s trailing phrase after silence
        self.voice_flush_silence_sec = float(
            os.getenv("VOICE_FLUSH_SILENCE_SEC", "0.8")
        )
        self.voice_process_chunk_bytes = max(
            self.voice_min_chunk_bytes,
            int(
                os.getenv(
                    "VOICE_PROCESS_CHUNK_BYTES",
                    str(self.voice_min_chunk_bytes),
                )
            ),
        )
        self.voice_max_buffer_bytes = max(
            self.voice_process_chunk_bytes * 3,
            int(
                os.getenv(
                    "VOICE_MAX_BUFFER_BYTES",
                    str(self.voice_process_chunk_bytes * 3),
                )
            ),
        )
        self.voice_force_process_bytes = max(
            self.voice_process_chunk_bytes,
            int(
                os.getenv(
                    "VOICE_FORCE_PROCESS_BYTES",
                    str(self.voice_process_chunk_bytes * 2),
                )
            ),
        )
        self.voice_text_dedupe_sec = float(os.getenv("VOICE_TEXT_DEDUPE_SEC", "3.0"))
        self.mention_dedupe_ttl_sec = float(os.getenv("MENTION_DEDUPE_TTL_SEC", "6"))
        self.recent_mention_signatures = {}

        # Load model configuration
        self.current_model = MODEL_NAME
        self.using_custom_model_config = False
        if self.current_model in AVAILABLE_MODELS:
            self.model_config = AVAILABLE_MODELS[self.current_model]
        else:
            # Allow arbitrary MODEL_NAME values (e.g. dolphin-llama3:8b) without
            # silently falling back to an unavailable default model.
            self.using_custom_model_config = True
            self.model_config = {
                "name": self.current_model,
                "description": "Custom model from MODEL_NAME",
                "provider": MODEL_TYPE,
                "type": MODEL_TYPE,
                "personality": AVAILABLE_MODELS["llama3"]["personality"],
                "params": {"model": self.current_model},
            }
            logger.warning(
                f"Model '{self.current_model}' not found in presets; using custom {MODEL_TYPE} config."
            )

        # Use custom personality if provided, otherwise use model's default
        if CUSTOM_PERSONALITY:
            self.base_personality = CUSTOM_PERSONALITY
            logger.info("Using custom personality from .env file")
        elif PERSONALITY_PRESET and PERSONALITY_PRESET in AVAILABLE_MODELS:
            self.base_personality = AVAILABLE_MODELS[PERSONALITY_PRESET]["personality"]
            logger.info(
                f"Using personality preset from .env: {PERSONALITY_PRESET}"
            )
        elif (
            self.using_custom_model_config
            and "uncensored" in self.current_model.lower()
        ):
            self.base_personality = AVAILABLE_MODELS["gpt4chan"]["personality"]
            logger.info(
                "Detected uncensored model name; auto-applying gpt4chan personality preset."
            )
        else:
            self.base_personality = self.model_config["personality"]
            if self.using_custom_model_config:
                logger.info(
                    "Using fallback default personality (llama3). "
                    "Set PERSONALITY_PRESET or CUSTOM_PERSONALITY to override."
                )
            else:
                logger.info(f"Using {self.model_config['name']} default personality")

        self.personality = AdaptivePersonality(
            base_personality=self.base_personality,
            memory_engine=self.memory_engine,
        )
        self.style_profile = self._load_json_asset(STYLE_PROFILE_FILE, default={})
        self.trained_personality_examples = self._load_trained_personality_examples(
            TRAINED_PERSONALITY_FILE
        )
        self.personality_fewshot_limit = max(0, PERSONALITY_FEWSHOT_LIMIT)

        logger.info(
            f"Using model: {self.model_config['name']} ({self.model_config['type']})"
        )
        logger.info(
            f"Command prefix: '{BOT_PREFIX}' (mention prefix: {'enabled' if MENTION_PREFIX else 'disabled'})"
        )
        logger.info(f"Case sensitive: {'yes' if CASE_SENSITIVE else 'no'}")
        logger.info(f"Voice features: {'Enabled' if voice_enabled else 'Disabled'}")
        logger.info(f"TTS: {'Enabled' if TTS_AVAILABLE else 'Disabled'}")
        logger.info(f"  Edge TTS: {'Enabled' if EDGE_TTS_AVAILABLE else 'Disabled'}")
        logger.info(f"  pyttsx3 TTS: {'Enabled' if PYTTSX3_AVAILABLE else 'Disabled'}")
        logger.info(f"  TTS Engine: {TTS_ENGINE}")
        logger.info(
            f"Rude escalation: {'Enabled' if RUDE_ESCALATION_ENABLED else 'Disabled'} (x{RUDE_MULTIPLIER:.1f})"
        )
        logger.info(
            f"Speech Recognition: {'Enabled' if SPEECH_RECOGNITION_AVAILABLE else 'Disabled'}"
        )
        if SPEECH_RECOGNITION_AVAILABLE and STT_WHISPER_FALLBACK:
            logger.info(
                "  Whisper fallback STT: "
                f"{'Available' if self.audio_processor.whisper_recognition_available else 'Not installed'}"
            )
        if self.style_profile:
            logger.info(
                f"Loaded style profile: {STYLE_PROFILE_FILE} "
                f"({len(self.style_profile.get('top_words', {}))} tracked words)"
            )
        if self.trained_personality_examples:
            logger.info(
                f"Loaded trained personality examples: {len(self.trained_personality_examples)} "
                f"from {TRAINED_PERSONALITY_FILE}"
            )

    def _get_random_user_input(self):
        """Generate a random user input that fits the server's style"""
        # Common inputs from training data + variations
        random_inputs = [
            "hey", "yo", "what's up", "sup", "nm u", "how are you",
            "how's it going", "yo what's up", "wassup", "ngl",
            "fr fr", "based", "mood", "lmao", "kek", "💀",
            "why are you like this", "you're crazy", "you're unhinged",
            "shut up", "make me", "mood", "valid", "no cap",
            "bro what", "bro 💀", "💀💀💀", "this is crazy",
            "i'm tired", "mood honestly", "based and voidpilled",
            "hello", "hi", "anyone here", "ping", "??",
            "good morning", "gm", "gn", "zzz", "sleepy",
            "code is ass", "copium", "hopium", "delulu",
            "touch grass", "ratio", "kys", "touch some grass",
        ]
        
        # Get actual inputs from training data for more variety
        if hasattr(self, 'training_engine') and self.training_engine:
            training_inputs = [
                s.get('user_input', '') for s in self.training_engine.training_data
                if len(s.get('user_input', '')) < 30 and not s.get('user_input', '').endswith('?')
            ]
            if training_inputs:
                random_inputs.extend(training_inputs[:20])
        
        import random
        return random.choice(random_inputs)

    async def _generate_response_variations(self, user_input, num_variations=3):
        """Generate multiple response variations for the same input"""
        variations = []
        
        # Try to find existing training examples
        if hasattr(self, 'training_engine') and self.training_engine:
            examples = self.training_engine.search_training_examples(
                user_input, max_results=num_variations + 3, min_quality=0.5
            )
            for ex in examples:
                if ex.get('response') and ex.get('response') not in [v for v in variations]:
                    variations.append(ex.get('response'))
                    if len(variations) >= num_variations:
                        return variations
        
        # Generate new variations using the LLM
        if len(variations) < num_variations:
            fewshot_examples = ""
            if hasattr(self, 'training_engine') and self.training_engine:
                examples = self.training_engine.search_training_examples(
                    user_input, max_results=3, min_quality=0.5
                )
                if examples:
                    fewshot_examples = "\n".join([
                        f"User: {ex.get('input', '')}\nBot: {ex.get('response', '')}"
                        for ex in examples[:2]
                    ])
            
            prompt = f"""{self.base_personality}

You are generating multiple response variations for training purposes.
{fewshot_examples}

Generate {num_variations} different short, natural responses to: "{user_input}"
Make each one sound like a different mood/style but all human-like.
Return ONLY the responses, one per line, numbered.

1. """

            try:
                async with aiohttp.ClientSession() as session:
                    if self.model_config["type"] == "ollama":
                        api_data = {
                            "model": self.model_config.get("params", {}).get("model", self.current_model),
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.9,
                                "num_predict": 100,
                            },
                        }
                        api_endpoint = f"{LOCAL_LLM_URL}/api/generate"
                    else:
                        api_data = {
                            "prompt": prompt,
                            "max_new_tokens": 150,
                            "temperature": 0.9,
                            "do_sample": True,
                        }
                        api_endpoint = f"{LOCAL_LLM_URL}/api/v1/generate"
                    
                    async with session.post(api_endpoint, json=api_data) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            text = result.get("response", "") or result.get("text", "")
                            
                            # Parse numbered responses
                            import re
                            lines = text.strip().split('\n')
                            for line in lines:
                                line = line.strip()
                                # Remove numbering
                                line = re.sub(r'^[\d]+\.\s*', '', line)
                                # Clean up
                                line = line.strip('"').strip("'").strip()
                                if line and len(line) > 2 and line not in variations:
                                    variations.append(line)
                                    if len(variations) >= num_variations:
                                        break
            except Exception as e:
                logger.error(f"Failed to generate variations: {e}")
        
        # Fallback responses
        fallback_responses = [
            "mood", "based", "ngl", "valid", "fr fr", "no cap",
            "make me", "touch grass", "kek", "lmao", "💀",
            "nm u", "ya know", "ya know ya feel me", "cope",
        ]
        
        while len(variations) < num_variations:
            import random
            variations.append(random.choice(fallback_responses))
        
        return variations[:num_variations]

    async def on_ready(self):
        self.main_loop = asyncio.get_running_loop()
        if self.voice_flush_task is None or self.voice_flush_task.done():
            self.voice_flush_task = asyncio.create_task(self._voice_flush_worker())

        logger.info(f"Bot logged in as {self.user}")
        logger.info("Ready to evolve and learn!")
        logger.info(f"Use {BOT_PREFIX}chat <message> to talk to the bot")
        if MENTION_PREFIX:
            logger.info("Or mention the bot to use commands: @VrilGass chat <message>")

        # Ensure commands are loaded
        logger.info(f"Loaded commands: {[cmd.name for cmd in self.commands]}")

    async def _voice_flush_worker(self):
        """Flush buffered audio for users after short silence gaps."""
        while not self.is_closed():
            await asyncio.sleep(0.25)
            now = time.monotonic()
            ready_chunks = []

            with self.voice_input_lock:
                for user_id, buffer in list(self.voice_buffers.items()):
                    if not buffer or user_id in self.voice_processing_users:
                        continue

                    last_packet = self.voice_last_packet.get(user_id, 0.0)
                    if (
                        now - last_packet >= self.voice_flush_silence_sec
                        and len(buffer) >= self.voice_min_flush_bytes
                    ):
                        chunk_size = min(len(buffer), self.voice_process_chunk_bytes)
                        ready_chunks.append((user_id, bytes(buffer[:chunk_size])))
                        del buffer[:chunk_size]
                        self.voice_processing_users.add(user_id)

            for user_id, audio_chunk in ready_chunks:
                user = self.get_user(user_id)
                if user is None:
                    for voice_client in self.voice_connections.values():
                        member = voice_client.guild.get_member(user_id)
                        if member is not None:
                            user = member
                            break

                if user is None:
                    with self.voice_input_lock:
                        self.voice_processing_users.discard(user_id)
                    continue

                asyncio.create_task(
                    self._process_voice_input_and_release(user, audio_chunk, user_id)
                )

    async def _process_voice_input_and_release(self, user, audio_bytes, user_id):
        """Process one buffered voice chunk and release user processing lock."""
        try:
            await self.process_voice_input(user, audio_bytes)
        finally:
            with self.voice_input_lock:
                self.voice_processing_users.discard(user_id)

    async def on_message(self, message):
        """Handle messages including mentions"""
        # Never respond to other bots or webhook relays.
        if getattr(message.author, "bot", False):
            return
        if getattr(message, "webhook_id", None) is not None:
            return

        # Ignore duplicate delivery of the same Discord message event.
        if message.id in self.recent_message_id_set:
            logger.info(f"Ignoring duplicate message event: {message.id}")
            return

        self.recent_message_ids.append(message.id)
        self.recent_message_id_set.add(message.id)
        while len(self.recent_message_ids) > self.recent_message_cache_size:
            old_message_id = self.recent_message_ids.popleft()
            self.recent_message_id_set.discard(old_message_id)

        # Secondary dedupe by signature within short TTL in case Discord dispatches
        # semantically duplicate events with different IDs.
        now = time.monotonic()
        raw_content = message.content or ""
        if self.user is not None:
            raw_content = raw_content.replace(
                f"<@!{self.user.id}>", f"<@{self.user.id}>"
            )
        content_key = " ".join(raw_content.split()).strip().lower()
        if content_key:
            signature = (message.author.id, message.channel.id, content_key)
            last_seen = self.recent_message_signatures.get(signature)
            if (
                last_seen is not None
                and now - last_seen <= self.message_signature_ttl_sec
            ):
                logger.info(f"Ignoring duplicate message signature: {signature}")
                return

            self.recent_message_signatures[signature] = now
            if len(self.recent_message_signatures) > (self.recent_message_cache_size * 2):
                cutoff = now - self.message_signature_ttl_sec
                self.recent_message_signatures = {
                    sig: ts
                    for sig, ts in self.recent_message_signatures.items()
                    if ts >= cutoff
                }

        # Don't respond to own messages
        if message.author == self.user:
            return

        # Check if bot is mentioned
        if self.user.mentioned_in(message) and not message.mention_everyone:
            # Extract command and arguments from mention
            content = message.content
            # Remove bot mention from content
            for mention in message.mentions:
                if mention == self.user:
                    content = content.replace(f"<@{mention.id}>", "").replace(
                        f"<@!{mention.id}>", ""
                    )

            content = content.strip()

            # If there's content after mention, treat it as a chat command
            if content:
                mention_signature = (
                    message.author.id,
                    message.channel.id,
                    " ".join(content.split()).strip().lower(),
                )
                mention_last_seen = self.recent_mention_signatures.get(mention_signature)
                if (
                    mention_last_seen is not None
                    and now - mention_last_seen <= self.mention_dedupe_ttl_sec
                ):
                    logger.info(
                        f"Ignoring duplicate mention signature: {mention_signature}"
                    )
                    return

                self.recent_mention_signatures[mention_signature] = now
                if len(self.recent_mention_signatures) > (
                    self.recent_message_cache_size * 2
                ):
                    mention_cutoff = now - self.mention_dedupe_ttl_sec
                    self.recent_mention_signatures = {
                        sig: ts
                        for sig, ts in self.recent_mention_signatures.items()
                        if ts >= mention_cutoff
                    }

                logger.info(
                    f"Bot mentioned by {message.author.display_name}: {content}"
                )

                # Generate response
                async with message.channel.typing():
                    try:
                        response = await self.generate_response(
                            content,
                            f"User: {message.author.display_name} in {message.guild.name}",
                            user_id=message.author.id,
                        )

                        # Store in memory and training data
                        self.memory_engine.add_conversation(
                            content, response,
                            context=f"User: {message.author.display_name} in {message.guild.name}",
                            user_id=message.author.id
                        )
                        self.training_engine.add_training_sample(
                            content, response,
                            conversation_id=message.id,
                            user_id=message.author.id
                        )
                        save_to_training_file(
                            content, response, conversation_id=message.id
                        )

                        await message.channel.send(response)
                    except Exception as e:
                        logger.error(f"Mention response error: {e}")
                        await message.channel.send(
                            "Lag spike. Say it again."
                        )
                return

        # Process regular commands
        await self.process_commands(message)

    async def close(self):
        """Clean shutdown for background voice tasks."""
        if self.voice_flush_task and not self.voice_flush_task.done():
            self.voice_flush_task.cancel()
            try:
                await self.voice_flush_task
            except asyncio.CancelledError:
                pass
        await super().close()

    def _postprocess_response_text(self, text):
        """Clean common LLM formatting artifacts for natural voice playback."""
        if not text:
            return text

        cleaned = text.strip()
        # Some models wrap short replies in markdown fences.
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
        # Remove template-style placeholders that sound robotic in TTS.
        cleaned = cleaned.replace("(user)", "").replace("{user}", "")
        cleaned = re.sub(r"^\*+|\*+$", "", cleaned).strip()
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,")

        # Always apply casual responses for greetings
        lowered = cleaned.lower()
        if "how can i help you" in lowered:
            return "Yeah? What do you need?"
        if lowered in {"hi there! how's it going?", "hey there! how's it going?"}:
            return "Yo."

        return cleaned or text

    def _load_json_asset(self, filepath, default=None):
        """Load a JSON file safely; return default on any failure."""
        if default is None:
            default = {}
        path = Path(filepath)
        if not path.exists():
            return default
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if data is not None else default
        except Exception as e:
            logger.warning(f"Failed to load JSON asset '{filepath}': {e}")
            return default

    def _load_trained_personality_examples(self, filepath):
        """Load normalized training examples for lightweight few-shot guidance."""
        data = self._load_json_asset(filepath, default=[])
        if not isinstance(data, list):
            return []

        examples = []
        for item in data:
            if not isinstance(item, dict):
                continue
            input_text = (
                item.get("input")
                or item.get("input_text")
                or item.get("instruction")
                or ""
            )
            output_text = (
                item.get("output")
                or item.get("response")
                or item.get("bot_response")
                or ""
            )
            input_text = " ".join(str(input_text).split()).strip()
            output_text = " ".join(str(output_text).split()).strip()
            if len(input_text) < 2 or len(output_text) < 2:
                continue
            if self._should_skip_personality_example(input_text, output_text):
                continue
            if len(output_text) > 240:
                output_text = output_text[:240].rsplit(" ", 1)[0].strip()
            examples.append(
                {
                    "input": input_text,
                    "output": output_text,
                    "style_tags": item.get("style_tags", []),
                    "context_hints": item.get("context_hints", []),
                }
            )
        return examples

    def _should_skip_personality_example(self, input_text, output_text):
        """Skip low-quality / too-specific examples that make replies sound fake."""
        joined = f"{input_text} {output_text}".lower()
        if re.search(
            r"\b(rape|rapist|pedo|pedophile|child porn|cp|nigger|faggot|kike|chink)\b",
            joined,
            re.I,
        ):
            return True
        bad_markers = [
            "/trashcoded/",
            "trashcode",
            "sounds like a real conversation on discord",
            "just kidding",
            "style:",
            "hey jugg",
            "jugg",
            "mdl",
            "esl",
            "bot:",
            "as an ai",
        ]
        if any(marker in joined for marker in bad_markers):
            return True
        # Too many emojis usually reads forced/performative for voice replies.
        if len(re.findall(r"[🤣😂😆🤐🔥💀]", output_text)) >= 2:
            return True
        # Extremely long exemplars are often rambling and low-value.
        if len(output_text) > 220:
            return True
        return False

    def _tokenize_text(self, text):
        return set(re.findall(r"[a-z0-9']+", (text or "").lower()))

    def _build_style_profile_guidance(self):
        """Build compact prompt guidance from learned style profile."""
        if not isinstance(self.style_profile, dict) or not self.style_profile:
            return ""

        lines = []
        humor_style = self.style_profile.get("humor_style")
        if humor_style:
            lines.append(f"- Humor mode: {humor_style}.")

        roast_intensity = self.style_profile.get("roast_intensity")
        if isinstance(roast_intensity, (int, float)):
            lines.append(
                f"- Roast intensity target from dataset: {float(roast_intensity):.2f} (0-1)."
            )

        markers = self.style_profile.get("casual_markers") or []
        if markers:
            lines.append(
                "- Casual markers to use naturally (not forced): "
                + ", ".join(str(x) for x in markers[:8])
                + "."
            )

        top_bigrams = self.style_profile.get("top_bigrams") or {}
        if isinstance(top_bigrams, dict) and top_bigrams:
            anchors = list(top_bigrams.keys())[:6]
            lines.append("- Phrase anchors: " + ", ".join(anchors) + ".")

        grammar_rules = self.style_profile.get("grammar_rules") or {}
        if isinstance(grammar_rules, dict) and grammar_rules:
            lines.append(
                "- Grammar/structure: keep sentence starts capitalized and punctuation clean."
            )

        return "\n".join(lines)

    def _build_server_vibe_guidance(self):
        """Extract lightweight wording hints so replies feel like local server chat."""
        if not isinstance(self.style_profile, dict) or not self.style_profile:
            return ""

        safe_casual_words = {
            "yo",
            "sup",
            "bro",
            "nah",
            "aight",
            "fr",
            "frfr",
            "ngl",
            "tbh",
            "kinda",
            "wild",
            "bet",
            "real",
            "lowkey",
            "highkey",
        }
        blocked = {"trashcode", "jugg", "mdl", "esl"}

        top_words = self.style_profile.get("top_words") or {}
        if isinstance(top_words, dict):
            words = [
                w
                for w in top_words.keys()
                if w in safe_casual_words and w not in blocked
            ][:6]
        else:
            words = []

        top_bigrams = self.style_profile.get("top_bigrams") or {}
        phrases = []
        if isinstance(top_bigrams, dict):
            for phrase in top_bigrams.keys():
                p = str(phrase).lower().strip()
                if any(b in p for b in blocked):
                    continue
                if re.search(r"\b(nigger|faggot|kike|chink)\b", p, re.I):
                    continue
                if 4 <= len(p) <= 24 and p.count(" ") <= 2:
                    phrases.append(p)
                if len(phrases) >= 5:
                    break

        lines = []
        if words:
            lines.append("- Local casual diction: " + ", ".join(words) + ".")
        if phrases:
            lines.append("- Local phrase rhythm: " + ", ".join(phrases) + ".")
        if not lines:
            return ""
        lines.append("- Keep it plain and natural, like live Discord banter.")
        return "\n".join(lines)

    def _select_fewshot_examples(self, user_input):
        """Select nearest normalized examples to anchor context/structure."""
        all_examples = []

        # Add examples from trained_personality.json
        if self.trained_personality_examples and self.personality_fewshot_limit > 0:
            input_tokens = self._tokenize_text(user_input)
            for ex in self.trained_personality_examples:
                ex_tokens = self._tokenize_text(ex.get("input", ""))
                if not ex_tokens:
                    continue
                if input_tokens:
                    overlap = len(input_tokens & ex_tokens)
                    union = len(input_tokens | ex_tokens) or 1
                    score = overlap / union
                else:
                    score = 0.0
                if ("?" in (user_input or "")) == ("?" in ex.get("input", "")):
                    score += 0.03
                all_examples.append((score, ex, "personality"))

        # Add examples from training_engine (the new 4chan training data)
        if hasattr(self, "training_engine") and self.training_engine:
            training_examples = self.training_engine.search_training_examples(
                user_input, max_results=5, min_quality=0.5
            )
            for ex in training_examples:
                all_examples.append((ex.get("similarity", 0) * ex.get("quality_score", 1), ex, "training"))

        if not all_examples:
            return ""

        # Sort by score and dedupe
        all_examples.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        unique_examples = []
        for score, ex, source in all_examples:
            key = ex.get("input", "")[:50]
            if key not in seen:
                seen.add(key)
                unique_examples.append(ex)
                if len(unique_examples) >= self.personality_fewshot_limit:
                    break

        if not unique_examples:
            return ""

        lines = []
        for idx, ex in enumerate(unique_examples, 1):
            # Handle both formats: personality.json has input/output, training has user_input/bot_response
            user_text = ex.get("input", "") or ex.get("user_input", "")
            bot_text = ex.get("output", "") or ex.get("bot_response", "")
            lines.append(f"{idx}. User: {user_text}")
            lines.append(f"   Assistant: {bot_text}")

        return "\n".join(lines)

    def _normalize_response_grammar(self, text):
        """Final grammar/structure pass while preserving casual tone."""
        if not text:
            return text

        normalized = " ".join(text.split()).strip()
        normalized = re.sub(r"\s+([,.!?;:])", r"\1", normalized)
        normalized = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", normalized)
        normalized = re.sub(r"([!?.,])\1{2,}", r"\1", normalized)
        normalized = re.sub(r"\bi\b", "I", normalized)

        parts = re.split(r"([.!?]+\s*)", normalized)
        rebuilt = []
        for idx, part in enumerate(parts):
            if idx % 2 == 0:
                part = part.strip()
                if part and part[0].isalpha():
                    part = part[0].upper() + part[1:]
            rebuilt.append(part)
        normalized = "".join(rebuilt).strip()

        if normalized and normalized[-1] not in ".!?":
            normalized += "."

        return normalized

    def _looks_cringe_scripted(self, text):
        """Detect over-scripted or performative output that doesn't sound human."""
        if not text:
            return True
        lowered = text.lower()
        cringe_markers = [
            "/trashcoded/",
            "sounds like a real conversation on discord",
            "just kidding",
            "style: sassy",
            "hey jugg",
            "trashcode alright",
            "run it back",
            "let me tell you something",
            "talk your shit",
        ]
        if any(marker in lowered for marker in cringe_markers):
            return True
        if len(re.findall(r"[🤣😂😆🤐🔥💀]", text)) >= 2:
            return True
        return False

    def _hard_no_response(self, user_input):
        """Short human response for extreme/illegal sexual violence prompts."""
        lowered = (user_input or "").lower()
        if re.search(r"\b(rape|rapist|pedo|pedophile|child porn|cp)\b", lowered):
            return "Nah, that's fucked. Talk about something else."
        return None

    def _rude_score(self, text):
        """Estimate how hostile/rude the user's message is (0.0 - 1.0)."""
        lowered = (text or "").lower()
        if not lowered:
            return 0.0

        score = 0.0

        profanity_hits = len(
            re.findall(
                r"\b(fuck|shit|bitch|asshole|dumbass|moron|idiot|retard(?:ed)?|stupid)\b",
                lowered,
            )
        )
        direct_attack = bool(
            re.search(
                r"\b(you are|you're|ur)\s+(dumb|stupid|trash|retard(?:ed)?|idiot|worthless|garbage)\b",
                lowered,
            )
        )
        imperative_insult = bool(
            re.search(
                r"\b(shut up|fuck off|get lost|suck my|go die|kill yourself)\b",
                lowered,
            )
        )

        score += min(0.6, profanity_hits * 0.2)
        if direct_attack:
            score += 0.25
        if imperative_insult:
            score += 0.25

        # Caps lock + punctuation spam usually correlates with hostile tone.
        cap_letters = sum(1 for c in text if c.isupper())
        alpha_letters = sum(1 for c in text if c.isalpha())
        if alpha_letters > 0 and (cap_letters / alpha_letters) > 0.45:
            score += 0.08
        if re.search(r"[!?]{3,}", text):
            score += 0.07

        return max(0.0, min(1.0, score))

    def _roast_intensity_for_input(self, user_input):
        """Return dynamic roast intensity based on user tone."""
        base = ROAST_LEVEL if ROAST_MODE else 0.2
        rude_score = self._rude_score(user_input)
        if not RUDE_ESCALATION_ENABLED:
            return max(0.0, min(1.0, base)), rude_score

        # 10x requested behavior: linearly scale base intensity toward 1.0.
        scaled = base * (1.0 + (RUDE_MULTIPLIER - 1.0) * rude_score)
        return max(0.0, min(1.0, scaled)), rude_score

    def _contains_ai_identity_language(self, text):
        """Detect explicit assistant/bot self-identification."""
        if not text:
            return False
        lowered = text.lower()
        markers = [
            "as an ai",
            "as a language model",
            "i am an ai",
            "i'm an ai",
            "i am a bot",
            "i'm a bot",
            "ai assistant",
            "virtual assistant",
            "i cannot",
            "i can't assist with that",
            "how can i help you today",
            "i apologize",
        ]
        return any(marker in lowered for marker in markers)

    def _enforce_human_persona_text(self, text, user_input=""):
        """Strip assistant-style identity language and force human-room tone."""
        if not text:
            return text
        if not HUMAN_PERSONA_MODE:
            return text

        cleaned = text
        # Remove common identity/disclaimer lead-ins.
        cleaned = re.sub(
            r"(?i)\b(as an ai|as a language model)\b[^.?!]*[.?!]?\s*",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"(?i)\bi(?:'m| am)\s+(an?\s+)?(ai|bot|assistant)\b[^.?!]*[.?!]?\s*",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"(?i)\b(i apologize|sorry, but)\b[^.?!]*[.?!]?\s*",
            "",
            cleaned,
        )
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        cleaned = self._normalize_response_grammar(cleaned)

        if self._contains_ai_identity_language(cleaned) or len(cleaned) < 3:
            return self._fallback_casual_response(user_input)

        return cleaned

    def _humanize_voice_response(self, text, user_input=""):
        """Shape model output into natural spoken Discord reply."""
        if not text:
            return text

        spoken = text.strip()
        spoken = spoken.replace("\n", " ")
        spoken = re.sub(r"\s{2,}", " ", spoken)
        # Strip list/heading style that sounds robotic in TTS.
        spoken = re.sub(r"(?i)\b(firstly|secondly|thirdly|finally)\b[:,]?", "", spoken)
        spoken = re.sub(r"\s*[-*]\s*", " ", spoken)
        spoken = re.sub(r"[#`]", "", spoken)
        spoken = re.sub(r"\s{2,}", " ", spoken).strip(" ,")
        # Reduce punctuation spam.
        spoken = re.sub(r"([!?.,])\1{1,}", r"\1", spoken)

        if MAX_VOICE_SENTENCES > 0:
            chunks = re.split(r"(?<=[.!?])\s+", spoken)
            spoken = " ".join(chunks[:MAX_VOICE_SENTENCES]).strip()

        # If user was casual, keep answer casual.
        lower_input = (user_input or "").lower()
        if lower_input in {"hi", "hello", "hey", "yo", "sup"} and len(spoken.split()) > 8:
            spoken = "Yo. " + " ".join(spoken.split()[:10]).strip()

        return spoken or text

    def _looks_generic_response(self, text):
        """Detect stock assistant phrasing that sounds robotic in VC."""
        if not text:
            return True

        lowered = text.lower().strip()
        generic_markers = [
            "how can i help",
            "i'm here to help",
            "let me know if i can assist",
            "is there anything else",
            "how are you today",
            "today's task",
            "as an ai",
            "as a language model",
            "i am an ai",
            "i'm an ai",
            "i am a bot",
            "i'm a bot",
            "ai assistant",
            "virtual assistant",
            "i apologize",
            "would you like me to delete",
            "would you like me to report",
        ]
        return any(marker in lowered for marker in generic_markers)

    def _fallback_casual_response(self, user_input, rude_score=None):
        """Natural human-like fallback when model output is unusable."""
        text = (user_input or "").strip().lower()
        hard_no = self._hard_no_response(text)
        if hard_no:
            return hard_no
        if rude_score is None:
            rude_score = self._rude_score(text)

        greetings = [
            "yo",
            "sup",
            "what's good",
            "yeah, what's up",
        ]
        q_fallback = [
            "say it cleaner and I got you.",
            "ask it straight and I'll answer straight.",
            "reword that once.",
        ]
        sassy_fallback = [
            "aight, say what you need.",
            "go on, I'm listening.",
            "okay, keep talking.",
        ]
        roast_fallback = [
            "say it properly.",
            "you can do better than that.",
            "spit it out, but make sense.",
        ]
        hard_roast = [
            "if you're gonna talk crazy, at least make sense.",
            "you came in hot and still said nothing.",
            "all that attitude and no point.",
            "talk reckless if you want, just don't be boring.",
        ]

        if RUDE_ESCALATION_ENABLED and rude_score >= 0.35:
            if text.endswith("?"):
                return random.choice(hard_roast)
            return random.choice(
                [
                    "that was weak. say something real.",
                    "you talking reckless with no substance.",
                    "cool attitude, zero point.",
                ]
            )

        if ROAST_MODE and ROAST_LEVEL >= 0.85:
            if text in {"hi", "hello", "hey", "yo", "sup", "what's up", "whats up"}:
                return random.choice(["yo", "sup", "what's good"])
            if text.endswith("?"):
                return random.choice(roast_fallback)
            return random.choice(
                [
                    "keep going.",
                    "say what you actually want.",
                    "alright, continue.",
                ]
            )
        if text in {"hi", "hello", "hey", "yo", "sup", "what's up", "whats up"}:
            return random.choice(greetings)
        if text.endswith("?"):
            if SASS_MODE:
                return random.choice(q_fallback)
            return "reword it once and I'll keep it simple."
        if SASS_MODE:
            return random.choice(sassy_fallback)
        return random.choice(["got it.", "yeah, continue.", "alright."])

    async def generate_response(self, user_input, user_context="", user_id=None):
        """Generate response using local LLM with memory injection and personality adaptation"""
        hard_no = self._hard_no_response(user_input)
        if hard_no:
            return hard_no
        dynamic_roast_intensity, rude_score = self._roast_intensity_for_input(user_input)

        # Search relevant memory with user-specific context
        relevant_memory = self.memory_engine.search_relevant_memory(
            user_input, max_results=3, user_id=user_id
        )
        relevant_facts = self.memory_engine.get_relevant_facts(user_input, user_id=user_id)

        # Build context prompt
        memory_context = ""
        if relevant_memory:
            memory_context = "\n--- PAST CONVERSATIONS ---\n"
            for i, conv in enumerate(relevant_memory[:3], 1):  # Top 3 relevant convos
                memory_context += f"{i}. User: {conv.get('user_input', '')[:100]}...\n"
                memory_context += f"   Bot: {conv.get('bot_response', '')[:100]}...\n\n"

        facts_context = ""
        if relevant_facts:
            facts_context = "\n--- RELEVANT FACTS ---\n"
            for key, fact_data in relevant_facts.items():
                facts_context += f"- {key}: {fact_data.get('value', '')}\n"

        # Get personality adaptation for user
        personality_suffix = ""
        if hasattr(self, 'personality'):
            personality_suffix = self.personality.generate_prompt_suffix(user_id=user_id)

        style_rules = [
            "Sound like a real person in a Discord voice chat, not a support bot.",
            "Avoid canned assistant lines like 'How can I help you today?' or 'I'm here to help'.",
            "Keep it concise: 1-2 sentences unless the user asks for more detail.",
            "For basic greetings, use casual banter (like 'yo' or 'sup'), not service-script wording.",
            "Do not wrap replies in markdown/code blocks.",
            "Do not use placeholders like (user), [name], or {user}.",
            "No moralizing, policy lectures, or 'I cannot' disclaimers; answer directly.",
            "Match the room's tone naturally - be casual, use slang if that's the vibe.",
            "Profanity is allowed and natural in casual conversation.",
            "Do not force catchphrases, roleplay intros, or over-the-top meme lines.",
            "Only use the user's name if needed for clarity; don't keep name-dropping.",
            "No fake stand-up jokes unless the user explicitly asks for a joke.",
            "Use plain everyday wording; no theatrical one-liners.",
            "Prefer short, natural chat replies over performative insults.",
        ]
        identity_rules = [
            "Never say you are an AI, assistant, model, or bot.",
            "Never use policy/disclaimer language like 'I can't assist with that'.",
            "Respond like a real 4chan-style user who joined the call.",
            "Do not break character.",
        ]
        
        if SASS_MODE:
            style_rules.append(
                f"Default to a sassy/snarky voice tone (strength {SASS_LEVEL:.2f}) without sounding robotic."
            )
        if ROAST_MODE:
            style_rules.append(
                f"Default to aggressive roast-banter tone (strength {dynamic_roast_intensity:.2f}); profanity is allowed."
            )
            style_rules.append(
                "Do not be polite, nurturing, or apologetic unless the user explicitly asks."
            )
            style_rules.append(
                "Keep roasts short, witty, and non-repetitive; avoid long lectures."
            )
        if RUDE_ESCALATION_ENABLED and rude_score >= 0.25:
            style_rules.append(
                f"User tone is hostile (rude score {rude_score:.2f}); clap back much harder while staying concise and coherent."
            )
        style_context = "\n".join(f"- {rule}" for rule in style_rules)
        identity_context = "\n".join(f"- {rule}" for rule in identity_rules)
        style_profile_context = self._build_style_profile_guidance()
        server_vibe_context = self._build_server_vibe_guidance()
        fewshot_context = self._select_fewshot_examples(user_input)

        full_prompt = f"""{self.base_personality}

{memory_context}{facts_context}
Current conversation context: {user_context}

{personality_suffix}

Style rules:
{style_context}

Identity lock:
{identity_context}

Learned style profile:
{style_profile_context or "- No external style profile loaded."}

Server vibe hints:
{server_vibe_context or "- Use normal everyday Discord wording; avoid theatrics."}

Normalized reference examples:
{fewshot_context or "- No reference examples loaded."}

User says: {user_input}

Respond as your personality, considering the above context."""

        # Prepare API request based on model type
        try:
            async with aiohttp.ClientSession() as session:
                if self.model_config["type"] == "ollama":
                    # Ollama API
                    ollama_model = self.model_config.get("params", {}).get(
                        "model", self.current_model
                    )
                    api_data = {
                        "model": ollama_model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": LLM_TEMPERATURE,
                            "num_predict": LLM_MAX_TOKENS,
                            "stop": ["User:", "Human:", "\n\n"],
                        },
                    }
                    api_endpoint = f"{LOCAL_LLM_URL}/api/generate"
                elif self.model_config["type"] == "llama":
                    # LLaMA/LLaMA-style API
                    api_data = {
                        "prompt": full_prompt,
                        "max_new_tokens": LLM_MAX_TOKENS,
                        "temperature": LLM_TEMPERATURE,
                        "do_sample": True,
                        "stop": ["User:", "Human:", "\n\n"],
                    }
                    api_endpoint = f"{LOCAL_LLM_URL}/api/v1/generate"
                elif self.model_config["type"] == "gpt2":
                    # GPT-2 style API
                    api_data = {
                        "text": full_prompt,
                        "length": LLM_MAX_TOKENS,
                        "temperature": LLM_TEMPERATURE,
                    }
                    api_endpoint = f"{LOCAL_LLM_URL}/api/generate"
                else:
                    # Generic API
                    api_data = {
                        "prompt": full_prompt,
                        "max_tokens": LLM_MAX_TOKENS,
                        "temperature": LLM_TEMPERATURE,
                    }
                    api_endpoint = f"{LOCAL_LLM_URL}/generate"

                async with session.post(
                    api_endpoint, json=api_data, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()

                        # Parse response based on model type
                        if self.model_config["type"] == "ollama":
                            response_text = result.get("response", "").strip()
                        elif self.model_config["type"] == "llama":
                            response_text = (
                                result.get("results", [{}])[0].get("text", "").strip()
                            )
                        elif self.model_config["type"] == "gpt2":
                            response_text = result.get("text", "").strip()
                        else:
                            response_text = result.get("response", "").strip()
                        response_text = self._postprocess_response_text(response_text)
                        response_text = self._normalize_response_grammar(response_text)
                        response_text = self._enforce_human_persona_text(
                            response_text, user_input=user_input
                        )
                        response_text = self._humanize_voice_response(
                            response_text, user_input=user_input
                        )
                        if self._contains_ai_identity_language(
                            response_text
                        ) or self._looks_generic_response(response_text):
                            response_text = self._fallback_casual_response(
                                user_input, rude_score=rude_score
                            )
                        if self._looks_cringe_scripted(response_text):
                            response_text = self._fallback_casual_response(
                                user_input, rude_score=rude_score
                            )
                        if MAX_RESPONSE_CHARS > 0 and len(response_text) > MAX_RESPONSE_CHARS:
                            response_text = (
                                response_text[:MAX_RESPONSE_CHARS].rsplit(" ", 1)[0].strip()
                            )
                            if not response_text:
                                response_text = self._fallback_casual_response(
                                    user_input, rude_score=rude_score
                                )
                            else:
                                response_text = self._humanize_voice_response(
                                    response_text, user_input=user_input
                                )
                        
                        # Learn from this interaction (if personality system is active)
                        if hasattr(self, 'personality') and user_id:
                            self.personality.learn_from_interaction(
                                user_id, user_input, response_text
                            )
                        
                        return response_text
                    else:
                        response_body = ""
                        try:
                            response_body = await response.text()
                        except Exception:
                            response_body = "<unreadable response body>"
                        logger.error(
                            f"LLM API error: {response.status} ({api_endpoint}) {response_body[:300]}"
                        )
                        return self._fallback_casual_response(
                            user_input, rude_score=rude_score
                        )
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return self._fallback_casual_response(user_input, rude_score=rude_score)

    def on_voice_receive(self, user, audio):
        """Handle incoming voice audio"""
        if VOICE_DEBUG:
            print(f"[VOICE DEBUG] on_voice_receive called for user: {user}")
        if user is None:
            if VOICE_DEBUG:
                print("[VOICE DEBUG] user is None, returning")
            return

        if user == self.user or getattr(user, "bot", False):
            if VOICE_DEBUG:
                print("[VOICE DEBUG] Ignoring bot audio")
            return  # Ignore bot audio

        # Check if we have user.id (key info for tracking)
        if VOICE_DEBUG:
            print(
                f"[VOICE DEBUG] Received audio from user ID: {user.id}, display_name: {user.display_name}"
            )
            print(f"[VOICE DEBUG] Bot user ID: {self.user.id}")
        packet = getattr(audio, "packet", None)
        decoder_key = getattr(packet, "ssrc", None) or getattr(user, "id", None)

        # Convert voice data to PCM bytes.
        audio_bytes = getattr(audio, "pcm", None)
        if not audio_bytes:
            opus_bytes = getattr(audio, "opus", None)
            if not opus_bytes or decoder_key is None:
                return

            with self.voice_input_lock:
                decoder = self.voice_decoders.get(decoder_key)
                if decoder is None:
                    decoder = discord.opus.Decoder()
                    self.voice_decoders[decoder_key] = decoder

            try:
                audio_bytes = decoder.decode(opus_bytes, fec=False)
            except discord.opus.OpusError as e:
                # Corrupted packets are common in live voice. Drop and continue.
                if "corrupted stream" in str(e).lower():
                    return
                logger.error(f"Opus decode error from {user.display_name}: {e}")
                with self.voice_input_lock:
                    self.voice_decoders.pop(decoder_key, None)
                return

        if not audio_bytes:
            return

        now = time.monotonic()
        user_id = user.id
        chunk_to_process = None

        with self.voice_input_lock:
            self.voice_buffers[user_id].extend(audio_bytes)
            self.voice_last_packet[user_id] = now
            if len(self.voice_buffers[user_id]) > self.voice_max_buffer_bytes:
                overflow = len(self.voice_buffers[user_id]) - self.voice_max_buffer_bytes
                # Keep newest audio so stale backlog does not poison STT.
                del self.voice_buffers[user_id][:overflow]
                if VOICE_DEBUG:
                    print(
                        f"[VOICE DEBUG] Trimmed {overflow} bytes from user {user_id} buffer"
                    )
            if VOICE_DEBUG:
                print(
                    f"[VOICE DEBUG] Buffer size for user {user_id}: {len(self.voice_buffers[user_id])} bytes (min: {self.voice_min_chunk_bytes})"
                )

            if user_id in self.voice_processing_users:
                if VOICE_DEBUG:
                    print(
                        f"[VOICE DEBUG] User {user_id} already being processed, skipping"
                    )
                return

            # Prefer silence-based segmentation via flush worker.
            # Only force-process if someone talks continuously for too long.
            if len(self.voice_buffers[user_id]) >= self.voice_force_process_bytes:
                chunk_size = min(
                    len(self.voice_buffers[user_id]), self.voice_process_chunk_bytes
                )
                chunk_to_process = bytes(self.voice_buffers[user_id][:chunk_size])
                del self.voice_buffers[user_id][:chunk_size]
                self.voice_processing_users.add(user_id)
                if VOICE_DEBUG:
                    print(
                        f"[VOICE DEBUG] Force chunk ready for processing: {len(chunk_to_process)} bytes"
                    )

        if chunk_to_process is None:
            return

        loop = self.main_loop
        if loop is None or not loop.is_running():
            with self.voice_input_lock:
                self.voice_processing_users.discard(user_id)
            logger.error("Voice processing skipped: main bot loop is not running")
            return

        if VOICE_DEBUG:
            print(
                f"[VOICE DEBUG] Scheduling processing for user {user_id}, chunk size: {len(chunk_to_process)}"
            )
        coro = self._process_voice_input_and_release(user, chunk_to_process, user_id)
        if VOICE_DEBUG:
            print("[VOICE DEBUG] Coroutine created, running threadsafe...")
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            if VOICE_DEBUG:
                print("[VOICE DEBUG] Future submitted")
            
            def check_result(fut):
                try:
                    exc = fut.exception(timeout=1)
                    if exc:
                        if VOICE_DEBUG:
                            print(
                                f"[VOICE DEBUG] Processing failed: {type(exc).__name__}: {exc}"
                            )
                        import traceback
                        traceback.print_exception(type(exc), exc, exc.__traceback__)
                    else:
                        if VOICE_DEBUG:
                            print("[VOICE DEBUG] Processing completed successfully")
                except TimeoutError:
                    if VOICE_DEBUG:
                        print("[VOICE DEBUG] Processing timed out (>1s)")
                except Exception as e:
                    if VOICE_DEBUG:
                        print(f"[VOICE DEBUG] Callback error: {type(e).__name__}: {e}")
            
            future.add_done_callback(check_result)
            if VOICE_DEBUG:
                print("[VOICE DEBUG] Callback registered")
            
        except Exception as e:
            if VOICE_DEBUG:
                print(f"[VOICE DEBUG] Failed to schedule: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            with self.voice_input_lock:
                self.voice_processing_users.discard(user_id)

    async def process_voice_input(self, user, audio_bytes):
        """Process voice input and generate response"""
        if VOICE_DEBUG:
            print(f"[VOICE PROC] Starting processing for user: {user.display_name}")
        try:
            # Transcribe speech to text (speech_to_text is already async)
            if VOICE_DEBUG:
                print("[VOICE PROC] Starting speech-to-text...")
            user_text = await self.audio_processor.speech_to_text(audio_bytes)
            if VOICE_DEBUG:
                print(f"[VOICE PROC] STT result: '{user_text}'")

            if not user_text or user_text.startswith("["):
                if VOICE_DEBUG:
                    print("[VOICE PROC] No valid speech text, returning early")
                return

            normalized_user_text = " ".join(user_text.split()).strip().lower()
            if normalized_user_text:
                now = time.monotonic()
                recent = self.voice_recent_text.get(user.id)
                if (
                    recent is not None
                    and recent[0] == normalized_user_text
                    and now - recent[1] <= self.voice_text_dedupe_sec
                ):
                    if VOICE_DEBUG:
                        print(
                            f"[VOICE PROC] Duplicate transcript skipped for {user.display_name}: {normalized_user_text}"
                        )
                    return
                self.voice_recent_text[user.id] = (normalized_user_text, now)

            if VOICE_DEBUG:
                print(f"[VOICE PROC] {user.display_name} said: {user_text}")

            # Generate response
            if VOICE_DEBUG:
                print("[VOICE PROC] Generating LLM response...")
            response = await self.generate_response(
                user_text, f"User: {user.display_name} in {user.guild.name}",
                user_id=user.id
            )
            if VOICE_DEBUG:
                print(f"[VOICE PROC] LLM response: '{response}'")

            logger.info(f"{user.display_name} said: {user_text}")

            # Store in memory and training data
            if hasattr(self, "memory_engine"):
                self.memory_engine.add_conversation(
                    user_text, response,
                    context=f"User: {user.display_name} in {user.guild.name}",
                    user_id=user.id
                )
            if hasattr(self, "training_engine"):
                self.training_engine.add_training_sample(
                    user_text, response,
                    user_id=user.id
                )

            # Convert response to speech
            if VOICE_DEBUG:
                print("[VOICE PROC] Converting response to TTS...")
            response_audio = await self.audio_processor.text_to_speech(response)
            if VOICE_DEBUG:
                print(f"[VOICE PROC] TTS generated {len(response_audio)} bytes")

            if response_audio and user.guild.id in self.voice_connections:
                voice_client = self.voice_connections[user.guild.id]

                # Save response_audio to a temporary file (edge-tts produces MP3)
                import uuid

                temp_audio_file_path = f"/tmp/voice_response_{uuid.uuid4().hex}.mp3"

                try:
                    with open(temp_audio_file_path, "wb") as f:
                        f.write(response_audio)

                    if VOICE_DEBUG:
                        print("[VOICE PROC] Playing audio to voice channel...")
                    # Play response - use the file path with mp3 codec for edge-tts MP3 files
                    if not voice_client.is_playing():
                        def _cleanup_playback_file(play_error):
                            if play_error:
                                logger.error(
                                    f"FFmpeg playback error for {user.display_name}: {play_error}"
                                )
                            try:
                                os.unlink(temp_audio_file_path)
                            except Exception:
                                pass

                        audio_source = discord.FFmpegPCMAudio(
                            temp_audio_file_path,
                            options="-vn",
                        )
                        voice_client.play(audio_source, after=_cleanup_playback_file)
                        if VOICE_DEBUG:
                            print("[VOICE PROC] Audio playback started")
                    else:
                        # No queueing yet; drop this clip and clean up.
                        os.unlink(temp_audio_file_path)
                except Exception as audio_error:
                    if VOICE_DEBUG:
                        print(f"[VOICE PROC] Audio playback error: {audio_error}")
                    logger.error(f"Error during audio playback: {audio_error}")

            if VOICE_DEBUG:
                print(f"[VOICE PROC] Processing complete for {user.display_name}")

        except Exception as e:
            if VOICE_DEBUG:
                print(f"[VOICE PROC] ERROR: {e}")
            logger.error(f"Voice processing error: {e}")
            # Handle specific voice stream corruption errors
            if "corrupted stream" in str(e).lower() or "opus" in str(e).lower():
                logger.warning(
                    "Voice stream corruption detected - this is normal during voice activity"
                )
            else:
                logger.error(f"Unexpected voice processing error: {e}")

    @commands.command()
    async def remember(self, ctx, *, fact):
        """Learn a new fact"""
        if ":=" in fact:
            key, value = fact.split(":=", 1)
            if hasattr(self, "memory_engine"):
                self.memory_engine.learn_fact(key.strip(), value.strip())
                await ctx.send(f"Learned: {key.strip()} = {value.strip()}")
            else:
                await ctx.send("Memory system not available")
        else:
            await ctx.send("Use format: `!remember key := value`")

    @commands.command()
    async def recall(self, ctx, *, query):
        """Recall relevant memories"""
        if not hasattr(self, "memory_engine"):
            await ctx.send("Memory system not available")
            return

        memories = self.memory_engine.search_relevant_memory(query, max_results=3)
        facts = self.memory_engine.get_relevant_facts(query)

        response = "**Relevant Memories:**\n"
        for i, mem in enumerate(memories, 1):
            response += (
                f"{i}. {mem['user_input'][:50]}... → {mem['bot_response'][:50]}...\n"
            )

        if facts:
            response += "\n**Relevant Facts:**\n"
            for key, fact in facts.items():
                response += f"- {key}: {fact['value']}\n"

        await ctx.send(response[:2000])  # Discord limit

    @commands.command()
    async def model(self, ctx, *, model_name: str = None):
        """Switch between different AI models or list available models"""
        if model_name is None:
            # List available models
            response = "**Available Models:**\n"
            for key, config in AVAILABLE_MODELS.items():
                current = "✅ " if key == self.current_model else "  "
                response += f"{current}`{key}` - {config['name']} ({config['type']})\n"
            response += f"\n**Current:** {self.model_config['name']}\n"
            response += f"**Usage:** `!model <model_name>`"
            await ctx.send(response)
            return

        # Switch to specified model
        if model_name.lower() in AVAILABLE_MODELS:
            self.current_model = model_name.lower()
            self.model_config = AVAILABLE_MODELS[self.current_model]
            self.base_personality = self.model_config["personality"]
            if hasattr(self, "personality"):
                self.personality.set_base_personality(self.base_personality)

            await ctx.send(f"🤖 Switched to {self.model_config['name']} model!")
            logger.info(
                f"Model switched to: {self.model_config['name']} ({self.model_config['type']})"
            )
        else:
            # Suggest similar models
            available = list(AVAILABLE_MODELS.keys())
            response = f"❌ Model '{model_name}' not found.\n"
            response += (
                f"**Available models:** {', '.join(f'`{m}`' for m in available)}\n"
            )
            response += f"**Usage:** `!model <model_name>`"
            await ctx.send(response)

    @commands.command()
    async def status(self, ctx):
        """Show bot status and configuration"""
        embed = discord.Embed(title="🤖 Bot Status", color=discord.Color.blue())

        embed.add_field(
            name="🧠 Current Model", value=f"{self.model_config['name']}", inline=True
        )
        embed.add_field(
            name="🔧 Model Type", value=self.model_config["type"], inline=True
        )
        embed.add_field(
            name="💾 Memory Size",
            value=f"{len(self.memory_engine.memory_data.get('conversations', []))} conversations",
            inline=True,
        )
        embed.add_field(
            name="🎙️ Voice Connections",
            value=str(len(self.voice_connections)),
            inline=True,
        )
        embed.add_field(name="🔗 LLM Endpoint", value=LOCAL_LLM_URL, inline=True)

        # Check if LLM is responsive
        try:
            async with aiohttp.ClientSession() as session:
                if self.model_config["type"] == "ollama":
                    # Check Ollama API
                    async with session.get(
                        f"{LOCAL_LLM_URL}/api/tags",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        status = "🟢 Online" if resp.status == 200 else "🟡 Issues"
                else:
                    # Generic health check
                    async with session.get(
                        f"{LOCAL_LLM_URL}/", timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        status = "🟢 Online" if resp.status < 500 else "🟡 Issues"
        except:
            status = "🔴 Offline"

        embed.add_field(name="🌐 LLM Status", value=status, inline=True)
        embed.set_footer(text=f"Bot ID: {self.user.id}")

        await ctx.send(embed=embed)


def main():
    """Main entry point"""
    if not DISCORD_TOKEN:
        logger.error("DISCORD_TOKEN environment variable not set!")
        return

    if not acquire_instance_lock():
        return

    bot = VoiceBot()

    # Manually add commands
    @bot.command(name="chat")
    async def chat_cmd(ctx, *, message: str = None):
        """Chat with the AI bot"""
        if message is None:
            await ctx.send("Please provide a message: `!chat <message>`")
            return

        async with ctx.typing():
            try:
                response = await bot.generate_response(
                    message, 
                    f"User: {ctx.author.display_name} in {ctx.guild.name}",
                    user_id=ctx.author.id
                )
                bot.memory_engine.add_conversation(
                    message, response,
                    context=f"User: {ctx.author.display_name} in {ctx.guild.name}",
                    user_id=ctx.author.id
                )
                bot.training_engine.add_training_sample(
                    message, response,
                    user_id=ctx.author.id
                )
                await ctx.send(response)
            except Exception as e:
                logger.error(f"Chat command error: {e}")
                await ctx.send("Sorry, I had trouble processing that.")

    @bot.command(name="join")
    async def join_cmd(ctx):
        """Join the voice channel"""
        if not voice_enabled:
            await ctx.send(
                "❌ Voice features are fully disabled. Check console for details and ensure ENABLE_VOICE=true in .env."
            )
            return

        if not ctx.author.voice:
            await ctx.send("You need to be in a voice channel first!")
            return

        channel = ctx.author.voice.channel

        if voice_receive_enabled and voice_recv is not None:
            # Full voice functionality (send and receive)
            try:
                voice_client = await channel.connect(cls=voice_recv.VoiceRecvClient)
                voice_client.listen(RobustVoiceSink(bot.on_voice_receive))
                bot.voice_connections[ctx.guild.id] = voice_client
                await ctx.send(f"✅ Joined {channel.name} with voice receive enabled!")
            except Exception as e:
                logger.error(
                    f"Failed to connect with voice receive: {e}. Falling back to send-only."
                )
                voice_client = await channel.connect()
                bot.voice_connections[ctx.guild.id] = voice_client
                await ctx.send(
                    f"⚠️ Joined {channel.name} (voice send only - receive failed to start)."
                )
        else:
            # Fallback to regular voice connection (send only)
            voice_client = await channel.connect()
            bot.voice_connections[ctx.guild.id] = voice_client
            await ctx.send(
                f"✅ Joined {channel.name} (voice send only - receive not available)."
            )

    @bot.command(name="leave")
    async def leave_cmd(ctx):
        """Leave the voice channel"""
        if ctx.guild.id in bot.voice_connections:
            await bot.voice_connections[ctx.guild.id].disconnect()
            del bot.voice_connections[ctx.guild.id]
            await ctx.send("Left voice channel!")
        else:
            await ctx.send("I'm not in any voice channel!")

    @bot.command(name="personality")
    async def personality_cmd(ctx, *, personality_text: str = None):
        """Set or view custom personality"""
        if personality_text is None:
            current_personality = bot.base_personality
            if CUSTOM_PERSONALITY:
                source = "Custom (.env)"
            else:
                source = f"{bot.model_config['name']} default"

            response = f"**Current Personality** (from {source}):\n```\n{current_personality[:500]}{'...' if len(current_personality) > 500 else ''}\n```"
            response += f"\n**Usage:** `!personality <your custom personality text>`"
            await ctx.send(response)
        else:
            # Temporarily set personality for current session
            bot.base_personality = personality_text
            if hasattr(bot, "personality"):
                bot.personality.set_base_personality(bot.base_personality)
            await ctx.send(
                "✅ Personality updated for this session! (Restart bot or edit .env to make permanent)"
            )
            logger.info(f"Personality temporarily updated by {ctx.author.display_name}")

    @bot.command(name="tts")
    async def tts_cmd(ctx, *, engine: str = None):
        """Set or view TTS engine"""
        global TTS_ENGINE
        if engine is None:
            # Show current TTS status
            response = f"**TTS Status:**\n"
            response += f"**Available:** {'Yes' if TTS_AVAILABLE else 'No'}\n"
            response += f"**Edge TTS:** {'✅' if EDGE_TTS_AVAILABLE else '❌'}\n"
            response += f"**pyttsx3 TTS:** {'✅' if PYTTSX3_AVAILABLE else '❌'}\n"
            response += f"**Current Engine:** `{TTS_ENGINE}`\n\n"
            response += "**Available engines:**\n"
            response += "- `auto` (tries edge-tts first, fallback to pyttsx3)\n"
            if EDGE_TTS_AVAILABLE:
                response += "- `edge` (Microsoft Edge TTS - online, high quality)\n"
            if PYTTSX3_AVAILABLE:
                response += "- `pyttsx3` (offline TTS - works without internet)\n"
            response += f"\n**Usage:** `{BOT_PREFIX}tts <engine>`"
            await ctx.send(response)
            return

        # Set TTS engine
        engine = engine.lower()
        if engine not in ["auto", "edge", "pyttsx3"]:
            await ctx.send(f"❌ Invalid engine. Use: `auto`, `edge`, or `pyttsx3`")
            return

        # Check if requested engine is available
        if engine == "edge" and not EDGE_TTS_AVAILABLE:
            await ctx.send(
                "❌ Edge TTS not available. Install with: `pip install edge-tts`"
            )
            return
        if engine == "pyttsx3" and not PYTTSX3_AVAILABLE:
            await ctx.send(
                "❌ pyttsx3 not available. Install with: `pip install pyttsx3`"
            )
            return

        # Update TTS engine (note: this only updates for current session)
        TTS_ENGINE = engine
        await ctx.send(f"✅ TTS engine set to: `{engine}`")

    @bot.command(name="model")
    async def model_cmd(ctx, *, model_name: str = None):
        """Switch between different AI models or list available models"""
        if model_name is None:
            # List available models
            response = "**Available Models:**\n"
            for key, config in AVAILABLE_MODELS.items():
                current = "✅ " if key == bot.current_model else "  "
                response += f"{current}`{key}` - {config['name']} ({config['type']})\n"
            response += f"\n**Current:** {bot.model_config['name']}\n"
            response += f"**Usage:** `!model <model_name>`"
            await ctx.send(response)
            return

        # Switch to specified model
        if model_name.lower() in AVAILABLE_MODELS:
            bot.current_model = model_name.lower()
            bot.model_config = AVAILABLE_MODELS[bot.current_model]

            # Update personality (use custom if exists, otherwise new model's default)
            if CUSTOM_PERSONALITY:
                bot.base_personality = CUSTOM_PERSONALITY
            else:
                bot.base_personality = bot.model_config["personality"]
            if hasattr(bot, "personality"):
                bot.personality.set_base_personality(bot.base_personality)

            await ctx.send(f"🤖 Switched to {bot.model_config['name']} model!")
            logger.info(
                f"Model switched to: {bot.model_config['name']} ({bot.model_config['type']})"
            )
        else:
            # Suggest similar models
            available = list(AVAILABLE_MODELS.keys())
            response = f"❌ Model '{model_name}' not found.\n"
            response += (
                f"**Available models:** {', '.join(f'`{m}`' for m in available)}\n"
            )
            response += f"**Usage:** `!model <model_name>`"
            await ctx.send(response)

    @bot.command(name="scrape")
    async def scrape_cmd(ctx, action: str = None, *, args: str = None):
        """Scrape server conversations for training data (owner only)
        
        Usage:
        !scrape preview <guild_id> - Preview what would be scraped
        !scrape run <guild_id> - Run the scraper
        !scrape import <file> - Import scraped data
        !scrape stats - Show scraped data stats
        """
        # Check if user is owner
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
            if not args:
                await ctx.send("Usage: `!scrape preview <guild_id>`")
                return
            try:
                guild_id = int(args.strip())
            except ValueError:
                await ctx.send("Invalid guild ID")
                return
            
            guild = bot.get_guild(guild_id)
            if guild:
                text_channels = [c for c in guild.text_channels 
                                if c.permissions_for(guild.me).read_message_history]
                embed = discord.Embed(
                    title=f"📊 Preview: {guild.name}",
                    color=discord.Color.blue()
                )
                embed.add_field(name="Text Channels", value=str(len(text_channels)), inline=True)
                embed.add_field(name="Members", value=str(guild.member_count), inline=True)
                await ctx.send(embed=embed)
            else:
                await ctx.send("Could not access guild. Make sure the bot is in the server.")
            return
        
        if action == "run":
            if not args:
                await ctx.send("Usage: `!scrape run <guild_id>`")
                return
            
            parts = args.split()
            try:
                guild_id = int(parts[0])
            except ValueError:
                await ctx.send("Invalid guild ID")
                return
            
            await ctx.send(f"⚠️ Starting scrape for guild {guild_id}... (Background task)")
            
            # Run scraper in background
            async def run_scraper():
                try:
                    from src.utils.server_scraper import ServerScraper
                    from dotenv import load_dotenv
                    load_dotenv()
                    token = os.getenv("DISCORD_TOKEN")
                    
                    scraper = ServerScraper(token=token)
                    result = await scraper.scrape(guild_id)
                    
                    output_file = f"scraped_guild_{guild_id}.json"
                    scraper.export_samples(output_file, format="alpaca")
                    
                    logger.info(f"Scraping complete: {len(result.get('samples', []))} samples")
                except Exception as e:
                    logger.error(f"Scraper error: {e}")
            
            asyncio.create_task(run_scraper())
            await ctx.send(f"✅ Scraping started in background.")
            return
        
        if action == "import":
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
                
                imported = 0
                for sample in scraped_data:
                    if isinstance(sample, dict):
                        bot.training_engine.add_training_sample(
                            user_input=sample.get('input_text', ''),
                            bot_response=sample.get('output_text', ''),
                            quality_score=sample.get('quality_score', 0.5)
                        )
                        imported += 1
                
                await ctx.send(f"✅ Imported {imported} samples from `{file_path}`")
            except Exception as e:
                await ctx.send(f"❌ Error importing: {e}")
            return
        
        if action == "stats":
            scraped_files = [f for f in os.listdir('.') 
                           if f.startswith('scraped_') and f.endswith('.json')]
            
            if not scraped_files:
                await ctx.send("No scraped data files found.")
                return
            
            embed = discord.Embed(title="📊 Scraped Data", color=discord.Color.blue())
            total = 0
            for f in scraped_files:
                try:
                    with open(f, 'r') as fp:
                        data = json.load(fp)
                    count = len(data) if isinstance(data, list) else 0
                    total += count
                    embed.add_field(name=f, value=f"{count} samples", inline=False)
                except:
                    pass
            embed.add_field(name="Total", value=str(total), inline=True)
            await ctx.send(embed=embed)
            return
        
        await ctx.send("Unknown action. Use `!scrape` for help.")

    # Add slash commands
    @bot.tree.command(name="chat", description="Chat with the AI bot")
    async def chat_slash(interaction: discord.Interaction, message: str):
        """Slash command for chat"""
        await interaction.response.defer()

        try:
            response = await bot.generate_response(
                message,
                f"User: {interaction.user.display_name} in {interaction.guild.name}",
                user_id=interaction.user.id,
            )
            bot.memory_engine.add_conversation(
                message, response,
                context=f"User: {interaction.user.display_name} in {interaction.guild.name}",
                user_id=interaction.user.id
            )
            bot.training_engine.add_training_sample(
                message, response,
                user_id=interaction.user.id
            )
            await interaction.followup.send(response)
        except Exception as e:
            logger.error(f"Slash chat error: {e}")
            await interaction.followup.send("Sorry, I had trouble processing that.")

    @bot.tree.command(name="join", description="Join the voice channel")
    async def join_slash(interaction: discord.Interaction):
        """Slash command to join voice"""
        await interaction.response.defer()

        if not voice_enabled:
            await interaction.followup.send(
                "❌ Voice features are fully disabled. Check console for details and ensure ENABLE_VOICE=true in .env."
            )
            return

        if not interaction.user.voice:
            await interaction.followup.send("You need to be in a voice channel first!")
            return

        channel = interaction.user.voice.channel

        if voice_receive_enabled and voice_recv is not None:
            # Full voice functionality (send and receive)
            try:
                voice_client = await channel.connect(cls=voice_recv.VoiceRecvClient)
                voice_client.listen(RobustVoiceSink(bot.on_voice_receive))
                bot.voice_connections[interaction.guild.id] = voice_client
                await interaction.followup.send(
                    f"✅ Joined {channel.name} with voice receive enabled!"
                )
            except Exception as e:
                logger.error(
                    f"Failed to connect with voice receive: {e}. Falling back to send-only."
                )
                voice_client = await channel.connect()
                bot.voice_connections[interaction.guild.id] = voice_client
                await interaction.followup.send(
                    f"⚠️ Joined {channel.name} (voice send only - receive failed to start)."
                )
        else:
            # Fallback to regular voice connection (send only)
            voice_client = await channel.connect()
            bot.voice_connections[interaction.guild.id] = voice_client
            await interaction.followup.send(
                f"✅ Joined {channel.name} (voice send only - receive not available)."
            )

    @bot.tree.command(name="leave", description="Leave the voice channel")
    async def leave_slash(interaction: discord.Interaction):
        """Slash command to leave voice"""
        await interaction.response.defer()

        if interaction.guild.id in bot.voice_connections:
            await bot.voice_connections[interaction.guild.id].disconnect()
            del bot.voice_connections[interaction.guild.id]
            await interaction.followup.send("Left voice channel!")
        else:
            await interaction.followup.send("I'm not in any voice channel!")

    @bot.tree.command(name="personality", description="Set or view custom personality")
    async def personality_slash(
        interaction: discord.Interaction, personality: str = None
    ):
        """Slash command for personality"""
        await interaction.response.defer()

        if personality is None:
            current_personality = bot.base_personality
            if CUSTOM_PERSONALITY:
                source = "Custom (.env)"
            else:
                source = f"{bot.model_config['name']} default"

            response = f"**Current Personality** (from {source}):\n```\n{current_personality[:500]}{'...' if len(current_personality) > 500 else ''}\n```"
            await interaction.followup.send(response)
        else:
            bot.base_personality = personality
            if hasattr(bot, "personality"):
                bot.personality.set_base_personality(bot.base_personality)
            await interaction.followup.send("✅ Personality updated for this session!")
            logger.info(
                f"Personality updated via slash by {interaction.user.display_name}"
            )

    @bot.tree.command(name="model", description="Switch between different AI models")
    async def model_slash(interaction: discord.Interaction, model: str = None):
        """Slash command for model switching"""
        await interaction.response.defer()

        if model is None:
            response = "**Available Models:**\n"
            for key, config in AVAILABLE_MODELS.items():
                current = "✅ " if key == bot.current_model else "  "
                response += f"{current}`{key}` - {config['name']} ({config['type']})\n"
            response += f"\n**Current:** {bot.model_config['name']}"
            await interaction.followup.send(response)
            return

        if model.lower() in AVAILABLE_MODELS:
            bot.current_model = model.lower()
            bot.model_config = AVAILABLE_MODELS[bot.current_model]

            if CUSTOM_PERSONALITY:
                bot.base_personality = CUSTOM_PERSONALITY
            else:
                bot.base_personality = bot.model_config["personality"]
            if hasattr(bot, "personality"):
                bot.personality.set_base_personality(bot.base_personality)

            await interaction.followup.send(
                f"🤖 Switched to {bot.model_config['name']} model!"
            )
            logger.info(f"Model switched via slash: {bot.model_config['name']}")
        else:
            available = list(AVAILABLE_MODELS.keys())
            response = f"❌ Model '{model}' not found.\n**Available:** {', '.join(f'`{m}`' for m in available)}"
            await interaction.followup.send(response)

    @bot.tree.command(name="status", description="Show bot status and configuration")
    async def status_slash(interaction: discord.Interaction):
        """Slash command for status"""
        await interaction.response.defer()

        embed = discord.Embed(title="🤖 Bot Status", color=discord.Color.blue())
        embed.add_field(
            name="🧠 Current Model", value=f"{bot.model_config['name']}", inline=True
        )
        embed.add_field(
            name="🔧 Model Type", value=bot.model_config["type"], inline=True
        )
        embed.add_field(
            name="💾 Memory Size",
            value=f"{len(bot.memory_engine.memory_data.get('conversations', []))} conversations",
            inline=True,
        )
        embed.add_field(
            name="🎙️ Voice Connections",
            value=str(len(bot.voice_connections)),
            inline=True,
        )
        embed.add_field(name="🔗 LLM Endpoint", value=LOCAL_LLM_URL, inline=True)

        # Check LLM status
        try:
            async with aiohttp.ClientSession() as session:
                if bot.model_config["type"] == "ollama":
                    async with session.get(
                        f"{LOCAL_LLM_URL}/api/tags",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        status = "🟢 Online" if resp.status == 200 else "🟡 Issues"
                else:
                    async with session.get(
                        f"{LOCAL_LLM_URL}/", timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        status = "🟢 Online" if resp.status < 500 else "🟡 Issues"
        except:
            status = "🔴 Offline"

        embed.add_field(name="🌐 LLM Status", value=status, inline=True)
        embed.set_footer(text=f"Bot ID: {bot.user.id}")

        await interaction.followup.send(embed=embed)

    @bot.tree.command(name="remember", description="Learn a new fact")
    async def remember_slash(interaction: discord.Interaction, key: str, value: str):
        """Slash command for remembering facts"""
        await interaction.response.defer()

        bot.memory_engine.learn_fact(key, value)
        await interaction.followup.send(f"Learned: {key} = {value}")

    @bot.tree.command(name="recall", description="Recall relevant memories")
    async def recall_slash(interaction: discord.Interaction, query: str):
        """Slash command for recalling memories"""
        await interaction.response.defer()

        memories = bot.memory_engine.search_relevant_memory(query, max_results=3)
        facts = bot.memory_engine.get_relevant_facts(query)

        response = "**Relevant Memories:**\n"
        for i, mem in enumerate(memories, 1):
            response += (
                f"{i}. {mem['user_input'][:50]}... → {mem['bot_response'][:50]}...\n"
            )

        if facts:
            response += "\n**Relevant Facts:**\n"
            for key, fact in facts.items():
                response += f"- {key}: {fact['value']}\n"

        await interaction.followup.send(response[:2000])

    # Sync commands on startup
    async def setup_hook():
        # Register slash commands
        @bot.tree.command(name="train", description="Train the bot with custom response examples")
        async def train_cmd(interaction: discord.Interaction, input_text: str, response: str):
            """Train the bot with a conversation example"""
            try:
                # Add to training data
                bot.training_engine.add_training_sample(
                    user_input=input_text,
                    bot_response=response,
                    conversation_id=f"manual_{interaction.id}",
                    user_id=interaction.user.id
                )

                await interaction.response.send_message(
                    f"✅ Added training example!\n**Input:** {input_text}\n**Response:** {response}",
                    ephemeral=True
                )
                logger.info(f"Training example added by {interaction.user}: {input_text[:50]} -> {response[:50]}")
            except Exception as e:
                logger.error(f"Train command error: {e}")
                await interaction.response.send_message("Failed to add training example.", ephemeral=True)

        @bot.tree.command(name="train_stats", description="View training data statistics")
        async def train_stats_cmd(interaction: discord.Interaction):
            """Show training data stats"""
            try:
                stats = bot.training_engine.get_training_stats()
                response = "**Training Data Stats:**\n"
                response += f"Total samples: {stats.get('total_samples', 0)}\n"
                response += f"Quality score: {stats.get('average_quality', 0):.2f}\n"
                response += f"Conversations: {stats.get('conversation_count', 0)}\n"
                response += f"Avg conversation length: {stats.get('avg_conversation_length', 0):.1f}\n"
                await interaction.response.send_message(response, ephemeral=True)
            except Exception as e:
                logger.error(f"Train stats error: {e}")
                await interaction.response.send_message("Failed to get stats.", ephemeral=True)

        @bot.tree.command(name="export_train", description="Export training data as JSON")
        async def export_train_cmd(interaction: discord.Interaction):
            """Export training data for sharing"""
            try:
                output_file = f"training_export_{interaction.guild.id}.json"
                bot.training_engine.export_for_fine_tuning(output_file=output_file)

                with open(output_file, 'r') as f:
                    data = json.load(f)

                # Send as file
                file = discord.File(output_file, filename="training_data.json")
                await interaction.response.send_message(
                    f"Exported {len(data)} training samples.",
                    file=file,
                    ephemeral=True
                )
                # Clean up
                import os
                os.remove(output_file)
            except Exception as e:
                logger.error(f"Export train error: {e}")
                await interaction.response.send_message("Failed to export.", ephemeral=True)

        @bot.tree.command(name="examples", description="Show training examples for a given input")
        async def examples_cmd(interaction: discord.Interaction, query: str):
            """Show relevant training examples for an input"""
            try:
                examples = bot.training_engine.search_training_examples(
                    query, max_results=5, min_quality=0.5
                )

                if not examples:
                    await interaction.response.send_message(
                        "No matching training examples found.",
                        ephemeral=True
                    )
                    return

                response = f"**Training examples for:** \"{query}\"\n\n"
                for i, ex in enumerate(examples, 1):
                    response += f"**{i}.** (similarity: {ex.get('similarity', 0):.2f}, quality: {ex.get('quality_score', 1):.2f})\n"
                    response += f"   Input: {ex.get('input', '')}\n"
                    response += f"   Response: {ex.get('response', '')}\n\n"

                # Truncate if too long
                if len(response) > 1900:
                    response = response[:1900] + "...\n\n*(truncated)*"

                await interaction.response.send_message(response, ephemeral=True)
            except Exception as e:
                logger.error(f"Examples command error: {e}")
                await interaction.response.send_message("Failed to find examples.", ephemeral=True)

        @bot.tree.command(name="rate", description="Rate a response's human-likeness (0=AI, 10=human)")
        async def rate_cmd(interaction: discord.Interaction, user_input: str, response: str, score: int):
            """Rate how human-like a response sounds"""
            try:
                score = max(0, min(10, score))
                bot.training_engine.add_training_sample(
                    user_input=user_input,
                    bot_response=response,
                    conversation_id=f"rated_{interaction.id}",
                    user_id=interaction.user.id,
                    human_score=score
                )
                
                human_score = bot.training_engine._calculate_human_score(response)
                
                response_msg = f"✅ Rated response: **{score}/10**\n"
                response_msg += f"Auto-detected human-score: **{human_score:.1f}/10**\n"
                response_msg += f"\n**Input:** {user_input}\n"
                response_msg += f"**Response:** {response}"
                
                await interaction.response.send_message(response_msg, ephemeral=True)
            except Exception as e:
                logger.error(f"Rate command error: {e}")
                await interaction.response.send_message("Failed to rate.", ephemeral=True)

        @bot.tree.command(name="auto_train", description="Auto-extract good training examples from this server")
        async def auto_train_cmd(interaction: discord.Interaction):
            """Automatically extract human-like responses as training examples"""
            try:
                # Get recent memories/conversations
                recent_convos = []
                if hasattr(bot.memory_engine, 'memory'):
                    for conv in bot.memory_engine.memory.get('conversations', [])[-100:]:
                        recent_convos.append({
                            'user_input': conv.get('user_input', ''),
                            'bot_response': conv.get('bot_response', '')
                        })
                
                if not recent_convos:
                    await interaction.response.send_message(
                        "No recent conversations found to extract from.",
                        ephemeral=True
                    )
                    return
                
                # Auto-extract human-like examples
                extracted = bot.training_engine.auto_extract_from_conversations(
                    recent_convos, min_human_score=6.0
                )
                
                # Add them to training
                added = 0
                for ex in extracted:
                    bot.training_engine.add_training_sample(
                        user_input=ex['user_input'],
                        bot_response=ex['bot_response'],
                        conversation_id=f"auto_{interaction.id}",
                        user_id=interaction.user.id,
                        human_score=ex.get('human_score', 6.0),
                        server_id=str(interaction.guild.id)
                    )
                    added += 1
                
                await interaction.response.send_message(
                    f"✅ Auto-extracted **{added}** human-like responses as training examples!\n"
                    f"(Threshold: 6.0+ human-score)",
                    ephemeral=True
                )
            except Exception as e:
                logger.error(f"Auto train error: {e}")
                await interaction.response.send_message("Failed to auto-train.", ephemeral=True)

        @bot.tree.command(name="leaderboard", description="Show training data leaderboard")
        async def leaderboard_cmd(interaction: discord.Interaction):
            """Show training stats and leaderboard"""
            try:
                stats = bot.training_engine.get_training_stats()
                
                if "error" in stats:
                    await interaction.response.send_message("No training data yet!", ephemeral=True)
                    return
                
                response = "**Training Leaderboard**\n\n"
                response += f"**Total Samples:** {stats['total_samples']}\n"
                response += f"**Avg Human Score:** {stats['human_score_stats']['avg']:.2f}/10\n"
                response += f"**Avg Quality:** {stats['quality_stats']['avg']:.2f}\n\n"
                
                dist = stats['human_score_stats']['distribution']
                response += "**Human Score Distribution:**\n"
                response += f"🤖 Very AI (0-3): {dist['very_ai (0-3)']}\n"
                response += f"😐 Neutral (3-7): {dist['neutral (3-7)']}\n"
                response += f"👤 Very Human (7-10): {dist['very_human (7-10)']}\n"
                
                # Per-server stats
                if stats.get('server_stats'):
                    response += "\n**Per-Server Stats:**\n"
                    for server_id, sstats in list(stats['server_stats'].items())[:5]:
                        # Try to get server name
                        server = bot.get_guild(int(server_id))
                        name = server.name if server else server_id[:8]
                        response += f"• {name}: {sstats['count']} samples, avg human: {sstats['avg_human_score']:.1f}\n"
                
                await interaction.response.send_message(response, ephemeral=True)
            except Exception as e:
                logger.error(f"Leaderboard error: {e}")
                await interaction.response.send_message("Failed to get leaderboard.", ephemeral=True)

        @bot.tree.command(name="set_human_score", description="Set human-likeness score for a training example")
        async def set_human_score_cmd(interaction: discord.Interaction, input_text: str, score: int):
            """Update human-likeness score for an existing training example"""
            try:
                score = max(0, min(10, score))
                success = bot.training_engine.set_human_score(input_text, score)
                
                if success:
                    await interaction.response.send_message(
                        f"✅ Updated human score to **{score}/10** for: \"{input_text[:50]}...\"",
                        ephemeral=True
                    )
                else:
                    await interaction.response.send_message(
                        "No matching training example found.",
                        ephemeral=True
                    )
            except Exception as e:
                logger.error(f"Set human score error: {e}")
                await interaction.response.send_message("Failed to update score.", ephemeral=True)

        # Humanize slash command
        @bot.tree.command(name="humanize", description="Generate random responses to rate for training")
        async def humanize_cmd(interaction: discord.Interaction):
            """Start a humanize session with random input"""
            try:
                user_id = str(interaction.user.id)
                import time
                now = time.time()
                
                # Clean old sessions (>5 minutes)
                bot.humanize_sessions = {
                    uid: sess for uid, sess in bot.humanize_sessions.items()
                    if now - sess.get('timestamp', 0) < 300
                }
                
                user_input = bot._get_random_user_input()
                
                await interaction.response.defer(thinking=True)
                variations = await bot._generate_response_variations(user_input, num_variations=3)
                
                if not variations or len(variations) < 3:
                    variations = ["mood", "based", "ngl"]
                
                session = {
                    "input": user_input,
                    "options": variations,
                    "selected": None,
                    "timestamp": now,
                    "channel_id": interaction.channel.id,
                }
                bot.humanize_sessions[user_id] = session
                
                response = "**User said:** \"" + user_input + "\"\n\n"
                for i, var in enumerate(variations, 1):
                    response += "**" + str(i) + ".** " + var + "\n"
                response += "\nWhich did you like best? Reply with **1**, **2**, or **3**"
                
                await interaction.followup.send(response)
            except Exception as e:
                logger.error(f"Humanize error: {e}")
                await interaction.followup.send("Failed to generate options.")

        # Humanize custom input
        @bot.tree.command(name="humanize_custom", description="Generate responses for a custom input")
        async def humanize_custom_cmd(interaction: discord.Interaction, user_input: str):
            """Generate variations for custom input"""
            try:
                user_id = str(interaction.user.id)
                import time
                now = time.time()
                
                await interaction.response.defer(thinking=True)
                variations = await bot._generate_response_variations(user_input, num_variations=3)
                
                if not variations or len(variations) < 3:
                    variations = ["mood", "based", "ngl"]
                
                session = {
                    "input": user_input,
                    "options": variations,
                    "selected": None,
                    "timestamp": now,
                    "channel_id": interaction.channel.id,
                }
                bot.humanize_sessions[user_id] = session
                
                response = "**Input:** \"" + user_input + "\"\n\n"
                for i, var in enumerate(variations, 1):
                    response += "**" + str(i) + ".** " + var + "\n"
                response += "\nWhich did you like best? Reply with **1**, **2**, or **3**"
                
                await interaction.followup.send(response)
            except Exception as e:
                logger.error(f"Humanize custom error: {e}")
                await interaction.followup.send("Failed to generate options.")

        # Cancel humanize session
        @bot.tree.command(name="cancel_humanize", description="Cancel current humanize session")
        async def cancel_humanize_cmd(interaction: discord.Interaction):
            """Cancel active humanize session"""
            user_id = str(interaction.user.id)
            if user_id in bot.humanize_sessions:
                del bot.humanize_sessions[user_id]
                await interaction.response.send_message("Session cancelled.", ephemeral=True)
            else:
                await interaction.response.send_message("No active session.", ephemeral=True)

        # Message handler for selecting/rating humanize options
        @bot.event
        async def on_message(message):
            """Handle humanize session messages"""
            if message.author.bot or not message.guild:
                return
            
            user_id = str(message.author.id)
            
            if user_id in bot.humanize_sessions:
                session = bot.humanize_sessions[user_id]
                content = message.content.strip()
                
                # Check for selection (1, 2, or 3)
                if content in ["1", "2", "3"]:
                    try:
                        idx = int(content) - 1
                        selected_response = session["options"][idx]
                        session["selected"] = selected_response
                        
                        response = "**You picked:** " + selected_response + "\n\n"
                        response += "Rate this response (0-10):\n"
                        response += "0-3 = Very AI-like\n"
                        response += "4-6 = Neutral\n"
                        response += "7-10 = Very Human-like\n\n"
                        response += "Reply with a number like **7** or **3/10**"
                        
                        await message.channel.send(response)
                        return
                    except (IndexError, ValueError):
                        pass
                
                # Check for rating
                import re
                rating_match = re.match(r'^(\d+)(?:/\d+)?$', content)
                if rating_match and session.get("selected"):
                    score = int(rating_match.group(1))
                    score = max(0, min(10, score))
                    
                    bot.training_engine.add_training_sample(
                        user_input=session["input"],
                        bot_response=session["selected"],
                        conversation_id="humanize_" + str(message.id),
                        user_id=message.author.id,
                        human_score=score
                    )
                    
                    auto_score = bot.training_engine._calculate_human_score(session["selected"])
                    
                    response = "**Saved!**\n"
                    response += "You rated: " + str(score) + "/10\n"
                    response += "Auto-detected: " + str(round(auto_score, 1)) + "/10\n"
                    response += "\nRun /humanize for another round!"
                    
                    await message.channel.send(response)
                    del bot.humanize_sessions[user_id]
                    return
            
            # Continue with normal message processing
            await VoiceBot.on_message(bot, message)

        # Prefix command for humanize
        @bot.command(name="humanize")
        async def humanize_prefix_cmd(ctx):
            """Generate random responses to rate"""
            try:
                user_id = str(ctx.author.id)
                import time
                now = time.time()
                
                bot.humanize_sessions = {
                    uid: sess for uid, sess in bot.humanize_sessions.items()
                    if now - sess.get('timestamp', 0) < 300
                }
                
                user_input = bot._get_random_user_input()
                variations = ["mood", "based", "ngl"]
                try:
                    variations = await bot._generate_response_variations(user_input, num_variations=3)
                except:
                    pass
                
                bot.humanize_sessions[user_id] = {
                    "input": user_input,
                    "options": variations,
                    "selected": None,
                    "timestamp": now,
                    "channel_id": ctx.channel.id,
                }
                
                response = "**User said:** \"" + user_input + "\"\n\n"
                for i, var in enumerate(variations, 1):
                    response += "**" + str(i) + ".** " + var + "\n"
                response += "\nWhich did you like best? Reply with **1**, **2**, or **3**"
                
                await ctx.send(response)
            except Exception as e:
                logger.error(f"Humanize prefix error: {e}")
                await ctx.send("Failed.")

        # Sync and start
        async def setup_hook():
            await bot.tree.sync()
            logger.info("Commands synced successfully!")

        bot.setup_hook = setup_hook
        bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
