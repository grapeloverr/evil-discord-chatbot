"""
Simplified memory system for Discord bot
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryManager:
    """Simplified memory system with SQLite persistence"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = Path(config.get("file", "memory.db"))
        self.max_conversations = config.get("max_conversations", 1000)
        self.max_message_length = config.get("max_message_length", 2000)
        self.persistence = config.get("persistence", True)
        
        if self.persistence:
            self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        channel_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        user_input TEXT NOT NULL,
                        bot_response TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        context TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS facts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT UNIQUE NOT NULL,
                        value TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conversations_channel_user 
                    ON conversations(channel_id, user_id)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conversations_timestamp 
                    ON conversations(timestamp)
                """)
                
                conn.commit()
                logger.info(f"Memory database initialized: {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize memory database: {e}")
            raise
    
    def add_conversation(self, channel_id: str, user_id: str, user_input: str, 
                      bot_response: str, context: str = ""):
        """Add a conversation to memory"""
        try:
            # Truncate long messages
            user_input = user_input[:self.max_message_length]
            bot_response = bot_response[:self.max_message_length]
            
            if self.persistence:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO conversations 
                        (channel_id, user_id, user_input, bot_response, context)
                        VALUES (?, ?, ?, ?, ?)
                    """, (channel_id, user_id, user_input, bot_response, context))
                    
                    # Clean up old conversations if exceeding limit
                    conn.execute("""
                        DELETE FROM conversations 
                        WHERE id NOT IN (
                            SELECT id FROM conversations 
                            ORDER BY timestamp DESC 
                            LIMIT ?
                        )
                    """, (self.max_conversations,))
                    
                    conn.commit()
            else:
                # In-memory mode (not implemented for now)
                logger.debug("Memory persistence disabled, conversation not stored")
                
        except Exception as e:
            logger.error(f"Failed to add conversation to memory: {e}")
    
    def get_recent_conversations(self, channel_id: str, user_id: str, 
                             limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversations for a user in a channel"""
        try:
            if not self.persistence:
                return []
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT user_input, bot_response, timestamp, context
                    FROM conversations
                    WHERE channel_id = ? AND user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (channel_id, user_id, limit))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get recent conversations: {e}")
            return []
    
    def search_conversations(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search conversations by content"""
        try:
            if not self.persistence:
                return []
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT user_input, bot_response, timestamp, channel_id, user_id
                    FROM conversations
                    WHERE user_input LIKE ? OR bot_response LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", limit))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to search conversations: {e}")
            return []
    
    def learn_fact(self, key: str, value: str):
        """Store a fact in memory"""
        try:
            if not self.persistence:
                return
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO facts (key, value)
                    VALUES (?, ?)
                """, (key, value))
                conn.commit()
                
            logger.info(f"Learned fact: {key}")
            
        except Exception as e:
            logger.error(f"Failed to learn fact: {e}")
    
    def get_fact(self, key: str) -> Optional[str]:
        """Retrieve a fact from memory"""
        try:
            if not self.persistence:
                return None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT value FROM facts WHERE key = ?
                """, (key,))
                
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Failed to get fact: {e}")
            return None
    
    def search_facts(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search facts by content"""
        try:
            if not self.persistence:
                return []
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT key, value, timestamp
                    FROM facts
                    WHERE key LIKE ? OR value LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", limit))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to search facts: {e}")
            return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            if not self.persistence:
                return {"conversations": 0, "facts": 0}
            
            with sqlite3.connect(self.db_path) as conn:
                conv_count = conn.execute("""
                    SELECT COUNT(*) FROM conversations
                """).fetchone()[0]
                
                fact_count = conn.execute("""
                    SELECT COUNT(*) FROM facts
                """).fetchone()[0]
                
                return {
                    "conversations": conv_count,
                    "facts": fact_count,
                    "max_conversations": self.max_conversations,
                    "persistence_enabled": self.persistence
                }
                
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"conversations": 0, "facts": 0}
    
    def clear_memory(self, channel_id: str = None, user_id: str = None):
        """Clear memory for specific channel/user or all memory"""
        try:
            if not self.persistence:
                return
            
            with sqlite3.connect(self.db_path) as conn:
                if channel_id and user_id:
                    conn.execute("""
                        DELETE FROM conversations 
                        WHERE channel_id = ? AND user_id = ?
                    """, (channel_id, user_id))
                elif channel_id:
                    conn.execute("""
                        DELETE FROM conversations 
                        WHERE channel_id = ?
                    """, (channel_id,))
                else:
                    # Clear all conversations
                    conn.execute("DELETE FROM conversations")
                
                conn.commit()
                logger.info(f"Cleared memory for channel={channel_id}, user={user_id}")
                
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
