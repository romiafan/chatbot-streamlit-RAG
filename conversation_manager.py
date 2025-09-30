"""
Enhanced Conversation Memory System
Provides persistent storage, categorization, search, and management of chat conversations
"""

import json
import os
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import streamlit as st


@dataclass
class Conversation:
    """Represents a single conversation with metadata"""
    id: str
    title: str
    messages: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    category: str = "general"
    tags: Optional[List[str]] = None
    summary: str = ""
    message_count: int = 0
    total_tokens: int = 0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if not self.message_count:
            self.message_count = len(self.messages)


class ConversationManager:
    """Manages conversation storage, retrieval, and organization"""
    
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database for conversation storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    tags TEXT DEFAULT '[]',
                    summary TEXT DEFAULT '',
                    message_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_created_at 
                ON conversations(created_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_category 
                ON conversations(category)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_updated_at 
                ON conversations(updated_at)
            """)
    
    def save_conversation(self, conversation: Conversation) -> bool:
        """Save or update a conversation in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO conversations 
                    (id, title, messages, created_at, updated_at, category, tags, summary, message_count, total_tokens)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    conversation.id,
                    conversation.title,
                    json.dumps(conversation.messages),
                    conversation.created_at.isoformat(),
                    conversation.updated_at.isoformat(),
                    conversation.category,
                    json.dumps(conversation.tags),
                    conversation.summary,
                    conversation.message_count,
                    conversation.total_tokens
                ))
            return True
        except Exception as e:
            st.error(f"Failed to save conversation: {e}")
            return False
    
    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Load a specific conversation by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM conversations WHERE id = ?", 
                    (conversation_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return Conversation(
                        id=row[0],
                        title=row[1],
                        messages=json.loads(row[2]),
                        created_at=datetime.fromisoformat(row[3]),
                        updated_at=datetime.fromisoformat(row[4]),
                        category=row[5],
                        tags=json.loads(row[6]),
                        summary=row[7],
                        message_count=row[8],
                        total_tokens=row[9]
                    )
        except Exception as e:
            st.error(f"Failed to load conversation: {e}")
        return None
    
    def list_conversations(
        self, 
        category: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        search_term: str = ""
    ) -> List[Conversation]:
        """List conversations with optional filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM conversations WHERE 1=1"
                params = []
                
                if category:
                    query += " AND category = ?"
                    params.append(category)
                
                if search_term:
                    query += " AND (title LIKE ? OR summary LIKE ?)"
                    search_pattern = f"%{search_term}%"
                    params.extend([search_pattern, search_pattern])
                
                query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                conversations = []
                for row in rows:
                    conversations.append(Conversation(
                        id=row[0],
                        title=row[1],
                        messages=json.loads(row[2]),
                        created_at=datetime.fromisoformat(row[3]),
                        updated_at=datetime.fromisoformat(row[4]),
                        category=row[5],
                        tags=json.loads(row[6]),
                        summary=row[7],
                        message_count=row[8],
                        total_tokens=row[9]
                    ))
                
                return conversations
        except Exception as e:
            st.error(f"Failed to list conversations: {e}")
            return []
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            return True
        except Exception as e:
            st.error(f"Failed to delete conversation: {e}")
            return False
    
    def get_categories(self) -> List[str]:
        """Get all unique categories"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT DISTINCT category FROM conversations ORDER BY category")
                return [row[0] for row in cursor.fetchall()]
        except Exception:
            return ["general"]
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_conversations,
                        SUM(message_count) as total_messages,
                        SUM(total_tokens) as total_tokens,
                        AVG(message_count) as avg_messages_per_conversation,
                        COUNT(DISTINCT category) as total_categories
                    FROM conversations
                """)
                row = cursor.fetchone()
                
                if row:
                    return {
                        "total_conversations": row[0],
                        "total_messages": row[1] or 0,
                        "total_tokens": row[2] or 0,
                        "avg_messages_per_conversation": round(row[3] or 0, 1),
                        "total_categories": row[4]
                    }
        except Exception:
            pass
        
        return {
            "total_conversations": 0,
            "total_messages": 0,
            "total_tokens": 0,
            "avg_messages_per_conversation": 0,
            "total_categories": 0
        }
    
    def export_conversations(
        self, 
        format_type: str = "json",
        category: Optional[str] = None
    ) -> str:
        """Export conversations in specified format"""
        conversations = self.list_conversations(category=category, limit=1000)
        
        if format_type == "json":
            export_data = {
                "export_date": datetime.now().isoformat(),
                "total_conversations": len(conversations),
                "conversations": [asdict(conv) for conv in conversations]
            }
            return json.dumps(export_data, indent=2, default=str)
        
        elif format_type == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "ID", "Title", "Category", "Created", "Updated", 
                "Message Count", "Total Tokens", "Summary"
            ])
            
            # Write data
            for conv in conversations:
                writer.writerow([
                    conv.id,
                    conv.title,
                    conv.category,
                    conv.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    conv.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
                    conv.message_count,
                    conv.total_tokens,
                    conv.summary
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def import_conversations(self, import_data: str, format_type: str = "json") -> Tuple[int, int]:
        """Import conversations from exported data"""
        imported = 0
        errors = 0
        
        try:
            if format_type == "json":
                data = json.loads(import_data)
                conversations_data = data.get("conversations", [])
                
                for conv_data in conversations_data:
                    try:
                        # Convert string dates back to datetime
                        conv_data["created_at"] = datetime.fromisoformat(conv_data["created_at"])
                        conv_data["updated_at"] = datetime.fromisoformat(conv_data["updated_at"])
                        
                        conv = Conversation(**conv_data)
                        if self.save_conversation(conv):
                            imported += 1
                        else:
                            errors += 1
                    except Exception:
                        errors += 1
            
        except Exception as e:
            st.error(f"Import failed: {e}")
            return 0, 1
        
        return imported, errors


def generate_conversation_id(messages: List[Dict[str, Any]]) -> str:
    """Generate a unique conversation ID based on content"""
    if not messages:
        return hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:12]
    
    # Use first message content and timestamp for ID generation
    first_msg = messages[0].get("content", "")
    timestamp = str(datetime.now().timestamp())
    content_hash = hashlib.md5(f"{first_msg}_{timestamp}".encode()).hexdigest()
    return content_hash[:12]


def generate_conversation_title(messages: List[Dict[str, Any]]) -> str:
    """Generate an intelligent title based on the first user message"""
    if not messages:
        return "New Conversation"
    
    # Find first user message
    first_user_msg = None
    for msg in messages:
        if msg.get("role") == "user":
            first_user_msg = msg.get("content", "")
            break
    
    if not first_user_msg:
        return "New Conversation"
    
    # Truncate and clean up the message for title
    title = first_user_msg.strip()
    if len(title) > 50:
        title = title[:50] + "..."
    
    # Remove common question words and make it more title-like
    title = title.replace("?", "").replace("!", "").replace("\n", " ")
    
    return title if title else "New Conversation"


def auto_categorize_conversation(messages: List[Dict[str, Any]]) -> str:
    """Automatically categorize conversation based on content"""
    if not messages:
        return "general"
    
    # Combine all message content for analysis
    all_content = " ".join([
        msg.get("content", "").lower() 
        for msg in messages 
        if msg.get("role") == "user"
    ])
    
    # Simple keyword-based categorization
    categories = {
        "technical": ["code", "programming", "debug", "error", "function", "api", "database"],
        "research": ["analyze", "research", "study", "compare", "explain", "what is"],
        "creative": ["write", "create", "design", "generate", "story", "poem"],
        "business": ["strategy", "plan", "market", "sales", "revenue", "business"],
        "educational": ["learn", "teach", "tutorial", "example", "how to", "guide"],
        "support": ["help", "issue", "problem", "fix", "broken", "not working"]
    }
    
    for category, keywords in categories.items():
        if any(keyword in all_content for keyword in keywords):
            return category
    
    return "general"


@st.cache_resource
def get_conversation_manager() -> ConversationManager:
    """Get cached conversation manager instance"""
    db_path = os.getenv("CONVERSATION_DB_PATH", "conversations.db")
    return ConversationManager(db_path)