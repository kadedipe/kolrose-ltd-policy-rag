"""
Conversation Memory Module for Kolrose Policy Assistant
=========================================================
Handles chat history storage, retrieval, and context management
for natural follow-up conversations.
"""

import os
import re
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# ============================================================================
# CONVERSATION MODELS
# ============================================================================

class ConversationMessage:
    """Single message in a conversation"""
    
    def __init__(
        self,
        role: str,
        content: str,
        citations: Optional[List[str]] = None,
        sources: Optional[List[Dict]] = None,
        timestamp: Optional[str] = None,
    ):
        self.role = role
        self.content = content
        self.citations = citations or []
        self.sources = sources or []
        self.timestamp = timestamp or datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "citations": self.citations,
            "sources": self.sources,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMessage":
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            citations=data.get("citations", []),
            sources=data.get("sources", []),
            timestamp=data.get("timestamp"),
        )


class Conversation:
    """A single conversation session with message history"""
    
    def __init__(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        max_messages: int = 50,
    ):
        self.session_id = session_id
        self.user_id = user_id or "anonymous"
        self.title = title or "New Conversation"
        self.max_messages = max_messages
        self.messages: List[ConversationMessage] = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self._lock = threading.Lock()
    
    def add_message(
        self,
        role: str,
        content: str,
        citations: Optional[List[str]] = None,
        sources: Optional[List[Dict]] = None,
    ) -> ConversationMessage:
        """Add a message to the conversation"""
        with self._lock:
            message = ConversationMessage(
                role=role,
                content=content,
                citations=citations,
                sources=sources,
            )
            self.messages.append(message)
            
            # Trim old messages if exceeding max
            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages:]
            
            self.updated_at = datetime.now().isoformat()
            
            # Auto-generate title from first user message
            if not self.title or self.title == "New Conversation":
                if role == "user" and len(self.messages) <= 2:
                    self.title = content[:60] + "..." if len(content) > 60 else content
            
            return message
    
    def get_context_messages(self, last_n: int = 10) -> List[Dict]:
        """Get recent messages for context window"""
        recent = self.messages[-last_n:] if last_n > 0 else self.messages[-10:]
        return [m.to_dict() for m in recent]
    
    def get_context_for_rag(self, max_exchanges: int = 5) -> str:
        """Format recent conversation as context for RAG"""
        recent = self.messages[-(max_exchanges * 2):]
        
        if not recent:
            return ""
        
        parts = ["\n--- Previous Conversation ---"]
        for msg in recent:
            role_label = "User" if msg.role == "user" else "Assistant"
            parts.append(f"{role_label}: {msg.content}")
        
        return "\n".join(parts) + "\n--- End of History ---\n"
    
    def get_last_user_question(self) -> Optional[str]:
        """Get the most recent user question"""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None
    
    def get_last_assistant_response(self) -> Optional[str]:
        """Get the most recent assistant response"""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None
    
    def clear(self):
        """Clear all messages"""
        with self._lock:
            self.messages = []
            self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "message_count": len(self.messages),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    def summary(self) -> dict:
        """Get conversation summary without full messages"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "title": self.title,
            "message_count": len(self.messages),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_message": self.messages[-1].content[:100] if self.messages else None,
        }


# ============================================================================
# CONVERSATION STORE (In-Memory with TTL)
# ============================================================================

class ConversationStore:
    """Thread-safe in-memory conversation store with TTL"""
    
    def __init__(self, ttl_minutes: int = 120):
        self._conversations: Dict[str, Conversation] = {}
        self._user_conversations: Dict[str, List[str]] = defaultdict(list)
        self.ttl_minutes = ttl_minutes
        self._lock = threading.Lock()
    
    def create_conversation(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Conversation:
        """Create a new conversation"""
        with self._lock:
            if session_id is None:
                session_id = f"conv_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
            
            conv = Conversation(
                session_id=session_id,
                user_id=user_id,
                title=title,
            )
            
            self._conversations[session_id] = conv
            
            if user_id:
                self._user_conversations[user_id].append(session_id)
            
            return conv
    
    def get_conversation(self, session_id: str) -> Optional[Conversation]:
        """Get a conversation by session ID"""
        return self._conversations.get(session_id)
    
    def get_or_create_conversation(
        self,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> Conversation:
        """Get existing conversation or create new one"""
        conv = self.get_conversation(session_id)
        if conv is None:
            conv = self.create_conversation(user_id=user_id, session_id=session_id)
        return conv
    
    def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation"""
        with self._lock:
            conv = self._conversations.pop(session_id, None)
            if conv:
                if conv.user_id in self._user_conversations:
                    self._user_conversations[conv.user_id].remove(session_id)
                return True
            return False
    
    def get_user_conversations(self, user_id: str) -> List[Dict]:
        """Get all conversations for a user"""
        with self._lock:
            session_ids = self._user_conversations.get(user_id, [])
            conversations = []
            for sid in session_ids:
                conv = self._conversations.get(sid)
                if conv:
                    conversations.append(conv.summary())
            return sorted(conversations, key=lambda x: x.get("updated_at", ""), reverse=True)
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List[str]] = None,
        sources: Optional[List[Dict]] = None,
        user_id: Optional[str] = None,
    ) -> Optional[ConversationMessage]:
        """Add a message to a conversation"""
        conv = self.get_or_create_conversation(session_id, user_id)
        return conv.add_message(role, content, citations, sources)
    
    def get_stats(self) -> dict:
        """Get store statistics"""
        with self._lock:
            return {
                "total_conversations": len(self._conversations),
                "total_users": len(self._user_conversations),
                "ttl_minutes": self.ttl_minutes,
            }


# ============================================================================
# GLOBAL CONVERSATION STORE
# ============================================================================

conversation_store = ConversationStore(ttl_minutes=120)


# ============================================================================
# FOLLOW-UP QUESTION DETECTION
# ============================================================================

class FollowUpDetector:
    """Detects if a question is a follow-up to a previous conversation"""
    
    FOLLOW_UP_PATTERNS = [
        r"^(what about|how about|and)\s+(it|that|this|those|these|them)",
        r"^(is it|is that|is this|are they|are those)",
        r"^(can you|could you)\s+(explain|elaborate|clarify|detail)",
        r"^(tell me|give me)\s+(more|additional|further)",
        r"^(why|when|where|who|how)\??$",
        r"^(what|which)\s+(one|ones)\??$",
        r"^(what does|what do you)\s+(that|it|this)\s+(mean)",
        r"^(can you|could you)\s+(repeat|say that again)",
        r"^(and|also|additionally|furthermore|moreover)",
        r"^(what if|what about|how about)",
    ]
    
    @classmethod
    def is_follow_up(cls, question: str, conversation: Conversation) -> Tuple[bool, str]:
        """Check if a question is a follow-up. Returns (is_follow_up, context_to_add)"""
        if not conversation or not conversation.messages:
            return False, ""
        
        question_lower = question.lower().strip()
        
        # Check for short questions (likely follow-ups)
        if len(question.split()) <= 3:
            return True, conversation.get_context_for_rag(max_exchanges=3)
        
        # Check for follow-up patterns
        for pattern in cls.FOLLOW_UP_PATTERNS:
            if re.search(pattern, question_lower):
                return True, conversation.get_context_for_rag(max_exchanges=3)
        
        # Check for pronoun references at the start
        pronouns = ["it", "that", "this", "they", "them", "those", "these"]
        words = question_lower.split()
        if any(p in words[:3] for p in pronouns):
            return True, conversation.get_context_for_rag(max_exchanges=2)
        
        return False, ""


# ============================================================================
# MEMORY-ENHANCED RAG QUERY BUILDER
# ============================================================================

def build_context_with_memory(
    question: str,
    conversation: Conversation,
    max_history_exchanges: int = 5,
) -> str:
    """Build prompt context that includes conversation history for follow-up questions"""
    detector = FollowUpDetector()
    is_follow_up, history_context = detector.is_follow_up(question, conversation)
    
    if is_follow_up and history_context:
        return history_context
    
    if conversation and len(conversation.messages) > 0:
        return conversation.get_context_for_rag(max_exchanges=max_history_exchanges)
    
    return ""