import re
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

def clean_text(text: str) -> str:
    """Clean and normalize text for processing."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for better context preservation."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundaries
        if end < len(text):
            # Look for sentence endings
            for i in range(end, max(start + chunk_size - 100, start), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using simple word-based approximation."""
    # Simple approximation: 1 token ≈ 4 characters or 0.75 words
    words = text.split()
    return int(len(words) * 1.3)  # Approximate token count

def save_conversation_history(conversation: List[Dict], filepath: str = "data/conversation_history.json"):
    """Save conversation history to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, indent=2, ensure_ascii=False)

def load_conversation_history(filepath: str = "data/conversation_history.json") -> List[Dict]:
    """Load conversation history from JSON file."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract important keywords from text using simple methods."""
    # Simple keyword extraction without TextBlob dependency
    text = text.lower()
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'
    }
    
    # Split into words and filter
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count frequency
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_keywords[:max_keywords]]

def calculate_similarity_score(text1: str, text2: str) -> float:
    """Calculate simple text similarity using word overlap."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """Format timestamp for display."""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def create_metadata(text: str, source: str = "unknown") -> Dict[str, Any]:
    """Create metadata for text chunks."""
    return {
        "source": source,
        "timestamp": format_timestamp(),
        "length": len(text),
        "tokens": count_tokens(text),
        "keywords": ", ".join(extract_keywords(text, max_keywords=5))  # Convert list to string
    }

def validate_api_key(api_key: str) -> bool:
    """Basic validation for API keys."""
    if not api_key:
        return False
    if len(api_key) < 10:
        return False
    return True

def get_environment_variables() -> Dict[str, str]:
    """Get environment variables for the application."""
    from dotenv import load_dotenv
    load_dotenv()
    
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "HUGGINGFACE_API_KEY": os.getenv("HUGGINGFACE_API_KEY", ""),
    } 