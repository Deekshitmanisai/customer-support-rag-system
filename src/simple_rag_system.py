#!/usr/bin/env python3
"""
Simple RAG System for Customer Support
=====================================

A simplified RAG system that provides the basic functionality needed
for the Customer Support RAG System.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
import json
import os

class SimpleRAGSystem:
    """Simple RAG system for customer support articles."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the simple RAG system."""
        self.embedding_model = SentenceTransformer(embedding_model)
        self.articles = []
        self.embeddings = []
        
        # Use in-memory storage only (no ChromaDB dependency)
        self.article_embeddings = []
        self.article_metadata = []
        print("Using in-memory storage for RAG system")
    
    def add_article(self, title: str, content: str, category: str = None):
        """Add an article to the RAG system."""
        article_id = f"article_{len(self.articles)}"
        
        # Store locally for reference
        article_data = {
            "id": article_id,
            "title": title,
            "content": content,
            "category": category or "general"
        }
        self.articles.append(article_data)
        
        # Store embeddings in memory
        try:
            embedding = self.embedding_model.encode(content)
            self.article_embeddings.append(embedding)
            self.article_metadata.append({
                "id": article_id,
                "title": title,
                "content": content,
                "category": category or "general"
            })
        except Exception as e:
            print(f"Warning: Failed to create embedding: {e}")
    
    def search_articles(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for relevant articles."""
        try:
            # Search in memory using cosine similarity
            if not self.article_embeddings:
                return {
                    "results": [],
                    "query": query,
                    "total_results": 0
                }
            
            query_embedding = self.embedding_model.encode(query)
            similarities = []
            
            for i, embedding in enumerate(self.article_embeddings):
                similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
                similarities.append((similarity, i))
            
            # Sort by similarity and get top results
            similarities.sort(reverse=True)
            formatted_results = []
            
            for similarity, idx in similarities[:n_results]:
                metadata = self.article_metadata[idx]
                formatted_results.append({
                    "content": metadata["content"],
                    "metadata": {
                        "title": metadata["title"],
                        "category": metadata["category"]
                    },
                    "distance": 1 - similarity  # Convert similarity to distance
                })
            
            return {
                "results": formatted_results,
                "query": query,
                "total_results": len(formatted_results)
            }
        except Exception as e:
            print(f"Error searching articles: {e}")
            return {
                "results": [],
                "query": query,
                "total_results": 0,
                "error": str(e)
            }
    
    def get_article_count(self) -> int:
        """Get the total number of articles."""
        return len(self.articles)
    
    def get_articles(self) -> List[Dict[str, Any]]:
        """Get all articles."""
        return self.articles.copy()
    
    def clear_articles(self):
        """Clear all articles."""
        self.articles = []
        self.article_embeddings = []
        self.article_metadata = [] 