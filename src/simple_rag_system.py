#!/usr/bin/env python3
"""
Simple RAG System for Customer Support
=====================================

A simplified RAG system that provides the basic functionality needed
for the Customer Support RAG System.
"""

import chromadb
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
        
        # Initialize ChromaDB with error handling
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.chroma_client.get_or_create_collection(
                name="customer_support_articles",
                metadata={"hnsw:space": "cosine"}
            )
        except ValueError as e:
            # If there's a conflict, try with a different path
            import uuid
            unique_path = f"./chroma_db_{uuid.uuid4().hex[:8]}"
            self.chroma_client = chromadb.PersistentClient(path=unique_path)
            self.collection = self.chroma_client.get_or_create_collection(
                name="customer_support_articles",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            # Fallback to in-memory client if persistent fails
            print(f"Warning: Using in-memory ChromaDB due to error: {e}")
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.get_or_create_collection(
                name="customer_support_articles",
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_article(self, title: str, content: str, category: str = None):
        """Add an article to the RAG system."""
        article_id = f"article_{len(self.articles)}"
        
        # Add to ChromaDB
        self.collection.add(
            documents=[content],
            metadatas=[{"title": title, "category": category or "general"}],
            ids=[article_id]
        )
        
        # Store locally for reference
        self.articles.append({
            "id": article_id,
            "title": title,
            "content": content,
            "category": category
        })
    
    def search_articles(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search for relevant articles."""
        try:
            # Search in ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else 0.0
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
        self.collection.delete()
        self.collection = self.chroma_client.create_collection(
            name="customer_support_articles",
            metadata={"hnsw:space": "cosine"}
        ) 