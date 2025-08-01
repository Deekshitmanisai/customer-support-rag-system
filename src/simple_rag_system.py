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
        
        # Try to import and use ChromaDB
        try:
            import chromadb
            # Try in-memory client first (more reliable for deployment)
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.get_or_create_collection(
                name="customer_support_articles",
                metadata={"hnsw:space": "cosine"}
            )
            self.use_chromadb = True
            print("Using ChromaDB for storage")
        except Exception as e:
            print(f"Warning: ChromaDB not available, using fallback storage: {e}")
            self.use_chromadb = False
        
        # Fallback storage using simple list
        if not self.use_chromadb:
            self.article_embeddings = []
            self.article_metadata = []
            print("Using in-memory fallback storage")
    
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
        
        # Add to storage system
        if self.use_chromadb:
            try:
                self.collection.add(
                    documents=[content],
                    metadatas=[{"title": title, "category": category or "general"}],
                    ids=[article_id]
                )
            except Exception as e:
                print(f"Warning: Failed to add to ChromaDB: {e}")
        else:
            # Fallback: store embeddings in memory
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
            if self.use_chromadb:
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
            else:
                # Fallback: search in memory using cosine similarity
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
        if self.use_chromadb:
            try:
                self.collection.delete()
                self.collection = self.chroma_client.create_collection(
                    name="customer_support_articles",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                print(f"Warning: Failed to clear ChromaDB: {e}")
        else:
            self.article_embeddings = []
            self.article_metadata = [] 