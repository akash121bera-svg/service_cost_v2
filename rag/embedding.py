"""
Embedding utilities using LangChain.

This module provides embeddings using HuggingFace sentence transformers
via LangChain's community integration.

Usage:
    python -c "from rag.embedding import generate_embeddings; print(generate_embeddings(['hello', 'world']))"
"""

import os
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = PROJECT_ROOT / ".model_cache"

# Configure model caching directories
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR / "huggingface"))
os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_HOME",
    str(MODEL_CACHE_DIR / "sentence_transformers"),
)


def get_embedding_model():
    """
    Get the HuggingFace embeddings model.
    
    Returns a cached model instance using BAAI/bge-small-en-v1.5,
    which is fast and provides good semantic search results.
    
    Returns:
        HuggingFaceEmbeddings: Embedding model instance
    """
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder=str(MODEL_CACHE_DIR / "sentence_transformers"),
    )

def generate_embeddings(texts):
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
    
    Returns:
        List of embedding vectors
    """
    model = get_embedding_model()
    return model.embed_documents(texts)