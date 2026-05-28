"""
Vector store using LangChain FAISS.

This module provides FAISS vector storage using LangChain's community integration.
It wraps document chunks in LangChain Document objects for compatibility
with the RetrievalQA chain.

Usage:
    python -c "from rag.vector_store import create_vector_store; vs = create_vector_store(['chunk1', 'chunk2']); print(vs)"
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag.embedding import get_embedding_model


def create_vector_store(chunks):
    """
    Create a FAISS vector store from text chunks.
    
    Args:
        chunks: List of text strings to index
    
    Returns:
        FAISS vector store with embeddings
    """
    embeddings = get_embedding_model()
    
    # Wrap each chunk in a LangChain Document
    documents = [
        Document(page_content=chunk, metadata={"source": f"chunk_{i}"})
        for i, chunk in enumerate(chunks)
    ]
    
    return FAISS.from_documents(documents, embeddings)


def load_vector_store(chunks, index_path):
    """
    Create and save a FAISS vector store to disk.
    
    Args:
        chunks: List of text strings to index
        index_path: Directory path to save the index
    
    Returns:
        FAISS vector store (also saved to disk)
    """
    embeddings = get_embedding_model()
    
    documents = [
        Document(page_content=chunk, metadata={"source": f"chunk_{i}"})
        for i, chunk in enumerate(chunks)
    ]
    
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(index_path)
    
    return vector_store