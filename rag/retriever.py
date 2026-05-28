"""
Retriever utilities for LangChain FAISS.

This module provides simple retrieval functions that use
the vector store's similarity search capabilities.

Usage:
    from rag.vector_store import create_vector_store
    from rag.retriever import retrieve_context
    vs = create_vector_store(['chunk1', 'chunk2', 'chunk3'])
    results = retrieve_context('What is packaging?', vs, ['chunk1', 'chunk2', 'chunk3'])
"""

from rag.vector_store import create_vector_store


def retrieve_context(query, vector_store, chunks, top_k=3):
    """
    Retrieve relevant documents using similarity search.
    
    Args:
        query: Search query string
        vector_store: FAISS vector store instance
        chunks: List of text chunks (for reference)
        top_k: Number of results to return
    
    Returns:
        List of retrieved text chunks
    """
    results = vector_store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]