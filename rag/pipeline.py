"""
Uploaded-file RAG pipeline using LangChain.

This module provides a unified interface for:
- PDF/CSV chunking and vector storage
- FAISS vector search
- LLM-backed answers using Google Gemini

Usage:
    python -c "from rag.pipeline import build_gemini_rag_answer; print(build_gemini_rag_answer('What is packaging cost?', ['chunk1', 'chunk2'], 100))"
"""

import os

import fitz
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import LLMChain

from langchain_groq import ChatGroq

from rag.vector_store import create_vector_store

SERVICE_COST_PROMPT = PromptTemplate.from_template("""
You are a service costing assistant. Use the context to answer questions about vendors, rates, pricing, and comparisons.

Chat History:
{chat_history}

Context:
{context}

Deterministic Calculation (if available):
{structured_answer}

Original Question (for follow-up expansion):
{original_question}

Rephrased Question:
{rephrased_question}

Answer the rephrased question based on the context, chat history, and deterministic calculation. Make sure to reference the original question context when answering follow-up questions about quantity changes, comparisons, or vendor recommendations.
""")


def get_llm():
    return ChatGroq(
        model="llama-3.2-90b-vision-preview",
        temperature=0,
    )


def chunk_text(text, chunk_size=900, overlap=120):
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk (default: 900)
        overlap: Characters to overlap between chunks (default: 120)
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks


def dataframe_to_chunks(file_name, df):
    """
    Convert a pandas DataFrame into text chunks for RAG.
    
    Creates one chunk for the file header (columns),
    and one chunk per data row.
    
    Args:
        file_name: Name of the source CSV file
        df: Pandas DataFrame with rate card data
    
    Returns:
        List of text chunks
    """
    chunks = [
        f"Source file: {file_name}\n"
        f"Columns: {', '.join(str(column) for column in df.columns)}\n"
        "This file contains service-cost rate card data."
    ]
    
    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        row_values = "\n".join(f"{column}: {row[column]}" for column in df.columns)
        chunks.append(
            f"Source file: {file_name}\n"
            f"Row number: {row_number}\n"
            f"{row_values}"
        )
    
    return chunks


def pdf_to_chunks(file_name, file_bytes):
    """
    Extract text from a PDF and split into chunks.
    
    Args:
        file_name: Name of the source PDF file
        file_bytes: PDF file content as bytes
    
    Returns:
        List of text chunks extracted from PDF
    """
    document = fitz.open(stream=file_bytes, filetype="pdf")
    chunks = []
    
    for page_number, page in enumerate(document, start=1):
        text = page.get_text().strip()
        for chunk in chunk_text(text):
            chunks.append(
                f"Source file: {file_name}\n"
                f"Page number: {page_number}\n"
                f"{chunk}"
            )
    
    document.close()
    return chunks


def build_rag_chain(chunks):
    """
    Build a LangChain LLMChain for answering questions.
    
    Args:
        chunks: List of text context chunks
    
    Returns:
        Tuple of (chain, vector_store)
    """
    vector_store = create_vector_store(chunks)
    llm = get_llm()
    
    chain = LLMChain(llm=llm, prompt=SERVICE_COST_PROMPT, output_parser=StrOutputParser())
    
    return chain, vector_store


def build_gemini_rag_answer(question, context_chunks, quantity, structured_answer=None, chat_history=None):
    """
    Generate an answer using RAG with the configured LLM.
    
    Args:
        question: User question
        context_chunks: Retrieved context chunks from uploaded files
        quantity: Product quantity for costing
        structured_answer: Optional deterministic calculation result
        chat_history: Optional list of previous messages for follow-up context
    
    Returns:
        Generated answer string
    """
    if not context_chunks:
        return "I could not retrieve relevant context from the uploaded files."
    
    vector_store = create_vector_store(context_chunks)
    docs = vector_store.similarity_search(question, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=SERVICE_COST_PROMPT, output_parser=StrOutputParser())
    
    history_str = ""
    if chat_history:
        for msg in chat_history:
            if msg["role"] == "user":
                history_str += f"User: {msg['content']}\n"
            else:
                history_str += f"Assistant: {msg['content']}\n"
    
    is_follow_up = chat_history and len(chat_history) > 0
    
    if is_follow_up:
        rephrase_prompt = f"""Based on the chat history, rephrase this follow-up question to be a complete, self-contained question.

Chat History:
{history_str}

Follow-up Question: {question}

Rephrased Question (make it complete and self-contained):"""
        
        try:
            rephrase_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template("{question}"), output_parser=StrOutputParser())
            rephrased_question = rephrase_chain.invoke({"question": rephrase_prompt})
            rephrased_question = rephrased_question.get("text", rephrased_question) if isinstance(rephrased_question, dict) else rephrased_question
        except:
            rephrased_question = question
    else:
        rephrased_question = question
    
    original_question = chat_history[-1]["content"] if chat_history and chat_history[-1]["role"] == "user" else question
    
    context_info = structured_answer or "No deterministic calculation was required."
    
    input_dict = {
        "question": question,
        "original_question": original_question,
        "rephrased_question": rephrased_question,
        "context": context,
        "quantity": quantity,
        "chat_history": history_str or "No previous conversation.",
        "structured_answer": context_info,
    }
    
    try:
        result = chain.invoke(input_dict)
        return result.get("text", result) if isinstance(result, dict) else result
    except Exception as error:
        return f"Error generating answer: {error}"


# Compatibility Alias
build_groq_rag_answer = build_gemini_rag_answer


def retrieve_uploaded_context(question, chunks, top_k=3):
    """
    Retrieve relevant context chunks for a question using similarity search.
    
    Args:
        question: User question
        chunks: List of all text chunks
        top_k: Number of top results to return (default: 3)
    
    Returns:
        List of retrieved text chunks
    """
    if not chunks:
        return []
    
    vector_store = create_vector_store(chunks)
    docs = vector_store.similarity_search(question, k=top_k)
    
    return [doc.page_content for doc in docs]


def retrieve_context(state, chunks, top_k=5):
    """
    Evolved RAG Retrieval Tool.
    Retrieves context using local FAISS + HuggingFace embeddings
    and populates the shared state.
    """
    if not chunks:
        state.add_trace("RAG Retrieval Skipped: No uploaded document chunks", {"query": state.query})
        return []
        
    retrieved = retrieve_uploaded_context(state.query, chunks, top_k=top_k)
    state.retrieved_docs = retrieved
    
    # Track actions
    state.add_trace("RAG Retrieval Completed", {
        "retrieved_count": len(retrieved),
        "query": state.query
    })
    return retrieved


def build_uploaded_file_rag_answer(
    question: str,
    csv_dataframes: list,
    rag_chunks: list,
    quantity: int,
    structured_answer: str = None,
) -> tuple[str, list[dict]]:
    """
    Build answer using RAG pipeline with optional structured answer fallback.
    
    Args:
        question: User question
        csv_dataframes: List of (filename, dataframe) tuples
        rag_chunks: List of text chunks from uploaded files
        quantity: Product quantity for costing
        structured_answer: Pre-calculated deterministic result
    
    Returns:
        Tuple of (answer, table_rows)
    """
    table_rows = []
    
    context_chunks = retrieve_uploaded_context(question, rag_chunks)
    answer = build_gemini_rag_answer(
        question,
        context_chunks,
        quantity,
        structured_answer=structured_answer,
    )
    
    return answer, table_rows