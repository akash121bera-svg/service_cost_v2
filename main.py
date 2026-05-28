"""
FastAPI Backend Application.

Exposes REST API endpoints for:
1. Query processing via Orchestrator workflow.
2. File uploads (CSV and PDF) with server-side RAG chunk caching.
3. Cache clearing.
"""

import os
import io
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from dotenv import load_dotenv

from engine.orchestrator import execute_orchestrator_workflow
from rag.pipeline import dataframe_to_chunks, pdf_to_chunks
from engine.uploaded_costs import build_costing_engine_chunks
from engine.caching import GLOBAL_CACHE
from engine.vision_extractor import extract_rate_card_from_media

load_dotenv()

app = FastAPI(
    title="Hybrid Workflow Intelligence Procurement API",
    description="FastAPI service coordinating the Hybrid Workflow RAG pipeline.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory single-user global cache for uploaded files and chunks
GLOBAL_RAG_CHUNKS: List[str] = []
GLOBAL_CSV_DATAFRAMES: List[tuple] = []  # List of (filename, DataFrame)


class QueryRequest(BaseModel):
    query: str = Field(..., example="Which vendor is best for 1200 units?")
    quantity: int = Field(default=100, example=1200)
    chat_history: Optional[List[Dict[str, str]]] = Field(default=None, example=[])
    last_best_option: Optional[Dict[str, Any]] = Field(default=None)


class QueryResponse(BaseModel):
    answer: str
    state: Dict[str, Any]


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Procurement RAG Backend",
        "cached_files": [name for name, _ in GLOBAL_CSV_DATAFRAMES],
        "rag_chunks_count": len(GLOBAL_RAG_CHUNKS)
    }


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Accepts multiple CSV/PDF files, parses and chunks them into global cache."""
    global GLOBAL_RAG_CHUNKS, GLOBAL_CSV_DATAFRAMES
    
    uploaded_files_log = []
    
    try:
        for file in files:
            file_content = await file.read()
            filename = file.filename
            
            if filename.lower().endswith(".csv"):
                # Parse pandas DataFrame
                df = pd.read_csv(io.BytesIO(file_content))
                GLOBAL_CSV_DATAFRAMES.append((filename, df))
                GLOBAL_RAG_CHUNKS.extend(dataframe_to_chunks(filename, df))
                uploaded_files_log.append({"filename": filename, "type": "CSV"})
                
            elif filename.lower().endswith(".pdf"):
                # 1. OCR extract rate card table
                df = extract_rate_card_from_media(file_content, filename)
                if df is not None:
                    GLOBAL_CSV_DATAFRAMES.append((filename, df))
                # 2. Index PDF text RAG chunks
                GLOBAL_RAG_CHUNKS.extend(pdf_to_chunks(filename, file_content))
                uploaded_files_log.append({"filename": filename, "type": "PDF"})
                
            elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
                # 1. OCR extract rate card table
                df = extract_rate_card_from_media(file_content, filename)
                if df is not None:
                    GLOBAL_CSV_DATAFRAMES.append((filename, df))
                    GLOBAL_RAG_CHUNKS.extend(dataframe_to_chunks(filename, df))
                else:
                    GLOBAL_RAG_CHUNKS.append(
                        f"Source image: {filename}\nThis file represents an uploaded image without a structured rate card table."
                    )
                uploaded_files_log.append({"filename": filename, "type": "IMAGE"})
                
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file format: {filename}")
                
        return {
            "message": f"Successfully processed {len(files)} files.",
            "uploaded": uploaded_files_log,
            "total_cached_files": len(GLOBAL_CSV_DATAFRAMES),
            "total_rag_chunks": len(GLOBAL_RAG_CHUNKS)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
def run_query(payload: QueryRequest):
    """Triggers Orchestrator with query, history, and uploaded files context."""
    global GLOBAL_RAG_CHUNKS, GLOBAL_CSV_DATAFRAMES
    
    # Pre-add deterministic costing engine chunks into rag_chunks before query
    local_rag_chunks = list(GLOBAL_RAG_CHUNKS)
    
    if GLOBAL_CSV_DATAFRAMES:
        try:
            costing_chunks = build_costing_engine_chunks(GLOBAL_CSV_DATAFRAMES, payload.quantity)
            local_rag_chunks.extend(costing_chunks)
        except Exception as e:
            # Let RAG continue even if building pre-calculated chunks fails
            pass
            
    try:
        final_answer, state = execute_orchestrator_workflow(
            query=payload.query,
            quantity=payload.quantity,
            csv_dataframes=GLOBAL_CSV_DATAFRAMES,
            rag_chunks=local_rag_chunks,
            chat_history=payload.chat_history,
            last_best_option=payload.last_best_option
        )
        
        return QueryResponse(
            answer=final_answer,
            state=state.to_dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow Execution Error: {str(e)}")


@app.post("/api/clear")
def clear_cache():
    """Resets global server state and enterprise caching layer."""
    global GLOBAL_RAG_CHUNKS, GLOBAL_CSV_DATAFRAMES
    GLOBAL_RAG_CHUNKS = []
    GLOBAL_CSV_DATAFRAMES = []
    GLOBAL_CACHE.clear()
    return {"message": "Server-side files, chunks, and enterprise cache cleared successfully."}
