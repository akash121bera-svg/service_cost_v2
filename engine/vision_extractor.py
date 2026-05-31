"""
Multimodal Vision Extractor.

Coordinates multimodal extraction using Gemini-2.0-flash to parse text tables
from PDF documents and image screenshots (PNG, JPG, JPEG) into structured 
pandas DataFrames, allowing standard costing engines to evaluate them.
"""

import base64
import os
import io
import pandas as pd
from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from engine.caching import GLOBAL_CACHE


def get_llm():
    """Retrieve Groq Llama 4 Scout Vision LLM instance."""
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
    )


def extract_rate_card_from_media(file_bytes: bytes, file_name: str) -> Optional[pd.DataFrame]:
    """
    Base64 encodes media bytes (PDF/PNG/JPG/JPEG), calls Gemini Vision OCR,
    and returns a normalized pandas DataFrame matching our costing engine schema.
    """
    file_name_lower = file_name.lower()
    
    # Try retrieving from cache first to avoid repeating Vision API calls
    cached_df_dict = GLOBAL_CACHE.get("vision_df", file_name, len(file_bytes))
    if cached_df_dict is not None:
        try:
            return pd.DataFrame.from_dict(cached_df_dict)
        except Exception:
            pass
            
    # Resolve MIME Type
    if file_name_lower.endswith(".pdf"):
        mime_type = "application/pdf"
    elif file_name_lower.endswith(".png"):
        mime_type = "image/png"
    elif file_name_lower.endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    else:
        return None
        
    try:
        # Base64 encode the media bytes
        b64_data = base64.b64encode(file_bytes).decode("utf-8")
        
        # Build multimodal prompt
        prompt = (
            "Extract the service rate card table from this document/image. Return the table ONLY in clean CSV format.\n"
            "The CSV MUST have precisely these column headers:\n"
            "shipment_category,min_qty,max_qty,packaging_rate,sterilization_rate,logistics_rate,quality_rate,warehousing_rate\n\n"
            "Rules:\n"
            "1. shipment_category values MUST be normalized to lowercase: 'small', 'medium', or 'large'.\n"
            "2. If a column is missing or empty, leave it blank in the CSV row.\n"
            "3. If multiple categories exist, output one CSV row for each category (Small, Medium, Large).\n"
            "4. Return ONLY raw CSV text. Do not wrap in markdown ```csv blocks, do not write explanations, and do not add headers other than the one specified."
        )
        
        llm = get_llm()
        
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_data}"
                    }
                }
            ]
        )
        
        response = llm.invoke([message])
        content = response.content.strip()
        
        # Clean markdown code blocks if the model ignored our formatting instruction
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()
            
        if not content or "shipment_category" not in content:
            return None
            
        # Parse CSV string into Pandas DataFrame
        df = pd.read_csv(io.StringIO(content))
        
        # Validate required columns exist
        required_cols = ["shipment_category", "min_qty", "max_qty"]
        if not all(col in df.columns for col in required_cols):
            return None
            
        # Convert numeric columns safely
        numeric_cols = [
            "min_qty", "max_qty", "packaging_rate", "sterilization_rate",
            "logistics_rate", "quality_rate", "warehousing_rate"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
                
        # Cache the extracted DataFrame
        GLOBAL_CACHE.set("vision_df", df.to_dict(orient="list"), file_name, len(file_bytes))
        
        return df
    except Exception:
        # Graceful degradation - log and fail over safely
        return None
