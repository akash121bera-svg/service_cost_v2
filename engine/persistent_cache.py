import sqlite3
import json
import os
import logging
from typing import Tuple, List, Dict, Any, Optional

logger = logging.getLogger(__name__)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "query_cache.db")

def init_db():
    """Initialize the SQLite database and create the cache table if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                normalized_query TEXT UNIQUE NOT NULL,
                original_query TEXT NOT NULL,
                answer TEXT NOT NULL,
                table_rows_json TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to initialize SQLite cache database: {e}")
    finally:
        conn.close()

def _normalize(query: str) -> str:
    """Normalize query for consistent matching (lowercase, stripped space/punctuation)."""
    if not query:
        return ""
    q = query.strip().lower()
    # Strip common trailing characters
    while q and q[-1] in ('?', '!', '.', ',', ' ', '\n'):
        q = q[:-1]
    return q.strip()

import re
from difflib import SequenceMatcher

COMMON_STOPWORDS = {
    "who", "are", "is", "the", "a", "an", "for", "in", "at", "to", "of", "and", "or",
    "please", "can", "you", "i", "get", "show", "list", "tell", "me", "find", "search",
    "vendor", "vendors", "supplier", "suppliers", "available", "listed", "near", "nearby",
    "around", "about", "how", "many", "where", "what", "which", "with"
}

def get_similarity_score(q1: str, q2: str) -> float:
    """Compute a hybrid similarity score based on stopword-filtered Jaccard similarity and string sequence ratio."""
    words1 = [w for w in re.findall(r"\w+", q1.lower()) if w not in COMMON_STOPWORDS]
    words2 = [w for w in re.findall(r"\w+", q2.lower()) if w not in COMMON_STOPWORDS]
    
    if not words1 or not words2:
        return 0.0
        
    # 1. Word-level Jaccard similarity (order independent)
    set1, set2 = set(words1), set(words2)
    jaccard = len(set1 & set2) / len(set1 | set2)
    
    # 2. Structural sequence similarity
    clean_q1 = " ".join(words1)
    clean_q2 = " ".join(words2)
    seq_match = SequenceMatcher(None, clean_q1, clean_q2).ratio()
    
    # Hybrid score weighting: 70% word overlap, 30% sequence order
    return 0.7 * jaccard + 0.3 * seq_match

def get_cached_answer(query: str, threshold: float = 0.85) -> Optional[Tuple[str, List[Dict[str, Any]], str]]:
    """Retrieve answer and table rows from persistent cache if exists, supporting exact and fuzzy matches."""
    init_db()
    norm_q = _normalize(query)
    if not norm_q:
        return None

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        
        # 1. Try exact match first for performance
        cursor.execute(
            "SELECT answer, table_rows_json, original_query FROM query_cache WHERE normalized_query = ?",
            (norm_q,)
        )
        row = cursor.fetchone()
        if row:
            answer, rows_json, orig_q = row
            table_rows = json.loads(rows_json)
            logger.info(f"Persistent Database Cache Hit (Exact) for: '{query}' (matched '{orig_q}')")
            return answer, table_rows, orig_q

        # 2. Try fuzzy similarity match against all cached queries
        cursor.execute("SELECT normalized_query, original_query, answer, table_rows_json FROM query_cache")
        all_rows = cursor.fetchall()
        
        best_score = 0.0
        best_row = None
        
        for cached_norm, cached_orig, answer, rows_json in all_rows:
            score = get_similarity_score(norm_q, cached_norm)
            if score > best_score:
                best_score = score
                best_row = (answer, rows_json, cached_orig)
                
        if best_row and best_score >= threshold:
            answer, rows_json, cached_orig = best_row
            table_rows = json.loads(rows_json)
            logger.info(f"Persistent Database Cache Hit (Fuzzy: {best_score:.2f}) for: '{query}' (matched '{cached_orig}')")
            return answer, table_rows, cached_orig
            
    except Exception as e:
        logger.error(f"Error reading persistent cache: {e}")
    finally:
        conn.close()
    return None

def set_cached_answer(query: str, answer: str, table_rows: List[Dict[str, Any]]):
    """Persist new query, answer, and table rows to SQLite database."""
    init_db()
    norm_q = _normalize(query)
    if not norm_q or not answer:
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        rows_json = json.dumps(table_rows)
        cursor.execute(
            """
            INSERT OR REPLACE INTO query_cache (normalized_query, original_query, answer, table_rows_json)
            VALUES (?, ?, ?, ?)
            """,
            (norm_q, query, answer, rows_json)
        )
        conn.commit()
        logger.info(f"Successfully saved query to persistent database: '{query}'")
    except Exception as e:
        logger.error(f"Error writing to persistent cache: {e}")
    finally:
        conn.close()
