"""
Search Enrichment Tool.

Performs conditional procurement and supplier lookup using DuckDuckGo search by default,
and falls back to Tavily Search if DuckDuckGo fails or is unavailable.
"""

import os
import re
import requests
from typing import List, Dict, Any
from engine.shared_state import SharedState
from config.constants import SERVICE_CATEGORY_TERMS

GOOGLE_SEARCH_DOMAINS = [
    "indiamart.com",
    "tradeindia.com",
    "alibaba.com",
    "medicalexpo.com",
    "pharmacompass.com"
]


def extract_location(question: str) -> str:
    """Extract location parameters from the user query."""
    pin_code_match = re.search(r"\b\d{5,6}\b", question)
    if pin_code_match:
        return pin_code_match.group()

    location_match = re.search(
        r"\b(?:near|nearby|near to|in|around|to)\s+([a-z][a-z\s-]{2,})",
        question.lower(),
    )
    if location_match:
        return location_match.group(1).strip()
    return ""


def build_procurement_query(question: str) -> str:
    """Build an optimized search query using location and service cost terms."""
    location = extract_location(question)
    question_lower = question.lower()
    
    matched_categories = [
        term for term in SERVICE_CATEGORY_TERMS if term in question_lower
    ]
    categories_str = " ".join(set(matched_categories)) if matched_categories else ""
    location_str = f" near {location}" if location else ""

    return f"{categories_str} service cost supplier rate contact{location_str} {question}".strip()


def is_procurement_related(result_text: str) -> bool:
    """Filter search results to keep them relevant to service-costing and procurement."""
    text_lower = result_text.lower()
    service_markers = [
        "service", "cost", "quotation", "quote", "rate", "packaging",
        "sterilization", "logistics", "quality", "warehousing", "supplier", "vendor"
    ]
    unrelated_markers = [
        "electricity", "power.delhi.gov", "railway", "cement", "pipe", "fitting", "construction"
    ]
    has_marker = any(marker in text_lower for marker in service_markers)
    has_unrelated = any(unrelated in text_lower for u_marker in unrelated_markers for unrelated in [u_marker])
    return has_marker and not has_unrelated


def extract_contacts(text: str) -> str:
    """Extract phone numbers from search result snippets."""
    phone_pattern = r"(?:\+?\d[\d\s().-]{7,}\d)"
    matches = re.findall(phone_pattern, text)
    cleaned_matches = []
    
    for match in matches:
        cleaned = re.sub(r"\s+", " ", match).strip(" .,-")
        digits = re.sub(r"\D", "", cleaned)
        if 8 <= len(digits) <= 15 and cleaned not in cleaned_matches:
            cleaned_matches.append(cleaned)
            
    return ", ".join(cleaned_matches) if cleaned_matches else "Not found in search snippet"


def run_tavily_search(query: str, api_key: str) -> List[Dict[str, Any]]:
    """Query Tavily Search API as fallback."""
    url = "https://api.tavily.com/search"
    response = requests.post(
        url,
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
        },
        timeout=15,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    
    hits = []
    for res in results:
        title = res.get("title", "")
        content = res.get("content", "")
        link = res.get("url", "")
        combined = f"{title} {content}"
        
        # Calculate reliability
        reliability = "Medium: verify directly with vendor"
        if any(d in link for d in GOOGLE_SEARCH_DOMAINS):
            reliability = "High: trusted procurement portal"
            
        hits.append({
            "title": title,
            "url": link,
            "snippet": content,
            "contact_numbers_found": extract_contacts(combined),
            "reliability": reliability,
        })
    return hits


def run_ddg_search(query: str) -> List[Dict[str, Any]]:
    """Query DuckDuckGo text search (100% free fallback)."""
    from duckduckgo_search import DDGS
    domain_query = " OR ".join(f"site:{domain}" for domain in GOOGLE_SEARCH_DOMAINS)
    final_query = f"({domain_query}) {query}"
    
    hits = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(final_query, max_results=5)
            for item in results:
                title = item.get("title", "")
                link = item.get("href", "")
                snippet = item.get("body", "")
                combined = f"{title} {snippet}"
                
                hits.append({
                    "title": title,
                    "url": link,
                    "snippet": snippet,
                    "contact_numbers_found": extract_contacts(combined),
                    "reliability": "High: trusted procurement portal (via DDG)" if any(d in link for d in GOOGLE_SEARCH_DOMAINS) else "Medium",
                })
    except Exception:
        # Fallback to general query if restricted domains fail
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=5)
                for item in results:
                    title = item.get("title", "")
                    link = item.get("href", "")
                    snippet = item.get("body", "")
                    combined = f"{title} {snippet}"
                    
                    hits.append({
                        "title": title,
                        "url": link,
                        "snippet": snippet,
                        "contact_numbers_found": extract_contacts(combined),
                        "reliability": "Medium (Free DDG Search)",
                    })
        except Exception:
            pass
            
    return hits


def run_search_enrichment(state: SharedState):
    """
    Enrich state using DuckDuckGo search by default, and fallback to Tavily search.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    search_query = build_procurement_query(state.query)
    search_hits = []

    # 1. Try DuckDuckGo Search first (Default & Free)
    try:
        search_hits = run_ddg_search(search_query)
        if search_hits:
            state.add_trace("DuckDuckGo Free Search Executed (Default)", {"query": search_query})
    except Exception as e:
        state.add_trace("DuckDuckGo Search Failed", {"error": str(e)})

    # 2. Fallback to Tavily if DDG failed or returned zero results
    if not search_hits and tavily_api_key:
        try:
            search_hits = run_tavily_search(search_query, tavily_api_key)
            state.add_trace("Tavily Search Executed (Fallback)", {"query": search_query})
        except Exception as e:
            state.add_trace("Tavily Search Failed (Fallback)", {"error": str(e)})

    if not search_hits:
        state.add_trace("Search Enrichment Skipped: All search engines failed or no keys configured", {"query": search_query})
        return

    # Filter and format results
    filtered_hits = [hit for hit in search_hits if is_procurement_related(hit["snippet"] + " " + hit["title"])]
    
    state.web_search_results = filtered_hits or search_hits
    state.add_trace("Search Enrichment Results Cached", {
        "results_count": len(state.web_search_results)
    })
