"""
Search Enrichment Tool.

Performs conditional procurement and supplier lookup using DuckDuckGo search by default,
and falls back to Tavily Search if DuckDuckGo fails or is unavailable.
"""

import logging
import os
import re
import requests
import json
import concurrent.futures
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from engine.shared_state import SharedState
from config.constants import SERVICE_CATEGORY_TERMS

logger = logging.getLogger(__name__)
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("VENDOR_SEARCH_CONFIDENCE_THRESHOLD", "0.7"))
DEFAULT_MAX_RESULTS = int(os.getenv("VENDOR_SEARCH_MAX_RESULTS", "8"))

def load_trusted_domains() -> List[str]:
    """Load trusted B2B domains from configuration file."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "trusted_domains.json")
        with open(config_path, "r") as f:
            data = json.load(f)
            return data.get("trusted_domains", [])
    except Exception:
        return [
            "indiamart.com",
            "tradeindia.com",
            "alibaba.com",
            "medicalexpo.com",
            "pharmacompass.com"
        ]

GOOGLE_SEARCH_DOMAINS = load_trusted_domains()

DIRECTORY_DOMAINS = {
    "indiamart.com",
    "tradeindia.com",
    "alibaba.com",
    "medicalexpo.com",
    "pharmacompass.com",
    "justdial.com",
    "exportersindia.com",
    "sulekha.com",
    "yellowpages.com",
}

LOW_QUALITY_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "youtube.com",
    "amazon.com",
    "flipkart.com",
    "wikipedia.org",
}

AD_OR_DIRECTORY_MARKERS = [
    "sponsored",
    "advertisement",
    "classified",
    "directory",
    "yellow pages",
    "justdial",
    "indiamart",
    "tradeindia",
    "exportersindia",
    "alibaba",
]

VENDOR_RELEVANCE_MARKERS = [
    "vendor",
    "supplier",
    "manufacturer",
    "service provider",
    "company",
    "contact",
    "packaging",
    "sterilization",
    "logistics",
    "warehousing",
    "medical",
    "healthcare",
    "biologics",
    "fda",
    "iso",
    "gmp",
]


class HTMLTextExtractor(HTMLParser):
    """Custom HTML parser to extract readable text while ignoring scripts, styles, and headers/footers."""
    def __init__(self):
        super().__init__()
        self.result = []
        self.ignore = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'head', 'meta', 'header', 'footer', 'nav', 'noscript', 'select', 'option'):
            self.ignore = True

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'head', 'meta', 'header', 'footer', 'nav', 'noscript', 'select', 'option'):
            self.ignore = False

    def handle_data(self, data):
        if not self.ignore:
            cleaned = data.strip()
            if cleaned:
                cleaned = re.sub(r'\s+', ' ', cleaned)
                self.result.append(cleaned)

    def get_text(self) -> str:
        return "\n".join(self.result)


def scrape_url(url: str, timeout: int = 5) -> Optional[str]:
    """Scrape a URL and extract text using pure Python HTML parser."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            encoding = response.apparent_encoding or "utf-8"
            html_content = response.content.decode(encoding, errors="ignore")
            parser = HTMLTextExtractor()
            parser.feed(html_content)
            extracted_text = parser.get_text()
            clean_text = "\n".join([line.strip() for line in extracted_text.split("\n") if line.strip()])
            return clean_text[:4000]
    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
    return None


def enrich_hits_with_scraping(hits: List[Dict[str, Any]], max_to_scrape: int = 3) -> List[Dict[str, Any]]:
    """Scrape matched web pages concurrently to extract rich contents."""
    if not hits:
        return hits

    hits_to_scrape = hits[:max_to_scrape]
    logger.info(f"Concurrently scraping top {len(hits_to_scrape)} search results...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_to_scrape) as executor:
        future_to_hit = {
            executor.submit(scrape_url, hit["url"], timeout=6): hit
            for hit in hits_to_scrape
        }
        
        for future in concurrent.futures.as_completed(future_to_hit):
            hit = future_to_hit[future]
            try:
                scraped_text = future.result()
                if scraped_text and len(scraped_text) > 100:
                    hit["content"] = scraped_text
                    combined_text = f"{hit['title']} {scraped_text}"
                    hit["contact_numbers_found"] = extract_contacts(combined_text)
                    logger.info(f"Successfully scraped and enriched URL: {hit['url']}")
            except Exception as e:
                logger.warning(f"Failed parallel scrape for {hit['url']}: {e}")
                
    return hits


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


def _domain_from_url(url: str) -> str:
    """Return a normalized domain for a search-result URL."""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return parsed.netloc.lower().removeprefix("www.")


def _result_text(result: Dict[str, Any]) -> str:
    """Combine the common text fields from a normalized search result."""
    return " ".join(
        [
            str(result.get("title", "")),
            str(result.get("snippet", "")),
            str(result.get("content", "")),
            str(result.get("url", "")),
        ]
    ).lower()


def _is_directory_or_ad_result(result: Dict[str, Any]) -> bool:
    """Detect directory, marketplace, sponsored, or ad-like search results."""
    text = _result_text(result)
    domain = _domain_from_url(result.get("url", ""))
    return domain in DIRECTORY_DOMAINS or any(marker in text for marker in AD_OR_DIRECTORY_MARKERS)


def _is_low_quality_domain(result: Dict[str, Any]) -> bool:
    """Detect domains that are usually weak vendor-discovery signals."""
    domain = _domain_from_url(result.get("url", ""))
    return any(domain == bad or domain.endswith(f".{bad}") for bad in LOW_QUALITY_DOMAINS)


def _is_relevant_vendor_result(query: str, result: Dict[str, Any]) -> bool:
    """Return whether a search result looks relevant to the vendor query."""
    text = _result_text(result)
    query_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", query.lower())
        if len(term) > 2
    }
    matched_query_terms = sum(1 for term in query_terms if term in text)
    has_vendor_marker = any(marker in text for marker in VENDOR_RELEVANCE_MARKERS)
    return (matched_query_terms >= 2 or has_vendor_marker) and is_procurement_related(text)


def _is_company_website(result: Dict[str, Any]) -> bool:
    """Return whether a result appears to be a direct company/vendor website."""
    domain = _domain_from_url(result.get("url", ""))
    if not domain or domain in DIRECTORY_DOMAINS or _is_low_quality_domain(result):
        return False
    return "." in domain and not _is_directory_or_ad_result(result)


def _deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate results by canonical URL/domain and title."""
    seen = set()
    deduped = []
    for result in results:
        url = str(result.get("url", "")).rstrip("/").lower()
        title = str(result.get("title", "")).strip().lower()
        key = url or f"{_domain_from_url(url)}:{title}"
        if key and key not in seen:
            seen.add(key)
            deduped.append(result)
    return deduped


def evaluate_search_results(
    query: str,
    results: List[Dict[str, Any]],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    search_error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate DuckDuckGo vendor-search quality and decide whether Tavily is needed.

    The score considers result count, query/vendor relevance, direct company websites,
    duplicates, directories/ads, unrelated pages, and domain quality.
    """
    if search_error:
        return {
            "confidence": 0.0,
            "reason": f"DuckDuckGo search failed: {search_error}",
            "should_fallback": True,
        }

    total_results = len(results)
    if total_results == 0:
        return {
            "confidence": 0.0,
            "reason": "DuckDuckGo returned no results.",
            "should_fallback": True,
        }

    deduped_results = _deduplicate_results(results)
    duplicate_count = total_results - len(deduped_results)
    relevant_results = [
        result for result in deduped_results if _is_relevant_vendor_result(query, result)
    ]
    company_results = [
        result for result in relevant_results if _is_company_website(result)
    ]
    directory_or_ad_count = sum(1 for result in deduped_results if _is_directory_or_ad_result(result))
    low_quality_count = sum(1 for result in deduped_results if _is_low_quality_domain(result))
    unrelated_count = len(deduped_results) - len(relevant_results)

    relevant_count = len(relevant_results)
    unique_count = len(deduped_results)
    mostly_directories_or_unrelated = (
        directory_or_ad_count + unrelated_count
    ) / max(unique_count, 1) >= 0.6

    result_score = min(unique_count / 5.0, 1.0)
    relevance_score = min(relevant_count / 3.0, 1.0)
    company_score = min(len(company_results) / 2.0, 1.0)
    duplicate_penalty = min(duplicate_count / max(total_results, 1), 1.0)
    quality_penalty = min((directory_or_ad_count + low_quality_count + unrelated_count) / max(unique_count, 1), 1.0)

    confidence = (
        0.25 * result_score
        + 0.35 * relevance_score
        + 0.25 * company_score
        + 0.15 * (1.0 - quality_penalty)
        - 0.10 * duplicate_penalty
    )
    confidence = round(max(0.0, min(1.0, confidence)), 3)

    fallback_reasons = []
    if relevant_count < 3:
        fallback_reasons.append(f"fewer than 3 relevant vendor results ({relevant_count})")
    if not company_results:
        fallback_reasons.append("no direct company/vendor websites found")
    if mostly_directories_or_unrelated:
        fallback_reasons.append("results are mostly directories, ads, or unrelated pages")
    if confidence < confidence_threshold:
        fallback_reasons.append(
            f"confidence {confidence:.2f} is below threshold {confidence_threshold:.2f}"
        )

    return {
        "confidence": confidence,
        "reason": "; ".join(fallback_reasons) if fallback_reasons else "DuckDuckGo results meet vendor-search quality threshold.",
        "should_fallback": bool(fallback_reasons),
    }


def search_tavily(
    query: str,
    api_key: Optional[str] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> List[Dict[str, Any]]:
    """Query Tavily Search API and normalize vendor-search results."""
    api_key = api_key or os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("Missing TAVILY_API_KEY in environment.")

    url = "https://api.tavily.com/search"
    response = requests.post(
        url,
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
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
            "content": content,
            "contact_numbers_found": extract_contacts(combined),
            "reliability": reliability,
            "source": "tavily",
        })
    return hits


def search_duckduckgo(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict[str, Any]]:
    """Query DuckDuckGo text search prioritizing trusted procurement domains, and fallback to general search."""
    from duckduckgo_search import DDGS

    hits = []
    
    # 1. Try search restricted to trusted procurement/supplier portals first
    if "-site:" in query:
        logger.info("Exclusion operators detected. Skipping restricted search phase to prioritize general vendor sites.")
    else:
        try:
            domain_query = " OR ".join(f"site:{domain}" for domain in GOOGLE_SEARCH_DOMAINS)
            final_query = f"({domain_query}) {query}"
            logger.info(f"Executing restricted DDG search query: {final_query}")
            with DDGS() as ddgs:
                results = ddgs.text(final_query, max_results=max_results)
                for item in results:
                    title = item.get("title", "")
                    link = item.get("href", "")
                    snippet = item.get("body", "") or item.get("snippet", "")
                    combined = f"{title} {snippet}"
                    
                    hits.append({
                        "title": title,
                        "url": link,
                        "snippet": snippet,
                        "content": snippet,
                        "contact_numbers_found": extract_contacts(combined),
                        "reliability": "High: trusted procurement portal (via DDG)" if any(d in link for d in GOOGLE_SEARCH_DOMAINS) else "Medium: verify directly with vendor",
                        "source": "duckduckgo",
                    })
        except Exception as e:
            logger.warning(f"Restricted DuckDuckGo search failed: {e}")

    # 2. Fallback to general query if restricted search returned zero results or failed
    if not hits:
        logger.info(f"Restricted search returned no results. Falling back to general DDG query: {query}")
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                for item in results:
                    title = item.get("title", "")
                    link = item.get("href", "")
                    snippet = item.get("body", "") or item.get("snippet", "")
                    combined = f"{title} {snippet}"
                    
                    hits.append({
                        "title": title,
                        "url": link,
                        "snippet": snippet,
                        "content": snippet,
                        "contact_numbers_found": extract_contacts(combined),
                        "reliability": "High: direct vendor website (via DDG)" if _is_company_website({"url": link, "title": title, "snippet": snippet}) else "Medium: verify directly with vendor",
                        "source": "duckduckgo",
                    })
        except Exception as exc:
            raise RuntimeError(f"DuckDuckGo general search failed: {exc}") from exc
            
    return hits



def vendor_search_with_fallback(
    query: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_results: int = DEFAULT_MAX_RESULTS,
    tavily_api_key: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Search vendors with DuckDuckGo first and Tavily only when quality is insufficient.

    Returns normalized results and metadata containing source, fallback status,
    confidence, fallback reason, and result count.
    """
    ddg_results: List[Dict[str, Any]] = []
    ddg_error: Optional[str] = None

    try:
        ddg_results = search_duckduckgo(query, max_results=max_results)
    except Exception as exc:
        ddg_error = str(exc)

    evaluation = evaluate_search_results(
        query=query,
        results=ddg_results,
        confidence_threshold=confidence_threshold,
        search_error=ddg_error,
    )

    metadata = {
        "source": "duckduckgo",
        "fallback_used": False,
        "confidence": evaluation["confidence"],
        "fallback_reason": evaluation["reason"],
        "results_count": len(ddg_results),
    }

    if not evaluation["should_fallback"]:
        logger.info(
            "Vendor search completed",
            extra={
                "query": query,
                "source": "duckduckgo",
                "confidence": evaluation["confidence"],
                "fallback_reason": evaluation["reason"],
                "results_count": len(ddg_results),
            },
        )
        return ddg_results, metadata

    try:
        tavily_results = search_tavily(
            query=query,
            api_key=tavily_api_key,
            max_results=max_results,
        )
        tavily_evaluation = evaluate_search_results(
            query=query,
            results=tavily_results,
            confidence_threshold=confidence_threshold,
        )
        metadata.update(
            {
                "source": "tavily",
                "fallback_used": True,
                "confidence": tavily_evaluation["confidence"],
                "fallback_reason": tavily_evaluation["reason"],
                "results_count": len(tavily_results),
            }
        )
        logger.info(
            "Vendor search completed with fallback",
            extra={
                "query": query,
                "source": "tavily",
                "confidence": tavily_evaluation["confidence"],
                "fallback_reason": tavily_evaluation["reason"],
                "results_count": len(tavily_results),
            },
        )
        return tavily_results, metadata
    except Exception as exc:
        metadata["fallback_error"] = str(exc)
        logger.warning(
            "Vendor search fallback failed",
            extra={
                "query": query,
                "source": "duckduckgo",
                "confidence": evaluation["confidence"],
                "fallback_reason": f"{evaluation['reason']}; Tavily failed: {exc}",
                "results_count": len(ddg_results),
            },
        )
        return ddg_results, metadata


def run_tavily_search(query: str, api_key: str) -> List[Dict[str, Any]]:
    """Backward-compatible wrapper for Tavily search."""
    return search_tavily(query=query, api_key=api_key, max_results=5)


def run_ddg_search(query: str) -> List[Dict[str, Any]]:
    """Backward-compatible wrapper for DuckDuckGo search."""
    return search_duckduckgo(query=query, max_results=5)


def run_search_enrichment(state: SharedState):
    """
    Enrich state using DuckDuckGo search by default, and fallback to Tavily search.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    search_query = build_procurement_query(state.query)
    search_hits, metadata = vendor_search_with_fallback(
        query=search_query,
        tavily_api_key=tavily_api_key,
    )
    state.search_metadata = metadata
    state.add_trace("Vendor Search Executed", {
        "query": search_query,
        "source": metadata["source"],
        "confidence": metadata["confidence"],
        "fallback_used": metadata["fallback_used"],
        "fallback_reason": metadata.get("fallback_reason", ""),
        "results_count": metadata["results_count"],
    })

    if not search_hits:
        state.add_trace("Search Enrichment Skipped: All search engines failed or no keys configured", {"query": search_query})
        return

    # Concurrently scrape the top 3 matches (respecting the 2-30s latency budget)
    state.add_trace("Concurrently scraping top vendor web URLs")
    enriched_hits = enrich_hits_with_scraping(search_hits, max_to_scrape=3)

    # Filter and format results using both snippets and rich scraped content
    filtered_hits = [
        hit for hit in enriched_hits 
        if is_procurement_related(hit["snippet"] + " " + hit["title"] + " " + hit.get("content", ""))
    ]
    
    state.web_search_results = filtered_hits or enriched_hits
    state.add_trace("Search Enrichment Results Cached with Scraped Text", {
        "results_count": len(state.web_search_results)
    })

