import streamlit as st
import pandas as pd
import re
import os
import io
import requests
import json
from typing import Optional

def load_trusted_domains() -> list[str]:
    """Load trusted B2B domains from configuration file."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config", "trusted_domains.json")
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

TRUSTED_DOMAINS = load_trusted_domains()
from dotenv import load_dotenv
from engine.category_selector import get_shipment_category
from engine.uploaded_costs import (
    SERVICE_RATE_NAMES,
    build_comparison_factor_rows,
    build_costing_engine_chunks,
    calculate_uploaded_costs,
    find_vendor_matches,
    get_answer_quantity,
    get_category_quantity,
    get_factor_rate_value,
    get_vendor_name,
)
from engine.semantic_costs import (
    build_semantic_breakdown_rows,
    build_semantic_breakdown_rows_by_category,
    is_detailed_audit_question,
)
from engine.query_planner import build_query_plan
from engine.search_enrichment import vendor_search_with_fallback
from rag.pipeline import (
    build_gemini_rag_answer,
    dataframe_to_chunks,
    pdf_to_chunks,
    retrieve_uploaded_context,
)
from engine.vision_extractor import extract_rate_card_from_media
from config.constants import (
    TAVILY_SEARCH_URL,
    TAVILY_SEARCH_LIMIT_PER_SESSION,
    SERVICE_COST_TERMS,
    WEB_SEARCH_TERMS,
    SERVICE_EXPLAIN_TERMS,
    SERVICE_CATEGORY_TERMS,
    VENDOR_DISCOVERY_TERMS,
    STRUCTURED_ANSWER_TERMS,
    VENDOR_SUPPLIER_TERMS,
    SPECIFIC_RATE_TERMS,
    AVAILABLE_VENDOR_TERMS,
    REASON_TERMS,
    COMPARISON_FACTOR_TERMS,
    COMPARE_OPTION_TERMS,
    BEST_CHEAPEST_TERMS,
    BEST_FOR_QUANTITY_TERMS,
    QUANTITY_CHANGE_TERMS,
)

load_dotenv()

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
    "justdial",
    "yellowpages"
}

def is_directory_domain(url: str) -> bool:
    from urllib.parse import urlparse
    if not url:
        return False
    parsed = urlparse(url if "://" in url else f"https://{url}")
    domain = parsed.netloc.lower().removeprefix("www.")
    return any(d == domain or domain.endswith(f".{d}") or d in domain for d in DIRECTORY_DOMAINS)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def is_service_cost_question(question: str) -> bool:
    """Check if question is within service-costing domain."""
    question_lower = question.lower()
    return any(term in question_lower for term in SERVICE_COST_TERMS)


def is_vendor_discovery_question(question: str) -> bool:
    """Check if question requires vendor/supplier discovery."""
    question_lower = question.lower()
    return any(term in question_lower for term in VENDOR_DISCOVERY_TERMS)


def has_vendor_or_supplier_term(question: str) -> bool:
    """Check if question contains vendor/supplier terminology."""
    question_lower = question.lower()
    return any(term in question_lower for term in VENDOR_SUPPLIER_TERMS)


def has_service_category(question: str) -> bool:
    """Check if question mentions a service category."""
    question_lower = question.lower()
    return any(term in question_lower for term in SERVICE_CATEGORY_TERMS)


def has_location_hint(question: str) -> bool:
    """Check if question contains location information."""
    question_lower = question.lower()

    if "near me" in question_lower:
        return True

    has_pin_code = bool(re.search(r"\b\d{5,6}\b", question_lower))
    has_near_location = bool(
        re.search(
            r"\b(?:near|nearby|near to|in|around|to)\s+(?:[a-z][a-z\s-]{2,}?|\d{5,6})(?=\s+(?:for|with|using|via|and|contact|phone|vendor|vendors|supplier|suppliers)\b|[?.!,]|$)",
            question_lower,
        )
    )

    return has_pin_code or has_near_location


def is_location_vendor_search(question: str) -> bool:
    """Check if question is a location-based vendor search."""
    return has_vendor_or_supplier_term(question) and has_location_hint(question)


def is_explicit_web_vendor_request(question: str) -> bool:
    """Check if the user explicitly wants live web vendor discovery."""
    question_lower = question.lower()
    explicit_web_terms = [
        "tavily",
        "web search",
        "search web",
        "find vendors",
        "find vendor",
        "vendor list",
        "vendors list",
        "list vendors",
        "list vendor",
        "near",
        "nearby",
        "around",
        "contact",
        "phone",
        "mobile",
        "trusted",
        "reliable",
    ]
    return has_vendor_or_supplier_term(question) and any(
        term in question_lower for term in explicit_web_terms
    )


def validate_web_search_question(question: str) -> Optional[str]:
    """Validate question has required elements for web search."""
    if st.session_state.get("enable_web_search", False):
        return None

    if not is_vendor_discovery_question(question) and not is_explicit_web_vendor_request(question):
        return None

    question_lower = question.lower()
    location_optional_terms = [
        "external reputation",
        "certification",
        "certifications",
        "customer reviews",
        "regulatory history",
        "regulatory concerns",
        "public web",
        "public web sources",
        "publicly",
        "available publicly",
        "official website",
        "support email",
        "market benchmark",
        "market benchmarks",
        "market intelligence",
        "duckduckgo",
    ]
    
    has_cat_or_vendor = has_service_category(question) or has_vendor_or_supplier_term(question)
    has_loc_or_opt = has_location_hint(question) or any(term in question_lower for term in location_optional_terms)

    # Relaxed validation: block only if BOTH are entirely missing
    if not has_cat_or_vendor and not has_loc_or_opt:
        return "Please include a service category (e.g., packaging, logistics) or a location hint (e.g., city name, 'near me') before I search the web."

    return None


# =============================================================================
# DECISION FUNCTIONS
# =============================================================================

def should_use_web_search(question: str) -> bool:
    """Determine if web search is needed for this question."""
    if not is_service_cost_question(question):
        return False

    question_lower = question.lower()

    needs_search = any(term in question_lower for term in WEB_SEARCH_TERMS)
    needs_explanation = any(term in question_lower for term in SERVICE_EXPLAIN_TERMS)
    needs_location_vendor_search = is_location_vendor_search(question)
    needs_explicit_vendor_search = is_explicit_web_vendor_request(question)
    asks_specific_uploaded_rate = any(
        term in question_lower
        for term in SPECIFIC_RATE_TERMS
    )

    return (
        needs_location_vendor_search
        or needs_explicit_vendor_search
        or needs_search
        or (needs_explanation and not asks_specific_uploaded_rate)
    )


def should_use_structured_answer(question: str) -> bool:
    """Determine if structured CSV answer should be used."""
    question_lower = question.lower()
    structured_terms = (
        STRUCTURED_ANSWER_TERMS
        + AVAILABLE_VENDOR_TERMS
        + SPECIFIC_RATE_TERMS
        + COMPARISON_FACTOR_TERMS
        + BEST_CHEAPEST_TERMS
        + BEST_FOR_QUANTITY_TERMS
        + QUANTITY_CHANGE_TERMS
        + REASON_TERMS
        + SERVICE_RATE_NAMES
        + [
            "small",
            "medium",
            "large",
            "ranking",
            "rank",
            "recommend",
            "recommended",
            "cost-effective",
            "breakdown",
            "detailed",
            "economies of scale",
            "trend",
            "as quantity increases",
            "moving from",
            "reduction",
            "high-volume",
            "at scale",
            "transport",
            "transportation",
            "handling",
            "distributor",
            "inspection",
            "audit",
            "insurance",
            "inventory",
            "assurance",
            "overhead",
            "complete comparative analysis",
        ]
    )
    return any(term in question_lower for term in structured_terms)


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_phone_numbers(text: str) -> str:
    """Extract phone numbers from text."""
    phone_pattern = r"(?:\+?\d[\d\s().-]{7,}\d)"
    matches = re.findall(phone_pattern, text)

    cleaned_matches = []

    for match in matches:
        cleaned = re.sub(r"\s+", " ", match).strip(" .,-")
        digits = re.sub(r"\D", "", cleaned)

        if 8 <= len(digits) <= 15 and cleaned not in cleaned_matches:
            cleaned_matches.append(cleaned)

    return ", ".join(cleaned_matches)


def get_source_reliability(result: dict) -> str:
    """Determine reliability of search result source."""
    url = result.get("url", "").lower()
    score = result.get("score")

    if any(domain in url for domain in [".gov", ".edu", "iso.org", "fda.gov", "who.int"]):
        return "High: official or standards source"

    if score is not None and score >= 0.75:
        return "Medium-high: strong search match"

    if any(domain in url for domain in ["indiamart", "justdial", "tradeindia", "exportersindia"]):
        return "Medium: directory listing, verify directly"

    return "Medium: verify directly with vendor"


def extract_location_hint(question: str) -> str:
    """Extract location hint from question."""
    pin_code_match = re.search(r"\b\d{5,6}\b", question)

    if pin_code_match:
        return pin_code_match.group()

    location_match = re.search(
        r"\b(?:near|nearby|near to|in|around|to)\s+([a-z][a-z\s-]{2,}?)(?=\s+(?:for|with|using|via|and|contact|phone|vendor|vendors|supplier|suppliers)\b|[?.!,]|$)",
        question.lower(),
    )

    if location_match:
        return location_match.group(1).strip()

    return ""


# =============================================================================
# QUERY BUILDING FUNCTIONS
# =============================================================================

def get_requested_service_terms(question: str) -> str:
    """Extract service category terms from question."""
    question_lower = question.lower()
    service_aliases = {
        "medical suppy": "medical supply",
    }
    matched_terms = [
        service_aliases.get(term, term)
        for term in SERVICE_CATEGORY_TERMS
        if term in question_lower
    ]

    if matched_terms:
        return " ".join(sorted(set(matched_terms)))

    return ""


def build_service_vendor_search_query(question: str) -> str:
    """Build optimized, natural search query for search engines."""
    location = extract_location_hint(question)
    service_terms = get_requested_service_terms(question)
    
    # Strip common conversational phrases to build a clean search query
    clean_q = re.sub(r"\b(?:find|search|show|list|get|please|tell me|who are|where is|where can i find)\b", "", question, flags=re.IGNORECASE)
    clean_q = re.sub(r"[?!.,]", "", clean_q).strip()
    
    # Exclude B2B yellow pages and directories to prioritize direct supplier websites
    exclusion_suffix = " -site:indiamart.com -site:justdial.com -site:tradeindia.com -site:alibaba.com -site:yellowpages.com -site:sulekha.com -site:exportersindia.com"
    
    # If the clean question is already solid, use it. Otherwise construct a focused query.
    if len(clean_q.split()) >= 3:
        return f"{clean_q}{exclusion_suffix}"
        
    location_text = f" near {location}" if location else ""
    return f"{service_terms} service vendor supplier{location_text}{exclusion_suffix}".strip()


def is_service_related_search_result(result: dict) -> bool:
    """Filter search results for service-related content."""
    result_text = " ".join(
        [
            result.get("title", ""),
            result.get("content", ""),
            result.get("url", ""),
        ]
    ).lower()

    service_markers = [
        "service",
        "cost",
        "quotation",
        "quote",
        "rate",
        "packaging",
        "sterilization",
        "logistics",
        "quality",
        "warehousing",
        "warehouse",
        "medical",
        "supplies",
        "healthcare",
        "equipment",
        "supplier",
        "vendor",
    ]
    unrelated_markers = [
        "electricity",
        "power.delhi.gov",
        "railway",
        "cement",
        "pipe",
        "fitting",
        "construction",
    ]

    has_service_marker = any(marker in result_text for marker in service_markers)
    has_unrelated_marker = any(marker in result_text for marker in unrelated_markers)

    return has_service_marker and not has_unrelated_marker


# =============================================================================
# ANSWER BUILDING FUNCTIONS
# =============================================================================

def build_web_answer(question: str) -> tuple[str, list[dict]]:
    """Build answer from web search results using persistent database caching and quality fallbacks."""
    # 1. Check persistent SQLite database cache first!
    from engine.persistent_cache import get_cached_answer, set_cached_answer
    
    cached_res = get_cached_answer(question)
    if cached_res is not None:
        answer, table_rows, matched_query = cached_res
        if matched_query.lower().strip("?!. ") != question.lower().strip("?!. "):
            st.toast(f"⚡ Served from database cache (similar query match: '{matched_query}')")
        else:
            st.toast("⚡ Served from database cache")
        return answer, table_rows

    # 2. Cache Miss: Run live search and scraping pipeline
    search_query = build_service_vendor_search_query(question)
    location = extract_location_hint(question)

    try:
        results, metadata = vendor_search_with_fallback(search_query)
        
        # Concurrently scrape top results to extract phone numbers from actual page content
        from engine.search_enrichment import enrich_hits_with_scraping
        results = enrich_hits_with_scraping(results, max_to_scrape=3)
    except Exception as exc:
        return f"Vendor web search failed: {str(exc)}", []

    if not results:
        return "I could not find relevant web results for that question.", []

    engine_used = "DuckDuckGo" if metadata.get("source") == "duckduckgo" else "Tavily"
    table_rows = []

    # Filter service results
    service_results = [
        result
        for result in results
        if st.session_state.get("enable_web_search", False) or is_service_related_search_result(result)
    ]

    # Prioritize direct vendor/company websites by filtering out directories
    direct_vendors = [
        result for result in service_results
        if not is_directory_domain(result.get("url", ""))
    ]
    
    # Use direct vendors if we found any, otherwise fallback to all service results
    final_service_results = direct_vendors if len(direct_vendors) >= 1 else service_results

    if not final_service_results:
        return (
            f"I found web results using {engine_used}, but they did not look related to service-cost vendors, "
            "quotations, rates, packaging, sterilization, logistics, quality, warehousing, or benchmarks. "
            "Try adding a specific service category and location."
        ), []

    for result in final_service_results[:5]:
        result_text = " ".join(
            [
                result.get("title", ""),
                result.get("content", ""),
                result.get("url", ""),
            ]
        )

        table_rows.append(
            {
                "vendor_or_source": result.get("title", ""),
                "location": location or "Not specified",
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "contact_numbers_found": extract_phone_numbers(result_text) or "Not found in search result",
                "reliability": get_source_reliability(result),
                "snippet": result.get("content") or result.get("snippet", ""),
                "search_metadata": {
                    "source": metadata.get("source"),
                    "fallback_used": metadata.get("fallback_used", False),
                    "confidence": metadata.get("confidence", 0.0),
                },
            }
        )

    answer = (
        f"Here are live {engine_used} results for that vendor/location request. Contact numbers are shown "
        "only when they appear in the search result text. Verify each vendor directly before making a decision.\n\n"
        f"Search metadata: {json.dumps({'source': metadata.get('source'), 'fallback_used': metadata.get('fallback_used', False), 'confidence': metadata.get('confidence', 0.0)})}"
    )

    # 3. Store the new answer and table rows persistently in SQLite database for future reuse!
    set_cached_answer(question, answer, table_rows)

    return answer, table_rows


def _list_available_vendors(csv_dataframes: list) -> tuple[str, list[dict]]:
    """List all available vendors from uploaded CSV files."""
    vendor_names = [get_vendor_name(file_name) for file_name, _ in csv_dataframes]
    answer = "Available vendors are: " + ", ".join(vendor_names) + "."

    return answer, [
        {
            "vendor": get_vendor_name(file_name),
            "file": file_name,
        }
        for file_name, _ in csv_dataframes
    ]


def _resolve_answer_quantity(question: str, csv_dataframes: list, quantity: int) -> int:
    """Resolve quantity from the question, falling back to recent chat context."""
    context_quantity = st.session_state.get("last_answer_quantity", quantity)
    category_quantity = get_category_quantity(question, csv_dataframes, context_quantity)
    return get_answer_quantity(question, category_quantity)


def _has_explicit_quantity_context(question: str) -> bool:
    """Return True when the user named a quantity or shipment size."""
    if re.search(r"\b(?:small|medium|large)\b", question, re.IGNORECASE):
        return True

    return bool(
        re.search(
            r"\b\d+\s*(?:-|to)?\s*(?:unit|units|kit|kits)\b",
            question,
            re.IGNORECASE,
        )
    )


def _extract_quantity_range(question: str) -> Optional[tuple[int, int]]:
    """Extract a numeric order range such as 200-500 units."""
    range_match = re.search(
        r"\b(\d+)\s*(?:-|–|to)\s*(\d+)\s*(?:unit|units|kit|kits)?\b",
        question,
        re.IGNORECASE,
    )

    if not range_match:
        return None

    start_qty = int(range_match.group(1))
    end_qty = int(range_match.group(2))

    return (min(start_qty, end_qty), max(start_qty, end_qty))


def _get_optimized_range_quantity(question: str, fallback_quantity: int) -> Optional[int]:
    """Use the upper range bound for fixed shipment/period per-unit optimization."""
    quantity_range = _extract_quantity_range(question)

    if not quantity_range:
        return None

    question_lower = question.lower()
    has_optimization_intent = any(
        term in question_lower
        for term in ["optimized", "optimize", "minimize", "lowest", "cheapest", "best"]
    )

    if not has_optimization_intent:
        return fallback_quantity

    return quantity_range[1]


def _explain_vendor_choice(question_lower: str) -> Optional[tuple[str, list]]:
    """Explain why a specific vendor was selected."""
    previous_best = st.session_state.get("last_best_option")
    if not previous_best or not any(term in question_lower for term in REASON_TERMS):
        return None

    answer = (
        f"{previous_best['file']} was selected because it had the lowest total service cost "
        f"for {previous_best['quantity']} units: {previous_best['total_cost']}."
    )

    return answer, []


def _get_specific_rate(
    question: str,
    question_lower: str,
    csv_dataframes: list,
    answer_quantity: int,
) -> tuple[str, list[dict]]:
    """Get specific rate for requested service type."""
    requested_rate = next(
        (rate_name for rate_name in SERVICE_RATE_NAMES if rate_name in question_lower),
        None,
    )

    if not requested_rate:
        return None, []

    vendor_matches = find_vendor_matches(question, csv_dataframes)
    selected_dataframes = vendor_matches or csv_dataframes
    table_rows = []

    for file_name, df in selected_dataframes:
        row = get_shipment_category(df, answer_quantity)
        rate_value = get_factor_rate_value(row, requested_rate, answer_quantity)

        if rate_value is None:
            raise KeyError(f"{requested_rate}_rate")

        table_rows.append(
            {
                "vendor": get_vendor_name(file_name),
                "shipment_category": row.get("shipment_category", "N/A"),
                f"{requested_rate}_rate": round(rate_value, 2),
            }
        )

    if vendor_matches:
        answer = (
            f"The {requested_rate} cost for {answer_quantity} units is shown below "
            f"for the matched vendor."
        )
    elif "vendor" in question_lower:
        answer = (
            "I could not find that specific vendor in the uploaded CSVs. "
            f"I do not see a matching vendor name or `vendor_id` value, so here are the {requested_rate} rates for all uploaded vendors."
        )
    else:
        answer = (
            f"Here are the {requested_rate} rates for {answer_quantity} units "
            "from the uploaded vendors."
        )

    return answer, table_rows


def _get_requested_factors(question_lower: str) -> list[str]:
    """Return service factors mentioned in the question."""
    return [
        rate_name
        for rate_name in SERVICE_RATE_NAMES
        if rate_name in question_lower
    ]


def _rank_by_requested_factor(
    question: str,
    question_lower: str,
    csv_dataframes: list,
    answer_quantity: int,
) -> tuple[str, list[dict]]:
    """Rank vendors by a requested service factor or combined factor set."""
    requested_factors = _get_requested_factors(question_lower)

    if not requested_factors:
        return None, []

    asks_extreme = any(
        term in question_lower
        for term in [
            "lowest",
            "cheapest",
            "minimizing",
            "minimize",
            "highest",
            "most expensive",
            "efficient",
            "efficiency",
            "ranking",
            "rank",
            "recommend",
            "recommended",
            "best",
            "compare",
            "comparison",
            "between",
            "across",
        ]
    )

    if not asks_extreme:
        return None, []

    range_quantity = _get_optimized_range_quantity(question, answer_quantity)
    scoring_quantity = range_quantity or answer_quantity
    quantity_range = _extract_quantity_range(question)
    selected_dataframes = find_vendor_matches(question, csv_dataframes)
    table_rows = build_comparison_factor_rows(
        selected_dataframes or csv_dataframes,
        scoring_quantity,
    )
    scored_rows = []

    for row in table_rows:
        score = sum(
            row.get(f"{factor}_rate", 0)
            for factor in requested_factors
        )
        scored_row = {
            "vendor": row["vendor"],
            "file": row["file"],
            "shipment_category": row["shipment_category"],
            "quantity": scoring_quantity,
            "combined_rate": round(score, 2),
        }

        if quantity_range:
            scored_row["requested_quantity_range"] = f"{quantity_range[0]}-{quantity_range[1]}"

        for factor in requested_factors:
            scored_row[f"{factor}_rate"] = row.get(f"{factor}_rate")

        scored_rows.append(scored_row)

    reverse_sort = any(
        term in question_lower
        for term in ["highest", "most expensive"]
    )
    sorted_rows = sorted(
        scored_rows,
        key=lambda item: item["combined_rate"],
        reverse=reverse_sort,
    )
    selected = sorted_rows[0]
    direction = "highest" if reverse_sort else "lowest"
    factor_text = " + ".join(requested_factors)
    quantity_text = f"{scoring_quantity} units"

    if quantity_range and range_quantity:
        quantity_text = (
            f"{scoring_quantity} units within the requested "
            f"{quantity_range[0]}-{quantity_range[1]} unit range"
        )

    answer = (
        f"For {quantity_text}, **{selected['file']}** has the {direction} "
        f"{factor_text} cost at **{selected['combined_rate']}** per unit."
    )

    return answer, sorted_rows


def _rank_by_component_cost(
    question_lower: str,
    csv_dataframes: list,
    answer_quantity: int,
) -> tuple[str, list[dict]]:
    """Rank vendors by detailed component columns such as transport or audit."""
    component_groups = {
        "quality assurance": [
            "inspection_cost_per_shipment",
            "audit_cost_per_shipment",
            "documentation_cost_per_shipment",
        ],
        "inspection and audit": [
            "inspection_cost_per_shipment",
            "audit_cost_per_shipment",
        ],
        "operational overhead": [
            "transport_cost_per_shipment",
            "handling_cost_per_shipment",
            "distributor_fee_per_shipment",
            "inspection_cost_per_shipment",
            "audit_cost_per_shipment",
            "documentation_cost_per_shipment",
            "warehouse_rent_period",
            "insurance_cost_period",
            "inventory_handling_cost_period",
        ],
        "transportation": ["transport_cost_per_shipment"],
        "transport": ["transport_cost_per_shipment"],
        "distributor": ["distributor_fee_per_shipment"],
        "inspection": ["inspection_cost_per_shipment"],
        "audit": ["audit_cost_per_shipment"],
        "assurance": [
            "inspection_cost_per_shipment",
            "audit_cost_per_shipment",
            "documentation_cost_per_shipment",
        ],
        "overhead": [
            "transport_cost_per_shipment",
            "handling_cost_per_shipment",
            "distributor_fee_per_shipment",
            "inspection_cost_per_shipment",
            "audit_cost_per_shipment",
            "documentation_cost_per_shipment",
            "warehouse_rent_period",
            "insurance_cost_period",
            "inventory_handling_cost_period",
        ],
    }
    selected_columns = []
    label = None

    for component_name, columns in component_groups.items():
        if component_name in question_lower:
            selected_columns = columns
            label = component_name
            break

    if not selected_columns:
        return None, []

    table_rows = []

    for file_name, df in csv_dataframes:
        row = get_shipment_category(df, answer_quantity)
        component_total = sum(float(row[column]) for column in selected_columns)
        table_rows.append(
            {
                "vendor": get_vendor_name(file_name),
                "file": file_name,
                "shipment_category": row.get("shipment_category", "N/A"),
                "quantity": answer_quantity,
                f"{label}_cost": round(component_total, 2),
            }
        )

    reverse_sort = any(
        term in question_lower
        for term in ["highest", "most expensive"]
    )
    sorted_rows = sorted(
        table_rows,
        key=lambda item: item[f"{label}_cost"],
        reverse=reverse_sort,
    )
    direction = "highest" if reverse_sort else "lowest"
    selected = sorted_rows[0]

    answer = (
        f"For {answer_quantity} units, **{selected['file']}** has the {direction} "
        f"{label} cost at **{selected[f'{label}_cost']}**."
    )

    return answer, sorted_rows


def _build_vendor_breakdown(
    question: str,
    question_lower: str,
    csv_dataframes: list,
    answer_quantity: int,
) -> tuple[str, list[dict]]:
    """Build a deterministic factor breakdown for a matched vendor."""
    if not any(term in question_lower for term in ["breakdown", "detailed"]):
        return None, []

    vendor_matches = find_vendor_matches(question, csv_dataframes)

    if not vendor_matches:
        return None, []

    if not _has_explicit_quantity_context(question):
        semantic_answer, semantic_rows = build_semantic_breakdown_rows_by_category(
            question,
            vendor_matches,
        )

        if semantic_rows:
            answer = (
                "Here is the detailed cost breakdown across the matched vendor's "
                "shipment quantity bands. "
                + semantic_answer
            )
            return answer, semantic_rows

    semantic_answer, semantic_rows = build_semantic_breakdown_rows(
        question,
        vendor_matches,
        answer_quantity,
    )

    if semantic_rows:
        answer = (
            f"Here is the detailed cost breakdown for {answer_quantity} units "
            "from the matched vendor. "
            + semantic_answer
        )
        return answer, semantic_rows

    table_rows = build_comparison_factor_rows(vendor_matches, answer_quantity)
    answer = (
        f"Here is the detailed cost breakdown for {answer_quantity} units "
        "from the matched vendor."
    )

    return answer, table_rows


def _build_semantic_cost_answer(
    question: str,
    question_lower: str,
    csv_dataframes: list,
    answer_quantity: int,
) -> tuple[str, list[dict]]:
    """Build a hierarchical semantic cost response for sub-component terms."""
    if not is_detailed_audit_question(question_lower):
        return None, []

    semantic_answer, semantic_rows = build_semantic_breakdown_rows(
        question,
        csv_dataframes,
        answer_quantity,
    )

    if not semantic_rows:
        return None, []

    return semantic_answer, semantic_rows


def _get_vendor_total_cost(
    question: str,
    question_lower: str,
    csv_dataframes: list,
    answer_quantity: int,
) -> tuple[str, list[dict]]:
    """Return total service cost for a specifically named vendor."""
    asks_total_cost = (
        "total" in question_lower
        and "service cost" in question_lower
    )

    if not asks_total_cost:
        return None, []

    vendor_matches = find_vendor_matches(question, csv_dataframes)

    if not vendor_matches:
        return None, []

    table_rows = build_comparison_factor_rows(vendor_matches, answer_quantity)

    answer = (
        f"Here is the estimated total service cost for {answer_quantity} units "
        "from the matched vendor."
    )

    return answer, table_rows


def _compare_scale_change(
    question_lower: str,
    csv_dataframes: list,
) -> tuple[str, list[dict]]:
    """Compare per-unit reduction between shipment categories."""
    scale_terms = [
        "economies of scale",
        "trend",
        "as quantity increases",
        "moving from",
        "reduction",
        "high-volume",
        "at scale",
    ]

    if not any(term in question_lower for term in scale_terms):
        return None, []

    requested_factors = _get_requested_factors(question_lower)
    trend_factor = requested_factors[0] if requested_factors else None
    table_rows = []

    for file_name, df in csv_dataframes:
        small_qty = get_category_quantity("small", [(file_name, df)], 1)
        medium_qty = get_category_quantity("medium", [(file_name, df)], small_qty)
        large_qty = get_category_quantity("large", [(file_name, df)], medium_qty)

        if trend_factor:
            small_row = build_comparison_factor_rows([(file_name, df)], small_qty)[0]
            medium_row = build_comparison_factor_rows([(file_name, df)], medium_qty)[0]
            large_row = build_comparison_factor_rows([(file_name, df)], large_qty)[0]
            small_cost = small_row[f"{trend_factor}_rate"]
            medium_cost = medium_row[f"{trend_factor}_rate"]
            large_cost = large_row[f"{trend_factor}_rate"]
            cost_label = f"{trend_factor}_rate"
        else:
            small_cost = calculate_uploaded_costs([(file_name, df)], small_qty)[0]["total_cost"]
            medium_cost = calculate_uploaded_costs([(file_name, df)], medium_qty)[0]["total_cost"]
            large_cost = calculate_uploaded_costs([(file_name, df)], large_qty)[0]["total_cost"]
            cost_label = "total_cost"

        table_rows.append(
            {
                "vendor": get_vendor_name(file_name),
                "file": file_name,
                "small_qty": small_qty,
                "small_total_cost": small_cost,
                "medium_qty": medium_qty,
                "medium_total_cost": medium_cost,
                "large_qty": large_qty,
                "large_total_cost": large_cost,
                "trend_basis": cost_label,
                "small_to_large_reduction": round(small_cost - large_cost, 2),
                "medium_to_large_reduction": round(medium_cost - large_cost, 2),
            }
        )

    reduction_key = "medium_to_large_reduction" if "medium" in question_lower else "small_to_large_reduction"

    if any(term in question_lower for term in ["low-cost", "minimizing", "minimize"]):
        sorted_rows = sorted(table_rows, key=lambda item: item["large_total_cost"])
        answer = (
            f"For high-volume operations, **{sorted_rows[0]['file']}** has the lowest "
            f"Large-category {cost_label}: **{sorted_rows[0]['large_total_cost']}**."
        )
    elif "lowest" in question_lower and trend_factor:
        sorted_rows = sorted(table_rows, key=lambda item: item["large_total_cost"])
        answer = (
            f"**{sorted_rows[0]['file']}** has the lowest Large-category {cost_label}: "
            f"**{sorted_rows[0]['large_total_cost']}**."
        )
    else:
        sorted_rows = sorted(
            table_rows,
            key=lambda item: item[reduction_key],
            reverse=True,
        )
        answer = (
            f"**{sorted_rows[0]['file']}** shows the greatest {reduction_key.replace('_', ' ')} "
            f"at **{sorted_rows[0][reduction_key]}**."
        )

    return answer, sorted_rows


def _rank_landed_scalability_compliance(
    question_lower: str,
    csv_dataframes: list,
    answer_quantity: int,
    query_plan: Optional[dict] = None,
) -> tuple[str, list[dict]]:
    """Rank vendors across landed cost, shipment fit, and compliance overhead."""
    plan_criteria = set((query_plan or {}).get("criteria", []))
    has_planned_intent = (
        (query_plan or {}).get("intent") == "multi_criteria_vendor_ranking"
        and "shipment_scalability" in plan_criteria
        and len(plan_criteria & {"total_landed_cost", "compliance_overhead"}) >= 1
    )
    has_legacy_terms = all(
        term in question_lower
        for term in [
            "rank",
            "total landed cost",
            "shipment scalability",
            "compliance overhead",
        ]
    )

    if not has_planned_intent and not has_legacy_terms:
        return None, []

    table_rows = []

    for file_name, df in csv_dataframes:
        row = get_shipment_category(df, answer_quantity)
        factor_row = build_comparison_factor_rows(
            [(file_name, df)],
            answer_quantity,
        )[0]

        compliance_columns = [
            "inspection_cost_per_shipment",
            "audit_cost_per_shipment",
            "documentation_cost_per_shipment",
        ]
        compliance_overhead = sum(
            float(row[column])
            for column in compliance_columns
            if column in row
        )
        min_qty = int(row.get("min_qty", 0))
        max_qty = int(row.get("max_qty", 0))
        supports_quantity = min_qty <= answer_quantity <= max_qty
        available_capacity = max(max_qty - answer_quantity, 0)

        table_rows.append(
            {
                "vendor": get_vendor_name(file_name),
                "file": file_name,
                "shipment_category": row.get("shipment_category", "N/A"),
                "quantity": answer_quantity,
                "category_min_qty": min_qty,
                "category_max_qty": max_qty,
                "supports_quantity": supports_quantity,
                "available_capacity": available_capacity,
                "total_cost_per_unit": factor_row["total_cost"],
                "total_landed_cost": round(
                    factor_row["total_cost"] * answer_quantity,
                    2,
                ),
                "compliance_overhead_cost": round(compliance_overhead, 2),
                "compliance_overhead_per_unit": round(
                    compliance_overhead / answer_quantity,
                    2,
                ),
                "rank_basis": (
                    "total_landed_cost asc, compliance_overhead_per_unit asc, "
                    "available_capacity desc"
                ),
            }
        )

    sorted_rows = sorted(
        table_rows,
        key=lambda item: (
            not item["supports_quantity"],
            item["total_landed_cost"],
            item["compliance_overhead_per_unit"],
            -item["available_capacity"],
        ),
    )

    for rank, row in enumerate(sorted_rows, start=1):
        row["rank"] = rank

    best_row = sorted_rows[0]
    answer = (
        f"For {answer_quantity} units, **{best_row['file']}** ranks first on "
        "total landed cost, with shipment scalability and compliance overhead "
        "shown as supporting decision factors."
    )

    return answer, sorted_rows


def _complete_comparative_report(
    question_lower: str,
    csv_dataframes: list,
) -> tuple[str, list[dict]]:
    """Return all vendor/category factor rows for broad report questions."""
    if not (
        "complete" in question_lower
        and "comparative" in question_lower
        and "all shipment categories" in question_lower
    ):
        return None, []

    table_rows = []

    for file_name, df in csv_dataframes:
        if "shipment_category" not in df.columns:
            continue

        for _, category_row in df.iterrows():
            category = str(category_row["shipment_category"])
            category_qty = int(category_row["min_qty"])
            comparison_rows = build_comparison_factor_rows(
                [(file_name, df)],
                category_qty,
            )

            if comparison_rows:
                comparison_rows[0]["report_category"] = category
                table_rows.append(comparison_rows[0])

    answer = (
        "Here is the comparative deterministic cost report across all uploaded "
        "vendors and shipment categories."
    )

    return answer, table_rows


def _compare_factors(
    question_lower: str,
    csv_dataframes: list,
    answer_quantity: int,
) -> tuple[str, list[dict]]:
    """Handle comparison factor questions."""
    if not (
        any(term in question_lower for term in COMPARISON_FACTOR_TERMS)
        and any(term in question_lower for term in COMPARE_OPTION_TERMS)
    ):
        return None, []

    table_rows = build_comparison_factor_rows(csv_dataframes, answer_quantity)

    answer = (
        f"For this comparison at {answer_quantity} units, I consider the matching shipment "
        "category and these service-cost factors: packaging, sterilization, logistics, "
        "quality, and warehousing. The total cost is calculated from those factor rates "
        "for each uploaded vendor."
    )

    return answer, table_rows


def _find_best_option(
    question_lower: str,
    csv_dataframes: list,
    answer_quantity: int,
) -> tuple[str, list[dict]]:
    """Find best/cheapest option among vendors."""
    is_best_question = any(
        term in question_lower
        for term in (
            BEST_CHEAPEST_TERMS
            + BEST_FOR_QUANTITY_TERMS
            + QUANTITY_CHANGE_TERMS
            + [
                "ranking",
                "rank",
                "recommend",
                "recommended",
                "cost-effective",
            ]
        )
    )

    if not is_best_question:
        return None, []

    costs = calculate_uploaded_costs(csv_dataframes, answer_quantity)
    sorted_costs = sorted(costs, key=lambda item: item["total_cost"])
    best_option = sorted_costs[0]

    answer = (
        f"The better option for {answer_quantity} units is **{best_option['file']}** "
        f"with a total service cost of **{best_option['total_cost']}** "
        f"in the **{best_option['shipment_category']}** shipment category."
    )

    st.session_state["last_best_option"] = {
        **best_option,
        "quantity": answer_quantity,
    }

    return answer, sorted_costs


def build_structured_answer(
    question: str,
    csv_dataframes: list,
    quantity: int,
) -> tuple[str, list[dict]]:
    """Build structured answer from CSV data."""
    query_plan = build_query_plan(question)
    answer_quantity = _resolve_answer_quantity(question, csv_dataframes, quantity)
    question_lower = question.lower()
    table_rows = []

    if not is_service_cost_question(question):
        return (
            "I can help only with service-cost topics such as vendors, quotations, "
            "rates, packaging, sterilization, logistics, quality, warehousing, and benchmarks.",
            [],
        )

    if should_use_web_search(question):
        validation_error = validate_web_search_question(question)

        if validation_error:
            return validation_error, []

        return build_web_answer(question)

    if any(term in question_lower for term in AVAILABLE_VENDOR_TERMS):
        return _list_available_vendors(csv_dataframes)

    reason_answer = _explain_vendor_choice(question_lower)
    if reason_answer:
        return reason_answer

    report_answer, report_rows = _complete_comparative_report(
        question_lower, csv_dataframes
    )
    if report_answer:
        return report_answer, report_rows

    landed_rank_answer, landed_rank_rows = _rank_landed_scalability_compliance(
        question_lower, csv_dataframes, answer_quantity, query_plan
    )
    if landed_rank_answer:
        return landed_rank_answer, landed_rank_rows

    breakdown_answer, breakdown_rows = _build_vendor_breakdown(
        question, question_lower, csv_dataframes, answer_quantity
    )
    if breakdown_answer:
        return breakdown_answer, breakdown_rows

    semantic_answer, semantic_rows = _build_semantic_cost_answer(
        question, question_lower, csv_dataframes, answer_quantity
    )
    if semantic_answer:
        return semantic_answer, semantic_rows

    vendor_total_answer, vendor_total_rows = _get_vendor_total_cost(
        question, question_lower, csv_dataframes, answer_quantity
    )
    if vendor_total_answer:
        return vendor_total_answer, vendor_total_rows

    scale_answer, scale_rows = _compare_scale_change(
        question_lower, csv_dataframes
    )
    if scale_answer:
        return scale_answer, scale_rows

    component_answer, component_rows = _rank_by_component_cost(
        question_lower, csv_dataframes, answer_quantity
    )
    if component_answer:
        return component_answer, component_rows

    factor_rank_answer, factor_rank_rows = _rank_by_requested_factor(
        question, question_lower, csv_dataframes, answer_quantity
    )
    if factor_rank_answer:
        return factor_rank_answer, factor_rank_rows

    factor_answer, factor_rows = _compare_factors(
        question_lower, csv_dataframes, answer_quantity
    )
    if factor_answer:
        return factor_answer, factor_rows

    rate_answer, rate_rows = _get_specific_rate(
        question, question_lower, csv_dataframes, answer_quantity
    )
    if rate_answer:
        return rate_answer, rate_rows

    best_answer, best_rows = _find_best_option(
        question_lower, csv_dataframes, answer_quantity
    )
    if best_answer:
        return best_answer, best_rows

    answer = (
        "I can answer questions about available vendors, best option, total cost comparison, "
        "and individual rates like packaging, sterilization, logistics, quality, or warehousing."
    )

    return answer, []


def build_uploaded_file_rag_answer(
    question: str,
    csv_dataframes: list,
    rag_chunks: list,
    quantity: int,
) -> tuple[str, list[dict]]:
    """Build answer using RAG pipeline."""
    table_rows = []
    structured_answer = None

    if csv_dataframes and should_use_structured_answer(question):
        structured_answer, table_rows = build_structured_answer(
            question,
            csv_dataframes,
            quantity,
        )

        if structured_answer:
            return structured_answer, table_rows

    context_chunks = retrieve_uploaded_context(question, rag_chunks)
    answer = build_gemini_rag_answer(
        question,
        context_chunks,
        quantity,
        structured_answer=structured_answer,
    )

    return answer, table_rows


def render_web_search_results(rows: list[dict]) -> None:
    """Render web search results in a clean tabular UI."""
    if not rows:
        return

    st.markdown("### 📊 Live Vendor Search Results")
    
    table_data = []
    for row in rows:
        table_data.append({
            "Vendor / Source": row.get("vendor_or_source", "Unknown"),
            "Website": row.get("url", ""),
            "Phone Number": row.get("contact_numbers_found", "Not found"),
            "Reliability": row.get("reliability", "Medium"),
            "Details / Snippet": row.get("snippet", "")
        })
        
    df = pd.DataFrame(table_data)
    st.dataframe(
        df,
        column_config={
            "Website": st.column_config.LinkColumn("Website"),
            "Details / Snippet": st.column_config.TextColumn("Details / Snippet", width="large")
        },
        use_container_width=True,
        hide_index=True
    )


def render_cost_rows(rows: list[dict]) -> None:
    """Render standard comparison rows, semantic audit rows, or web search cards."""
    if not rows:
        return

    # Check if this is a web search result
    if "contact_numbers_found" in rows[0]:
        render_web_search_results(rows)
        return

    if rows[0].get("response_mode") != "detailed_audit":
        results_df = pd.DataFrame(rows)

        if "total_cost" in results_df.columns:
            results_df = results_df.sort_values("total_cost")

        st.dataframe(results_df)
        return

    results_df = pd.DataFrame(rows)
    visible_columns = [
        "category_min_qty",
        "category_max_qty",
        "quantity_basis",
        "component",
        "source_column",
        "availability",
        "per_unit_cost",
        "aggregate_rate",
    ]

    summary_columns = [
        "vendor",
        "file",
        "shipment_category",
        "quantity",
        "quantity_basis",
        "total_cost",
    ]
    summary_df = results_df[
        [column for column in summary_columns if column in results_df.columns]
    ].drop_duplicates()
    st.dataframe(summary_df)

    group_columns = ["category"]
    if "shipment_category" in results_df.columns and results_df["shipment_category"].nunique() > 1:
        group_columns.insert(0, "shipment_category")

    for group_key, category_df in results_df.groupby(group_columns, sort=False):
        category_name = group_key[-1] if isinstance(group_key, tuple) else group_key
        aggregate_rate = category_df["aggregate_rate"].dropna()
        aggregate_label = (
            f" - aggregate rate {aggregate_rate.iloc[0]}"
            if not aggregate_rate.empty
            else ""
        )
        shipment_label = ""
        if isinstance(group_key, tuple):
            shipment_label = f"{group_key[0]} / "

        with st.expander(f"{shipment_label}{category_name}{aggregate_label}", expanded=True):
            st.dataframe(
                category_df[
                    [column for column in visible_columns if column in category_df.columns]
                ]
            )


@st.cache_data(show_spinner=False)
def parse_uploaded_file(file_name: str, file_bytes: bytes) -> tuple[Optional[pd.DataFrame], list[str]]:
    """Parse one uploaded file (CSV, PDF, or Image) and extract costing data & RAG context."""
    file_name_lower = file_name.lower()

    if file_name_lower.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
        return df, dataframe_to_chunks(file_name, df)

    if file_name_lower.endswith(".pdf"):
        # 1. OCR extract rate card table if present
        df = extract_rate_card_from_media(file_bytes, file_name)
        # 2. Extract standard PDF text chunks for RAG
        chunks = pdf_to_chunks(file_name, file_bytes)
        return df, chunks

    if file_name_lower.endswith((".png", ".jpg", ".jpeg")):
        # 1. OCR extract rate card table from image
        df = extract_rate_card_from_media(file_bytes, file_name)
        # 2. Convert DataFrame rows into text chunks so they are searchable in RAG
        chunks = []
        if df is not None:
            chunks = dataframe_to_chunks(file_name, df)
        else:
            chunks = [f"Source image: {file_name}\nThis file represents an uploaded image without a structured rate card table."]
        return df, chunks

    return None, []


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="Service Costing Agent",
    layout="wide"
)

st.title("Service Costing Agent")

st.divider()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "tavily_search_count" not in st.session_state:
    st.session_state["tavily_search_count"] = 0

if "calculation_results" not in st.session_state:
    st.session_state["calculation_results"] = []

if "calculation_context" not in st.session_state:
    st.session_state["calculation_context"] = None

if "costing_engine_context_error" not in st.session_state:
    st.session_state["costing_engine_context_error"] = None

if "last_answer_quantity" not in st.session_state:
    st.session_state["last_answer_quantity"] = None

uploaded_files = st.file_uploader(
    "Upload Rate Cards (CSV, PDF, PNG, JPG, JPEG)",
    type=["csv", "pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

csv_dataframes = []
rag_chunks = []

if uploaded_files:
    st.success(
        f"Uploaded {len(uploaded_files)} file(s)"
    )

    for uploaded_file in uploaded_files:
        st.write(f"Uploaded: {uploaded_file.name}")
        parsed_df, parsed_chunks = parse_uploaded_file(
            uploaded_file.name,
            uploaded_file.getvalue(),
        )
        rag_chunks.extend(parsed_chunks)

        if parsed_df is not None:
            csv_dataframes.append((uploaded_file.name, parsed_df))

            label = "CSV Preview" if uploaded_file.name.lower().endswith(".csv") else "Extracted Rate Card"
            st.subheader(f"{label}: {uploaded_file.name}")
            st.dataframe(parsed_df)

quantity = st.number_input(
    "Enter Product Quantity",
    min_value=1,
    value=100
)

current_calculation_context = (
    tuple(uploaded_file.name for uploaded_file in uploaded_files or []),
    quantity,
)

if st.session_state["calculation_context"] != current_calculation_context:
    st.session_state["calculation_results"] = []
    st.session_state["calculation_context"] = current_calculation_context
    st.session_state["last_answer_quantity"] = quantity

if csv_dataframes:
    try:
        rag_chunks.extend(build_costing_engine_chunks(csv_dataframes, quantity))
        st.session_state["costing_engine_context_error"] = None
    except (KeyError, ValueError) as error:
        st.session_state["costing_engine_context_error"] = str(error)

if st.session_state["costing_engine_context_error"]:
    st.warning(
        "The uploaded files were parsed, but deterministic costing context could not be added "
        f"before embeddings: {st.session_state['costing_engine_context_error']}"
    )

if st.button("Calculate Service Cost"):
    if not csv_dataframes:
        st.session_state["calculation_results"] = []
        st.error(
            "Upload at least one CSV rate card before calculating. PDF uploads are shown, but PDF costing extraction is not implemented yet."
        )
    else:
        try:
            st.session_state["calculation_results"] = calculate_uploaded_costs(
                csv_dataframes,
                quantity,
            )
        except KeyError as error:
            st.session_state["calculation_results"] = []
            st.error(
                f"Missing required column {error}."
            )
        except ValueError as error:
            st.session_state["calculation_results"] = []
            st.error(
                str(error)
            )

for result in st.session_state["calculation_results"]:
    st.success(
        f"{result['file']}: Total service cost for {quantity} units is {result['total_cost']}"
    )

st.subheader("Agent Chat")

enable_web_search = st.checkbox(
    "🌐 Enable Web Search / Live Vendor Finding",
    key="enable_web_search",
    value=False,
    help="Bypass local calculations and query the web directly (DuckDuckGo + Tavily fallback) to find new suppliers, rates, contacts, or industry benchmarks."
)

for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        render_cost_rows(message.get("rows", []))

question = st.chat_input("Ask about service costs, vendors, rates, or nearby suppliers")

if question:
    answer_rows = []
    st.session_state["chat_history"].append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if not is_service_cost_question(question) and not enable_web_search:
            answer = (
                "I can help only with service-cost topics such as vendors, quotations, "
                "rates, packaging, sterilization, logistics, quality, warehousing, and benchmarks."
            )
            st.warning(answer)
        elif not rag_chunks and not should_use_web_search(question) and not enable_web_search:
            answer = "Upload at least one CSV or PDF before asking this service-cost question."
            st.error(answer)
        else:
            try:
                if should_use_web_search(question) or enable_web_search:
                    validation_error = validate_web_search_question(question)

                    if validation_error:
                        answer, costs = validation_error, []
                    else:
                        answer, costs = build_web_answer(question)
                else:
                    answer, costs = build_uploaded_file_rag_answer(
                        question,
                        csv_dataframes,
                        rag_chunks,
                        quantity,
                    )
                    if costs and isinstance(costs[0], dict) and "quantity" in costs[0]:
                        st.session_state["last_answer_quantity"] = int(costs[0]["quantity"])
                st.write(answer)
                answer_rows = costs
                render_cost_rows(costs)
            except KeyError as error:
                answer = f"Missing required column {error}."
                st.error(answer)
            except ValueError as error:
                answer = str(error)
                st.error(answer)
            except requests.RequestException as error:
                answer = f"Tavily search failed: {error}"
                st.error(answer)

    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": answer,
            "rows": answer_rows,
        }
    )
