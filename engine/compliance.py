"""
Compliance & Quality Auditing Engine.

Scans retrieved documents and web results for quality and regulatory certifications 
(e.g., ISO 9001, ISO 13485, FDA, CE Mark, GMP) to produce a structured compliance log
with numeric trust scoring and validation traces.
"""

import re
from typing import Dict, List, Any
from engine.shared_state import SharedState
from engine.caching import GLOBAL_CACHE

# Standardized regex rules for regulatory verification
STANDARD_CERTIFICATIONS = {
    "ISO 13485": r"\bISO\s*13485\b",
    "ISO 9001": r"\bISO\s*9001\b",
    "FDA Approved/Registered": r"\bFDA\b",
    "CE Mark": r"\bCE\s*(?:Mark|Marked|Certification)\b",
    "GMP": r"\bGMP\b|\bGood\s*Manufacturing\s*Practice\b",
}

# Trust scoring weights (normalized to 1.0 maximum)
CERT_WEIGHTS = {
    "ISO 13485": 0.35,
    "FDA Approved/Registered": 0.30,
    "CE Mark": 0.20,
    "GMP": 0.10,
    "ISO 9001": 0.05,
}


def calculate_trust_score(cert_matches: Dict[str, Dict[str, Any]]) -> float:
    """
    Compute a numerical trust score between 0.0 and 1.0 based on weights.
    """
    score = 0.0
    for cert_name, match_info in cert_matches.items():
        if match_info["status"] == "Verified":
            score += CERT_WEIGHTS.get(cert_name, 0.0)
    return round(score, 2)


def run_compliance_checks(state: SharedState):
    """
    Scans retrieved content and search snippets for standard certifications.
    Stores structured validation results for each vendor.
    Utilizes enterprise caching to optimize performance.
    """
    state.set_module_status("compliance", "running")
    
    # Read both local retrieved docs and external web search snippet content
    all_context_texts: List[str] = list(state.retrieved_docs)
    for search_hit in state.web_search_results:
        all_context_texts.append(search_hit.get("snippet", "") + " " + search_hit.get("title", ""))

    context_fingerprint = " ".join(all_context_texts)
    
    # Try retrieving from cache first to avoid repeating regex passes
    cached_report = GLOBAL_CACHE.get("compliance", context_fingerprint)
    if cached_report:
        state.compliance_results = cached_report
        state.set_module_status("compliance", "completed")
        state.add_trace("Compliance Scanning Completed (Cache Hit)", {
            "results_count": len(cached_report)
        })
        return

    compliance_report: Dict[str, Any] = {}
    
    # Gather vendor list from vendor profiles, costing results, or web results
    vendors = set()
    for profile in state.vendor_profiles:
        vendor_name = profile.get("vendor", profile.get("file", ""))
        if vendor_name:
            vendors.add(vendor_name.replace(".csv", "").replace("_rates", "").replace("_quotation", ""))
            
    for score in state.vendor_scores:
        if "vendor" in score:
            vendors.add(score["vendor"])
            
    # Default fallback vendors from query/state if none resolved yet
    if not vendors:
        vendors.add("General Vendor Profile")

    # Inspect certifications for each vendor
    for vendor in vendors:
        vendor_lower = vendor.lower()
        vendor_contexts = [text for text in all_context_texts if vendor_lower in text.lower()]
        
        # If no specific context exists, check all context
        if not vendor_contexts:
            vendor_contexts = all_context_texts
            
        cert_matches = {}
        combined_text = " ".join(vendor_contexts)
        
        for cert_name, pattern in STANDARD_CERTIFICATIONS.items():
            match = re.search(pattern, combined_text, re.IGNORECASE)
            cert_matches[cert_name] = {
                "status": "Verified" if match else "Not Found",
                "evidence": match.group(0) if match else None
            }
            
        trust_score = calculate_trust_score(cert_matches)
        
        # Resolve overall audit status label
        if trust_score >= 0.7:
            overall_status = "Highly Compliant"
        elif trust_score >= 0.3:
            overall_status = "Moderate Compliance"
        else:
            overall_status = "Unverified / High Risk"
            
        compliance_report[vendor] = {
            "certifications": cert_matches,
            "iso_13485": cert_matches["ISO 13485"]["status"] == "Verified",
            "fda": cert_matches["FDA Approved/Registered"]["status"] == "Verified",
            "ce_certified": cert_matches["CE Mark"]["status"] == "Verified",
            "gmp": cert_matches["GMP"]["status"] == "Verified",
            "iso_9001": cert_matches["ISO 9001"]["status"] == "Verified",
            "trust_score": trust_score,
            "overall_status": overall_status,
        }

    # Store in cache
    GLOBAL_CACHE.set("compliance", compliance_report, context_fingerprint)
    
    state.compliance_results = compliance_report
    state.set_module_status("compliance", "completed")
    state.add_trace("Compliance Scanning Completed", {
        "scanned_vendors": list(vendors),
        "results": compliance_report
    })

