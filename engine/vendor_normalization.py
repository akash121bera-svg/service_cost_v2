"""
Vendor Intelligence & Entity Normalization Schema.

Defines a standardized structure to represent vendors discovered either locally
from rate card CSVs or externally from web search directories, providing parsing
helpers to align various inputs into a normalized data contract.
"""

import re
from typing import Dict, Any, List, Optional


def calculate_pricing_score(total_cost: float, max_cost: float = 100000.0) -> float:
    """
    Generate a normalized pricing score from 0 (very expensive) to 100 (most cost-effective).
    """
    if total_cost <= 0:
        return 100.0
    
    # Simple relative scaling
    ratio = total_cost / max_cost
    score = 100.0 * (1.0 - ratio)
    return max(0.0, min(100.0, round(score, 1)))


def extract_lead_time_from_text(text: str) -> str:
    """
    Search for standard lead time patterns in text snippets.
    """
    patterns = [
        r"\b\d+\s*-\s*\d+\s*(?:day|days|week|weeks)\b",
        r"\bship\s*in\s*\d+\s*(?:day|days|week|weeks)\b",
        r"\blead\s*time\s*:\s*\d+\s*(?:day|days)\b",
        r"\b\d+\s*(?:day|days|week|weeks)\b"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return "Standard (Contact Vendor)"


def parse_vendor_profile(
    name: str,
    source: str,
    total_cost: Optional[float] = None,
    certifications: Optional[List[str]] = None,
    compliance_score: float = 0.0,
    contact_numbers: str = "Not found",
    additional_text: str = "",
    website: str = "N/A"
) -> Dict[str, Any]:
    """
    Normalize raw vendor fields into the unified enterprise vendor profile schema.
    """
    certs = certifications or []
    
    # Assess risks
    risk_flags = []
    if compliance_score < 0.5:
        risk_flags.append("Low regulatory compliance (unverified certs)")
    if source == "web_search" and contact_numbers == "Not found":
        risk_flags.append("Missing primary contact number")
    
    # Calculate pricing score if local costing was run
    pricing_score = 0.0
    if total_cost is not None:
        pricing_score = calculate_pricing_score(total_cost)
        
    lead_time = extract_lead_time_from_text(additional_text)

    return {
        "vendor_name": name,
        "website": website,
        "source": source,  # 'local_csv' or 'web_search'
        "certifications": certs,
        "lead_time": lead_time,
        "pricing_score": pricing_score,
        "compliance_score": round(compliance_score, 2),
        "risk_flags": risk_flags,
        "contact_numbers": contact_numbers,
    }
