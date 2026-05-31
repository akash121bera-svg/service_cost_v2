"""
Query planning for procurement costing questions.

The planner converts natural-language questions into a small execution plan.
LLM planning is optional; deterministic fallback keeps calculations reliable
and available in local/offline environments.
"""

import json
import os
import re


CRITERIA_ALIASES = {
    "total_landed_cost": [
        "total landed cost",
        "landed cost",
        "total procurement cost",
        "total service cost",
        "total cost",
    ],
    "shipment_scalability": [
        "shipment scalability",
        "scalability",
        "scale",
        "capacity",
        "high-volume",
        "high volume",
    ],
    "compliance_overhead": [
        "compliance overhead",
        "compliance cost",
        "compliance score",
        "compliance reliability",
        "strict compliance",
        "inspection",
        "audit",
        "documentation",
        "quality assurance",
        "regulatory transparency",
        "regulatory concerns",
    ],
    "packaging": ["packaging", "packaging cost"],
    "sterilization": ["sterilization", "sterilization cost"],
    "logistics": ["logistics", "logistics cost"],
    "transport": ["transport", "transportation"],
    "distributor_fee": ["distribution", "distributor", "distributor fee"],
    "quality": ["quality", "quality cost"],
    "warehousing": ["warehousing", "warehouse", "warehousing cost"],
    "lead_time": [
        "lead time",
        "fastest delivery",
        "urgent",
        "within 7 days",
        "emergency",
    ],
    "market_benchmark": [
        "market benchmark",
        "market benchmarks",
        "publicly available",
        "external market intelligence",
    ],
    "document_retrieval": [
        "uploaded",
        "retrieved vendor documents",
        "vendor quotation pdf",
        "tender requirements",
        "uploaded pdf",
        "uploaded scanned quotation image",
    ],
}


DETAIL_TERMS = [
    "breakdown",
    "detailed",
    "detail",
    "decompose",
    "decomposition",
    "component",
    "components",
]

RANK_TERMS = [
    "rank",
    "ranked",
    "ranking",
    "compare",
    "comparison",
    "lowest",
    "cheapest",
    "best",
    "better",
    "recommend",
    "recommended",
    "cost-efficient",
    "efficient",
    "efficiency",
    "identify",
]

WEB_TERMS = [
    "near",
    "nearby",
    "around",
    "find vendors",
    "find vendor",
    "search web",
    "contact",
    "phone",
    "trusted",
    "reliable",
    "external",
    "public",
    "publicly",
    "public web",
    "public web sources",
    "duckduckgo",
    "market benchmark",
    "market benchmarks",
    "market intelligence",
    "official website",
    "support email",
    "customer reviews",
    "reputation",
    "regulatory history",
    "regulatory concerns",
    "certifications",
    "verify whether",
    "available publicly",
]


PLANNER_PROMPT = """You are the planning layer for a procurement costing app.
Return only valid JSON. Do not calculate costs.

Allowed intents:
- vendor_list
- vendor_total
- best_vendor
- factor_ranking
- component_ranking
- detailed_breakdown
- multi_criteria_vendor_ranking
- scale_analysis
- comparative_report
- web_discovery
- fallback

Allowed criteria:
- total_landed_cost
- shipment_scalability
- compliance_overhead
- packaging
- sterilization
- logistics
- quality
- warehousing
- handling
- audit
- insurance
- warehouse_rent
- transport
- distributor_fee
- inspection
- documentation

Question: {question}

JSON shape:
{{
  "intent": "one_allowed_intent",
  "criteria": ["criterion"],
  "response_mode": "summary|detailed_audit",
  "needs_web": false,
  "confidence": 0.0
}}
"""


def _contains_any(question_lower, terms):
    return any(term in question_lower for term in terms)


def _extract_criteria(question_lower):
    criteria = []

    for criterion, aliases in CRITERIA_ALIASES.items():
        if any(alias in question_lower for alias in aliases):
            criteria.append(criterion)

    return criteria


def _fallback_plan(question):
    question_lower = question.lower()
    criteria = _extract_criteria(question_lower)
    asks_ranking = _contains_any(question_lower, RANK_TERMS)
    asks_detail = _contains_any(question_lower, DETAIL_TERMS)
    needs_web = _contains_any(question_lower, WEB_TERMS)

    if "lead_time" in criteria and "logistics" not in criteria:
        needs_web = True

    if needs_web:
        intent = "web_discovery"
    elif "uploaded" in question_lower or "retrieved vendor documents" in question_lower:
        intent = "comparative_report" if asks_ranking else "fallback"
    elif "shipment bands" in question_lower or "cost changes across" in question_lower:
        intent = "scale_analysis"
    elif "small" in question_lower and "medium" in question_lower and "large" in question_lower:
        intent = "comparative_report"
    elif (
        asks_ranking
        and "shipment_scalability" in criteria
        and len(set(criteria) & {"total_landed_cost", "compliance_overhead"}) >= 1
    ):
        intent = "multi_criteria_vendor_ranking"
    elif asks_detail:
        intent = "detailed_breakdown"
    elif "available vendor" in question_lower or "list vendor" in question_lower:
        intent = "vendor_list"
    elif _contains_any(question_lower, ["economies of scale", "trend", "as quantity increases", "moving from"]):
        intent = "scale_analysis"
    elif asks_ranking and criteria:
        intent = "factor_ranking"
    elif asks_ranking:
        intent = "best_vendor"
    elif "total" in question_lower and "service cost" in question_lower:
        intent = "vendor_total"
    else:
        intent = "fallback"

    response_mode = "detailed_audit" if asks_detail else "summary"

    return {
        "intent": intent,
        "criteria": criteria,
        "response_mode": response_mode,
        "needs_web": needs_web,
        "confidence": 0.65,
        "planner": "deterministic_fallback",
    }


def _parse_llm_json(raw_text):
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)

    if not match:
        return None

    return json.loads(match.group(0))


def _llm_plan(question):
    if not os.getenv("GROQ_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        return None

    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate
        from langchain_classic.chains import LLMChain
        from langchain_groq import ChatGroq
        from langchain_google_genai import ChatGoogleGenerativeAI

        google_key = os.environ.get("GOOGLE_API_KEY")
        gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        
        groq_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
        )
        
        if google_key and "your_" not in google_key:
            try:
                gemini_llm = ChatGoogleGenerativeAI(
                    model=gemini_model,
                    temperature=0,
                    google_api_key=google_key,
                    max_retries=1,
                )
                llm = gemini_llm.with_fallbacks([groq_llm])
            except Exception:
                llm = groq_llm
        else:
            llm = groq_llm

        chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(PLANNER_PROMPT),
            output_parser=StrOutputParser(),
        )
        result = chain.invoke({"question": question})
        raw_text = result.get("text", result) if isinstance(result, dict) else result
        plan = _parse_llm_json(str(raw_text))

        if not plan:
            return None

        fallback = _fallback_plan(question)
        plan.setdefault("criteria", fallback["criteria"])
        plan.setdefault("response_mode", fallback["response_mode"])
        plan.setdefault("needs_web", fallback["needs_web"])
        plan.setdefault("confidence", 0.8)
        plan["planner"] = "llm"
        return plan
    except Exception:
        return None


def build_query_plan(question):
    """Return an LLM-backed plan when possible, otherwise deterministic fallback."""
    llm_plan = _llm_plan(question)

    if llm_plan:
        return llm_plan

    return _fallback_plan(question)
