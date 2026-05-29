"""
Lightweight Orchestrator & Planner.

Main orchestration loop upgraded to a mature Hybrid Workflow Intelligence Architecture.
Features:
1. Structured execution plan contracts.
2. Parallel-safe dependency execution using thread pooling.
3. Isolated try-catch blocks with graceful failure recovery.
4. Automatic vendor profile intelligence normalization.
5. Dual-representation machine/synthesized response aggregation.
"""

import os
import re
import concurrent.futures
from typing import List, Dict, Any, Tuple, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import LLMChain

from engine.shared_state import SharedState
from engine.workflow_registry import WORKFLOW_REGISTRY
from engine.vendor_normalization import parse_vendor_profile
from engine.compliance import STANDARD_CERTIFICATIONS, CERT_WEIGHTS


CLASSIFIER_PROMPT = PromptTemplate.from_template("""
You are the planner of a procurement costing assistant. Your job is to classify the user's question and select which workflow components are required.

User Question: {query}

Supported Workflow Components:
1. RAG_RETRIEVAL: Required if user asks about custom specifications, uploaded documents/PDFs details, or internal files context.
2. COSTING: Required if user asks about calculating costs, rate cards, pricing lookup, comparison, or total service pricing.
3. WEB_SEARCH: Required if user asks for nearby suppliers, external market benchmarks, location-based vendor searches, or trusted vendor directories.
4. COMPLIANCE: Required if user asks about standards, certificates, FDA approval, CE marks, or ISO audits of vendors.

Respond with a comma-separated list of components required. Do not output anything else.
For example: RAG_RETRIEVAL, COSTING, COMPLIANCE
Or if just searching suppliers: WEB_SEARCH, COMPLIANCE
""")


SYNTHESIS_PROMPT = PromptTemplate.from_template("""
You are a senior service costing and procurement architect assistant. 
Your goal is to synthesize a professional, comprehensive, and clear procurement response based on the structured results computed by our pipeline.

Below is the shared runtime state containing all calculated costing numbers, rankings, search hits, and compliance validations.

=======================================================
SHARED PIPELINE STATE
=======================================================
Original Query: {query}
Quantity Requested: {quantity}

Calculated Costing Details:
{costing_results}

Vendor Scoring & Rankings (Lowest to Highest Cost):
{vendor_scores}

Compliance & Certification Check Results:
{compliance_results}

External Web Search Snippets (Market discovery/benchmarks):
{web_search_results}

Previous Context/Continuity:
{memory_context}
=======================================================

Instructions:
1. Ground your response STRICTLY on the structured data above.
2. Do NOT perform any additional calculations or invent pricing/scores. Use the deterministic costing_results and vendor_scores exactly as provided.
3. Present the calculated total service costs and individual factor rates (packaging, sterilization, logistics, quality, warehousing) clearly.
4. If semantic_breakdown_rows are present, explain the hierarchy between business terms and aggregate buckets. For example, handling belongs to logistics, audit belongs to quality, and insurance belongs to warehousing.
5. Synthesize compliance and quality audits professionally: mention who is certified (e.g., ISO, FDA) and who is not based on the compliance checks.
6. If web search results are present, summarize them as live external market benchmarks and supply contacts.
7. Provide a clear, actionable recommendation explaining the best option based on both costing efficiency and quality compliance.

Write a structured, elegant markdown response with bullet points and comparison tables where useful. Keep the tone executive and objective.
""")


def get_llm():
    """Retrieve Groq Llama 3.2 Vision LLM instance."""
    return ChatGroq(
        model="llama-3.2-90b-vision-preview",
        temperature=0,
    )


def classify_query(query: str) -> List[str]:
    """Classify query using Gemini to find matching workflow components."""
    try:
        llm = get_llm()
        chain = LLMChain(llm=llm, prompt=CLASSIFIER_PROMPT, output_parser=StrOutputParser())
        result = chain.invoke({"query": query})
        output = result.get("text", result) if isinstance(result, dict) else result
        
        components = [c.strip() for c in output.split(",") if c.strip()]
        return components
    except Exception:
        # Fallback query parsing if LLM classification fails
        components = ["RAG_RETRIEVAL", "COSTING"]
        query_lower = query.lower()
        if "near" in query_lower or "find" in query_lower or "web" in query_lower:
            components.append("WEB_SEARCH")
        if "iso" in query_lower or "fda" in query_lower or "compliance" in query_lower or "cert" in query_lower:
            components.append("COMPLIANCE")
        return components


def generate_workflow_contract(query: str, components: List[str]) -> Dict[str, Any]:
    """
    Build a formal execution contract outlining scheduled stages, needs, and execution order.
    """
    needs_rag = "RAG_RETRIEVAL" in components
    needs_web_search = "WEB_SEARCH" in components
    needs_costing = "COSTING" in components or needs_rag
    needs_compliance = "COMPLIANCE" in components
    
    # Establish staging sequence
    # Memory and retrieval/search occur in Stage 1
    # Costing, Scoring, and Auditing are safe to execute concurrently in Stage 2
    execution_order = ["MEM_LOAD"]
    if needs_rag:
        execution_order.append("RAG_RETRIEVAL")
    if needs_web_search:
        execution_order.append("WEB_SEARCH")
    if needs_costing:
        execution_order.append("COSTING")
        execution_order.append("VENDOR_LOGIC")
    if needs_compliance:
        execution_order.append("COMPLIANCE")
        
    # Standardize planning contracts
    return {
        "query_type": "procurement_comparison" if needs_costing and needs_compliance else "vendor_discovery" if needs_web_search else "document_query",
        "needs_rag": needs_rag,
        "needs_web_search": needs_web_search,
        "needs_costing": needs_costing,
        "needs_compliance": needs_compliance,
        "execution_order": execution_order
    }


def execute_step(
    step_name: str,
    state: SharedState,
    csv_dataframes: List[Tuple[str, Any]],
    rag_chunks: List[str],
    chat_history: Optional[List[Dict[str, str]]] = None,
    last_best_option: Optional[Dict[str, Any]] = None
):
    """
    Runs a registered pipeline module with strict error isolation and failure recovery.
    """
    state.set_module_status(step_name, "running")
    state.add_trace(f"Starting Step: {step_name}")
    try:
        func = WORKFLOW_REGISTRY.get_function(step_name)
        if func:
            if step_name == "MEM_LOAD":
                func(state, chat_history, last_best_option)
            elif step_name == "RAG_RETRIEVAL":
                func(state, rag_chunks)
            elif step_name in ("COSTING", "VENDOR_LOGIC"):
                func(state, csv_dataframes)
            else:
                func(state)
            state.set_module_status(step_name, "completed")
            state.add_trace(f"Step Completed: {step_name}")
        else:
            state.set_module_status(step_name, "skipped")
            state.add_trace(f"Step Skipped: {step_name} (Missing handler)")
    except Exception as e:
        state.set_module_status(step_name, "failed")
        state.add_error(step_name, str(e), fatal=False)
        state.add_trace(f"Step Failed: {step_name}", {"error": str(e)})


def execute_orchestrator_workflow(
    query: str,
    quantity: int,
    csv_dataframes: List[Tuple[str, Any]],
    rag_chunks: List[str],
    chat_history: Optional[List[Dict[str, str]]] = None,
    last_best_option: Optional[Dict[str, Any]] = None
) -> Tuple[str, SharedState]:
    """
    Coordinates execution of the procurement analysis pipeline.
    Utilizes dependency-aware parallel thread pools, vendor profile normalization, 
    and structured aggregation.
    """
    # 1. State Setup
    state = SharedState(query=query, quantity=quantity)
    state.add_trace("Orchestrator Executing Staged Workflow")
    
    # 2. Plan and Generate Contract
    components = classify_query(query)
    contract = generate_workflow_contract(query, components)
    plan = contract["execution_order"]
    
    state.workflow_plan = plan
    state.add_trace("Workflow Staged and Contract Generated", {"contract": contract})

    # 3. STAGE 1: Setup & Inbound Context (Sequential / Core Inputs)
    stage_1_steps = [s for s in plan if s in ("MEM_LOAD", "RAG_RETRIEVAL", "WEB_SEARCH")]
    for step in stage_1_steps:
        execute_step(
            step, state, csv_dataframes, rag_chunks, chat_history, last_best_option
        )

    # 4. STAGE 2: Parallel Audits & Costing Engine
    # Run Costing and Compliance concurrently to minimize CPU/IO wait times
    stage_2_parallel = [s for s in plan if s in ("COSTING", "COMPLIANCE")]
    
    if stage_2_parallel:
        state.add_trace("Launching Stage 2 Parallel Safe Operations", {"threads": stage_2_parallel})
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    execute_step, step, state, csv_dataframes, rag_chunks, chat_history, last_best_option
                ): step for step in stage_2_parallel
            }
            concurrent.futures.wait(futures)
            
    # Dependent scoring runs sequentially once Stage 2 costing completes
    if "VENDOR_LOGIC" in plan:
        execute_step(
            "VENDOR_LOGIC", state, csv_dataframes, rag_chunks, chat_history, last_best_option
        )

    # 5. STAGE 3: Intelligence Normalization & Entity Aggregation
    state.set_module_status("NORMALIZATION", "running")
    state.add_trace("Starting Step: NORMALIZATION")
    try:
        normalized_profiles = []
        
        # Normalize local CSV vendor cost models
        for row in state.vendor_scores:
            vendor_name = row.get("vendor", "")
            comp_data = state.compliance_results.get(vendor_name, {})
            certs = [cert for cert, val in comp_data.get("certifications", {}).items() if val.get("status") == "Verified"]
            comp_score = comp_data.get("trust_score", 0.0)
            
            profile = parse_vendor_profile(
                name=vendor_name,
                source="local_csv",
                total_cost=row.get("total_cost"),
                certifications=certs,
                compliance_score=comp_score,
                additional_text=state.query
            )
            normalized_profiles.append(profile)
            
        # Normalize web discovered supplier leads
        for search_hit in state.web_search_results:
            vendor_name = search_hit.get("title", "Web Discovered Vendor")
            snippet = search_hit.get("snippet", "")
            
            # Map regex matches to cert list
            certs = []
            for cert_name, pattern in STANDARD_CERTIFICATIONS.items():
                if re.search(pattern, snippet, re.IGNORECASE):
                    certs.append(cert_name)
            comp_score = sum(CERT_WEIGHTS.get(c, 0.0) for c in certs)
            
            profile = parse_vendor_profile(
                name=vendor_name,
                source="web_search",
                certifications=certs,
                compliance_score=comp_score,
                contact_numbers=search_hit.get("contact_numbers_found", "Not found"),
                additional_text=snippet,
                website=search_hit.get("url", "N/A")
            )
            normalized_profiles.append(profile)
            
        state.vendor_profiles = normalized_profiles
        state.set_module_status("NORMALIZATION", "completed")
        state.add_trace("Step Completed: NORMALIZATION", {"profile_count": len(normalized_profiles)})
    except Exception as e:
        state.set_module_status("NORMALIZATION", "failed")
        state.add_error("NORMALIZATION", str(e), fatal=False)
        state.add_trace("Step Failed: NORMALIZATION", {"error": str(e)})

    # Build Structured Output Schema
    state.structured_response = {
        "recommended_vendor": state.vendor_scores[0] if state.vendor_scores else {},
        "cost_breakdown": state.costing_results,
        "compliance_summary": state.compliance_results,
        "vendor_profiles": state.vendor_profiles,
        "metadata": state.execution_metadata,
        "errors": state.errors
    }

    # 6. STAGE 4: Synthesis Response Generation
    state.set_module_status("SYNTHESIS", "running")
    state.add_trace("Starting Step: SYNTHESIS")
    try:
        llm = get_llm()
        chain = LLMChain(llm=llm, prompt=SYNTHESIS_PROMPT, output_parser=StrOutputParser())
        
        input_data = {
            "query": state.query,
            "quantity": state.quantity,
            "costing_results": str(state.costing_results),
            "vendor_scores": str(state.vendor_scores),
            "compliance_results": str(state.compliance_results),
            "web_search_results": str(state.web_search_results),
            "memory_context": str(state.memory_context),
        }
        
        synthesis_result = chain.invoke(input_data)
        final_answer = synthesis_result.get("text", synthesis_result) if isinstance(synthesis_result, dict) else synthesis_result
        
        state.set_module_status("SYNTHESIS", "completed")
        state.finalize_metadata()
        state.add_trace("Synthesis Response Generated")
        return final_answer, state
    except Exception as e:
        state.set_module_status("SYNTHESIS", "failed")
        state.add_error("SYNTHESIS", str(e), fatal=True)
        state.finalize_metadata()
        
        error_msg = f"Synthesis stage failed: {str(e)}"
        state.add_trace("Synthesis Failed", {"error": str(e)})
        fallback_ans = f"I performed the calculations but could not generate the final explanation summary. Details:\n\n{state.vendor_scores}\n\nError: {error_msg}"
        return fallback_ans, state
