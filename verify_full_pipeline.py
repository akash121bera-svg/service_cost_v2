import sys
import os
import json
import logging
import pandas as pd
from dotenv import load_dotenv

# Set logging level
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# Ensure workspace root is in python path and load .env
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_ROOT)
load_dotenv(dotenv_path=os.path.join(WORKSPACE_ROOT, ".env"))

from engine.shared_state import SharedState
from engine.orchestrator import execute_orchestrator_workflow

def generate_mock_data():
    """Generate structured mock rate cards and unstructured document RAG chunks."""
    # 1. Biosafe Packaging CSV Rate Card
    df_biosafe = pd.DataFrame({
        "min_qty": [1, 201, 501],
        "max_qty": [200, 500, 10000],
        "shipment_category": ["Small", "Medium", "Large"],
        "packaging_rate": [2.5, 2.0, 1.5],
        "sterilization_rate": [1.2, 1.0, 0.8],
        "logistics_rate": [3.0, 2.5, 2.0],
        "quality_rate": [0.5, 0.4, 0.3],
        "warehousing_rate": [0.8, 0.6, 0.4],
        "vendor_id": ["V01", "V01", "V01"]
    })
    
    # 2. Carechain Services CSV Rate Card
    df_carechain = pd.DataFrame({
        "min_qty": [1, 201, 501],
        "max_qty": [200, 500, 10000],
        "shipment_category": ["Small", "Medium", "Large"],
        "packaging_rate": [2.8, 1.8, 1.3],
        "sterilization_rate": [1.0, 0.9, 0.7],
        "logistics_rate": [3.5, 2.2, 1.8],
        "quality_rate": [0.6, 0.3, 0.25],
        "warehousing_rate": [1.0, 0.5, 0.35],
        "vendor_id": ["V02", "V02", "V02"]
    })
    
    csv_dataframes = [
        ("Biosafe_Packaging_rates.csv", df_biosafe),
        ("Carechain_Services_rates.csv", df_carechain)
    ]
    
    # 3. PDF/Document text chunks for semantic FAISS RAG
    rag_chunks = [
        "Source file: Biosafe_Agreement.pdf\nBiosafe Packaging is fully ISO 13485 certified for medical device packaging. Their sterile cleanroom has been FDA registered since 2018.",
        "Source file: Carechain_Audit.pdf\nCarechain Services operates under CE mark compliance and GMP standards for all logistical warehousing.",
        "Source file: Terms.pdf\nAll shipments are subject to a minimum order quantity (MOQ) of 50 units. Product handling instructions limit high volume storage capacity to 50000 kits.",
    ]
    
    return csv_dataframes, rag_chunks

def execute_test_case(title, query, quantity, csv_dfs, rag_chunks, chat_history=None, last_best=None):
    """Run orchestrator end-to-end and display traces, timing metadata, and synthesized answers."""
    print("=" * 80)
    print(f"TEST CASE: {title}")
    print(f"Query: \"{query}\"")
    print(f"Quantity: {quantity}")
    print("=" * 80)
    
    # Run the full API-driven orchestrated workflow
    answer, state = execute_orchestrator_workflow(
        query=query,
        quantity=quantity,
        csv_dataframes=csv_dfs,
        rag_chunks=rag_chunks,
        chat_history=chat_history,
        last_best_option=last_best
    )
    
    print("\n--- Pipeline Observability Traces ---")
    for trace in state.execution_trace:
        print(f"[{trace['elapsed_ms']}ms] {trace['action']}: {trace['details']}")
        
    print("\n--- Module Observability Status ---")
    print(json.dumps(state.module_status, indent=2))
    
    if state.errors:
        print("\n--- Pipeline Errors Encountered ---")
        print(json.dumps(state.errors, indent=2))
        
    print("\n--- Execution Timing ---")
    print(json.dumps(state.execution_metadata, indent=2))
    
    print("\n--- Synthesized Executive Answer ---")
    print(answer)
    print("\n" + "#" * 80 + "\n")
    
    return state

def run_e2e_verification():
    """Trigger E2E workflow checks for all 3 critical entry pathways."""
    print("Starting Unified Multi-Modal Pipeline End-to-End Verification...\n")
    
    csv_dfs, rag_chunks = generate_mock_data()
    
    # CASE 1: Unified Costing + Compliance Trust Audit + Semantic FAISS RAG
    # This invokes CLASSIFIER, MEM_LOAD, RAG_RETRIEVAL, COSTING, COMPLIANCE, VENDOR_LOGIC, NORMALIZATION, and SYNTHESIS.
    execute_test_case(
        title="1. Hybrid Deterministic Costing + RAG Retrieval + Compliance Trust Auditing",
        query="Compare costing details and check ISO 13485 certifications for Biosafe and Carechain at 300 units.",
        quantity=300,
        csv_dfs=csv_dfs,
        rag_chunks=rag_chunks
    )
    
    # CASE 2: Prioritized B2B Web Search + Parallel Web Page Scraper + Phone Extraction
    # This invokes WEB_SEARCH, NORMALIZATION, and SYNTHESIS.
    # Note: Requires TAVILY_API_KEY inside .env for fallback search.
    execute_test_case(
        title="2. Prioritized B2B Discovery Search + Parallel Web Scraper & Dynamic Phone Extraction",
        query="Find medical packaging vendors near Mumbai and list their contact details.",
        quantity=100,
        csv_dfs=csv_dfs,
        rag_chunks=rag_chunks
    )
    
    # CASE 3: Lightweight Contextual Memory Follow-up switches
    # This validates memory rephrasing context.
    chat_history = [
        {"role": "user", "content": "What is the cost for 200 units from Biosafe Packaging?"},
        {"role": "assistant", "content": "Biosafe Packaging total cost for 200 units (Small Category) is $1700.00 ($8.50 per unit)."}
    ]
    last_best = {
        "vendor": "Biosafe Packaging",
        "file": "Biosafe_Packaging_rates.csv",
        "shipment_category": "Small",
        "quantity": 200,
        "total_cost": 1700.00
    }
    execute_test_case(
        title="3. Lightweight Conversational Memory Follow-up (Context Rephrasing)",
        query="What if the quantity changes to 500 units?",
        quantity=100, # Handled dynamically inside logic rephrase
        csv_dfs=csv_dfs,
        rag_chunks=rag_chunks,
        chat_history=chat_history,
        last_best=last_best
    )

if __name__ == "__main__":
    run_e2e_verification()
