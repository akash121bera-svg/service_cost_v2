"""
Helpers for answering chat questions from uploaded CSV rate cards.

This module keeps deterministic CSV logic outside Streamlit so local answers
such as vendor lists and best-option comparisons are easy to read and test.

Usage:
    from engine.uploaded_costs import calculate_uploaded_costs, build_comparison_factor_rows
    results = calculate_uploaded_costs(csv_dataframes, quantity)
"""

import re

from engine.category_selector import get_shipment_category
from engine.costing_engine import calculate_total_cost
from factors.logistics.logistics import calculate_logistics_cost
from factors.packaging.packaging import calculate_packaging_cost
from factors.quality.quality import calculate_quality_cost
from factors.sterilization.sterilization import calculate_sterilization_cost
from factors.warehousing.warehousing import calculate_warehousing_cost


# Service rate names matching CSV columns
SERVICE_RATE_NAMES = [
    "packaging",
    "sterilization",
    "logistics",
    "quality",
    "warehousing",
]


def get_answer_quantity(question, fallback_quantity):
    """
    Extract quantity from question text.
    
    Searches for a number in the question. Falls back to default if not found.
    
    Args:
        question: User question text
        fallback_quantity: Default quantity if none found in question
    
    Returns:
        int: Quantity to use for calculations
    """
    if re.search(r"\b(?:small|medium|large)\b", question, re.IGNORECASE):
        return fallback_quantity

    range_match = re.search(r"\b\d+\s*(?:-|–|to)\s*\d+\b", question)

    if range_match:
        return fallback_quantity

    unit_match = re.search(
        r"\b(\d+)\s*(?:-|–)?\s*(?:unit|units|kit|kits)\b",
        question,
        re.IGNORECASE,
    )

    if not unit_match:
        unit_match = re.search(
            r"\b(\d+)\s+(?:[a-z]+\s+){0,4}(?:unit|units|kit|kits)\b",
            question,
            re.IGNORECASE,
        )

    if unit_match:
        return int(unit_match.group(1))

    cleaned_question = re.sub(r"\biso\s+\d+\b", "", question, flags=re.IGNORECASE)
    cleaned_question = re.sub(r"\btop\s+\d+\b", "", cleaned_question, flags=re.IGNORECASE)

    match = re.search(r"\b\d+\b", cleaned_question)

    if match:
        return int(match.group())

    return fallback_quantity


def get_category_quantity(question, csv_dataframes, fallback_quantity):
    """
    Resolve category words like Small, Medium, or Large to a quantity.

    Uses the first uploaded rate card with a matching shipment_category and
    returns that row's min_qty so fixed shipment costs are calculated inside
    the requested band.
    """
    question_lower = question.lower()

    for category_name in ["small", "medium", "large"]:
        if category_name not in question_lower:
            continue

        for _, df in csv_dataframes:
            if "shipment_category" not in df.columns:
                continue

            matches = df[
                df["shipment_category"].astype(str).str.lower() == category_name
            ]

            if not matches.empty:
                return int(matches.iloc[0]["min_qty"])

    return fallback_quantity


def calculate_uploaded_costs(csv_dataframes, quantity):
    """
    Calculate total costs for all uploaded vendor CSVs.
    
    Args:
        csv_dataframes: List of (file_name, DataFrame) tuples
        quantity: Product quantity
    
    Returns:
        List of dicts with file, shipment_category, total_cost
    """
    results = []

    for file_name, df in csv_dataframes:
        row = get_shipment_category(df, quantity)
        total_cost = calculate_total_cost(row, quantity)

        results.append(
            {
                "file": file_name,
                "shipment_category": row.get("shipment_category", "N/A"),
                "total_cost": float(total_cost),
            }
        )

    return results


def get_vendor_name(file_name):
    """
    Extract vendor name from filename.
    
    Strips common suffixes like _quotation_rates.csv, _rates.csv, .csv
    
    Args:
        file_name: CSV filename
    
    Returns:
        str: Clean vendor name
    """
    return (
        file_name
        .replace("_quotation_rates.csv", "")
        .replace("_rates.csv", "")
        .replace(".csv", "")
    )


def get_rate_value(row, rate_name):
    """
    Get rate value for a specific service type.
    
    First checks for simple rate column, then calculates packaging from components.
    
    Args:
        row: DataFrame row
        rate_name: Service rate name (e.g., "packaging", "logistics")
    
    Returns:
        float: Rate value per unit, or None if not found
    """
    simple_column = f"{rate_name}_rate"

    if simple_column in row:
        return float(row[simple_column])

    if rate_name == "packaging":
        return float(calculate_packaging_cost(row))

    if rate_name == "sterilization":
        return float(calculate_sterilization_cost(row))

    return None


def get_factor_rate_value(row, rate_name, quantity):
    """
    Calculate a per-unit service factor from either simple or detailed columns.
    """
    simple_value = get_rate_value(row, rate_name)

    if simple_value is not None:
        return simple_value

    if rate_name == "logistics":
        return float(calculate_logistics_cost(row, quantity))

    if rate_name == "quality":
        return float(calculate_quality_cost(row, quantity))

    if rate_name == "warehousing":
        return float(calculate_warehousing_cost(row, quantity))

    return None


def find_vendor_matches(question, csv_dataframes):
    """
    Find vendors matching the question.
    
    Matches by vendor name in filename or vendor_id in CSV.
    
    Args:
        question: User question
        csv_dataframes: List of (file_name, DataFrame) tuples
    
    Returns:
        List of matching (file_name, DataFrame) tuples
    """
    question_lower = question.lower()
    normalized_question = re.sub(r"[^a-z0-9]", "", question_lower)
    vendor_matches = []

    for file_name, df in csv_dataframes:
        vendor_name = get_vendor_name(file_name)

        # Match by filename or vendor name
        normalized_vendor = re.sub(r"[^a-z0-9]", "", vendor_name.lower())
        normalized_file = re.sub(r"[^a-z0-9]", "", file_name.lower())

        if (
            vendor_name.lower() in question_lower
            or file_name.lower() in question_lower
            or normalized_vendor in normalized_question
            or normalized_file in normalized_question
        ):
            vendor_matches.append((file_name, df))
            continue

        # Match by vendor_id column
        if "vendor_id" in df.columns:
            matched_rows = df[
                df["vendor_id"].astype(str).str.lower().apply(
                    lambda vendor_id: vendor_id in question_lower
                )
            ]

            if not matched_rows.empty:
                vendor_matches.append((file_name, matched_rows))

    return vendor_matches


def build_comparison_factor_rows(csv_dataframes, quantity):
    """
    Build table rows for vendor comparison.
    
    Includes all individual rate factors for each vendor.
    
    Args:
        csv_dataframes: List of (file_name, DataFrame) tuples
        quantity: Product quantity
    
    Returns:
        List of dicts with vendor, file, shipment_category, rates, total_cost
    """
    costs = calculate_uploaded_costs(csv_dataframes, quantity)
    cost_lookup = {cost["file"]: cost for cost in costs}
    table_rows = []

    for file_name, df in csv_dataframes:
        row = get_shipment_category(df, quantity)
        table_row = {
            "vendor": get_vendor_name(file_name),
            "file": file_name,
            "shipment_category": row.get("shipment_category", "N/A"),
        }

        # Add each service rate
        for rate_name in SERVICE_RATE_NAMES:
            rate_value = get_factor_rate_value(row, rate_name, quantity)

            if rate_value is not None:
                table_row[f"{rate_name}_rate"] = round(rate_value, 2)

        # Add total cost
        table_row["total_cost"] = cost_lookup[file_name]["total_cost"]
        table_rows.append(table_row)

    return table_rows


def build_costing_engine_chunks(csv_dataframes, quantity):
    """
    Build deterministic costing chunks for RAG context.

    These chunks let the vector store index calculated totals and factor rates
    after document processing and before embedding generation.
    """
    table_rows = build_comparison_factor_rows(csv_dataframes, quantity)
    chunks = []

    for row in table_rows:
        rate_lines = []

        for rate_name in SERVICE_RATE_NAMES:
            rate_key = f"{rate_name}_rate"

            if rate_key in row:
                rate_lines.append(f"{rate_key}: {row[rate_key]}")

        chunks.append(
            "Source: deterministic costing engine\n"
            f"Vendor: {row['vendor']}\n"
            f"Source file: {row['file']}\n"
            f"Quantity: {quantity}\n"
            f"Shipment category: {row['shipment_category']}\n"
            + "\n".join(rate_lines)
            + f"\nTotal service cost: {row['total_cost']}"
        )

    if table_rows:
        sorted_rows = sorted(table_rows, key=lambda item: item["total_cost"])
        best_row = sorted_rows[0]
        comparison_lines = "\n".join(
            f"{row['vendor']} ({row['file']}): {row['total_cost']}"
            for row in sorted_rows
        )
        chunks.append(
            "Source: deterministic costing engine comparison\n"
            f"Quantity: {quantity}\n"
            f"Best option: {best_row['vendor']} ({best_row['file']})\n"
            f"Lowest total service cost: {best_row['total_cost']}\n"
            "Vendor totals sorted from lowest to highest:\n"
            f"{comparison_lines}"
        )

    return chunks


def run_costing_engine(state, csv_dataframes):
    """
    State-driven Costing Engine Wrapper.
    Calculates costs for all vendors in csv_dataframes using state.quantity.
    Saves outputs to state.costing_results.
    """
    if not csv_dataframes:
        state.add_trace("Costing Engine Skipped: No CSV rate cards uploaded")
        return {}

    state.add_trace("Costing Engine Execution Started", {"quantity": state.quantity})
    
    try:
        # Calculate comparison rows
        comparison_rows = build_comparison_factor_rows(csv_dataframes, state.quantity)
        
        # Save structured results
        state.costing_results = {
            "quantity": state.quantity,
            "comparison_rows": comparison_rows,
        }

        try:
            from engine.semantic_costs import (
                build_semantic_breakdown_rows,
                is_detailed_audit_question,
            )

            if is_detailed_audit_question(state.query.lower()):
                _, semantic_rows = build_semantic_breakdown_rows(
                    state.query,
                    csv_dataframes,
                    state.quantity,
                )

                if semantic_rows:
                    state.costing_results["semantic_breakdown_rows"] = semantic_rows
        except Exception as semantic_error:
            state.costing_results["semantic_breakdown_error"] = str(semantic_error)
        
        # Keep vendor profiles up to date in state
        state.vendor_profiles = [
            {"vendor": row["vendor"], "file": row["file"], "shipment_category": row["shipment_category"]}
            for row in comparison_rows
        ]
        
        state.add_trace("Costing Engine Calculations Completed", {
            "calculated_vendors_count": len(comparison_rows)
        })
        return state.costing_results
    except Exception as e:
        state.add_trace("Costing Engine Calculation Failed", {"error": str(e)})
        return {}


def run_vendor_logic(state, csv_dataframes):
    """
    State-driven Vendor Logic & Scoring Wrapper.
    Sorts vendors, recommends best vendor, and populates state.vendor_scores.
    """
    if not state.costing_results or "comparison_rows" not in state.costing_results:
        # Run costing first if not already run
        run_costing_engine(state, csv_dataframes)
        
    if not state.costing_results or "comparison_rows" not in state.costing_results:
        state.add_trace("Vendor Scoring Skipped: No costing results available")
        return []

    state.add_trace("Vendor Logic Execution Started")
    
    try:
        rows = state.costing_results["comparison_rows"]
        # Sort from lowest to highest total cost
        sorted_rows = sorted(rows, key=lambda item: item["total_cost"])
        
        state.vendor_scores = sorted_rows
        
        if sorted_rows:
            best_vendor = sorted_rows[0]
            # Record best option in memory as well
            from engine.memory import update_best_option
            update_best_option(state, best_vendor)
            
        state.add_trace("Vendor Scoring Completed", {
            "best_option": sorted_rows[0]["vendor"] if sorted_rows else None
        })
        return sorted_rows
    except Exception as e:
        state.add_trace("Vendor Scoring Failed", {"error": str(e)})
        return []
