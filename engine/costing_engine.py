"""
Costing engine that calculates total service cost.

This module combines all cost factors (packaging, sterilization, logistics,
quality, warehousing) to calculate the total service cost per unit.

Usage:
    from engine.costing_engine import calculate_total_cost
    total = calculate_total_cost(row, quantity)
"""

from factors.packaging.packaging import calculate_packaging_cost
from factors.sterilization.sterilization import calculate_sterilization_cost
from factors.logistics.logistics import calculate_logistics_cost
from factors.quality.quality import calculate_quality_cost
from factors.warehousing.warehousing import calculate_warehousing_cost


def calculate_total_cost(row, quantity):
    """
    Calculate total service cost per unit.
    
    Tries simple rate columns first (pre-calculated totals).
    If not available, calculates from individual factors.
    
    Args:
        row: DataFrame row with rate columns
        quantity: Product quantity for per-unit calculations
    
    Returns:
        float: Total cost per unit (rounded to 2 decimal places)
    """
    # Check for simple pre-calculated rates
    simple_rate_columns = [
        "packaging_rate",
        "sterilization_rate",
        "logistics_rate",
        "quality_rate",
        "warehousing_rate",
    ]

    if all(column in row for column in simple_rate_columns):
        total = sum(row[column] for column in simple_rate_columns)
        return round(total, 2)

    # Calculate from individual cost factors
    packaging = calculate_packaging_cost(row)
    sterilization = calculate_sterilization_cost(row)
    logistics = calculate_logistics_cost(row, quantity)
    quality = calculate_quality_cost(row, quantity)
    warehousing = calculate_warehousing_cost(row, quantity)

    total = (
        packaging
        + sterilization
        + logistics
        + quality
        + warehousing
    )

    return round(total, 2)