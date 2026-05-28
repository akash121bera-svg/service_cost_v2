"""
Quality cost factor.

Calculates quality cost per unit from inspection, audit, and documentation fees.

Usage:
    from factors.quality import calculate_quality_cost
    cost = calculate_quality_cost(row, quantity)
"""


def calculate_quality_cost(row, quantity):
    """
    Calculate quality cost per unit.
    
    (inspection + audit + documentation) / quantity
    
    Args:
        row: DataFrame row with inspection_cost_per_shipment, audit_cost_per_shipment, documentation_cost_per_shipment
        quantity: Product quantity
    
    Returns:
        float: Quality cost per unit
    """
    return (
        row["inspection_cost_per_shipment"]
        + row["audit_cost_per_shipment"]
        + row["documentation_cost_per_shipment"]
    ) / quantity