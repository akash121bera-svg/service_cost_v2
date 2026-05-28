"""
Logistics cost factor.

Calculates logistics cost per unit from transport, handling, and distributor fees.

Usage:
    from factors.logistics import calculate_logistics_cost
    cost = calculate_logistics_cost(row, quantity)
"""


def calculate_logistics_cost(row, quantity):
    """
    Calculate logistics cost per unit.
    
    (transport + handling + distributor_fee) / quantity
    
    Args:
        row: DataFrame row with transport_cost_per_shipment, handling_cost_per_shipment, distributor_fee_per_shipment
        quantity: Product quantity
    
    Returns:
        float: Logistics cost per unit
    """
    return (
        row["transport_cost_per_shipment"]
        + row["handling_cost_per_shipment"]
        + row["distributor_fee_per_shipment"]
    ) / quantity