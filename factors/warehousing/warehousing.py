"""
Warehousing cost factor.

Calculates warehousing cost per unit from rent, insurance, and handling fees.

Usage:
    from factors.warehousing import calculate_warehousing_cost
    cost = calculate_warehousing_cost(row, quantity)
"""


def calculate_warehousing_cost(row, quantity):
    """
    Calculate warehousing cost per unit.
    
    (rent + insurance + handling) / quantity
    
    Args:
        row: DataFrame row with warehouse_rent_period, insurance_cost_period, inventory_handling_cost_period
        quantity: Product quantity
    
    Returns:
        float: Warehousing cost per unit
    """
    return (
        row["warehouse_rent_period"]
        + row["insurance_cost_period"]
        + row["inventory_handling_cost_period"]
    ) / quantity