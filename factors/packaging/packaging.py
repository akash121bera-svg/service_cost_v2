"""
Packaging cost factor.

Calculates packaging cost from pouch, label, and carton rates.

Usage:
    from factors.packaging import calculate_packaging_cost
    cost = calculate_packaging_cost(row)
"""


def calculate_packaging_cost(row):
    """
    Calculate packaging cost per unit.
    
    Sums pouch, label, and carton rates per unit.
    
    Args:
        row: DataFrame row with pouch_rate_per_unit, label_rate_per_unit, carton_rate_per_unit
    
    Returns:
        float: Packaging cost per unit
    """
    return (
        row["pouch_rate_per_unit"]
        + row["label_rate_per_unit"]
        + row["carton_rate_per_unit"]
    )