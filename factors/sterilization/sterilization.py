"""
Sterilization cost factor.

Calculates sterilization cost per unit from batch cost and validation cost.

Usage:
    from factors.sterilization import calculate_sterilization_cost
    cost = calculate_sterilization_cost(row)
"""


def calculate_sterilization_cost(row):
    """
    Calculate sterilization cost per unit.
    
    (batch_cost + validation_cost) / units_per_batch
    
    Args:
        row: DataFrame row with sterilization_batch_cost, validation_cost, units_per_batch
    
    Returns:
        float: Sterilization cost per unit
    """
    return (
        row["sterilization_batch_cost"]
        + row["validation_cost"]
    ) / row["units_per_batch"]