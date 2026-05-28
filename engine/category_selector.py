"""
Shipment category selector.

Selects the appropriate rate row based on product quantity.

Usage:
    from engine.category_selector import get_shipment_category
    row = get_shipment_category(df, quantity)
"""


def get_shipment_category(df, quantity):
    """
    Find the matching shipment category for a quantity.
    
    Searches for a row where min_qty <= quantity <= max_qty.
    
    Args:
        df: DataFrame with min_qty and max_qty columns
        quantity: Product quantity
    
    Returns:
        DataFrame row with matching category
    
    Raises:
        ValueError: If no matching category found
    """
    matched = df[
        (df["min_qty"] <= quantity)
        &
        (df["max_qty"] >= quantity)
    ]

    if matched.empty:
        raise ValueError(
            f"No shipment category found for quantity {quantity}"
        )

    return matched.iloc[0]