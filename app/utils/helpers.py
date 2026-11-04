
def format_lakhs(value):
    """Format a numeric property price in Indian lakhs."""

    return f"₹{value:,.2f} Lakhs"


def format_rupees(value):
    """Format a numeric value in Indian rupees."""

    return f"₹{value:,.2f}"


def format_number(value):
    """Format a number with Indian-style comma grouping."""

    return f"{value:,.0f}"


def safe_percentage(value):
    """Convert a decimal metric into a percentage string."""

    return f"{value * 100:.2f}%"


def format_area(value):
    """Format property size in square feet."""

    return f"{value:,.0f} sq.ft."
