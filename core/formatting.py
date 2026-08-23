"""Value formatting helpers.

Formatting happens at the render boundary only. Everything upstream keeps
numbers as numbers, so nothing has to parse ``"$1,234.56"`` back into a float.
"""

from typing import Any, List, Optional

import pandas as pd


def format_value(
    value: Any, prefix: str = "$", suffix: str = "", decimals: int = 2
) -> str:
    """Format a number with a prefix/suffix, or return 'N/A' if it isn't one."""
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
        return f"{prefix}{float(value):,.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def format_compact_currency(value: Any) -> str:
    """Format a large currency figure as $1.23T / $45.6B / $789M."""
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    for threshold, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"${value / threshold:,.2f}{unit}"
    return f"${value:,.2f}"


def format_percent(value: Any, decimals: int = 2, signed: bool = False) -> str:
    """Format a decimal fraction (0.0432) as a percentage string (4.32%)."""
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
        sign = "+" if signed else ""
        return f"{float(value) * 100:{sign}.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_ratio(value: Any, kind: str) -> str:
    """Format a financial ratio according to its kind."""
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if kind == "currency":
        return format_compact_currency(value)
    if kind == "percentage":
        return f"{value * 100:.1f}%"
    if kind == "days":
        return f"{value:.1f} days"
    return f"{value:.2f}"


def remove_duplicates(items: List[str]) -> List[str]:
    """Drop duplicates while preserving order."""
    return list(dict.fromkeys(items))


def safe_float(value: Any) -> Optional[float]:
    """Coerce to float, or None if that isn't possible."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
