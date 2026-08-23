"""Financial statement parsing and ratio analysis.

Pure functions over DataFrames. Everything returned here is *numeric* — the
views format it. Nothing formats a number to a string and parses it back.
"""

from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.formatting import safe_float


# --------------------------------------------------------------------------
# Statement metric maps
# --------------------------------------------------------------------------

STATEMENT_METRICS: Dict[str, List[Tuple[str, List[str]]]] = {
    "income": [
        ("Total Revenue", ["total revenue", "totalrevenue", "revenue"]),
        ("Cost of Revenue", ["cost of revenue", "costofrevenue"]),
        ("Gross Profit", ["gross profit", "grossprofit"]),
        ("Operating Income", ["operating income", "operatingincome"]),
        ("EBIT", ["ebit"]),
        ("EBITDA", ["ebitda"]),
        ("Total Expenses", ["total expenses", "totalexpenses"]),
        ("Diluted EPS", ["diluted eps", "dilutedeps"]),
        ("Net Income", ["net income common stockholders", "net income", "netincome"]),
    ],
    "balance": [
        ("Total Assets", ["total assets", "totalassets"]),
        ("Current Assets", ["current assets", "total current assets"]),
        ("Total Liabilities", [
            "total liabilities net minority interest", "total liabilities", "totalliabilities",
        ]),
        ("Current Liabilities", ["current liabilities", "total current liabilities"]),
        ("Total Equity", [
            "total equity gross minority interest", "stockholders equity",
            "total stockholder equity", "total equity",
        ]),
        ("Total Capitalization", ["total capitalization", "totalcapitalization"]),
        ("Net Tangible Assets", ["net tangible assets", "nettangibleassets"]),
        ("Working Capital", ["working capital", "workingcapital"]),
        ("Invested Capital", ["invested capital", "investedcapital"]),
        ("Total Debt", ["total debt", "totaldebt"]),
        ("Inventory", ["inventory", "inventories"]),
        ("Net Receivables", ["accounts receivable", "net receivables", "receivables"]),
    ],
    "cashflow": [
        ("Operating Cash Flow", [
            "operating cash flow", "total cash from operating activities",
            "net cash provided by operating activities",
        ]),
        ("Investing Cash Flow", [
            "investing cash flow", "total cashflows from investing activities",
            "net cash used for investing activities",
        ]),
        ("Financing Cash Flow", [
            "financing cash flow", "total cash from financing activities",
            "net cash provided by financing activities",
        ]),
        ("End Cash Position", [
            "end cash position", "cash at end of period",
            "cash and cash equivalents at end of year",
        ]),
        ("Capital Expenditure", ["capital expenditure", "capital expenditures"]),
        ("Free Cash Flow", ["free cash flow"]),
    ],
}

BASE_METRICS = {
    "income": "Total Revenue",
    "balance": "Total Assets",
    "cashflow": "Operating Cash Flow",
}

_VERTICAL_BASE_ALIASES = {
    "income": ["total revenue", "revenue", "net sales", "total sales"],
    "balance": ["total assets", "assets"],
    "cashflow": [
        "operating cash flow", "cash from operations",
        "net cash provided by operating activities",
    ],
}


# --------------------------------------------------------------------------
# Ratio definitions — one table, rendered by one loop
# --------------------------------------------------------------------------
#
# bands: ordered (upper_bound, level, message). The first band whose upper bound
# the value falls below wins; a bound of None is the catch-all.

RATIO_SPECS: Dict[str, dict] = {
    "Current Ratio": dict(
        category="liquidity", kind="decimal",
        help="Current assets / current liabilities",
        formula="Current Ratio = Current Assets / Current Liabilities",
        bands=[
            (1.0, "error", "Below 1 — may struggle to meet short-term obligations"),
            (3.0, "success", "Healthy (1–3)"),
            (None, "warning", "Above 3 — possibly idle assets"),
        ],
    ),
    "Quick Ratio": dict(
        category="liquidity", kind="decimal",
        help="(Current assets − inventory) / current liabilities",
        formula="Quick Ratio = (Current Assets − Inventory) / Current Liabilities",
        bands=[
            (1.0, "error", "Below 1 — liquidity concern once inventory is excluded"),
            (2.0, "warning", "Moderate (1–2)"),
            (None, "success", "Strong (above 2)"),
        ],
    ),
    "Working Capital": dict(
        category="liquidity", kind="currency",
        help="Current assets − current liabilities",
        formula="Working Capital = Current Assets − Current Liabilities",
        bands=[
            (0.0, "error", "Negative — liquidity risk"),
            (None, "success", "Positive — short-term solvency OK"),
        ],
    ),
    "Gross Margin": dict(
        category="profitability", kind="percentage",
        help="Gross profit / revenue",
        formula="Gross Margin = Gross Profit / Revenue",
        bands=[
            (0.20, "error", "Below 20% — high cost of sales"),
            (0.40, "warning", "Moderate (20–40%)"),
            (None, "success", "Strong (above 40%)"),
        ],
    ),
    "Operating Margin": dict(
        category="profitability", kind="percentage",
        help="Operating income / revenue",
        formula="Operating Margin = Operating Income / Revenue",
        bands=[
            (0.10, "error", "Below 10% — heavy operating costs"),
            (0.20, "warning", "Moderate (10–20%)"),
            (None, "success", "Healthy (above 20%)"),
        ],
    ),
    "Net Margin": dict(
        category="profitability", kind="percentage",
        help="Net income / revenue",
        formula="Net Margin = Net Income / Revenue",
        bands=[
            (0.0, "error", "Negative — unprofitable"),
            (0.20, "warning", "Moderate (0–20%)"),
            (None, "success", "Excellent (above 20%)"),
        ],
    ),
    "ROA": dict(
        category="profitability", kind="percentage",
        help="Net income / total assets",
        formula="ROA = Net Income / Total Assets",
        bands=[
            (0.0, "error", "Negative — assets not generating profit"),
            (0.10, "warning", "Average (0–10%)"),
            (None, "success", "Strong (above 10%)"),
        ],
    ),
    "ROE": dict(
        category="profitability", kind="percentage",
        help="Net income / total equity",
        formula="ROE = Net Income / Total Equity",
        bands=[
            (0.0, "error", "Negative — equity is being eroded"),
            (0.15, "warning", "Average (0–15%)"),
            (None, "success", "High (above 15%)"),
        ],
    ),
    "DSO": dict(
        category="efficiency", kind="days",
        help="Days Sales Outstanding — how long customers take to pay",
        formula="DSO = Accounts Receivable / (Revenue / 365)",
        bands=[
            (30.0, "success", "Under 30 days — fast collections"),
            (60.0, "warning", "Average (30–60 days)"),
            (None, "error", "Over 60 days — slow collections"),
        ],
    ),
    "DPO": dict(
        category="efficiency", kind="days",
        help="Days Payable Outstanding — how long the company takes to pay suppliers",
        formula="DPO = Accounts Payable / (COGS / 365)",
        bands=[
            (30.0, "error", "Under 30 days — short supplier terms"),
            (90.0, "warning", "Normal (30–90 days)"),
            (None, "success", "Over 90 days — favourable credit terms"),
        ],
    ),
    "DIO": dict(
        category="efficiency", kind="days",
        help="Days Inventory Outstanding — how long stock sits before selling",
        formula="DIO = Inventory / (COGS / 365)",
        bands=[
            (30.0, "success", "Under 30 days — fast turnover"),
            (90.0, "warning", "Average (30–90 days)"),
            (None, "error", "Over 90 days — slow turnover"),
        ],
    ),
    "Cash Conversion Cycle": dict(
        category="efficiency", kind="days",
        help="DSO + DIO − DPO — days between paying for stock and collecting cash",
        formula="Cash Conversion Cycle = DSO + DIO − DPO",
        bands=[
            (0.0, "success", "Negative — suppliers effectively finance operations"),
            (60.0, "warning", "Reasonable (under 60 days)"),
            (None, "error", "Over 60 days — slow cash conversion"),
        ],
    ),
    "Debt/Equity": dict(
        category="leverage", kind="decimal",
        help="Total debt / total equity",
        formula="Debt/Equity = Total Debt / Total Equity",
        bands=[
            (1.0, "success", "Conservative (below 1)"),
            (2.0, "warning", "Moderate (1–2)"),
            (None, "error", "Above 2 — highly leveraged"),
        ],
    ),
    "Debt/Assets": dict(
        category="leverage", kind="decimal",
        help="Total debt / total assets",
        formula="Debt/Assets = Total Debt / Total Assets",
        bands=[
            (0.30, "success", "Conservative (below 0.3)"),
            (0.50, "warning", "Moderate (0.3–0.5)"),
            (None, "error", "Above 0.5 — debt-heavy balance sheet"),
        ],
    ),
}

RATIO_CATEGORIES = ("liquidity", "profitability", "efficiency", "leverage")

CATEGORY_LABELS = {
    "liquidity": "Liquidity",
    "profitability": "Profitability",
    "efficiency": "Efficiency",
    "leverage": "Leverage",
}


def ratios_in_category(category: str) -> List[str]:
    return [name for name, spec in RATIO_SPECS.items() if spec["category"] == category]


def interpret_ratio(name: str, value: Optional[float]) -> Optional[Tuple[str, str]]:
    """Return an (level, message) verdict for a ratio value, or None if unknown."""
    spec = RATIO_SPECS.get(name)
    value = safe_float(value)
    if spec is None or value is None:
        return None

    for upper, level, message in spec["bands"]:
        if upper is None or value < upper:
            return level, message
    return None


# --------------------------------------------------------------------------
# Statement shaping
# --------------------------------------------------------------------------

def format_financial_columns(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Relabel statement columns as years, newest first, dropping empty columns."""
    if df is None or df.empty:
        return None

    year_by_column = {}
    for col in df.columns:
        if df[col].isna().all():
            continue
        if isinstance(col, (pd.Timestamp, datetime)):
            year_by_column[col] = col.year
        else:
            head = str(col).split("-")[0]
            if head.isdigit():
                year_by_column[col] = int(head)

    if not year_by_column:
        return None

    ordered = sorted(year_by_column, key=lambda c: year_by_column[c], reverse=True)
    result = df[ordered].copy()
    result.columns = [str(year_by_column[c]) for c in ordered]
    return result


def filter_statement(
    statement: Optional[pd.DataFrame], statement_type: str
) -> Optional[pd.DataFrame]:
    """Reduce a raw yfinance statement to the named metrics we display."""
    if statement is None or statement.empty:
        return None

    metric_map = STATEMENT_METRICS.get(statement_type, [])
    lookup = {str(idx).lower(): idx for idx in statement.index}

    rows = {}
    for display_name, candidates in metric_map:
        for candidate in candidates:
            if candidate in lookup:
                rows[display_name] = statement.loc[lookup[candidate]]
                break

    if not rows:
        return None
    return pd.DataFrame(rows).T


def horizontal_analysis(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Year-over-year percentage change per metric, as numbers.

    Columns run newest to oldest, so column *i* holds the change from column
    *i+1*. The oldest column has no predecessor and is dropped.
    """
    df_sorted = format_financial_columns(df)
    if df_sorted is None or len(df_sorted.columns) < 2:
        return None

    numeric = df_sorted.apply(pd.to_numeric, errors="coerce")
    newer = numeric.iloc[:, :-1]
    older = numeric.iloc[:, 1:]
    older.columns = newer.columns

    with np.errstate(divide="ignore", invalid="ignore"):
        changes = (newer - older) / older.abs() * 100
    return changes.replace([np.inf, -np.inf], np.nan)


def vertical_analysis(
    df: Optional[pd.DataFrame], statement_type: str
) -> Optional[pd.DataFrame]:
    """Each metric as a percentage of the statement's base metric, as numbers."""
    df_sorted = format_financial_columns(df)
    if df_sorted is None:
        return None

    numeric = df_sorted.apply(pd.to_numeric, errors="coerce")
    aliases = _VERTICAL_BASE_ALIASES.get(statement_type, [])
    base_name = BASE_METRICS.get(statement_type, "").lower()
    candidates = [base_name] + aliases if base_name else aliases

    base_values = None
    for idx in numeric.index:
        if str(idx).lower() in candidates:
            base_values = numeric.loc[idx]
            break
    if base_values is None:
        for idx in numeric.index:
            idx_lower = str(idx).lower()
            if any(alias in idx_lower for alias in candidates):
                base_values = numeric.loc[idx]
                break
    if base_values is None:
        return None

    with np.errstate(divide="ignore", invalid="ignore"):
        percentages = numeric.div(base_values.replace(0, np.nan)) * 100
    return percentages.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------
# Ratio calculation
# --------------------------------------------------------------------------

def _lookup(
    df: pd.DataFrame, candidates: Sequence[str], year: str
) -> Optional[float]:
    """Find a statement line by name and return its value for ``year``.

    Exact (case-insensitive) matches are tried first across every candidate.
    Only then do we fall back to substrings, preferring the *shortest* matching
    label — otherwise looking up "Total Debt" can land on
    "Long Term Debt And Capital Lease Obligation".
    """
    if year not in df.columns:
        return None

    lookup = {str(idx).lower(): idx for idx in df.index}
    for candidate in candidates:
        key = candidate.lower()
        if key in lookup:
            return safe_float(df.loc[lookup[key], year])

    for candidate in candidates:
        key = candidate.lower()
        matches = [idx for label, idx in lookup.items() if key in label]
        if matches:
            best = min(matches, key=lambda idx: len(str(idx)))
            return safe_float(df.loc[best, year])

    return None


def _divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Safe division that returns None rather than raising on missing operands."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _subtract(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def calculate_financial_ratios(
    income_stmt: Optional[pd.DataFrame],
    balance_sheet: Optional[pd.DataFrame],
    cash_flow: Optional[pd.DataFrame],
) -> Optional[Dict[str, pd.DataFrame]]:
    """Compute every ratio for every common year.

    Returns numeric DataFrames keyed by category. Each ratio is computed
    independently, so one missing statement line (inventory, say — normal for
    banks and software companies) blanks that one ratio instead of aborting the
    rest of the year.
    """
    income_stmt = format_financial_columns(income_stmt)
    balance_sheet = format_financial_columns(balance_sheet)
    cash_flow = format_financial_columns(cash_flow)

    if any(df is None for df in (income_stmt, balance_sheet, cash_flow)):
        return None

    years = sorted(
        set(income_stmt.columns) & set(balance_sheet.columns) & set(cash_flow.columns),
        reverse=True,
    )
    if not years:
        return None

    frames = {
        category: pd.DataFrame(
            index=ratios_in_category(category), columns=years, dtype="float64"
        )
        for category in RATIO_CATEGORIES
    }

    for year in years:
        current_assets = _lookup(balance_sheet, ["Current Assets", "Total Current Assets"], year)
        current_liabilities = _lookup(balance_sheet, ["Current Liabilities", "Total Current Liabilities"], year)
        inventory = _lookup(balance_sheet, ["Inventory", "Inventories"], year)
        total_assets = _lookup(balance_sheet, ["Total Assets"], year)
        total_equity = _lookup(balance_sheet, [
            "Stockholders Equity", "Total Equity Gross Minority Interest",
            "Total Stockholder Equity", "Total Equity",
        ], year)
        total_debt = _lookup(balance_sheet, ["Total Debt", "Long Term Debt"], year)
        receivables = _lookup(balance_sheet, ["Accounts Receivable", "Net Receivables", "Receivables"], year)
        payables = _lookup(balance_sheet, ["Accounts Payable", "Payables"], year)

        revenue = _lookup(income_stmt, ["Total Revenue", "Revenue", "Net Sales"], year)
        gross_profit = _lookup(income_stmt, ["Gross Profit"], year)
        operating_income = _lookup(income_stmt, ["Operating Income", "Operating Profit"], year)
        net_income = _lookup(income_stmt, ["Net Income", "Net Income Common Stockholders"], year)
        cogs = _lookup(income_stmt, ["Cost Of Revenue", "Cost of Goods Sold"], year)

        daily_revenue = revenue / 365 if revenue else None
        daily_cogs = cogs / 365 if cogs else None

        dso = _divide(receivables, daily_revenue)
        dpo = _divide(payables, daily_cogs)
        dio = _divide(inventory, daily_cogs)
        ccc = None if None in (dso, dio, dpo) else dso + dio - dpo

        values = {
            "Current Ratio": _divide(current_assets, current_liabilities),
            "Quick Ratio": _divide(_subtract(current_assets, inventory), current_liabilities),
            "Working Capital": _subtract(current_assets, current_liabilities),
            "Gross Margin": _divide(gross_profit, revenue),
            "Operating Margin": _divide(operating_income, revenue),
            "Net Margin": _divide(net_income, revenue),
            "ROA": _divide(net_income, total_assets),
            "ROE": _divide(net_income, total_equity),
            "DSO": dso,
            "DPO": dpo,
            "DIO": dio,
            "Cash Conversion Cycle": ccc,
            "Debt/Equity": _divide(total_debt, total_equity),
            "Debt/Assets": _divide(total_debt, total_assets),
        }

        for name, value in values.items():
            category = RATIO_SPECS[name]["category"]
            frames[category].loc[name, year] = np.nan if value is None else value

    return frames


def latest_ratio_values(
    ratios: Dict[str, pd.DataFrame], year: str
) -> Dict[str, Optional[float]]:
    """Flatten the per-category frames into {ratio name: value} for one year."""
    values: Dict[str, Optional[float]] = {}
    for frame in ratios.values():
        if year in frame.columns:
            for name in frame.index:
                values[name] = safe_float(frame.loc[name, year])
    return values
