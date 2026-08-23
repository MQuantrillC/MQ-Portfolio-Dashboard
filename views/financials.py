"""Financial statement views: statements, common-size analysis, ratios, insights."""

from typing import Dict, Optional

import pandas as pd
import streamlit as st

from core.formatting import format_compact_currency, format_ratio
from core.ratios import (
    BASE_METRICS,
    CATEGORY_LABELS,
    RATIO_CATEGORIES,
    RATIO_SPECS,
    calculate_financial_ratios,
    filter_statement,
    format_financial_columns,
    horizontal_analysis,
    latest_ratio_values,
    interpret_ratio,
    ratios_in_category,
)
from data.market import fetch_all_statements
from views.styles import figure_note

STATEMENT_LABELS = {
    "income": "Income statement",
    "balance": "Balance sheet",
    "cashflow": "Cash flow",
}

# Low-alpha tints read correctly on both light and dark backgrounds, and the
# text colour is left to the theme rather than hardcoded to "green"/"red".
_POSITIVE_TINT = "background-color: rgba(46, 160, 87, 0.18)"
_NEGATIVE_TINT = "background-color: rgba(214, 64, 56, 0.18)"


def _tint_by_sign(value) -> str:
    if pd.isna(value):
        return ""
    if value > 0:
        return _POSITIVE_TINT
    if value < 0:
        return _NEGATIVE_TINT
    return ""


def _render_statement_table(df: pd.DataFrame) -> None:
    styled = (
        df.style
        .format(lambda v: format_compact_currency(v) if pd.notna(v) else "N/A")
    )
    st.dataframe(styled, use_container_width=True)


def _render_horizontal(df: pd.DataFrame) -> None:
    st.markdown("##### Horizontal analysis")
    changes = horizontal_analysis(df)
    if changes is None or changes.dropna(how="all").empty:
        st.info("Horizontal analysis needs at least two years of data.")
        return

    styled = (
        changes.style
        .map(_tint_by_sign)
        .format(lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A")
    )
    st.dataframe(styled, use_container_width=True)
    figure_note(
        "Year-over-year change. Each column shows the change from the previous "
        "year — 2024 is the change from 2023. Green is growth, red is decline."
    )


def _render_vertical(df: pd.DataFrame, statement_type: str) -> None:
    from core.ratios import vertical_analysis

    base_metric = BASE_METRICS.get(statement_type, "")
    st.markdown("##### Vertical analysis")
    percentages = vertical_analysis(df, statement_type)
    if percentages is None or percentages.dropna(how="all").empty:
        st.info(
            f"Vertical analysis needs a '{base_metric}' line, which isn't in this "
            f"statement."
        )
        return

    styled = percentages.style.format(lambda v: f"{v:.1f}%" if pd.notna(v) else "N/A")
    st.dataframe(styled, use_container_width=True)
    figure_note(f"Every line as a percentage of {base_metric}.")


def _render_statement(statement: Optional[pd.DataFrame], statement_type: str, ticker: str) -> None:
    label = STATEMENT_LABELS[statement_type]
    filtered = filter_statement(statement, statement_type)
    if filtered is None:
        st.warning(
            f"Couldn't load the {label.lower()} for {ticker}. The provider may not "
            f"publish it, or the request failed — check the server log."
        )
        return

    formatted = format_financial_columns(filtered)
    if formatted is None:
        st.warning(f"No dated columns in {ticker}'s {label.lower()}.")
        return

    _render_statement_table(formatted)
    _render_horizontal(filtered)
    _render_vertical(filtered, statement_type)


# --------------------------------------------------------------------------
# Ratios
# --------------------------------------------------------------------------

def _ratio_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Format a numeric ratio frame for display, one row-kind at a time."""
    display = pd.DataFrame(index=frame.index, columns=frame.columns, dtype="object")
    for name in frame.index:
        kind = RATIO_SPECS[name]["kind"]
        display.loc[name] = [format_ratio(v, kind) for v in frame.loc[name]]
    return display


def _render_ratio_tables(ratios: Dict[str, pd.DataFrame]) -> None:
    for category in RATIO_CATEGORIES:
        frame = ratios.get(category)
        if frame is None or frame.dropna(how="all").empty:
            continue

        st.markdown(f"##### {CATEGORY_LABELS[category]} ratios")
        st.dataframe(_ratio_display_frame(frame), use_container_width=True)
        formulas = " • ".join(
            RATIO_SPECS[name]["formula"] for name in ratios_in_category(category)
        )
        figure_note(formulas)


def _render_insights(ratios: Dict[str, pd.DataFrame], year: str) -> None:
    """Interpret the latest year's ratios against their benchmark bands.

    Driven entirely by the RATIO_SPECS table, so adding a ratio is a data change
    rather than another forty lines of if/elif.
    """
    values = latest_ratio_values(ratios, year)
    if not values:
        st.info("No ratio data available to interpret.")
        return

    st.markdown("#### Interpretation")
    st.caption(f"{year} ratios against common benchmark ranges.")

    for category in RATIO_CATEGORIES:
        names = [n for n in ratios_in_category(category) if values.get(n) is not None]
        if not names:
            continue

        with st.expander(f"{CATEGORY_LABELS[category]}", expanded=True):
            columns = st.columns(len(names))
            for column, name in zip(columns, names):
                spec = RATIO_SPECS[name]
                value = values[name]
                with column:
                    st.metric(name, format_ratio(value, spec["kind"]), help=spec["help"])
                    verdict = interpret_ratio(name, value)
                    if verdict is None:
                        continue
                    level, message = verdict
                    {"error": st.error, "warning": st.warning, "success": st.success}[level](message)


def _render_ratios(statements: Dict[str, Optional[pd.DataFrame]], ticker: str) -> None:
    # Ratios run against the *raw* statements, not the filtered display subset:
    # inputs like Accounts Payable and Cost Of Revenue aren't display lines, and
    # filtering first would make DPO and the cash conversion cycle permanently
    # blank.
    ratios = calculate_financial_ratios(
        statements.get("income"),
        statements.get("balance"),
        statements.get("cashflow"),
    )
    if ratios is None:
        st.warning(
            f"Couldn't calculate ratios for {ticker} — one or more statements is "
            f"missing or has no overlapping years."
        )
        return

    _render_ratio_tables(ratios)

    latest_year = next(
        (str(c) for c in ratios["liquidity"].columns), None
    )
    if latest_year:
        st.markdown("---")
        _render_insights(ratios, latest_year)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def render_financials(tickers) -> None:
    """Financial statements and ratios, one tab per ticker."""
    if not tickers:
        st.info("Create a portfolio to see financial statements.")
        return

    ticker_tabs = st.tabs(list(tickers))
    for tab, ticker in zip(ticker_tabs, tickers):
        with tab:
            with st.spinner(f"Loading {ticker} statements…"):
                statements = fetch_all_statements(ticker)

            statement_tabs = st.tabs(
                ["Income statement", "Balance sheet", "Cash flow", "Key ratios"]
            )
            with statement_tabs[0]:
                _render_statement(statements.get("income"), "income", ticker)
            with statement_tabs[1]:
                _render_statement(statements.get("balance"), "balance", ticker)
            with statement_tabs[2]:
                _render_statement(statements.get("cashflow"), "cashflow", ticker)
            with statement_tabs[3]:
                _render_ratios(statements, ticker)
