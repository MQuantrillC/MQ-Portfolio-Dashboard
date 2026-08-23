"""Tests for financial statement parsing and ratio analysis."""

import numpy as np
import pandas as pd
import pytest

from core.ratios import (
    calculate_financial_ratios,
    filter_statement,
    format_financial_columns,
    horizontal_analysis,
    interpret_ratio,
    latest_ratio_values,
    vertical_analysis,
)


@pytest.fixture
def income_statement():
    return pd.DataFrame(
        {
            pd.Timestamp("2024-12-31"): [1000.0, 400.0, 600.0, 200.0, 150.0],
            pd.Timestamp("2023-12-31"): [800.0, 360.0, 440.0, 120.0, 90.0],
        },
        index=["Total Revenue", "Cost Of Revenue", "Gross Profit",
               "Operating Income", "Net Income"],
    )


@pytest.fixture
def balance_sheet():
    return pd.DataFrame(
        {
            pd.Timestamp("2024-12-31"): [2000.0, 500.0, 250.0, 100.0, 800.0, 400.0, 120.0, 80.0],
            pd.Timestamp("2023-12-31"): [1800.0, 450.0, 300.0, 120.0, 700.0, 450.0, 100.0, 90.0],
        },
        index=["Total Assets", "Current Assets", "Current Liabilities", "Inventory",
               "Stockholders Equity", "Total Debt", "Accounts Receivable",
               "Accounts Payable"],
    )


@pytest.fixture
def cash_flow():
    return pd.DataFrame(
        {
            pd.Timestamp("2024-12-31"): [300.0, -100.0],
            pd.Timestamp("2023-12-31"): [250.0, -80.0],
        },
        index=["Operating Cash Flow", "Investing Cash Flow"],
    )


# --------------------------------------------------------------------------
# Column shaping
# --------------------------------------------------------------------------

def test_columns_become_years_newest_first(income_statement):
    result = format_financial_columns(income_statement)
    assert list(result.columns) == ["2024", "2023"]


def test_all_nan_columns_are_dropped():
    df = pd.DataFrame(
        {pd.Timestamp("2024-12-31"): [1.0], pd.Timestamp("2023-12-31"): [np.nan]},
        index=["Total Revenue"],
    )
    assert list(format_financial_columns(df).columns) == ["2024"]


def test_empty_statement_returns_none():
    assert format_financial_columns(pd.DataFrame()) is None
    assert format_financial_columns(None) is None


def test_filter_statement_maps_provider_names_to_display_names(balance_sheet):
    filtered = filter_statement(balance_sheet, "balance")
    assert "Total Assets" in filtered.index
    assert "Net Receivables" in filtered.index  # from "Accounts Receivable"


# --------------------------------------------------------------------------
# Common-size analysis
# --------------------------------------------------------------------------

def test_horizontal_analysis_computes_year_over_year_change(income_statement):
    changes = horizontal_analysis(income_statement)
    assert list(changes.columns) == ["2024"]
    assert changes.loc["Total Revenue", "2024"] == pytest.approx(25.0)


def test_horizontal_analysis_needs_two_years():
    single = pd.DataFrame({pd.Timestamp("2024-12-31"): [1.0]}, index=["Total Revenue"])
    assert horizontal_analysis(single) is None


def test_vertical_analysis_expresses_lines_as_a_share_of_revenue(income_statement):
    percentages = vertical_analysis(income_statement, "income")
    assert percentages.loc["Total Revenue", "2024"] == pytest.approx(100.0)
    assert percentages.loc["Gross Profit", "2024"] == pytest.approx(60.0)


def test_vertical_analysis_returns_none_without_a_base_metric():
    df = pd.DataFrame({pd.Timestamp("2024-12-31"): [5.0]}, index=["Something Else"])
    assert vertical_analysis(df, "income") is None


# --------------------------------------------------------------------------
# Ratios
# --------------------------------------------------------------------------

def test_ratios_are_computed_for_every_year(income_statement, balance_sheet, cash_flow):
    ratios = calculate_financial_ratios(income_statement, balance_sheet, cash_flow)
    assert set(ratios) == {"liquidity", "profitability", "efficiency", "leverage"}
    assert list(ratios["liquidity"].columns) == ["2024", "2023"]
    assert ratios["liquidity"].loc["Current Ratio", "2024"] == pytest.approx(2.0)
    assert ratios["profitability"].loc["Gross Margin", "2024"] == pytest.approx(0.6)
    assert ratios["leverage"].loc["Debt/Equity", "2024"] == pytest.approx(0.5)


def test_missing_inventory_blanks_only_the_quick_ratio(
    income_statement, balance_sheet, cash_flow
):
    """A bank has no inventory line.

    The original code computed ``current_assets - None``, raised TypeError, and
    the surrounding try/except skipped the rest of that year — silently wiping
    out DSO, DPO, DIO, the cash conversion cycle and both leverage ratios.
    """
    no_inventory = balance_sheet.drop(index=["Inventory"])
    ratios = calculate_financial_ratios(income_statement, no_inventory, cash_flow)

    assert ratios is not None
    assert pd.isna(ratios["liquidity"].loc["Quick Ratio", "2024"])
    # Everything computed after the quick ratio must survive.
    assert ratios["liquidity"].loc["Current Ratio", "2024"] == pytest.approx(2.0)
    assert ratios["leverage"].loc["Debt/Equity", "2024"] == pytest.approx(0.5)
    assert ratios["leverage"].loc["Debt/Assets", "2024"] == pytest.approx(0.2)
    assert not pd.isna(ratios["efficiency"].loc["DSO", "2024"])


def test_zero_denominator_yields_nan_not_an_exception(
    income_statement, balance_sheet, cash_flow
):
    zero_equity = balance_sheet.copy()
    zero_equity.loc["Stockholders Equity"] = 0.0
    ratios = calculate_financial_ratios(income_statement, zero_equity, cash_flow)
    assert pd.isna(ratios["leverage"].loc["Debt/Equity", "2024"])


def test_total_debt_lookup_prefers_the_exact_line():
    """Substring matching on 'Debt' used to land on the longest compound label."""
    balance = pd.DataFrame(
        {
            pd.Timestamp("2024-12-31"): [1000.0, 500.0, 250.0, 400.0, 900.0],
        },
        index=["Total Assets", "Current Assets", "Current Liabilities",
               "Total Debt", "Long Term Debt And Capital Lease Obligation"],
    )
    income = pd.DataFrame({pd.Timestamp("2024-12-31"): [1000.0]}, index=["Total Revenue"])
    cash = pd.DataFrame({pd.Timestamp("2024-12-31"): [100.0]}, index=["Operating Cash Flow"])

    ratios = calculate_financial_ratios(income, balance, cash)
    assert ratios["leverage"].loc["Debt/Assets", "2024"] == pytest.approx(0.4)


def test_missing_statement_returns_none(income_statement, balance_sheet):
    assert calculate_financial_ratios(income_statement, balance_sheet, None) is None


def test_no_overlapping_years_returns_none(balance_sheet, cash_flow):
    income = pd.DataFrame({pd.Timestamp("2019-12-31"): [1.0]}, index=["Total Revenue"])
    assert calculate_financial_ratios(income, balance_sheet, cash_flow) is None


def test_latest_ratio_values_flattens_categories(
    income_statement, balance_sheet, cash_flow
):
    ratios = calculate_financial_ratios(income_statement, balance_sheet, cash_flow)
    values = latest_ratio_values(ratios, "2024")
    assert values["Current Ratio"] == pytest.approx(2.0)
    assert values["ROE"] == pytest.approx(150.0 / 800.0)


# --------------------------------------------------------------------------
# Interpretation
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, value, expected_level",
    [
        ("Current Ratio", 0.8, "error"),
        ("Current Ratio", 2.0, "success"),
        ("Current Ratio", 4.0, "warning"),
        ("Net Margin", -0.05, "error"),
        ("Net Margin", 0.10, "warning"),
        ("Net Margin", 0.30, "success"),
        ("DSO", 20.0, "success"),
        ("DSO", 45.0, "warning"),
        ("DSO", 90.0, "error"),
        ("Debt/Equity", 0.5, "success"),
        ("Debt/Equity", 3.0, "error"),
        ("Cash Conversion Cycle", -10.0, "success"),
    ],
)
def test_interpret_ratio_bands(name, value, expected_level):
    level, message = interpret_ratio(name, value)
    assert level == expected_level
    assert message


def test_interpret_ratio_handles_missing_values():
    assert interpret_ratio("Current Ratio", None) is None
    assert interpret_ratio("Current Ratio", np.nan) is None
    assert interpret_ratio("Not A Ratio", 1.0) is None
