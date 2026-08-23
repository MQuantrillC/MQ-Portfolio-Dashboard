"""The portfolio setup form.

Collects widget input and returns a ``PortfolioConfig``. Nothing downstream
reads session state, so no view has to save-and-restore it around a call.
"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import streamlit as st

from core.config import (
    DEFAULT_RISK_FREE_RATE,
    PortfolioConfig,
    RISK_DESCRIPTIONS,
    RISK_LEVELS,
)
from core.formatting import remove_duplicates
from core.timeframes import clamp_end_date
from data.universe import fetch_sp500_universe

EARLIEST_START = date(2000, 1, 1)
DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA"]
MIN_HISTORY_DAYS = 30


def render_sidebar() -> Tuple[float, float]:
    """Sidebar assumptions. Returns (risk-free rate, max position weight)."""
    st.sidebar.markdown("### Assumptions")

    risk_free_rate = st.sidebar.number_input(
        "Risk-free rate (%)",
        min_value=0.0, max_value=20.0,
        value=DEFAULT_RISK_FREE_RATE * 100, step=0.25, format="%.2f",
        help="Used in the Sharpe ratio and when maximising risk-adjusted return. "
             "Roughly the 3-month Treasury yield.",
    ) / 100

    max_weight = st.sidebar.slider(
        "Maximum position size (%)",
        min_value=10, max_value=100, value=40, step=5,
        help="Caps any single holding. Without a cap, 'High risk' collapses to "
             "100% in whichever stock returned the most.",
    ) / 100

    return risk_free_rate, max_weight


def _select_tickers() -> List[str]:
    universe = fetch_sp500_universe()

    if universe.is_fallback:
        st.warning(
            f"Couldn't load the live S&P 500 constituent list — {universe.reason}. "
            f"Showing a fallback list of {len(universe.tickers)} major constituents."
        )

    display_by_ticker = {t: f"{t} — {name}" for t, name in universe.tickers.items()}
    ticker_by_display = {v: k for k, v in display_by_ticker.items()}

    defaults = [
        display_by_ticker[t] for t in DEFAULT_TICKERS if t in display_by_ticker
    ]

    selected = st.multiselect(
        "Stocks",
        options=list(display_by_ticker.values()),
        default=defaults,
        help="Pick two or more holdings to optimise across.",
    )
    return remove_duplicates([ticker_by_display[option] for option in selected])


def _select_risk_level() -> str:
    return st.radio(
        "Strategy",
        options=RISK_LEVELS,
        index=RISK_LEVELS.index("Moderate"),
        horizontal=True,
        format_func=lambda level: f"{level} — {RISK_DESCRIPTIONS[level]}",
        key="risk_level",
    )


def _custom_weights(tickers: List[str]) -> Optional[Dict[str, float]]:
    """Weight inputs for the Custom strategy.

    Returns None when the weights don't sum to 100%. The caller skips the
    analysis rather than calling ``st.stop()``, which used to blank the whole
    page — including the portfolio the user was already reading.
    """
    st.markdown("##### Custom weights")

    equal_weight = 100.0 / len(tickers)
    weights: Dict[str, float] = {}
    columns = st.columns(min(len(tickers), 4))

    for i, ticker in enumerate(tickers):
        with columns[i % len(columns)]:
            weights[ticker] = st.number_input(
                f"{ticker} (%)",
                min_value=0.0, max_value=100.0,
                value=equal_weight, step=1.0, format="%.1f",
                key=f"weight_{ticker}",
            )

    total = sum(weights.values())
    if abs(total - 100.0) > 0.01:
        st.warning(
            f"Weights currently total {total:.1f}%. Adjust them to 100% to run the "
            f"analysis — everything else on the page still works."
        )
        return None

    st.success("Weights total 100%.")
    return weights


def _select_dates() -> Tuple[date, date]:
    today = date.today()
    col1, col2 = st.columns(2)

    with col1:
        start = st.date_input(
            "Start date",
            value=date(today.year - 7, 1, 1),
            min_value=EARLIEST_START,
            max_value=today - timedelta(days=MIN_HISTORY_DAYS),
            format="YYYY-MM-DD",
        )
    with col2:
        end = st.date_input(
            "End date",
            value=today,
            min_value=EARLIEST_START + timedelta(days=MIN_HISTORY_DAYS),
            max_value=today,
            format="YYYY-MM-DD",
            help="Capped at today — the dashboard can't analyse dates that "
                 "haven't happened yet.",
        )

    return start, clamp_end_date(end, today)


def render_setup(risk_free_rate: float, max_weight: float) -> Optional[PortfolioConfig]:
    """Render the setup form and return a config, or None if it isn't ready."""
    st.markdown("### Portfolio setup")

    tickers = _select_tickers()
    risk_level = _select_risk_level()

    if not tickers:
        st.info("Select at least one stock to get started.")
        return None

    custom_weights = None
    if risk_level == "Custom":
        custom_weights = _custom_weights(tickers)
        if custom_weights is None:
            return None

    start, end = _select_dates()

    if (end - start).days < MIN_HISTORY_DAYS:
        st.warning(
            f"The date range needs to be at least {MIN_HISTORY_DAYS} days to "
            f"produce meaningful statistics."
        )
        return None

    if len(tickers) < 2 and risk_level != "Custom":
        st.info(
            "Optimisation across a single holding always returns 100%. Add another "
            "stock to see a real allocation."
        )

    return PortfolioConfig(
        tickers=tuple(tickers),
        risk_level=risk_level,
        start=start,
        end=end,
        custom_weights=custom_weights,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
    )
