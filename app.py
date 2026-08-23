"""MQ Portfolio Dashboard — application entry point.

Routing only. Data access lives in ``data/``, the maths in ``core/`` and the
rendering in ``views/``.
"""

import logging
import os

import streamlit as st

from data.analysis import analyze_portfolio
from data.market import fetch_profiles
from views.financials import render_financials
from views.market import render_market_overview
from views.montecarlo import render_monte_carlo
from views.portfolio import (
    render_benchmark,
    render_holdings,
    render_metrics,
    render_performance_chart,
    render_ticker_detail,
)
from views.setup import render_setup, render_sidebar
from views.styles import apply_page_style, render_footer

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

def _lazy_section(key: str, label: str, render, *args) -> None:
    """Render a section only once the user asks for it.

    ``st.tabs`` executes every tab's body on every rerun, even the hidden ones.
    That meant opening the page fetched three financial statements per ticker
    plus two dozen market quotes before anyone clicked anything. Gating each
    heavy section behind a button keeps the first paint fast.
    """
    state_key = f"loaded__{key}"
    if not st.session_state.get(state_key):
        if st.button(label, key=f"load__{key}", use_container_width=True):
            st.session_state[state_key] = True
            st.rerun()
        return
    render(*args)


def main() -> None:
    st.set_page_config(
        page_title="MQ Portfolio Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_page_style()

    st.title("📈 MQ Portfolio Dashboard")
    st.caption(
        "Mean-variance portfolio optimisation, benchmarking and fundamental "
        "analysis across the S&P 500."
    )

    _render_dashboard()
    render_footer()


def _render_dashboard() -> None:
    """The dashboard body.

    Separate from ``main`` so its early returns skip the rest of the analysis
    without also skipping the footer.
    """
    risk_free_rate, max_weight = render_sidebar()
    config = render_setup(risk_free_rate, max_weight)
    if config is None:
        return

    st.markdown("---")
    st.caption(
        f"**{', '.join(config.tickers)}** · {config.strategy_label} · "
        f"{config.start:%d %b %Y} to {config.end:%d %b %Y}"
    )

    with st.spinner("Fetching prices and optimising…"):
        analysis = analyze_portfolio(config)

    if analysis is None:
        st.error(
            "No price history was returned for any of the selected stocks over "
            "this date range. Try a different range, or check the server log for "
            "the underlying fetch error."
        )
        return

    with st.spinner("Fetching company profiles…"):
        profiles = fetch_profiles(list(config.tickers))

    render_metrics(analysis)
    render_holdings(analysis, profiles)
    render_benchmark(analysis)
    render_performance_chart(list(analysis.tickers))

    st.markdown("---")
    st.markdown("### Further analysis")
    st.caption("Each section loads its own data on demand.")

    tab_tickers, tab_financials, tab_monte_carlo, tab_market = st.tabs(
        ["Company detail", "Financials", "Monte Carlo", "Market overview"]
    )

    with tab_tickers:
        _lazy_section("tickers", "Load company detail", _render_ticker_tabs, analysis.tickers)
    with tab_financials:
        _lazy_section("financials", "Load financial statements", render_financials, analysis.tickers)
    with tab_monte_carlo:
        _lazy_section("monte_carlo", "Load Monte Carlo", render_monte_carlo, analysis)
    with tab_market:
        _lazy_section("market", "Load market overview", render_market_overview)


def _render_ticker_tabs(tickers) -> None:
    tabs = st.tabs(list(tickers))
    for tab, ticker in zip(tabs, tickers):
        with tab:
            render_ticker_detail(ticker)


if __name__ == "__main__":
    main()
