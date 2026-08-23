"""Portfolio analysis views: metrics, holdings, allocation, benchmark, charts."""

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st

from core.formatting import format_percent
from core.optimize import annualize_returns, annualize_volatility, portfolio_index
from core.timeframes import CHART_TIMEFRAMES
from data.analysis import PortfolioAnalysis
from data.market import (
    BENCHMARK_TICKER,
    fetch_annual_returns,
    fetch_close_prices,
    fetch_profiles,
    fetch_stock_history,
    fetch_timeframe_history,
)
from views.styles import figure_note

PALETTE = px.colors.qualitative.Set2 + px.colors.qualitative.Set3

TRANSPARENT_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def render_metrics(analysis: PortfolioAnalysis) -> None:
    """Headline portfolio metrics."""
    metrics = analysis.metrics
    if not metrics:
        st.warning("Not enough overlapping price history to compute portfolio metrics.")
        return

    st.markdown("### Portfolio metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Annual return",
            format_percent(metrics["annual_return"]),
            help="Arithmetic annualised return from daily data. This is the input "
                 "the Monte Carlo uses as its drift.",
        )
    with col2:
        st.metric(
            "CAGR",
            format_percent(metrics["cagr"]),
            help="Compound annual growth rate — what the portfolio actually "
                 "compounded at over the period.",
        )
    with col3:
        st.metric(
            "Annual volatility",
            format_percent(metrics["annual_vol"]),
            help="Annualised standard deviation of daily returns.",
        )
    with col4:
        st.metric(
            "Sharpe ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            help=f"(Annual return − risk-free rate) / volatility, at a "
                 f"{format_percent(metrics['risk_free_rate'])} risk-free rate. "
                 f"Adjust it in the sidebar.",
        )
    with col5:
        st.metric(
            "Max drawdown",
            format_percent(metrics["max_drawdown"]),
            help="Largest peak-to-trough fall over the period.",
        )


# --------------------------------------------------------------------------
# Holdings table
# --------------------------------------------------------------------------

def _holdings_frame(
    analysis: PortfolioAnalysis, profiles: Dict[str, Dict]
) -> pd.DataFrame:
    """Build the holdings table as numbers. Formatting happens at render time."""
    rows = []
    weights = analysis.weight_by_ticker

    for ticker in analysis.config.tickers:
        profile = profiles.get(ticker, {})
        row = {
            "Ticker": ticker,
            "Name": profile.get("shortName", ticker),
            "Weight": weights.get(ticker, 0.0),
            "Current price": np.nan,
            "Start price": np.nan,
            "Change": np.nan,
            "Expected return": np.nan,
            "Volatility": np.nan,
            "Market cap": pd.to_numeric(profile.get("marketCap"), errors="coerce"),
            "P/E": pd.to_numeric(profile.get("trailingPE"), errors="coerce"),
            "Beta": pd.to_numeric(profile.get("beta"), errors="coerce"),
        }

        if ticker in analysis.prices.columns:
            closes = analysis.prices[ticker].dropna()
            ticker_returns = analysis.returns[ticker]
            if len(closes) >= 2:
                row["Start price"] = float(closes.iloc[0])
                row["Current price"] = float(closes.iloc[-1])
                row["Change"] = float(closes.iloc[-1] / closes.iloc[0] - 1)
            row["Expected return"] = annualize_returns(ticker_returns)
            row["Volatility"] = annualize_volatility(ticker_returns)

        rows.append(row)

    return pd.DataFrame(rows)


def render_holdings(analysis: PortfolioAnalysis, profiles: Dict[str, Dict]) -> None:
    """Holdings table plus allocation and sector pie charts."""
    st.markdown("### Holdings")

    if analysis.missing:
        st.warning(
            f"No price history for {', '.join(analysis.missing)} over this date "
            f"range, so they're excluded from the optimisation and shown at 0%. "
            f"This is a data gap, not an allocation decision."
        )

    if analysis.truncated_by:
        st.info(
            f"{analysis.truncated_by} only has price history from "
            f"{analysis.start:%d %b %Y}, so the whole portfolio is measured from "
            f"then rather than {analysis.config.start:%d %b %Y}. Every figure "
            f"below covers {analysis.start:%d %b %Y} to {analysis.end:%d %b %Y}.",
            icon="ℹ️",
        )

    frame = _holdings_frame(analysis, profiles)

    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Weight": st.column_config.NumberColumn(
                format="percent", help="Share of the portfolio"
            ),
            "Current price": st.column_config.NumberColumn(format="$%.2f"),
            "Start price": st.column_config.NumberColumn(
                format="$%.2f", help="Close on the first trading day of the range"
            ),
            "Change": st.column_config.NumberColumn(
                format="percent", help="Total change over the period"
            ),
            "Expected return": st.column_config.NumberColumn(
                format="percent", help="Annualised return over the period"
            ),
            "Volatility": st.column_config.NumberColumn(
                format="percent", help="Annualised standard deviation"
            ),
            "Market cap": st.column_config.NumberColumn(format="compact"),
            "P/E": st.column_config.NumberColumn(format="%.1f"),
            "Beta": st.column_config.NumberColumn(
                format="%.2f", help="Versus the S&P 500. Estimated from two years "
                                    "of returns when the provider doesn't supply it."
            ),
        },
    )
    figure_note(
        "Columns are sortable. Expected return and volatility are annualised over "
        "the selected date range, not forward-looking forecasts."
    )

    allocated = frame[frame["Weight"] > 0]
    if allocated.empty:
        return

    col1, col2 = st.columns(2)
    with col1:
        _render_pie(
            values=allocated["Weight"] * 100,
            names=allocated["Ticker"],
            title="Allocation by holding",
        )
    with col2:
        sector_weights: Dict[str, float] = {}
        for ticker, weight in zip(allocated["Ticker"], allocated["Weight"]):
            sector = profiles.get(ticker, {}).get("sector") or "Unknown"
            sector_weights[sector] = sector_weights.get(sector, 0.0) + weight * 100
        _render_pie(
            values=list(sector_weights.values()),
            names=list(sector_weights.keys()),
            title="Allocation by sector",
        )


def _render_pie(values, names, title: str) -> None:
    fig = px.pie(
        values=values,
        names=names,
        title=title,
        hole=0.5,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Weight: %{value:.1f}%<extra></extra>",
        marker=dict(line=dict(color="rgba(128,128,128,0.35)", width=1)),
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        **TRANSPARENT_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# Benchmark comparison
# --------------------------------------------------------------------------

def render_benchmark(analysis: PortfolioAnalysis) -> None:
    """Portfolio versus the S&P 500, both normalised to 100 at the start."""
    st.markdown("### Portfolio vs. S&P 500")

    config = analysis.config
    benchmark = fetch_close_prices((BENCHMARK_TICKER,), config.start, config.end)
    if benchmark.empty:
        st.warning("Couldn't fetch S&P 500 data for this date range.")
        return

    # Weight each holding's growth multiple, not its share price — otherwise a
    # high-priced stock dominates the line regardless of its actual weight.
    portfolio_line = portfolio_index(analysis.prices, analysis.weights)
    if portfolio_line.empty:
        st.warning("Not enough overlapping price history to draw the comparison.")
        return

    # Both lines must start on the same day, or the comparison is meaningless.
    # The portfolio window can be shorter than requested when a holding listed
    # partway through the range.
    benchmark_close = benchmark[BENCHMARK_TICKER].reindex(
        portfolio_line.index, method="ffill"
    ).dropna()
    if benchmark_close.empty:
        st.warning("Couldn't align S&P 500 data with the portfolio's date range.")
        return
    benchmark_line = benchmark_close / benchmark_close.iloc[0] * 100

    tab_chart, tab_yearly = st.tabs(["Performance", "Yearly breakdown"])

    with tab_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_line.index, y=portfolio_line, mode="lines", name="Portfolio",
            line=dict(width=2.5),
            hovertemplate="Portfolio: %{y:.1f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=benchmark_line.index, y=benchmark_line, mode="lines", name="S&P 500",
            line=dict(width=2, dash="dash", color="rgba(128,128,128,0.9)"),
            hovertemplate="S&P 500: %{y:.1f}<extra></extra>",
        ))
        date_range = f"{analysis.start:%d %b %Y} to {analysis.end:%d %b %Y}"
        fig.update_layout(
            title=f"Growth of 100 · {date_range}",
            xaxis_title="Date",
            yaxis_title="Value (start = 100)",
            height=480,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **TRANSPARENT_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

        final_portfolio = float(portfolio_line.iloc[-1])
        final_benchmark = float(benchmark_line.iloc[-1])
        col1, col2 = st.columns(2)
        col1.metric("Portfolio", f"{final_portfolio:,.1f}",
                    delta=f"{final_portfolio - 100:+.1f} vs. start")
        col2.metric("S&P 500", f"{final_benchmark:,.1f}",
                    delta=f"{final_benchmark - final_portfolio:+.1f} vs. portfolio",
                    delta_color="inverse")

    with tab_yearly:
        annual = fetch_annual_returns(config.tickers, config.start, config.end)
        if annual.empty:
            st.info("Not enough history to break performance down by year.")
            return

        fig = px.bar(
            annual,
            x="Year", y="Annual Return (%)", color="Ticker",
            barmode="group",
            color_discrete_sequence=PALETTE,
            title="Calendar-year returns by holding",
        )
        fig.update_traces(marker_line_width=0, opacity=0.9)
        fig.update_layout(
            height=460,
            hovermode="x unified",
            xaxis=dict(showgrid=False, dtick=1),
            yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.5)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **TRANSPARENT_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)
        figure_note(
            "Each year's return is measured from the previous year-end close, so "
            "the first year of your range is included."
        )


# --------------------------------------------------------------------------
# Performance chart
# --------------------------------------------------------------------------

def render_performance_chart(tickers: List[str]) -> None:
    """Normalised and per-ticker price charts over a selectable timeframe."""
    st.markdown("### Performance chart")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        timeframe = st.selectbox(
            "Timeframe",
            options=CHART_TIMEFRAMES,
            index=CHART_TIMEFRAMES.index("1M"),
            key="chart_timeframe",
        )

    history = {}
    for ticker in tickers:
        hist = fetch_timeframe_history(ticker, timeframe)
        if hist is not None and not hist.empty:
            history[ticker] = hist

    if not history:
        st.warning(
            f"No price data available over the {timeframe} window. Intraday "
            f"timeframes have nothing to show before the first trades of a "
            f"session — try a longer window."
        )
        return

    tab_normalized, tab_price = st.tabs(["Normalised (%)", "Price ($)"])

    with tab_normalized:
        fig = go.Figure()
        for i, (ticker, hist) in enumerate(history.items()):
            normalized = (hist["Close"] / hist["Close"].iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(
                x=hist.index, y=normalized, mode="lines", name=ticker,
                line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
                hovertemplate=f"<b>{ticker}</b>: %{{y:+.2f}}%<extra></extra>",
            ))
        fig.update_layout(
            title=f"Relative performance · {timeframe}",
            xaxis_title="Date", yaxis_title="Change (%)",
            height=480, hovermode="x unified",
            yaxis=dict(zerolinecolor="rgba(128,128,128,0.5)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **TRANSPARENT_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_price:
        ticker_tabs = st.tabs(list(history.keys()))
        for i, (tab, (ticker, hist)) in enumerate(zip(ticker_tabs, history.items())):
            with tab:
                closes = hist["Close"]
                start_price = float(closes.iloc[0])
                end_price = float(closes.iloc[-1])
                change = end_price - start_price
                pct = change / start_price * 100 if start_price else 0.0

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist.index, y=closes, mode="lines", name=ticker,
                    line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
                    hovertemplate=f"<b>{ticker}</b>: $%{{y:.2f}}<extra></extra>",
                    showlegend=False,
                ))
                fig.update_layout(
                    title=(
                        f"{ticker} · {timeframe}<br>"
                        f"<sub>{start_price:,.2f} → {end_price:,.2f} "
                        f"({change:+,.2f}, {pct:+.2f}%)</sub>"
                    ),
                    xaxis_title="Date", yaxis_title="Price ($)",
                    height=480, hovermode="x",
                    **TRANSPARENT_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# Per-ticker detail
# --------------------------------------------------------------------------

def render_ticker_detail(ticker: str) -> None:
    """Company profile, recent price history and 52-week range for one ticker."""
    profile = fetch_profiles([ticker]).get(ticker, {})
    if profile.get("dataUnavailable"):
        st.error(f"Couldn't fetch company information for {ticker}.")
        return

    st.markdown("#### Company")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Name", profile.get("shortName", ticker))
        officers = profile.get("companyOfficers") or []
        ceo = officers[0].get("name", "N/A") if officers else "N/A"
        st.metric("CEO", ceo)
        employees = pd.to_numeric(profile.get("fullTimeEmployees"), errors="coerce")
        st.metric("Employees", "N/A" if pd.isna(employees) else f"{int(employees):,}")
    with col2:
        st.metric("Sector", profile.get("sector") or "N/A")
        st.metric("Industry", profile.get("industry") or "N/A")
        city, state = profile.get("city"), profile.get("state") or profile.get("country")
        st.metric("Headquarters", ", ".join(p for p in (city, state) if p) or "N/A")

    st.markdown("#### Price history")
    periods = {"1 month": "1mo", "6 months": "6mo", "1 year": "1y", "5 years": "5y"}
    rows = []
    for label, period in periods.items():
        hist = fetch_stock_history(ticker, period=period)
        if hist is None or hist.empty:
            continue
        closes = hist["Close"]
        rows.append({
            "Period": label,
            "Price then": float(closes.iloc[0]),
            "Price now": float(closes.iloc[-1]),
            "Change": float(closes.iloc[-1] / closes.iloc[0] - 1),
        })

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price then": st.column_config.NumberColumn(format="$%.2f"),
                "Price now": st.column_config.NumberColumn(format="$%.2f"),
                "Change": st.column_config.NumberColumn(format="percent"),
            },
        )
    else:
        st.info("No price history available.")

    st.markdown("#### 52-week range")
    high = pd.to_numeric(profile.get("fiftyTwoWeekHigh"), errors="coerce")
    low = pd.to_numeric(profile.get("fiftyTwoWeekLow"), errors="coerce")
    col1, col2 = st.columns(2)
    col1.metric("High", "N/A" if pd.isna(high) else f"${high:,.2f}")
    col2.metric("Low", "N/A" if pd.isna(low) else f"${low:,.2f}")
