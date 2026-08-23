"""Market overview: indices, commodities, FX, crypto and sector performance."""

from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from core.timeframes import (
    MARKET_TIMEFRAMES,
    TIMEFRAME_LABELS,
    get_sentiment,
    sentiment_tooltip,
)
from data.market import get_price_and_change
from views.styles import figure_note

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
}

COMMODITIES = {
    "Gold": "GC=F",
    "Crude oil (WTI)": "CL=F",
    "Silver": "SI=F",
    "Natural gas": "NG=F",
    "Brent crude": "BZ=F",
    "Copper": "HG=F",
}

FOREX = {
    "EUR/USD": ("EURUSD=X", "US dollars per euro"),
    "USD/JPY": ("JPY=X", "Japanese yen per US dollar"),
    "GBP/USD": ("GBPUSD=X", "US dollars per pound"),
    "USD/CAD": ("CAD=X", "Canadian dollars per US dollar"),
    "AUD/USD": ("AUDUSD=X", "US dollars per Australian dollar"),
    "USD index": ("DX-Y.NYB", "Dollar strength against a basket of currencies. "
                              "Above 100 is a strong dollar."),
}

CRYPTO = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}

VIX_BANDS = (
    (15, "🟢 Low volatility — market confidence"),
    (25, "🟡 Moderate volatility — typical movement"),
    (None, "🔴 High volatility — fear and uncertainty"),
)

YIELD_BANDS = (
    (3, "🟢 Accommodative — cheaper borrowing, growth-friendly"),
    (4, "🟡 Neutral — balanced policy outlook"),
    (None, "🔴 Restrictive — higher borrowing costs"),
)


def _band_label(value: float, bands) -> str:
    for upper, label in bands:
        if upper is None or value < upper:
            return label
    return ""


def _quote_metric(
    label: str,
    result: Optional[Tuple[float, float]],
    help_text: str = "",
    prefix: str = "$",
    decimals: int = 2,
) -> None:
    """Render one quote.

    ``st.metric``'s delta handles the up/down arrow and colour, so it follows the
    viewer's theme and doesn't rely on colour alone to carry meaning.
    """
    if result is None:
        st.metric(label, "N/A", help=help_text)
        st.caption("Data unavailable")
        return

    price, change = result
    st.metric(
        label,
        f"{prefix}{price:,.{decimals}f}",
        delta=f"{change:+.2f}%",
        help=help_text or None,
    )


def _render_quote_grid(assets: Dict[str, str], timeframe: str, prefix: str = "$") -> None:
    columns = st.columns(len(assets))
    for column, (name, ticker) in zip(columns, assets.items()):
        with column:
            _quote_metric(name, get_price_and_change(ticker, timeframe), prefix=prefix)


def _render_key_indicators(timeframe: str) -> None:
    st.markdown("#### Key indicators")
    col1, col2, col3 = st.columns(3)

    with col1:
        spy = get_price_and_change("SPY", timeframe)
        _quote_metric("S&P 500 (SPY)", spy, help_text=sentiment_tooltip(timeframe))
        if spy is not None:
            icon, label = get_sentiment(spy[1], timeframe)
            st.caption(f"{icon} {label}")

    with col2:
        vix = get_price_and_change("^VIX", timeframe)
        _quote_metric(
            "VIX volatility index", vix, prefix="",
            help_text="Expected 30-day volatility of the S&P 500. Below 15 is calm, "
                      "above 25 is fearful.",
        )
        if vix is not None:
            st.caption(_band_label(vix[0], VIX_BANDS))

    with col3:
        tnx = get_price_and_change("^TNX", timeframe)
        _quote_metric(
            "10-year Treasury yield", tnx, prefix="",
            help_text="The benchmark US government borrowing rate. Higher yields "
                      "generally pressure equity valuations.",
        )
        if tnx is not None:
            st.caption(_band_label(tnx[0], YIELD_BANDS))


def _render_forex(timeframe: str) -> None:
    columns = st.columns(len(FOREX))
    for column, (name, (ticker, description)) in zip(columns, FOREX.items()):
        with column:
            result = get_price_and_change(ticker, timeframe)
            decimals = 2 if "index" in name.lower() else 4
            _quote_metric(name, result, help_text=description, prefix="", decimals=decimals)


def _render_sectors(timeframe: str) -> None:
    st.markdown("#### Sector performance")

    rows = []
    for etf, sector in SECTOR_ETFS.items():
        result = get_price_and_change(etf, timeframe)
        if result is not None:
            rows.append({"Sector": sector, "Change": result[1] / 100, "ETF": etf})

    if not rows:
        st.warning("No sector data available for this timeframe.")
        return

    frame = pd.DataFrame(rows).sort_values("Change", ascending=False)
    label = TIMEFRAME_LABELS.get(timeframe, timeframe)

    fig = px.bar(
        frame,
        x="Sector",
        y="Change",
        color="Change",
        color_continuous_scale=["#D64038", "#B9BEC7", "#2EA057"],
        color_continuous_midpoint=0,
        text="Change",
        hover_data={"ETF": True, "Change": ":.2%"},
    )
    fig.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
        marker_line_width=0,
    )
    fig.update_layout(
        height=460,
        xaxis_title=None,
        yaxis_title=f"{label} change",
        yaxis=dict(tickformat=".0%", zerolinecolor="rgba(128,128,128,0.5)"),
        xaxis=dict(tickangle=-40),
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=90),
    )
    st.plotly_chart(fig, use_container_width=True)
    figure_note(
        "Based on SPDR sector ETFs: "
        + ", ".join(f"{etf} ({sector})" for etf, sector in SECTOR_ETFS.items())
        + "."
    )


def render_market_overview() -> None:
    """The full market overview panel."""
    st.markdown("### Market overview")

    timeframe = st.selectbox(
        "Timeframe",
        options=MARKET_TIMEFRAMES,
        index=MARKET_TIMEFRAMES.index("1M"),
        key="market_overview_timeframe",
    )

    with st.spinner("Loading market data…"):
        _render_key_indicators(timeframe)

        st.markdown("#### Commodities, FX and crypto")
        tab_commodities, tab_forex, tab_crypto = st.tabs(["Commodities", "FX", "Crypto"])
        with tab_commodities:
            _render_quote_grid(COMMODITIES, timeframe)
        with tab_forex:
            _render_forex(timeframe)
        with tab_crypto:
            _render_quote_grid(CRYPTO, timeframe)

        _render_sectors(timeframe)
