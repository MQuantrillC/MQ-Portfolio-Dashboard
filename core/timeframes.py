"""Timeframe handling and market-sentiment labelling.

Pure functions: a timeframe string in, dates or labels out.
"""

from datetime import date
from typing import Optional, Tuple

CHART_TIMEFRAMES = ("1D", "5D", "1M", "6M", "1Y", "YTD", "5Y", "10Y")
MARKET_TIMEFRAMES = ("5D", "1M", "6M", "1Y", "YTD", "5Y", "10Y")

# Timeframes whose default interval is finer than a day.
INTRADAY_TIMEFRAMES = ("1D", "5D")

# timeframe -> (yfinance period, yfinance interval)
_PERIOD_MAP = {
    "1D": ("1d", "1m"),
    "5D": ("5d", "5m"),
    "1M": ("1mo", "1d"),
    "6M": ("6mo", "1d"),
    "1Y": ("1y", "1d"),
    "YTD": ("ytd", "1d"),
    "5Y": ("5y", "1wk"),
    "10Y": ("10y", "1mo"),
}

TIMEFRAME_LABELS = {
    "1D": "1-Day",
    "5D": "5-Day",
    "1M": "1-Month",
    "6M": "6-Month",
    "1Y": "1-Year",
    "YTD": "Year-to-Date",
    "5Y": "5-Year",
    "10Y": "10-Year",
}

# timeframe -> (bullish threshold %, bearish threshold %, bull label, bear label, neutral label)
_SENTIMENT_BANDS = {
    "1D": (1.0, -1.0, "Up Day", "Down Day", "Flat"),
    "5D": (1.5, -1.5, "Bullish Week", "Bearish Week", "Sideways"),
    "1M": (3.0, -3.0, "Strong Rally", "Down Month", "Neutral"),
    "6M": (10.0, -10.0, "Strong Rally", "Downtrend", "Moderate"),
    "YTD": (15.0, -15.0, "Strong YTD", "Weak YTD", "Neutral YTD"),
    "1Y": (8.0, -8.0, "Strong Year", "Bearish Year", "Flat Year"),
    "5Y": (35.0, 5.0, "Strong Growth", "Underperforming", "Moderate"),
    "10Y": (80.0, 10.0, "Strong Decade", "Weak Decade", "Flat Decade"),
}


def get_period_from_timeframe(timeframe: str) -> Tuple[str, str]:
    """Map a UI timeframe to a yfinance (period, interval) pair."""
    return _PERIOD_MAP.get(timeframe, ("1mo", "1d"))


def history_requests(timeframe: str) -> Tuple[Tuple[str, str], ...]:
    """Ordered (period, interval) attempts for fetching a timeframe's history.

    Always expressed as a yfinance *period*, never a calendar start/end range.
    A period resolves to trading sessions; a date range doesn't. Asking for
    one-minute bars between yesterday and today returns nothing over a weekend,
    and the daily-interval retry returns exactly one row — which renders as a
    flat, single-point chart on a microsecond-wide axis.

    Later entries are progressively coarser fallbacks, used when a market
    holiday or a data gap leaves the first attempt with too few points to plot.
    """
    period, interval = get_period_from_timeframe(timeframe)

    attempts = [(period, interval)]
    if interval != "1d":
        attempts.append((period, "1d"))
    if timeframe in INTRADAY_TIMEFRAMES:
        # A long weekend can empty a single session entirely.
        attempts.append(("1mo", "1d"))

    return tuple(dict.fromkeys(attempts))


def clamp_end_date(end: date, today: Optional[date] = None) -> date:
    """Never ask for data from the future.

    Selecting the current year yields 31 December, which is months away and makes
    chart titles read as though they cover time that hasn't happened.
    """
    today = today or date.today()
    return min(end, today)


def get_sentiment(change_pct: float, timeframe: str) -> Tuple[str, str]:
    """Return an (icon, label) pair describing performance over a timeframe.

    Always a 2-tuple — callers unpack it as one.
    """
    bands = _SENTIMENT_BANDS.get(timeframe)
    if bands is None:
        return ("🟢", "Up") if change_pct >= 0 else ("🔴", "Down")

    bullish, bearish, bull_label, bear_label, neutral_label = bands
    if change_pct >= bullish:
        return "🟢", bull_label
    if change_pct <= bearish:
        return "🔴", bear_label
    return "🟡", neutral_label


def sentiment_tooltip(timeframe: str) -> str:
    """Human-readable description of the thresholds behind ``get_sentiment``."""
    bands = _SENTIMENT_BANDS.get(timeframe)
    if bands is None:
        return "Performance over the selected period."

    bullish, bearish, bull_label, bear_label, neutral_label = bands
    label = TIMEFRAME_LABELS.get(timeframe, timeframe)
    return (
        f"S&P 500 (SPY) {label} performance\n"
        f"🟢 {bull_label}: {bullish:+.1f}% or more\n"
        f"🔴 {bear_label}: {bearish:+.1f}% or less\n"
        f"🟡 {neutral_label}: between the two"
    )
