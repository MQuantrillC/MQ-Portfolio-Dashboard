"""Tests for timeframe resolution and market-sentiment labelling."""

from datetime import date

import pytest

from core.formatting import (
    format_compact_currency,
    format_percent,
    format_ratio,
    format_value,
    remove_duplicates,
    safe_float,
)
from core.timeframes import (
    CHART_TIMEFRAMES,
    INTRADAY_TIMEFRAMES,
    MARKET_TIMEFRAMES,
    clamp_end_date,
    get_period_from_timeframe,
    get_sentiment,
    history_requests,
)


# --------------------------------------------------------------------------
# get_sentiment — the three-element tuple bug
# --------------------------------------------------------------------------

@pytest.mark.parametrize("timeframe", list(CHART_TIMEFRAMES) + ["unrecognised", ""])
def test_sentiment_always_returns_exactly_two_values(timeframe):
    """Callers unpack this as ``icon, label``.

    The original fallback branch read
    ``return "i", "Normal" if change >= 0 else "!", "Caution"``, which Python
    parses as a *three*-element tuple — so it raised ValueError. It only avoided
    firing because the timeframe list had been narrowed to the handled values.
    """
    icon, label = get_sentiment(0.0, timeframe)
    assert isinstance(icon, str) and isinstance(label, str)


def test_sentiment_is_bullish_above_the_threshold():
    assert get_sentiment(5.0, "1M")[1] == "Strong Rally"


def test_sentiment_is_bearish_below_the_threshold():
    assert get_sentiment(-5.0, "1M")[1] == "Down Month"


def test_sentiment_is_neutral_between_thresholds():
    assert get_sentiment(1.0, "1M")[1] == "Neutral"


def test_long_horizon_thresholds_are_both_positive():
    """Over 5 years, a +2% total return is underperformance, not neutrality."""
    assert get_sentiment(2.0, "5Y")[1] == "Underperforming"
    assert get_sentiment(50.0, "5Y")[1] == "Strong Growth"


# --------------------------------------------------------------------------
# Date ranges
# --------------------------------------------------------------------------

def test_every_ui_timeframe_maps_to_a_period():
    for timeframe in set(CHART_TIMEFRAMES) | set(MARKET_TIMEFRAMES):
        period, interval = get_period_from_timeframe(timeframe)
        assert period and interval


def test_history_requests_never_uses_a_calendar_range():
    """Fetches must be expressed as yfinance periods, not start/end dates.

    A one-day calendar range returns zero 1-minute bars over a weekend, and
    exactly one row once it retries at daily resolution — which rendered as a
    flat point on a microsecond-wide axis.
    """
    for timeframe in CHART_TIMEFRAMES:
        attempts = history_requests(timeframe)
        assert attempts, f"{timeframe} has no fetch strategy"
        for period, interval in attempts:
            assert isinstance(period, str) and period
            assert isinstance(interval, str) and interval


def test_first_attempt_is_the_preferred_resolution():
    assert history_requests("1D")[0] == ("1d", "1m")
    assert history_requests("YTD")[0] == ("ytd", "1d")


def test_intraday_timeframes_fall_back_to_daily_bars():
    """A market holiday can leave a single session empty."""
    for timeframe in INTRADAY_TIMEFRAMES:
        attempts = history_requests(timeframe)
        assert len(attempts) > 1
        assert attempts[-1][1] == "1d"


def test_daily_timeframes_do_not_add_a_redundant_retry():
    """1M is already daily; retrying the identical request wastes a round trip."""
    assert history_requests("1M") == (("1mo", "1d"),)


def test_history_requests_are_deduplicated():
    for timeframe in CHART_TIMEFRAMES:
        attempts = history_requests(timeframe)
        assert len(set(attempts)) == len(attempts)


def test_unknown_timeframe_still_returns_a_usable_request():
    assert history_requests("nonsense") == (("1mo", "1d"),)


def test_clamp_end_date_never_returns_the_future():
    today = date(2026, 8, 22)
    assert clamp_end_date(date(2026, 12, 31), today) == today
    assert clamp_end_date(date(2024, 6, 1), today) == date(2024, 6, 1)


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def test_format_value_handles_missing_input():
    assert format_value(None) == "N/A"
    assert format_value(float("nan")) == "N/A"
    assert format_value("not a number") == "N/A"


def test_format_value_adds_thousands_separators():
    assert format_value(1234567.891) == "$1,234,567.89"


def test_compact_currency_scales_by_magnitude():
    assert format_compact_currency(3_400_000_000_000) == "$3.40T"
    assert format_compact_currency(45_600_000_000) == "$45.60B"
    assert format_compact_currency(789_000_000) == "$789.00M"
    assert format_compact_currency(512) == "$512.00"


def test_format_percent_signs_and_precision():
    assert format_percent(0.0432) == "4.32%"
    assert format_percent(0.0432, signed=True) == "+4.32%"
    assert format_percent(None) == "N/A"


def test_format_ratio_by_kind():
    assert format_ratio(1.234, "decimal") == "1.23"
    assert format_ratio(0.421, "percentage") == "42.1%"
    assert format_ratio(45.6, "days") == "45.6 days"
    assert format_ratio(None, "decimal") == "N/A"


def test_remove_duplicates_preserves_order():
    assert remove_duplicates(["B", "A", "B", "C", "A"]) == ["B", "A", "C"]


def test_safe_float_coerces_or_returns_none():
    assert safe_float("3.5") == 3.5
    assert safe_float(None) is None
    assert safe_float("abc") is None
