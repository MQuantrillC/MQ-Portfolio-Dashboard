"""Market data access.

The only module that talks to yfinance. Every public function is cached, so
nothing here fires on a plain Streamlit rerun.

Failures are logged, not swallowed silently — a blank panel in the UI should be
traceable to a line in the server log.
"""

import logging
import time
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st
import yfinance as yf

from core.optimize import calculate_beta
from core.timeframes import history_requests

log = logging.getLogger(__name__)

QUOTE_TTL = 1800        # 30 minutes — intraday quotes
STATEMENT_TTL = 86400   # 24 hours — financial statements change quarterly
BENCHMARK_TICKER = "SPY"

_MEANINGFUL_INFO_FIELDS = (
    "currentPrice", "regularMarketPrice", "shortName", "longName",
    "marketCap", "beta", "sector", "industry",
)


def _backoff(attempt: int) -> None:
    """Exponential backoff between retries, in place of proxy rotation."""
    time.sleep(0.25 * (2 ** attempt))


def _looks_meaningful(info: object) -> bool:
    return (
        isinstance(info, dict)
        and len(info) > 5
        and any(info.get(key) not in (None, 0, "") for key in _MEANINGFUL_INFO_FIELDS)
    )


# --------------------------------------------------------------------------
# Price history
# --------------------------------------------------------------------------

@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def fetch_stock_history(
    ticker: str,
    period: Optional[str] = None,
    interval: str = "1d",
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    max_retries: int = 3,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV history for one ticker, retrying on transient failures."""
    start_str = start.strftime("%Y-%m-%d") if isinstance(start, (date, datetime)) else start
    end_str = end.strftime("%Y-%m-%d") if isinstance(end, (date, datetime)) else end

    if not (start_str and end_str) and not period:
        log.warning("fetch_stock_history(%s): needs either a period or start+end", ticker)
        return None

    for attempt in range(max_retries):
        try:
            ticker_obj = yf.Ticker(ticker)
            if start_str and end_str:
                hist = ticker_obj.history(
                    start=start_str, end=end_str, interval=interval, timeout=15
                )
            else:
                hist = ticker_obj.history(period=period, interval=interval, timeout=15)

            if hist is not None and not hist.empty:
                return hist
            log.info("fetch_stock_history(%s): empty result on attempt %d", ticker, attempt + 1)
        except Exception as exc:  # noqa: BLE001 - yfinance raises a wide variety
            log.warning("fetch_stock_history(%s) attempt %d failed: %s", ticker, attempt + 1, exc)

        if attempt < max_retries - 1:
            _backoff(attempt)

    return None


@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def fetch_close_prices(
    tickers: Tuple[str, ...],
    start: Union[date, datetime],
    end: Union[date, datetime],
) -> pd.DataFrame:
    """Closing prices for several tickers, aligned on a shared date index.

    Columns follow ``tickers`` order; tickers with no data are dropped. Callers
    must read the resulting column order rather than assuming it.
    """
    closes = {}
    for ticker in tickers:
        hist = fetch_stock_history(ticker, start=start, end=end)
        if hist is not None and not hist.empty:
            closes[ticker] = hist["Close"]

    if not closes:
        return pd.DataFrame()

    frame = pd.DataFrame(closes)
    ordered = [t for t in tickers if t in frame.columns]
    return frame[ordered].dropna(how="all")


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily percentage returns from a price frame."""
    if prices.empty:
        return pd.DataFrame()
    return prices.pct_change().dropna(how="all").dropna(axis=1, how="all").dropna()


@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def fetch_timeframe_history(ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
    """History for a UI timeframe, using yfinance periods rather than dates.

    Walks the fallback chain from ``history_requests`` until a request comes
    back with enough points to plot. Anything with fewer than two rows is
    treated as no data: one point can't show a change, and plotting it produces
    a flat line on a nonsensical axis.
    """
    for period, interval in history_requests(timeframe):
        hist = fetch_stock_history(ticker, period=period, interval=interval)
        if hist is not None and len(hist.get("Close", pd.Series(dtype="float64")).dropna()) >= 2:
            return hist
        log.info(
            "%s: period=%s interval=%s gave too few points for %s",
            ticker, period, interval, timeframe,
        )

    log.warning("no usable history for %s over %s", ticker, timeframe)
    return None


@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def get_price_and_change(ticker: str, timeframe: str) -> Optional[Tuple[float, float]]:
    """Latest price and percentage change over a timeframe."""
    hist = fetch_timeframe_history(ticker, timeframe)
    if hist is None or hist.empty:
        return None

    closes = hist["Close"].dropna()
    if len(closes) < 2:
        return None

    start_price = float(closes.iloc[0])
    end_price = float(closes.iloc[-1])
    if start_price == 0:
        return None

    return end_price, (end_price - start_price) / start_price * 100


# --------------------------------------------------------------------------
# Company profile
# --------------------------------------------------------------------------

def _fetch_raw_info(ticker: str, max_retries: int = 2) -> Optional[Dict]:
    """Try ``.info``, then ``.fast_info``, then reconstruct from recent history."""
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(ticker).info
            if _looks_meaningful(info):
                return dict(info)
        except Exception as exc:  # noqa: BLE001
            log.warning("info fetch for %s attempt %d failed: %s", ticker, attempt + 1, exc)
        if attempt < max_retries - 1:
            _backoff(attempt)

    try:
        fast = yf.Ticker(ticker).fast_info
        last_price = getattr(fast, "last_price", None)
        if last_price:
            return {
                "currentPrice": last_price,
                "regularMarketPrice": last_price,
                "marketCap": getattr(fast, "market_cap", None),
                "shortName": ticker,
                "longName": ticker,
                "quoteType": "EQUITY",
                "currency": getattr(fast, "currency", "USD"),
            }
    except Exception as exc:  # noqa: BLE001
        log.warning("fast_info fetch for %s failed: %s", ticker, exc)

    hist = fetch_stock_history(ticker, period="5d")
    if hist is not None and not hist.empty:
        closes = hist["Close"]
        info = {
            "currentPrice": float(closes.iloc[-1]),
            "regularMarketPrice": float(closes.iloc[-1]),
            "shortName": ticker,
            "longName": ticker,
            "quoteType": "EQUITY",
            "currency": "USD",
        }
        if len(closes) > 1:
            info["previousClose"] = float(closes.iloc[-2])
        return info

    log.error("all info strategies failed for %s", ticker)
    return None


def _first_present(data: Dict, keys: Tuple[str, ...], positive_only: bool = False):
    """First key in ``keys`` whose value is usable, or None."""
    for key in keys:
        value = data.get(key)
        if value in (None, "", 0):
            continue
        if positive_only:
            number = pd.to_numeric(value, errors="coerce")
            if pd.isna(number) or number <= 0:
                continue
        return value
    return None


def validate_stock_data(data: Optional[Dict], ticker: str) -> Dict:
    """Fill in canonical field names from yfinance's many aliases."""
    if not data:
        return {
            "shortName": ticker, "longName": ticker,
            "quoteType": "EQUITY", "currency": "USD",
            "dataUnavailable": True,
        }

    result = dict(data)
    aliases = {
        "currentPrice": ("currentPrice", "regularMarketPrice", "previousClose", "open"),
        "shortName": ("shortName", "longName", "displayName", "symbol"),
        "beta": ("beta", "betaThreeYear", "betaFiveYear", "beta3Year", "beta5Year"),
        "trailingPE": ("trailingPE", "forwardPE", "priceToEarningsTrailing12Months"),
        "marketCap": ("marketCap",),
        "sector": ("sector", "sectorKey", "sectorDisp"),
        "industry": ("industry", "industryKey", "industryDisp"),
        "fiftyTwoWeekHigh": ("fiftyTwoWeekHigh", "yearHigh"),
        "fiftyTwoWeekLow": ("fiftyTwoWeekLow", "yearLow"),
        "fullTimeEmployees": ("fullTimeEmployees", "employees"),
    }
    numeric_fields = {"currentPrice", "beta", "trailingPE", "marketCap",
                      "fiftyTwoWeekHigh", "fiftyTwoWeekLow"}

    for field, keys in aliases.items():
        if not result.get(field):
            value = _first_present(result, keys, positive_only=field in numeric_fields)
            if value is not None:
                result[field] = value

    if not result.get("shortName"):
        result["shortName"] = ticker
    if not result.get("marketCap"):
        shares = pd.to_numeric(result.get("sharesOutstanding"), errors="coerce")
        price = pd.to_numeric(result.get("currentPrice"), errors="coerce")
        if not pd.isna(shares) and not pd.isna(price) and shares > 0 and price > 0:
            result["marketCap"] = float(shares * price)

    result.setdefault("quoteType", "EQUITY")
    result.setdefault("currency", "USD")
    return result


@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def fetch_market_returns(period: str = "2y") -> Optional[pd.Series]:
    """Daily returns for the benchmark — cached once and reused for every beta."""
    hist = fetch_stock_history(BENCHMARK_TICKER, period=period)
    if hist is None or hist.empty:
        return None
    return hist["Close"].pct_change().dropna()


@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def fetch_beta(ticker: str) -> Optional[float]:
    """Beta against the benchmark, computed from two years of daily returns."""
    if ticker == BENCHMARK_TICKER:
        return 1.0

    market_returns = fetch_market_returns()
    stock_hist = fetch_stock_history(ticker, period="2y")
    if market_returns is None or stock_hist is None or stock_hist.empty:
        return None

    return calculate_beta(stock_hist["Close"].pct_change().dropna(), market_returns)


@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def fetch_stock_profile(ticker: str) -> Dict:
    """Validated, enriched company profile for one ticker.

    Cached, because it can make several network calls. Uncached, it re-fired on
    every widget interaction.
    """
    profile = validate_stock_data(_fetch_raw_info(ticker), ticker)

    if not profile.get("currentPrice") or not profile.get("fiftyTwoWeekHigh"):
        year_hist = fetch_stock_history(ticker, period="1y")
        if year_hist is not None and not year_hist.empty:
            profile.setdefault("currentPrice", float(year_hist["Close"].iloc[-1]))
            if not profile.get("fiftyTwoWeekHigh"):
                profile["fiftyTwoWeekHigh"] = float(year_hist["High"].max())
            if not profile.get("fiftyTwoWeekLow"):
                profile["fiftyTwoWeekLow"] = float(year_hist["Low"].min())

    if not profile.get("beta"):
        beta = fetch_beta(ticker)
        if beta is not None:
            profile["beta"] = beta
            profile["betaEstimated"] = True

    return profile


def fetch_profiles(tickers: List[str]) -> Dict[str, Dict]:
    """Profiles for several tickers.

    Sequential on purpose. The previous version ran three threads that each
    called ``yf.set_config()``, which mutates a process-wide singleton — workers
    overwrote each other's settings mid-request. Since every call here is cached,
    the sequential path only pays full cost on a cold cache.
    """
    return {ticker: fetch_stock_profile(ticker) for ticker in tickers}


# --------------------------------------------------------------------------
# Financial statements
# --------------------------------------------------------------------------

_STATEMENT_ATTRS = {
    ("balance", "Annual"): "balance_sheet",
    ("balance", "Quarterly"): "quarterly_balance_sheet",
    ("income", "Annual"): "income_stmt",
    ("income", "Quarterly"): "quarterly_income_stmt",
    ("cashflow", "Annual"): "cashflow",
    ("cashflow", "Quarterly"): "quarterly_cashflow",
}


@st.cache_data(ttl=STATEMENT_TTL, show_spinner=False)
def fetch_financial_statement(
    ticker: str,
    statement_type: str,
    period: str = "Annual",
    max_retries: int = 2,
) -> Optional[pd.DataFrame]:
    """Fetch one financial statement, dropping columns that are mostly empty."""
    attr = _STATEMENT_ATTRS.get((statement_type, period))
    if attr is None:
        log.error("unknown statement type %r / period %r", statement_type, period)
        return None

    for attempt in range(max_retries):
        try:
            data = getattr(yf.Ticker(ticker), attr)
            if data is not None and not data.empty and len(data.columns) > 0:
                filtered = data.loc[:, data.isna().mean() < 0.7]
                if not filtered.empty:
                    return filtered
            log.info("%s %s: empty statement on attempt %d", ticker, statement_type, attempt + 1)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s %s fetch attempt %d failed: %s", ticker, statement_type, attempt + 1, exc)

        if attempt < max_retries - 1:
            _backoff(attempt)

    return None


def fetch_all_statements(ticker: str, period: str = "Annual") -> Dict[str, Optional[pd.DataFrame]]:
    """Income statement, balance sheet and cash flow for one ticker."""
    return {
        name: fetch_financial_statement(ticker, name, period)
        for name in ("income", "balance", "cashflow")
    }


# --------------------------------------------------------------------------
# Annual returns
# --------------------------------------------------------------------------

@st.cache_data(ttl=QUOTE_TTL, show_spinner=False)
def fetch_annual_returns(
    tickers: Tuple[str, ...],
    start: Union[date, datetime],
    end: Union[date, datetime],
) -> pd.DataFrame:
    """Calendar-year returns per ticker, including the first year in the range.

    History is fetched from a year before ``start`` so the first requested year
    has a prior year-end close to compare against. Without that, the first year
    silently vanished from the chart.
    """
    extended_start = datetime(start.year - 1, 1, 1)
    rows = []

    for ticker in tickers:
        hist = fetch_stock_history(ticker, start=extended_start, end=end)
        if hist is None or hist.empty:
            continue

        year_end = hist["Close"].resample("YE").last().dropna()
        changes = year_end.pct_change().dropna() * 100
        for timestamp, change in changes.items():
            if start.year <= timestamp.year <= end.year:
                rows.append({
                    "Year": timestamp.year,
                    "Ticker": ticker,
                    "Annual Return (%)": float(change),
                })

    if not rows:
        return pd.DataFrame(columns=["Year", "Ticker", "Annual Return (%)"])
    return pd.DataFrame(rows).sort_values(["Year", "Ticker"]).reset_index(drop=True)
