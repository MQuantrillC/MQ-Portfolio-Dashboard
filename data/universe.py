"""The selectable stock universe.

Scrapes the S&P 500 constituent list from Wikipedia, and reports honestly when
it can't. The old version swallowed the failure and quietly served a 50-name
fallback while the UI still said "S&P 500".
"""

import logging
from typing import Dict, NamedTuple

import pandas as pd
import requests
import streamlit as st

log = logging.getLogger(__name__)

UNIVERSE_TTL = 86400  # 24 hours
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Used only when the scrape fails. The UI says so when this is what's on screen.
FALLBACK_UNIVERSE: Dict[str, str] = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.", "NVDA": "NVIDIA Corporation", "TSLA": "Tesla Inc.",
    "META": "Meta Platforms Inc.", "BRK-B": "Berkshire Hathaway Inc.",
    "UNH": "UnitedHealth Group Inc.", "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase & Co.", "V": "Visa Inc.", "PG": "Procter & Gamble Co.",
    "XOM": "Exxon Mobil Corporation", "HD": "Home Depot Inc.",
    "CVX": "Chevron Corporation", "MA": "Mastercard Inc.", "PFE": "Pfizer Inc.",
    "ABBV": "AbbVie Inc.", "BAC": "Bank of America Corp.", "KO": "Coca-Cola Co.",
    "AVGO": "Broadcom Inc.", "PEP": "PepsiCo Inc.",
    "TMO": "Thermo Fisher Scientific Inc.", "COST": "Costco Wholesale Corp.",
    "DIS": "Walt Disney Co.", "ABT": "Abbott Laboratories", "WMT": "Walmart Inc.",
    "CRM": "Salesforce Inc.", "MRK": "Merck & Co. Inc.", "NFLX": "Netflix Inc.",
    "ADBE": "Adobe Inc.", "ACN": "Accenture Plc", "NKE": "Nike Inc.",
    "LLY": "Eli Lilly and Co.", "DHR": "Danaher Corporation",
    "TXN": "Texas Instruments Inc.", "NEE": "NextEra Energy Inc.",
    "VZ": "Verizon Communications Inc.", "BMY": "Bristol-Myers Squibb Co.",
    "QCOM": "QUALCOMM Inc.", "PM": "Philip Morris International Inc.",
    "T": "AT&T Inc.", "UPS": "United Parcel Service Inc.",
    "RTX": "RTX Corporation", "SCHW": "Charles Schwab Corp.",
    "HON": "Honeywell International Inc.", "LOW": "Lowe's Companies Inc.",
    "AMD": "Advanced Micro Devices Inc.", "AMGN": "Amgen Inc.",
}


class Universe(NamedTuple):
    """The tradable list, plus whether it came from the live source."""

    tickers: Dict[str, str]
    is_fallback: bool
    reason: str = ""


def _parse_constituents(tables: list) -> Dict[str, str]:
    """Pull {ticker: company name} out of the Wikipedia constituents table."""
    if not tables:
        return {}

    df = tables[0]
    ticker_col = next((c for c in df.columns if "symbol" in str(c).lower()), None)
    name_col = next(
        (c for c in df.columns if "security" in str(c).lower() or "company" in str(c).lower()),
        None,
    )
    if ticker_col is None or name_col is None:
        return {}

    constituents = {}
    for ticker, name in zip(df[ticker_col], df[name_col]):
        ticker = str(ticker).strip().upper()
        name = str(name).strip()
        if ticker and ticker != "NAN" and len(ticker) <= 6 and name and name.lower() != "nan":
            constituents[ticker] = name
    return constituents


@st.cache_data(ttl=UNIVERSE_TTL, show_spinner=False)
def fetch_sp500_universe() -> Universe:
    """Fetch the S&P 500 constituents, falling back visibly if that fails."""
    try:
        response = requests.get(
            WIKIPEDIA_URL, timeout=10, headers={"User-Agent": USER_AGENT}
        )
        response.raise_for_status()
        constituents = _parse_constituents(pd.read_html(response.content))
        if constituents:
            return Universe(constituents, is_fallback=False)
        reason = "the constituents table wasn't in the expected format"
        log.warning("S&P 500 scrape returned no usable rows")
    except ImportError as exc:
        # pandas.read_html needs lxml (or html5lib + bs4). Missing it used to
        # look identical to a network failure.
        reason = f"an HTML parser is missing ({exc})"
        log.error("read_html dependency missing: %s", exc)
    except requests.RequestException as exc:
        reason = "Wikipedia couldn't be reached"
        log.warning("S&P 500 scrape network failure: %s", exc)
    except (ValueError, KeyError) as exc:
        reason = "the constituents table couldn't be parsed"
        log.warning("S&P 500 scrape parse failure: %s", exc)

    return Universe(dict(FALLBACK_UNIVERSE), is_fallback=True, reason=reason)
