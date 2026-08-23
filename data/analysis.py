"""Assembles market data and portfolio maths into one result object.

This is the seam between ``data`` (network) and ``core`` (pure maths). Views
call ``analyze_portfolio`` once and render the result, instead of each view
re-fetching and re-optimising the same numbers.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import PortfolioConfig
from core.optimize import (
    calculate_portfolio_metrics,
    get_optimized_weights,
    normalize_custom_weights,
)
from data.market import daily_returns, fetch_close_prices


@dataclass(frozen=True)
class PortfolioAnalysis:
    """One portfolio's prices, weights and metrics, ready to render."""

    config: PortfolioConfig
    prices: pd.DataFrame
    returns: pd.DataFrame
    tickers: Tuple[str, ...]
    missing: Tuple[str, ...]
    weights: np.ndarray
    metrics: Dict[str, float]
    # First date each holding actually traded, before the frame was aligned.
    first_trade_dates: Dict[str, date]

    @property
    def weight_by_ticker(self) -> Dict[str, float]:
        return dict(zip(self.tickers, self.weights))

    @property
    def start(self) -> date:
        """First date actually analysed, which can be later than requested."""
        return self.returns.index[0].date()

    @property
    def end(self) -> date:
        return self.returns.index[-1].date()

    @property
    def truncated_by(self) -> Optional[str]:
        """The holding that shortened the window, if one did.

        A stock that listed partway through the requested range forces every
        other holding to be measured from its first trading day — otherwise
        there'd be no covariance to compute. Worth telling the user, since the
        dates they picked are no longer the dates being analysed.

        Read from ``first_trade_dates``, captured *before* the price frame is
        aligned. After alignment every column shares a first date, so there's
        nothing left to tell them apart.
        """
        if (self.start - self.config.start).days <= 7:
            return None
        if not self.first_trade_dates:
            return None
        return max(self.first_trade_dates.items(), key=lambda kv: kv[1])[0]


def analyze_portfolio(config: PortfolioConfig) -> Optional[PortfolioAnalysis]:
    """Fetch prices, optimise weights and compute metrics for a config.

    Returns None when no ticker had usable price history.
    """
    prices = fetch_close_prices(config.tickers, config.start, config.end)
    if prices.empty:
        return None

    returns = daily_returns(prices)
    if returns.empty:
        return None

    # The returns frame defines the ticker order everything else must follow.
    tickers = tuple(returns.columns)
    missing = tuple(t for t in config.tickers if t not in tickers)

    if config.risk_level == "Custom" and config.custom_weights:
        weights = normalize_custom_weights(config.custom_weights, tickers)
    else:
        weights = get_optimized_weights(
            mean_returns=returns.mean() * 252,
            cov_matrix=returns.cov() * 252,
            risk_level=config.risk_level,
            risk_free_rate=config.risk_free_rate,
            max_weight=config.max_weight,
        )

    metrics = calculate_portfolio_metrics(returns, weights, config.risk_free_rate)

    # Capture listing dates before aligning, while the columns still differ.
    first_trade_dates = {}
    for ticker in tickers:
        first = prices[ticker].first_valid_index()
        if first is not None:
            first_trade_dates[ticker] = first.date()

    return PortfolioAnalysis(
        config=config,
        # Align prices to the returns index so every view measures the same
        # window. Left un-aligned, a late listing leaves NaNs at the start and
        # the benchmark chart renders blank.
        prices=prices.loc[returns.index, list(tickers)],
        returns=returns,
        tickers=tickers,
        missing=missing,
        weights=weights,
        metrics=metrics,
        first_trade_dates=first_trade_dates,
    )
