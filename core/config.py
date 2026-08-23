"""Configuration objects passed explicitly between layers.

Nothing here touches Streamlit, the network, or global state. Views build a
``PortfolioConfig`` from the widgets and hand it down, so the analysis functions
never read session state themselves.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Tuple

# Risk-free rate used for the Sharpe ratio when the live short rate is
# unavailable. Overridable from the sidebar.
DEFAULT_RISK_FREE_RATE = 0.04

# Trading days per year, used to annualise daily returns.
TRADING_DAYS = 252

RISK_LEVELS = ("Low", "Moderate", "High", "Custom")

RISK_DESCRIPTIONS = {
    "Low": "Minimise portfolio volatility",
    "Moderate": "Maximise Sharpe ratio",
    "High": "Maximise return, capped per position",
    "Custom": "Set your own weights",
}

RISK_STRATEGY_LABELS = {
    "Low": "Low risk — minimising portfolio volatility",
    "Moderate": "Moderate risk — maximising Sharpe ratio",
    "High": "High risk — maximising expected return under a position cap",
    "Custom": "Custom — manual weight selection",
}


@dataclass(frozen=True)
class PortfolioConfig:
    """Everything the analysis layer needs to describe one portfolio."""

    tickers: Tuple[str, ...]
    risk_level: str
    start: date
    end: date
    custom_weights: Optional[Dict[str, float]] = None
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    max_weight: float = 0.40

    @property
    def strategy_label(self) -> str:
        return RISK_STRATEGY_LABELS.get(self.risk_level, self.risk_level)

    def cache_key(self) -> tuple:
        """Hashable identity, so views can tell one portfolio from another."""
        weights = tuple(sorted((self.custom_weights or {}).items()))
        return (
            self.tickers,
            self.risk_level,
            self.start,
            self.end,
            weights,
            self.risk_free_rate,
            self.max_weight,
        )
